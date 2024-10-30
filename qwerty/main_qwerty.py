import os
import torch
import torch.distributed as dist
from collections import defaultdict
from copy import deepcopy
from tqdm import tqdm

import sys
sys.path.append('/mnt/petrelfs/shaojie/code/Q-DiT')
from qdit.modelutils import quantize_model, quantize_model_gptq,  add_act_quant_wrapper
from qdit.datautils import get_loader
from qwerty.distributed import init_distributed_mode
from qwerty.utils_qdit import parse_option, init_data, init_env, init_model, save_ckpt
from qwerty.utils_qwerty import FeatureDataset, lienar_regression, CompensationBlock
from qwerty.utils_traineval import dit_generator, sample

LINEAR_COMPENSATION_SAMPLES = 4096
# LINEAR_COMPENSATION_SAMPLES = 512

@torch.no_grad()
def generate_compensation_model(args):
    init_distributed_mode(args)
    rank, device, logger, experiment_dir = init_env(args)
    checkpoint_dir = f"{experiment_dir}/checkpoints" 
    model, state_dict, diffusion, vae = init_model(args, device)
    loader, sampler = init_data(args, rank, logger)

    model.eval()

    args.weight_group_size = eval(args.weight_group_size)
    args.act_group_size = eval(args.act_group_size)
    args.weight_group_size = [args.weight_group_size] * len(model.blocks) if isinstance(args.weight_group_size, int) else args.weight_group_size
    args.act_group_size = [args.act_group_size] * len(model.blocks) if isinstance(args.act_group_size, int) else args.act_group_size

    scales = defaultdict(lambda: None)
    q_model = add_act_quant_wrapper(deepcopy(model), device=device, args=args, scales=scales)
    if args.qwerty_ckpt:
        gptq_dataloader = get_loader(args.calib_data_path, nsamples=1)
    else:
        gptq_dataloader = get_loader(args.calib_data_path, nsamples=256)
    q_model = quantize_model_gptq(q_model, device=device, args=args, dataloader=gptq_dataloader)


    if args.qwerty_ckpt:
        for block_id in range(len(q_model.blocks)):
            W = torch.zeros((1152, 1152))
            b = torch.zeros(1152)
            q_model.blocks[block_id] = CompensationBlock(W=W, b=b, r2_score=-1, block=q_model.blocks[block_id], linear_init=True if block_id >= args.start_block else False, local_rank=rank, block_id=block_id, logger=logger)
        q_model.load_state_dict(torch.load(args.qwerty_ckpt)['model'])
        q_model.cuda()
    else:
        output_x = torch.zeros(size=[0,], device=device)
        output_c = torch.zeros(size=[0,], device=device)

        # logger.info(f"Training for {args.epochs} epochs...")
        sampler.set_epoch(0)
        for i, (x, y) in tqdm(enumerate(loader)):
            x = x.to(device)
            y = y.to(device)
            with torch.no_grad():
                x = vae.encode(x).latent_dist.sample().mul_(0.18215)
            # t = torch.randint(0, 1, (x.shape[0],), device=device)
            t = torch.randint(0, diffusion.num_timesteps, (x.shape[0],), device=device)

            noise = torch.randn_like(x)
            x_t = diffusion.q_sample(x, t, noise=noise)

            x_t = model.x_embedder(x_t) + model.pos_embed
            t = model.t_embedder(t)
            y = model.y_embedder(y, False)
            c = t + y

            output_x = torch.cat([output_x, x_t.detach()], dim=0)
            output_c = torch.cat([output_c, c.detach()], dim=0)

            if i >= (LINEAR_COMPENSATION_SAMPLES // args.batch_size // args.world_size - 1):
                break
        
        feature_set = FeatureDataset(output_x.detach().cpu(), output_c.detach().cpu())
        feature_loader = torch.utils.data.DataLoader(feature_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

        output_x_previous = output_x
        output_c_previous = output_c

        for block_id in range(len(q_model.blocks)):

            feature_set.X = output_x_previous.detach().cpu()
            feature_set.Y = output_c_previous.detach().cpu()

            block_q = q_model.blocks[block_id]
            block_origin = model.blocks[block_id]
            output_full_precision = torch.zeros(size=[0, ], device=device)
            output_quant = torch.zeros(size=[0, ], device=device)
            output_t_ = torch.zeros(size=[0, ], device=device)

            for i, (x_out, c_out) in tqdm(enumerate(feature_loader)):
                x_out = x_out.cuda()
                c_out = c_out.cuda()

                full_precision_out = block_origin(x_out, c_out)
                quant_out = block_q(x_out, c_out)

                output_t_ = torch.cat([output_t_, x_out.detach()], dim=0)
                output_full_precision = torch.cat([output_full_precision, full_precision_out.detach()], dim=0)
                output_quant = torch.cat([output_quant, quant_out.detach()], dim=0)

                torch.cuda.synchronize()
                if i >= (LINEAR_COMPENSATION_SAMPLES // args.batch_size  // args.world_size - 1):
                    break

            assert torch.sum((output_x_previous - output_t_).abs()) < 1e-3
            
            W, b, r2_score = lienar_regression(output_t_, output_full_precision - output_quant, block_id=block_id, logger=logger)

            q_model.blocks[block_id] = CompensationBlock(W=W, b=b, r2_score=r2_score, block=q_model.blocks[block_id], linear_init=True if block_id >= args.start_block else False, local_rank=rank, block_id=block_id, logger=logger)
            q_model.cuda()
            qwerty_block = q_model.blocks[block_id]

            output_x_previous = torch.zeros(size=[0, ], device=device)
            quant_error = []
            qwerty_error = []
            for i, (x_out, c_out) in tqdm(enumerate(feature_loader)):
                x_out = x_out.cuda()
                c_out = c_out.cuda()

                previous_out = qwerty_block(x_out, c_out)
                output_x_previous = torch.cat([output_x_previous, previous_out.detach()], dim=0)

                quant_out = block_q(x_out, c_out)
                full_precision_out = block_origin(x_out, c_out)
                quant_error.append((quant_out - full_precision_out).abs().mean())
                qwerty_error.append((previous_out - full_precision_out).abs().mean())

                torch.cuda.synchronize()
                if i >= (LINEAR_COMPENSATION_SAMPLES // args.batch_size // args.world_size - 1):
                    break

            quant_error = torch.cat(quant_error).mean()
            qwerty_error = torch.cat(qwerty_error).mean()
            logger.info(f"Quantization error: {quant_error.item():.4f}; Qwerty error (L1 distance): {qwerty_error.item():.4f}")
        save_ckpt(q_model, args, checkpoint_dir, logger)
    
    mode = args.mode
    if mode == "sample":
        q_model.eval()
        model_string_name = args.model.replace("/", "-")
        ckpt_string_name = os.path.basename(args.ckpt).replace(".pt", "") if args.ckpt else "pretrained"
        folder_name = f"{model_string_name}-{ckpt_string_name}-size-{args.image_size}-vae-{args.vae}-" \
                    f"cfg-{args.cfg_scale}-seed-{args.global_seed}"
        sample_folder_dir = f"{experiment_dir}/{folder_name}"
        sample(args, q_model, vae, diffusion, sample_folder_dir)
    
    elif mode == "gen":
        q_model.eval()
        latent_size = args.image_size // 8
        diffusion_gen = dit_generator('250', latent_size=latent_size, device=device)
        diffusion_gen.forward_val(vae, model.forward, q_model.forward, cfg=False, name=f"{experiment_dir}/gen", logger=logger)
    
    logger.info("Done!")
    return q_model


if __name__ == "__main__":
    args = parse_option()
    generate_compensation_model(args)