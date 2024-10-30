import os
import math
import random
import torch
from tqdm import tqdm
import numpy as np
import torch.distributed as dist
from PIL import Image
from time import time
import matplotlib.pyplot as plt
from torchvision.utils import save_image

from diffusion.gaussian_diffusion import _extract_into_tensor
import diffusion.gaussian_diffusion as gd
from diffusion.respace import space_timesteps

from qwerty.utils_qdit import init_data

def create_npz_from_sample_folder(sample_dir, num=50_000):
    """
    Builds a single .npz file from a folder of .png samples.
    """
    samples = []
    for i in tqdm(range(num), desc="Building .npz file from samples"):
        sample_pil = Image.open(f"{sample_dir}/{i:06d}.png")
        sample_np = np.asarray(sample_pil).astype(np.uint8)
        samples.append(sample_np)
    samples = np.stack(samples)
    assert samples.shape == (num, samples.shape[1], samples.shape[2], 3)
    npz_path = f"{sample_dir}.npz"
    np.savez(npz_path, arr_0=samples)
    print(f"Saved .npz file to {npz_path} [shape={samples.shape}].")
    return npz_path


def sample(args, model_pq, vae, diffusion, sample_folder_dir):
    seed = dist.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    n = args.batch_size
    # n = args.per_proc_batch_size
    global_batch_size = n * dist.get_world_size()
    rank = dist.get_rank()
    device = rank % torch.cuda.device_count()
    latent_size = args.image_size // 8
    using_cfg = args.cfg_scale > 1.0
    # if rank == 0:
    os.makedirs(sample_folder_dir, exist_ok=True)
    print(f"Saving .png samples at {sample_folder_dir}")

    # To make things evenly-divisible, we'll sample a bit more than we need and then discard the extra samples:
    total_samples = int(math.ceil(args.num_fid_samples / global_batch_size) * global_batch_size)
    if rank == 0:
        print(f"Total number of images that will be sampled: {total_samples}")
    assert total_samples % dist.get_world_size() == 0, "total_samples must be divisible by world_size"
    samples_needed_this_gpu = int(total_samples // dist.get_world_size())
    assert samples_needed_this_gpu % n == 0, "samples_needed_this_gpu must be divisible by the per-GPU batch size"
    iterations = int(samples_needed_this_gpu // n)
    pbar = range(iterations)
    pbar = tqdm(pbar) if rank == 0 else pbar
    total = 0
    for _ in pbar:
        # Sample inputs:
        z = torch.randn(n, model_pq.in_channels, latent_size, latent_size, device=device)
        y = torch.randint(0, args.num_classes, (n,), device=device)

        # Setup classifier-free guidance:
        if using_cfg:
            z = torch.cat([z, z], 0)
            y_null = torch.tensor([1000] * n, device=device)
            y = torch.cat([y, y_null], 0)
            model_kwargs = dict(y=y, cfg_scale=args.cfg_scale)
            sample_fn = model_pq
            # sample_fn = model_pq.forward_with_cfg
        else:
            model_kwargs = dict(y=y)
            sample_fn = model_pq
            # sample_fn = model_pq.forward

        # Sample images:
        samples = diffusion.p_sample_loop(
            sample_fn, z.shape, z, clip_denoised=False, model_kwargs=model_kwargs, progress=False, device=device
        )
        if using_cfg:
            samples, _ = samples.chunk(2, dim=0)  # Remove null class samples

        samples = vae.decode(samples / 0.18215).sample
        samples = torch.clamp(127.5 * samples + 128.0, 0, 255).permute(0, 2, 3, 1).to("cpu", dtype=torch.uint8).numpy()

        # Save samples to disk as individual .png files
        for i, sample in enumerate(samples):
            index = i * dist.get_world_size() + rank + total
            Image.fromarray(sample).save(f"{sample_folder_dir}/{index:06d}.png")
        total += global_batch_size

    # Make sure all processes have finished saving their samples before attempting to convert to .npz
    dist.barrier()
    if rank == 0:
        create_npz_from_sample_folder(sample_folder_dir, args.num_fid_samples)
        print("Done.")
    dist.barrier()
    dist.destroy_process_group()


class dit_generator:
    def __init__(self, timestep_respacing, latent_size, device):
        # create_diffusion
        betas = gd.get_named_beta_schedule('linear', 1000)
        use_timesteps=space_timesteps(1000, timestep_respacing)

        # SpacedDiffusion
        self.use_timesteps = set(use_timesteps)
        self.timestep_map = []
        self.original_num_steps = len(betas)
        last_alpha_cumprod = 1.0
        new_betas = []
        self.set_alpha(betas)
        for i, alpha_cumprod in enumerate(self.alphas_cumprod):
            if i in self.use_timesteps:
                new_betas.append(1 - alpha_cumprod / last_alpha_cumprod)
                last_alpha_cumprod = alpha_cumprod
                self.timestep_map.append(i)
        self.set_alpha(new_betas)
        betas = np.array(new_betas, dtype=np.float64)

        # GaussianDiffusion
        alphas = 1.0 - betas
        self.alphas_cumprod = np.cumprod(alphas, axis=0)
        self.alphas_cumprod_prev = np.append(1.0, self.alphas_cumprod[:-1])
        self.posterior_variance = (betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod))
        self.posterior_log_variance_clipped = np.log(
            np.append(self.posterior_variance[1], self.posterior_variance[1:])
        ) if len(self.posterior_variance) > 1 else np.array([])
        self.sqrt_recip_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod - 1)
        self.posterior_mean_coef1 = (
            betas * np.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
            (1.0 - self.alphas_cumprod_prev) * np.sqrt(alphas) / (1.0 - self.alphas_cumprod)
        )
        self.num_timesteps = int(betas.shape[0])
        self.latent_size = latent_size
        self.device = device

    def set_alpha(self, betas):
        # GaussianDiffusion
        betas = np.array(betas, dtype=np.float64)
        self.betas = betas
        alphas = 1.0 - betas
        self.alphas_cumprod = np.cumprod(alphas, axis=0)

    def forward_val(self, vae, model, model_pq, cfg=False, name="sample_pq", save=True, args=None, logger=None):
        # sample
        class_labels = [207, 360, 387, 974, 88, 979, 417, 279]
        z, model_kwargs = self.pre_process(class_labels, cfg=cfg, args=args)

        # p_sample_loop_progressive
        img = z
        img_pq = z
        indices = list(range(self.num_timesteps))[::-1]
        indices_tqdm = tqdm(indices)
        mse_loss = []
        for i in indices_tqdm:
            t = torch.tensor([i] * z.shape[0], device=self.device)
            with torch.no_grad():
                # SpacedDiffusion
                map_tensor = torch.tensor(self.timestep_map, device=t.device, dtype=t.dtype)
                new_ts = map_tensor[t]
                # p_mean_variance
                model_output = model(img, new_ts, **model_kwargs)
                model_output_pq = model_pq(img_pq, new_ts, **model_kwargs)

                # model_output, feat = model(img, new_ts, **model_kwargs, distill=True)
                # model_output_pq, feat_pq = model_pq(img_pq, new_ts, **model_kwargs, distill=True)

                img, img_pq = self.post_process(t, img, img_pq, model_output, model_output_pq)

                mse_loss.append(torch.mean((model_output - model_output_pq) ** 2).cpu().numpy())

                if save:
                    with open('mse_and_mav.csv', 'a') as file:
                        file.write(f"{i},{torch.mean((model_output - model_output_pq) ** 2)},{model_output.abs().mean()}\n")                    

        samples = img    
        samples, _ = samples.chunk(2, dim=0)  # Remove null class samples
        samples = vae.decode(samples / 0.18215).sample

        samples_pq = img_pq
        samples_pq, _ = samples_pq.chunk(2, dim=0)  # Remove null class samples
        samples_pq = vae.decode(samples_pq / 0.18215).sample

        if save:
            plt.figure(figsize=(10, 5))
            plt.plot(indices[::-1], mse_loss, label='MSE Loss')
            plt.xlabel('Time Step')
            plt.ylabel('MSE Loss')
            plt.title('MSE Loss Over Time Steps')
            plt.legend()
            plt.grid(True)
            plt.savefig('mse_loss_over_time_steps.png')
                    
            save_image(samples, name+'.png', nrow=4, normalize=True, value_range=(-1, 1))
            save_image(samples_pq, name+'_compress.png', nrow=4, normalize=True, value_range=(-1, 1))
            logger.info(f"Original saved as {name}.png, compressed saved as {name}_compress.png")

    def pre_process(self, class_labels, cfg=False, args=None):
        n = len(class_labels)
        z = torch.randn(n, 4, self.latent_size, self.latent_size, device=self.device)
        y = torch.tensor(class_labels, device=self.device)
        if cfg:
            z = torch.cat([z, z], 0)
            y_null = torch.tensor([1000] * n, device=self.device)
            y = torch.cat([y, y_null], 0)
            model_kwargs = dict(y=y, cfg_scale=args.cfg_scale)
        else:
            model_kwargs = dict(y=y)
        return z, model_kwargs


    def post_process(self, t, img, img_pq, model_output, model_output_pq):
        # p_mean_variance
        B, C = img.shape[:2]
        model_output, model_var_values = torch.split(model_output, C, dim=1)
        min_log = _extract_into_tensor(self.posterior_log_variance_clipped, t, img.shape)
        max_log = _extract_into_tensor(np.log(self.betas), t, img.shape)
        frac = (model_var_values + 1) / 2
        model_log_variance = frac * max_log + (1 - frac) * min_log
        pred_xstart = self._predict_xstart_from_eps(x_t=img, t=t, eps=model_output)
        model_mean, _, _ = self.q_posterior_mean_variance(x_start=pred_xstart, x_t=img, t=t)

        model_output_pq, model_var_values_pq = torch.split(model_output_pq, C, dim=1)
        frac_pq = (model_var_values_pq + 1) / 2
        model_log_variance_pq = frac_pq * max_log + (1 - frac_pq) * min_log
        pred_xstart_pq = self._predict_xstart_from_eps(x_t=img_pq, t=t, eps=model_output_pq)
        model_mean_pq, _, _ = self.q_posterior_mean_variance(x_start=pred_xstart_pq, x_t=img_pq, t=t)

        # p_sample
        noise = torch.randn_like(img)
        nonzero_mask = (
            (t != 0).float().view(-1, *([1] * (len(img.shape) - 1)))
        )  # no noise when t == 0
        sample = model_mean + nonzero_mask * torch.exp(0.5 * model_log_variance) * noise
        sample_pq = model_mean_pq + nonzero_mask * torch.exp(0.5 * model_log_variance_pq) * noise

        return sample, sample_pq
    

    def _predict_xstart_from_eps(self, x_t, t, eps):
        return (
            _extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * eps
        )
    

    def q_posterior_mean_variance(self, x_start, x_t, t):
        posterior_mean = (
            _extract_into_tensor(self.posterior_mean_coef1, t, x_t.shape) * x_start
            + _extract_into_tensor(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = _extract_into_tensor(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = _extract_into_tensor(
            self.posterior_log_variance_clipped, t, x_t.shape
        )
        return posterior_mean, posterior_variance, posterior_log_variance_clipped
    

def train(args, logger, model, vae, diffusion, checkpoint_dir):
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0)
    rank = dist.get_rank()
    device = rank % torch.cuda.device_count()
    loader, sampler = init_data(args, rank, logger)

    model.train()  # important! This enables embedding dropout for classifier-free guidance
    train_steps = 0
    log_steps = 0
    running_loss = 0
    start_time = time()

    logger.info(f"Training for {args.epochs} epochs...")
    for epoch in range(args.epochs):
        sampler.set_epoch(epoch)
        logger.info(f"Beginning epoch {epoch}...")
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            with torch.no_grad():
                # Map input images to latent space + normalize latents:
                x = vae.encode(x).latent_dist.sample().mul_(0.18215)
            t = torch.randint(0, diffusion.num_timesteps, (x.shape[0],), device=device)
            model_kwargs = dict(y=y)
            # loss_dict = diffusion.training_losses_distill(model_pq, model, x, t, model_kwargs)
            loss_dict = diffusion.training_losses(model, x, t, model_kwargs)
            loss = loss_dict["loss"].mean()
            opt.zero_grad()
            loss.backward()
            opt.step()

            # Log loss values:
            running_loss += loss.item()
            log_steps += 1
            train_steps += 1
            if train_steps % args.log_every == 0:
                # Measure training speed:
                torch.cuda.synchronize()
                end_time = time()
                steps_per_sec = log_steps / (end_time - start_time)
                # Reduce loss history over all processes:
                avg_loss = torch.tensor(running_loss / log_steps, device=device)
                dist.all_reduce(avg_loss, op=dist.ReduceOp.SUM)
                avg_loss = avg_loss.item() / dist.get_world_size()
                logger.info(f"(step={train_steps:07d}) Train Loss: {avg_loss:.4f}, Train Steps/Sec: {steps_per_sec:.2f}")
                # Reset monitoring variables:
                running_loss = 0
                log_steps = 0
                start_time = time()

            # Save DiT checkpoint:
            if train_steps % args.ckpt_every == 0 and train_steps > 0:
                if rank == 0:
                    checkpoint = {
                        "model": model.module.state_dict(),
                        "opt": opt.state_dict(),
                        "args": args
                    }
                    checkpoint_path = f"{checkpoint_dir}/{train_steps:07d}.pt"
                    torch.save(checkpoint, checkpoint_path)
                    logger.info(f"Saved checkpoint to {checkpoint_path}")
                dist.barrier()

    model.eval()  # important! This disables randomized embedding dropout
    # do any sampling/FID calculation/etc. with ema (or model) in eval mode ...

    logger.info("Done!")