import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import Dataset
import torch.distributed as dist

def gather_tensor_from_multi_processes(input, world_size):
    if world_size == 1:
        return input
    torch.cuda.synchronize()
    gathered_tensors = [torch.zeros_like(input) for _ in range(world_size)]
    dist.all_gather(gathered_tensors, input)
    gathered_tensors = torch.cat(gathered_tensors, dim=0)
    torch.cuda.synchronize()

    return gathered_tensors


class FeatureDataset(Dataset):
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, item):
        return self.X[item], self.Y[item]
    

class CompensationBlock(nn.Module):
    def __init__(self, W, b, r2_score, block, linear_init=True, local_rank=0, block_id=None, logger=None):
        super(CompensationBlock, self).__init__()
        self.block = block

        self.lora_weight = nn.Parameter(torch.zeros((W.size(0), W.size(1))))
        self.lora_bias = nn.Parameter(torch.zeros(W.size(1)))

        if linear_init and (r2_score > 0):
            self.lora_weight.data.copy_(W)
            self.lora_bias.data.copy_(b)
            if local_rank == 0:
                logger.info('block {} using linear init'.format(block_id))
        else:
            nn.init.zeros_(self.lora_weight)
            nn.init.zeros_(self.lora_bias)
            if local_rank == 0:
                logger.info('block {} using lora init'.format(block_id))

    def forward(self, x, c):
        out = self.block(x, c)
        out = out + x @ self.lora_weight + self.lora_bias

        return out
    

def lienar_regression(X, Y, block_id=0, logger=None):
    X = X.reshape(-1, X.size(-1))

    X_add_one = torch.cat([X, torch.ones(size=[X.size(0), ], device=X.device).reshape(-1, 1)], dim=-1)
    Y = Y.reshape(-1, Y.size(-1))

    logger.info('the shape of X_add_one is {}, Y is {}'.format(X_add_one.size(), Y.size()))
    X_add_one_T = X_add_one.t()
    # W_overall = torch.inverse(X_add_one_T @ X_add_one) @ X_add_one_T @ Y 
    # W_overall = torch.pinverse(X_add_one_T @ X_add_one) @ X_add_one_T @ Y 
    W_overall = torch.linalg.lstsq(X_add_one.cpu(), Y.cpu()).solution.cuda()

    W = W_overall[:-1, :]
    b = W_overall[-1, :]
    Y_pred = X @ W + b
    abs_loss = (Y - Y_pred).abs().mean()

    ss_tot = torch.sum((Y - Y.mean(dim=0)).pow(2))
    ss_res = torch.sum((Y - Y_pred).pow(2))
    r2_score = 1 - ss_res / ss_tot

    logger.info('block : {}      abs : {:.6f}      r2 : {:.3f}'.format(block_id, abs_loss, r2_score))
    return W, b, r2_score

