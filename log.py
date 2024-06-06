import os
import wandb
import torch
import torchvision
from torch_utils import distributed as dist

class Logger:
    def __init__(self, project_name, exp_name, tags, run_dir, device=torch.device('cuda')):
        self.device = device

        if dist.get_rank() == 0:
            wandb.init(sync_tensorboard=False,
                    project=project_name,
                    name = exp_name,
                    tags = tags,
                    job_type="CleanRepo",
                    dir = run_dir
            )
    
    def log_iter(self, cur_iter, loss, lr):
        wandb.log({"train/loss": loss.mean()}, step=cur_iter)
        wandb.log({"train/lr": lr}, step=cur_iter)

    def log_eval(self, cur_iter, name, value):
        wandb.log({f'eval/{name}': value}, step=cur_iter)