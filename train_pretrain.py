import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
import math
import time
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.optim import AdamW
from torch.utils.data import DataLoader
from contextlib import nullcontext
from model import LMConfig, SLM
from dataset import PretrainParquetIterableDataset

class Trainer:
    def __init__(self, args):
        self.args = args
        self.setup_distributed()
        self.init_device()
        self.init_model()
        self.load_data()
        self.setup_optimizer()
        self.scaler = torch.cuda.amp.GradScaler(enabled=args.dtype != 'float32')
        self.global_step = 0  # 跟踪全局步数

    def setup_distributed(self):
        self.ddp = int(os.environ.get("RANK", -1)) != -1
        if self.ddp:
            dist.init_process_group(backend="nccl")
            self.ddp_rank = int(os.environ["RANK"])
            self.ddp_local_rank = int(os.environ["LOCAL_RANK"])
            self.ddp_world_size = int(os.environ["WORLD_SIZE"])
            torch.cuda.set_device(self.ddp_local_rank)
            self.device = f"cuda:{self.ddp_local_rank}"
        else:
            self.device = self.args.device

    def init_device(self):
        self.device_type = "cuda" if "cuda" in self.device else "cpu"
        self.ctx = nullcontext() if self.device_type == "cpu" else torch.cuda.amp.autocast()

    def init_model(self):
        config = LMConfig(
            dim=self.args.dim,
            n_layers=self.args.n_layers,
            max_seq_len=self.args.max_seq_len
        )
        print(config)
        self.model = SLM(config).to(self.device)
        if self.ddp:
            self.model = DistributedDataParallel(
                self.model,
                device_ids=[self.ddp_local_rank],
                output_device=self.ddp_local_rank
            )
        self.logger(f"Total parameters: {self.count_parameters() / 1e6:.2f}M")

    def load_data(self):
        dataset = PretrainParquetIterableDataset(
            parquet_folder=self.args.data_dir,
            max_length=self.args.max_seq_len
        )
        sampler = None
        if self.ddp:
            sampler = torch.utils.data.distributed.DistributedSampler(dataset)
        self.train_loader = DataLoader(
            dataset,
            batch_size=self.args.batch_size,
            pin_memory=True,
            num_workers=self.args.num_workers,
            sampler=sampler
        )

    def setup_optimizer(self):
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=self.args.learning_rate,
            weight_decay=self.args.weight_decay
        )
        # 使用固定的大T_max，或根据epochs和预估的steps_per_epoch设置
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=self.args.epochs * 1000  # 假设每epoch大约1000步
        )

    def count_parameters(self):
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)

    def logger(self, message):
        if not self.ddp or self.ddp_rank == 0:
            print(message)

    def train_epoch(self, epoch):
        self.model.train()
        start_time = time.time()
        
        for step, (X, Y, mask) in enumerate(self.train_loader):
            self.global_step += 1
            X, Y, mask = X.to(self.device), Y.to(self.device), mask.to(self.device)
            lr = self.scheduler.get_last_lr()[0]

            with self.ctx:
                outputs = self.model(X)
                loss = self.compute_loss(outputs.logits, Y, mask)
                
            self.scaler.scale(loss).backward()
            
            if (step + 1) % self.args.accumulation_steps == 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.grad_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True)
                self.scheduler.step()

            if self.global_step % self.args.log_interval == 0:
                self.logger(
                    f'Epoch {epoch+1}/{self.args.epochs} | '
                    f'Step {self.global_step} | '
                    f'Loss {loss.item():.3f} | '
                    f'LR {lr:.2e} | '
                    f'Time {time.time() - start_time:.2f}s'
                )

            if (self.global_step % self.args.save_interval == 0) and (not self.ddp or self.ddp_rank == 0):
                self.save_checkpoint(epoch, self.global_step)

    def compute_loss(self, logits, targets, mask):
        loss_fct = nn.CrossEntropyLoss(reduction='none')
        loss = loss_fct(
            logits.view(-1, logits.size(-1)),
            targets.view(-1)
        ).view(targets.size())
        return (loss * mask).sum() / mask.sum()

    def save_checkpoint(self, epoch, step):
        checkpoint = {
            'model': self.model.module.state_dict() if self.ddp else self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'epoch': epoch,
            'step': step
        }
        torch.save(checkpoint, f"{self.args.out_dir}/checkpoint_{epoch}_{step}.pt")

    def train(self):
        for epoch in range(self.args.epochs):
            self.train_epoch(epoch)
            if not self.ddp or self.ddp_rank == 0:
                self.save_checkpoint(epoch, self.global_step)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="./parquet_parts")
    parser.add_argument("--out_dir", type=str, default="./output")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--accumulation_steps", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=5e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--max_seq_len", type=int, default=2048)
    parser.add_argument("--dim", type=int, default=768)
    parser.add_argument("--n_layers", type=int, default=4)
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--save_interval", type=int, default=1000)
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()

    trainer = Trainer(args)
    trainer.train()