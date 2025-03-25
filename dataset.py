import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import IterableDataset, DataLoader
from glob import glob
import math
import gc

class PretrainParquetIterableDataset(IterableDataset):
    def __init__(self, parquet_folder, max_length=2048, files_per_batch=2):
        super().__init__()
        self.parquet_folder = parquet_folder
        self.max_length = max_length
        self.files_per_batch = files_per_batch
        self.parquet_files = sorted(glob(os.path.join(parquet_folder, "*.parquet")))

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:  # 单进程处理
            files = self.parquet_files
        else:  # 多进程分配文件
            per_worker = math.ceil(len(self.parquet_files) / worker_info.num_workers)
            worker_id = worker_info.id
            start = worker_id * per_worker
            end = min(start + per_worker, len(self.parquet_files))
            files = self.parquet_files[start:end]

        # 分批次加载文件
        for i in range(0, len(files), self.files_per_batch):
            batch_files = files[i:i+self.files_per_batch]
            df = self.load_batch(batch_files)
            for _, row in df.iterrows():
                input_ids = np.array(row['input_ids'][:self.max_length])
                loss_mask = np.array(row['attention_mask'][:self.max_length])
                X = input_ids[:-1].astype(np.int64)
                Y = input_ids[1:].astype(np.int64)
                mask = loss_mask[1:].astype(np.int64)
                yield torch.from_numpy(X), torch.from_numpy(Y), torch.from_numpy(mask)
            del df  # 释放当前批次内存
            gc.collect()

    def load_batch(self, batch_files):
        """加载并合并多个Parquet文件"""
        dfs = []
        for file in batch_files:
            df = pd.read_parquet(file)
            dfs.append(df)
        return pd.concat(dfs, ignore_index=True)

# 使用示例
if __name__ == "__main__":
    dataset = PretrainParquetIterableDataset(
        parquet_folder="./parquet_parts",
        files_per_batch=2
    )
    print(len(dataset))
    dataloader = DataLoader(dataset, batch_size=8, num_workers=0)

    for epoch in range(3):
        for X_batch, Y_batch, mask_batch in dataloader:
            # 训练代码
            print(f"Batch shapes: {X_batch.shape}, {Y_batch.shape}, {mask_batch.shape}")
            # 模拟训练步骤
            break  # 仅用于测试