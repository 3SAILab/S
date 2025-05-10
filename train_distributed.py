import torch
import torch.distributed as dist
from torch.utils.data import Dataset
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
import os
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.nn.utils import clip_grad_norm_
from train_utils import plot_loss, plot_lr, print_model_parameters
from config import config
from custom_llama_model import CustomLlamaModel, CustomLlamaConfig

class PretrainDataset(Dataset):
    def __init__(self, data_path, max_seq_len):
        self.data = np.memmap(data_path, dtype=np.uint32, mode='r')
        self.max_seq_len = max_seq_len
        # 计算可以形成的完整序列的数量
        self.num_sequences = len(self.data) // self.max_seq_len  # 修改计算方式

    def __len__(self):
        return self.num_sequences

    def __getitem__(self, idx):
        start_idx = idx * self.max_seq_len
        # 输入序列
        input_seq = torch.from_numpy(self.data[start_idx : start_idx + self.max_seq_len].astype(np.int64))
        # 目标序列 (下一个token预测)
        target_seq = torch.from_numpy(self.data[start_idx + 1 : start_idx + self.max_seq_len + 1].astype(np.int64))
        return input_seq, target_seq

def setup_ddp(rank, world_size, target_gpu_id):
    # Note: MASTER_ADDR and MASTER_PORT are environment variables that need to be set.
    # For local training on a single machine, 'localhost' and a free port are common.
    # Ensure 'localhost' and '12355' are appropriate for your setup.
    # If running in a cluster, these would be set by the cluster management system
    # or need to be configured to point to the rank 0 process.
    os.environ['MASTER_ADDR'] = os.environ.get('MASTER_ADDR', 'localhost')
    os.environ['MASTER_PORT'] = os.environ.get('MASTER_PORT', '12355')
    
    # Initialize the process group
    # backend: 'nccl' is recommended for NVIDIA GPUs, 'gloo' for CPUs
    # Adjust backend if necessary, e.g., if not using NCCL-enabled GPUs.
    backend = 'nccl' if torch.cuda.is_available() else 'gloo'
    dist.init_process_group(backend=backend, rank=rank, world_size=world_size)
    
    if backend == 'nccl': # Only set device if using CUDA
        torch.cuda.set_device(target_gpu_id) # Crucial: Set to the specific target GPU for this process

def cleanup_ddp():
    dist.destroy_process_group()

def train(rank, world_size, all_selected_gpu_ids, data_path, 
          batch_size, epochs, learning_rate, num_workers, 
          use_amp, gradient_accumulation_steps, output_dir, min_lr, max_grad_norm):
    current_process_gpu_id = all_selected_gpu_ids[rank]
    print(f"Rank {rank}: Initializing distributed setup for GPU {current_process_gpu_id}...")
    setup_ddp(rank, world_size, current_process_gpu_id)
    print(f"Rank {rank}: Distributed setup complete for GPU {current_process_gpu_id}.")

    dataset = PretrainDataset(data_path, config.model.max_seq_len)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True, drop_last=False)
    dataloader = DataLoader(dataset, batch_size=batch_size, sampler=sampler, pin_memory=True, num_workers=num_workers, persistent_workers=True if num_workers > 0 else False)
    print(f"Rank {rank}: DataLoader prepared for GPU {current_process_gpu_id}.")

    model_config_obj = CustomLlamaConfig(
        vocab_size=config.model.vocab_size,
        hidden_size=config.model.hidden_size,
        intermediate_size=config.model.intermediate_size,
        num_hidden_layers=config.model.num_hidden_layers,
        num_attention_heads=config.model.num_attention_heads,
        num_key_value_heads=config.model.num_key_value_heads,
        head_dim=config.model.hidden_size // config.model.num_attention_heads,
        max_seq_len=config.model.max_seq_len,
        rms_norm_eps=config.model.rms_norm_eps,
        rope_theta=config.model.rope_theta
    )
    model = CustomLlamaModel(model_config_obj).to(current_process_gpu_id)
    
    ddp_model = DDP(model, device_ids=[current_process_gpu_id], output_device=current_process_gpu_id)
    print(f"Rank {rank}: Model prepared and wrapped with DDP on GPU {current_process_gpu_id}.")

    if rank == 0:
        print(f"Rank {rank}: Printing model parameters...")
        print_model_parameters(ddp_model.module)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(ddp_model.parameters(), lr=learning_rate, weight_decay=config.training.weight_decay)

    scaler = torch.amp.GradScaler(enabled=use_amp)

    num_update_steps_per_epoch = len(dataloader) // gradient_accumulation_steps
    if len(dataloader) % gradient_accumulation_steps != 0:
        num_update_steps_per_epoch += 1
    num_total_training_steps = num_update_steps_per_epoch * epochs
    
    scheduler = CosineAnnealingLR(optimizer, T_max=num_total_training_steps, eta_min=min_lr)

    if rank == 0:
        print(f"Rank 0: Total training optimization steps (T_max for scheduler): {num_total_training_steps}")

    start_epoch = 0
    global_micro_batch_step_rank0 = 0 # This will be updated if resuming
    # Will store the path to the checkpoint file if resuming and the number of micro-batches processed in that epoch by rank 0
    actual_checkpoint_to_load = None 
    initial_micro_batches_processed_in_resumed_epoch = 0

    if config.training.resume_from_checkpoint and isinstance(config.training.resume_from_checkpoint, str) and config.training.resume_from_checkpoint.strip():
        actual_checkpoint_to_load = config.training.resume_from_checkpoint.strip()
        if rank == 0:
            print(f"Rank 0: Resume is enabled. Attempting to load from checkpoint path: {actual_checkpoint_to_load}")

        # resume_info_tensor: [load_flag (0 or 1), epoch_val, step_val, epoch_micro_batches_processed_val]
        resume_info_tensor = torch.zeros(4, dtype=torch.long, device=f'cuda:{current_process_gpu_id}')

        if rank == 0:
            if os.path.exists(actual_checkpoint_to_load):
                try:
                    print(f"Rank 0: Attempting to load checkpoint metadata from: {actual_checkpoint_to_load}")
                    temp_checkpoint_meta = torch.load(actual_checkpoint_to_load, map_location='cpu', weights_only=False)
                    resume_info_tensor[0] = 1 # Success flag
                    resume_info_tensor[1] = temp_checkpoint_meta['epoch'] # Current epoch index to resume from
                    resume_info_tensor[2] = temp_checkpoint_meta['global_step'] # Resumed global optimizer step
                    resume_info_tensor[3] = temp_checkpoint_meta.get('epoch_micro_batches_processed', 0) # Micro-batches processed in that epoch
                    print(f"Rank 0: Checkpoint metadata loaded. Will resume from epoch {resume_info_tensor[1].item()}, global_step {resume_info_tensor[2].item()}, micro_batch offset in epoch: {resume_info_tensor[3].item()}.")
                except Exception as e:
                    print(f"Rank 0: Failed to load checkpoint metadata from {actual_checkpoint_to_load}. Error: {e}. Starting fresh.")
                    resume_info_tensor.fill_(0) # Failure flag and clear other values
                    actual_checkpoint_to_load = None
            else:
                print(f"Rank 0: Checkpoint file {actual_checkpoint_to_load} not found. Starting fresh.")
                resume_info_tensor.fill_(0)
                actual_checkpoint_to_load = None
        
        dist.broadcast(resume_info_tensor, src=0)

        load_successful_flag = resume_info_tensor[0].item() == 1 and actual_checkpoint_to_load is not None
        resumed_epoch_from_cp = resume_info_tensor[1].item()
        resumed_global_step_from_cp = resume_info_tensor[2].item()
        initial_micro_batches_processed_in_resumed_epoch = resume_info_tensor[3].item()

        if load_successful_flag:
            try:
                print(f"Rank {rank}: Loading full checkpoint {actual_checkpoint_to_load} to device {current_process_gpu_id}")
                checkpoint = torch.load(actual_checkpoint_to_load, map_location=f'cuda:{current_process_gpu_id}', weights_only=False)
                
                ddp_model.module.load_state_dict(checkpoint['model_state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                if use_amp and 'scaler_state_dict' in checkpoint:
                    scaler.load_state_dict(checkpoint['scaler_state_dict'])
                
                start_epoch = resumed_epoch_from_cp
                global_micro_batch_step_rank0 = resumed_global_step_from_cp 
                # initial_micro_batches_processed_in_resumed_epoch is already set from broadcast
                
                if rank == 0:
                    all_losses_rank0 = checkpoint.get('all_losses_rank0', [])
                    all_lrs_rank0 = checkpoint.get('all_lrs_rank0', [])
                    all_plot_steps_rank0 = checkpoint.get('all_plot_steps_rank0', [])
                    print(f"Rank 0: Resumed plotting data. "
                          f"Losses: {len(all_losses_rank0)} points, "
                          f"LRs: {len(all_lrs_rank0)} points, "
                          f"Steps: {len(all_plot_steps_rank0)} points.")
                
                print(f"Rank {rank}: Successfully resumed from checkpoint. Starting epoch: {start_epoch}, Global Step: {global_micro_batch_step_rank0}, Micro-batch offset for this epoch: {initial_micro_batches_processed_in_resumed_epoch}")
            except Exception as e:
                print(f"Rank {rank}: Error loading full checkpoint state on this rank. Error: {e}. Starting fresh as a fallback.")
                start_epoch = 0 
                global_micro_batch_step_rank0 = 0
                initial_micro_batches_processed_in_resumed_epoch = 0
                if rank == 0: all_losses_rank0, all_lrs_rank0, all_plot_steps_rank0 = [], [], []
                load_successful_flag = False # Ensure we don't try to skip batches later
        else:
            if rank != 0: 
                 print(f"Rank {rank}: Following Rank 0's signal, starting fresh as checkpoint was not loaded or was invalid.")
            # Reset states if rank 0 indicated failure but other ranks thought it was fine
            start_epoch = 0
            global_micro_batch_step_rank0 = 0
            initial_micro_batches_processed_in_resumed_epoch = 0
            if rank == 0: all_losses_rank0, all_lrs_rank0, all_plot_steps_rank0 = [], [], []
            
    else:
        if rank == 0:
            print("Rank 0: Not resuming from checkpoint as 'resume_from_checkpoint' is not set or is an empty string.")
            all_losses_rank0, all_lrs_rank0, all_plot_steps_rank0 = [], [], [] # Initialize for fresh run
        # Ensure other ranks also know not to skip (already 0 by default)
        load_successful_flag = False # Explicitly set for clarity for later logic

    print(f"Rank {rank}: Starting training from epoch {start_epoch} (0-indexed) for {epochs - start_epoch} more epochs on GPU {current_process_gpu_id}...")
    optimizer.zero_grad(set_to_none=True)

    # Initialize plotting lists if not resuming (rank 0 only)
    if rank == 0 and not load_successful_flag:
        all_losses_rank0 = []
        all_lrs_rank0 = []
        all_plot_steps_rank0 = []

    for epoch in range(start_epoch, epochs):
        sampler.set_epoch(epoch)
        epoch_loss_sum_rank_local = 0.0 # Sum of losses for newly processed batches in this epoch on this rank
        num_newly_processed_micro_batches_in_epoch_rank_local = 0

        current_epoch_skip_count = 0
        # Determine skip count only for the first epoch after a successful resume
        if load_successful_flag and epoch == start_epoch:
            current_epoch_skip_count = initial_micro_batches_processed_in_resumed_epoch

        if rank == 0:
            tqdm_bar_format = "{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {postfix}]"
            data_iterator = tqdm(dataloader,
                                 desc=f"Epoch {epoch+1}/{epochs} [GPU {current_process_gpu_id}]", # User-facing epoch (1-indexed)
                                 unit="batch",
                                 dynamic_ncols=True,
                                 leave=False,
                                 bar_format=tqdm_bar_format,
                                 initial=current_epoch_skip_count, # Set initial for tqdm
                                 total=len(dataloader)) # Total for this rank's dataloader
        else:
            data_iterator = dataloader
        
        num_fwd_passes_since_last_opt_step = 0
        for i, (inputs, targets) in enumerate(data_iterator): # i is 0-indexed for this rank's dataloader
            # Skip batches if resuming the first epoch and i is less than the processed count
            if load_successful_flag and epoch == start_epoch and i < current_epoch_skip_count:
                continue

            inputs, targets = inputs.to(current_process_gpu_id, non_blocking=True), targets.to(current_process_gpu_id, non_blocking=True)

            with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=use_amp):
                outputs = ddp_model(inputs)
                loss = criterion(outputs.view(-1, config.model.vocab_size), targets.view(-1))
            
            current_micro_batch_loss = loss.item()
            epoch_loss_sum_rank_local += current_micro_batch_loss
            num_newly_processed_micro_batches_in_epoch_rank_local += 1
            
            if gradient_accumulation_steps > 1:
                loss = loss / gradient_accumulation_steps
            
            scaler.scale(loss).backward()
            num_fwd_passes_since_last_opt_step += 1

            if rank == 0: # Append to plot lists for every *newly processed* micro_batch
                all_losses_rank0.append(current_micro_batch_loss)
                all_lrs_rank0.append(scheduler.get_last_lr()[0])
                all_plot_steps_rank0.append(global_micro_batch_step_rank0) # GMS *before* this optimizer step

            # Optimizer step condition
            # Check if it's time for an optimizer step OR if it's the very last batch of the epoch and there are pending gradients
            is_last_batch_in_epoch = (i + 1 == len(dataloader))
            if num_fwd_passes_since_last_opt_step % gradient_accumulation_steps == 0 or \
               (is_last_batch_in_epoch and num_fwd_passes_since_last_opt_step > 0):
                if use_amp:
                    scaler.unscale_(optimizer)
                clip_grad_norm_(ddp_model.parameters(), max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step() 
                optimizer.zero_grad(set_to_none=True)

                if rank == 0:
                    global_micro_batch_step_rank0 +=1 

                    if config.logging.log_interval > 0 and global_micro_batch_step_rank0 % config.logging.log_interval == 0:
                        current_lr_for_log = scheduler.get_last_lr()[0]
                        print(f"\nRank {rank} [Epoch {epoch+1}, GlobalStep {global_micro_batch_step_rank0}]: "
                              f"Micro-batch loss: {current_micro_batch_loss:.4f}, LR: {current_lr_for_log:.8e}")
                        
                        interim_loss_plot_file = os.path.join(config.paths.output_dir, 
                                                              f"loss_interim_gpu{current_process_gpu_id}.png")
                        interim_lr_plot_file = os.path.join(config.paths.output_dir,
                                                            f"lr_interim_gpu{current_process_gpu_id}.png")
                        
                        if all_plot_steps_rank0: 
                            plot_loss(all_losses_rank0, all_plot_steps_rank0, 
                                      output_file=interim_loss_plot_file, 
                                      beta=config.logging.plot_loss_beta)
                            plot_lr(all_lrs_rank0, all_plot_steps_rank0, 
                                    output_file=interim_lr_plot_file)
                            print(f"Rank {rank}: Interim plots saved for step {global_micro_batch_step_rank0}.")

                    if config.logging.save_interval > 0 and global_micro_batch_step_rank0 % config.logging.save_interval == 0:
                        checkpoint_save_path = os.path.join(config.paths.output_dir, f"checkpoint_epoch{epoch}_step{global_micro_batch_step_rank0}.pt") # Use current epoch index
                        checkpoint = {
                            'epoch': epoch, # Current epoch index
                            'global_step': global_micro_batch_step_rank0, # Optimizer steps completed
                            'epoch_micro_batches_processed': i + 1, # Micro-batches processed in *this* epoch by this rank's dataloader
                            'model_state_dict': ddp_model.module.state_dict(), 
                            'optimizer_state_dict': optimizer.state_dict(),
                            'scheduler_state_dict': scheduler.state_dict(),
                            'model_config': model_config_obj, 
                            'training_config': config.training,
                            'all_losses_rank0': all_losses_rank0, # Rank 0 only data
                            'all_lrs_rank0': all_lrs_rank0,       # Rank 0 only data
                            'all_plot_steps_rank0': all_plot_steps_rank0 # Rank 0 only data
                        }
                        if use_amp:
                            checkpoint['scaler_state_dict'] = scaler.state_dict()
                        
                        torch.save(checkpoint, checkpoint_save_path)
                        print(f"\nRank {rank}: Checkpoint saved to {checkpoint_save_path} at epoch {epoch} (0-indexed), global_step {global_micro_batch_step_rank0}, micro_batch {i+1} of current epoch.")
                
                num_fwd_passes_since_last_opt_step = 0 # Reset for next accumulation cycle

            if rank == 0:
                data_iterator.set_postfix(loss=f"{current_micro_batch_loss:.6f}")
        
        if rank == 0 and isinstance(data_iterator, tqdm):
            data_iterator.close()
        
        # Calculate average epoch loss based on newly processed batches
        # For a global average, need to gather num_newly_processed_micro_batches and epoch_loss_sum from all ranks
        # For simplicity here, just printing rank-local average
        avg_epoch_loss_rank_local = epoch_loss_sum_rank_local / num_newly_processed_micro_batches_in_epoch_rank_local if num_newly_processed_micro_batches_in_epoch_rank_local > 0 else 0
        
        # Gather losses and counts from all ranks for a global average epoch loss (optional, more complex)
        # Example:
        # loss_sum_tensor = torch.tensor([epoch_loss_sum_rank_local], device=current_process_gpu_id)
        # processed_count_tensor = torch.tensor([num_newly_processed_micro_batches_in_epoch_rank_local], device=current_process_gpu_id)
        # dist.all_reduce(loss_sum_tensor, op=dist.ReduceOp.SUM)
        # dist.all_reduce(processed_count_tensor, op=dist.ReduceOp.SUM)
        # global_avg_epoch_loss = loss_sum_tensor.item() / processed_count_tensor.item() if processed_count_tensor.item() > 0 else 0

        if rank == 0: # Rank 0 prints its local average or the global average if computed
            epoch_summary_prefix = f"Epoch {epoch+1}/{epochs} Summary [GPU {current_process_gpu_id}]" # User-facing epoch
            print(f"{epoch_summary_prefix}: Average Micro-Batch Loss (this rank, new batches): {avg_epoch_loss_rank_local:.4f}, Final LR: {scheduler.get_last_lr()[0]:.2e}")
            # if global_avg_epoch_loss calculated: print(f"Global Avg Loss: {global_avg_epoch_loss:.4f}")

    if rank == 0:
        print(f"Rank {rank}: Generating loss and LR plots...")
        loss_plot_filename = os.path.join(config.paths.output_dir, f"loss_curve_final_gpu{current_process_gpu_id}.png") # Final suffix
        lr_plot_filename = os.path.join(config.paths.output_dir, f"lr_curve_final_gpu{current_process_gpu_id}.png")   # Final suffix
        
        if all_plot_steps_rank0: # Ensure data exists for plotting
            plot_loss(all_losses_rank0, all_plot_steps_rank0, 
                    output_file=loss_plot_filename, 
                    beta=config.logging.plot_loss_beta)
            plot_lr(all_lrs_rank0, all_plot_steps_rank0, 
                    output_file=lr_plot_filename)
            print(f"Rank {rank}: Final plots saved.")
        else:
            print(f"Rank {rank}: No data to plot for final curves.")

    print(f"Rank {rank}: Training finished on GPU {current_process_gpu_id}.")
    cleanup_ddp()
    print(f"Rank {rank}: Distributed environment cleaned up for GPU {current_process_gpu_id}.")

if __name__ == "__main__":

    if config.training.use_amp and not torch.cuda.is_bf16_supported():
        print(f"Rank {dist.get_rank() if dist.is_initialized() else 0}: Warning: --use_amp is True, but bfloat16 is not supported on this device. Disabling AMP.")
        config.training.use_amp = False
    elif config.training.use_amp:
        print(f"Rank {dist.get_rank() if dist.is_initialized() else 0}: AMP with bfloat16 is enabled.")

    if not dist.is_initialized() or dist.get_rank() == 0: 
        print("\n--- Applied Configuration ---")
        # Custom print for better readability
        for section_name, section_config in config.__dict__.items():
            print(f"  {section_name.capitalize()}:")
            if hasattr(section_config, '__dict__'): # Check if it's a dataclass instance
                for key, value in section_config.__dict__.items():
                    print(f"    {key}: {value}")
            else:
                print(f"    {section_config}") # Should not happen with current Config structure
            print() # Add a blank line between sections
        print("---------------------------\n")

    if not os.path.exists(config.paths.output_dir):
        try:
            # Only rank 0 should make directories generally, but with exist_ok=True, it's safer for all
            # However, to be absolutely safe and avoid race conditions if not on a networked FS:
            if not dist.is_initialized() or dist.get_rank() == 0:
                os.makedirs(config.paths.output_dir, exist_ok=True)
                print(f"MainProcess/Rank 0: Created output directory: {config.paths.output_dir}")
            if dist.is_initialized(): # Barrier to ensure dir is created before other ranks proceed
                dist.barrier()
        except OSError as e:
            if not os.path.isdir(config.paths.output_dir):
                raise e # Raise if it's not a directory after trying
    else:
        if not dist.is_initialized() or dist.get_rank() == 0:
            print(f"MainProcess/Rank 0: Output directory {config.paths.output_dir} already exists.")

    num_available_gpus = torch.cuda.device_count()
    actual_gpus_to_use = []
    final_world_size = 0

    if config.training.gpu_ids:
        try:
            parsed_ids = [int(g.strip()) for g in config.training.gpu_ids.split(',') if g.strip()]
            if not parsed_ids:
                raise ValueError("GPU IDs list cannot be empty if --gpu_ids is provided.")
            if len(parsed_ids) != len(set(parsed_ids)):
                raise ValueError("GPU IDs must be unique.")

            invalid_ids = [gid for gid in parsed_ids if not (0 <= gid < num_available_gpus)]
            if invalid_ids:
                is_are = "is" if len(invalid_ids) == 1 else "are"
                s_ = "" if len(invalid_ids) == 1 else "s"
                range_str = f"0 to {num_available_gpus-1}" if num_available_gpus > 0 else "none available"
                raise ValueError(f"GPU ID{s_} {', '.join(map(str, invalid_ids))} {is_are} invalid. Available GPU IDs range from {range_str}.")

            actual_gpus_to_use = parsed_ids
            final_world_size = len(actual_gpus_to_use)
            print(f"Info: Using GPU IDs specified by --gpu_ids: {config.training.gpu_ids}. Effective world size will be {final_world_size}.")

        except ValueError as e:
            print(f"Error parsing --gpu_ids '{config.training.gpu_ids}': {e}")
            if num_available_gpus > 0:
                print(f"Available GPU IDs: {list(range(num_available_gpus))}")
            else:
                print("No CUDA-capable GPUs detected on this system.")
            exit(1)
    else: # --gpu_ids was not provided, default to all available GPUs
        if num_available_gpus == 0:
            final_world_size = 0
            actual_gpus_to_use = [] 
        else:
            final_world_size = num_available_gpus
            actual_gpus_to_use = list(range(final_world_size))
            print(f"Info: --gpu_ids not specified. Defaulting to use all {num_available_gpus} available GPUs.")

    if final_world_size == 0:
        print("Error: No GPUs selected or available for training. Distributed training requires at least 1 GPU.")
        if num_available_gpus == 0:
            print("Details: No CUDA-capable GPUs were detected on this system.")
        else:
            print(f"Details: Effective world size is 0 based on arguments. Available GPUs: {num_available_gpus}. Please check --world_size and --gpu_ids settings.")
        exit(1)
        
    data_file_path = config.paths.pretrain_data_file
    if not os.path.exists(data_file_path):
        print(f"Error: Data file {data_file_path} not found. Please run process.py first.")
        exit(1)
        
    print(f"Starting distributed training with {final_world_size} processes, using GPU IDs: {actual_gpus_to_use}.")
    mp.spawn(train, 
             args=(
                final_world_size, 
                actual_gpus_to_use, 
                data_file_path, 
                config.training.batch_size, 
                config.training.num_epochs, 
                config.training.lr, 
                config.training.num_workers,
                config.training.use_amp,
                config.training.gradient_accumulation_steps,
                config.paths.output_dir,
                config.training.min_lr,
                config.training.max_grad_norm
            ),
             nprocs=final_world_size,
             join=True) 
