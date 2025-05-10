import os
import yaml
from dataclasses import dataclass
from typing import Optional, List

# Dataclass definitions (PathConfig, TrainingConfig, LoggingConfig, ModelConfig, Config)
# should remain the same as you provided in the initial prompt.
# Make sure they are defined above load_config.

@dataclass
class PathConfig:
    tokenizer_path: str
    pretrain_data_file: str
    output_dir: str
    default_checkpoint_file: str
    final_model_file: str
    loss_plot_file: str
    # Add hf_cache_dir if you want it to be part of the config structure
    hf_cache_dir: Optional[str] = None


@dataclass
class TrainingConfig:
    num_epochs: int
    batch_size: int
    num_workers: int
    gradient_accumulation_steps: int
    lr: float
    min_lr: float
    weight_decay: float
    max_grad_norm: float
    seed: int
    gpu_ids: str
    world_size: int
    use_amp: bool
    resume_from_checkpoint: Optional[str] = None


@dataclass
class LoggingConfig:
    log_interval: int
    save_interval: int
    plot_loss_beta: float = 0.1


@dataclass
class ModelConfig:
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    vocab_size: int 
    max_seq_len: int
    rms_norm_eps: float = 1e-5
    rope_theta: float = 10000.0


@dataclass
class Config:
    paths: PathConfig
    training: TrainingConfig
    logging: LoggingConfig
    model: ModelConfig


def load_config(config_path='config.yaml') -> Config:
    """加载配置文件并返回配置对象"""
    with open(config_path, 'r', encoding='utf-8') as f:
        raw_config = yaml.safe_load(f)

    # Helper to get and cast value, providing more robust type handling
    def get_config_value(config_section, key, expected_type=None, default=None, is_critical=False):
        value = raw_config.get(config_section, {}).get(key, default)
        if value is None and default is not None: # if key is missing but default is provided
             value = default
        elif value is None and is_critical:
             raise ValueError(f"Critical config '{config_section}.{key}' is missing and has no default.")


        if value is not None and expected_type is not None:
            try:
                if expected_type == bool and isinstance(value, str): # Special handling for bools from strings
                    if value.lower() in ['true', 'yes', '1']:
                        return True
                    elif value.lower() in ['false', 'no', '0']:
                        return False
                    else:
                        raise ValueError(f"Cannot convert string '{value}' to bool.")
                return expected_type(value)
            except (ValueError, TypeError) as e:
                error_msg = (f"Config Error: Could not convert '{config_section}.{key}' value '{value}' (type: {type(value).__name__}) "
                             f"to {expected_type}. Error: {e}.")
                if is_critical:
                    raise ValueError(error_msg) from e
                else:
                    print(f"Warning: {error_msg} Using default: {default}")
                    return default # Return the original default, not the potentially mis-typed value
        # If no expected_type, or value is None and no default to process
        return value

    # 构建配置对象
    paths_conf = raw_config.get('paths', {})
    training_conf = raw_config.get('training', {})
    logging_conf = raw_config.get('logging', {})
    model_conf = raw_config.get('model', {})

    config_obj = Config(
        paths=PathConfig(
            tokenizer_path=get_config_value('paths', 'tokenizer_path', str, is_critical=True),
            pretrain_data_file=get_config_value('paths', 'pretrain_data_file', str, is_critical=True),
            output_dir=get_config_value('paths', 'output_dir', str, is_critical=True),
            default_checkpoint_file=get_config_value('paths', 'default_checkpoint_file', str, default="custom_llama_checkpoint.pt"),
            final_model_file=get_config_value('paths', 'final_model_file', str, default="custom_llama_final.pt"),
            loss_plot_file=get_config_value('paths', 'loss_plot_file', str, default="custom_pretrain_loss_curve.png"),
            hf_cache_dir=get_config_value('paths', 'hf_cache_dir', str, default=None)
        ),
        training=TrainingConfig(
            num_epochs=get_config_value('training', 'num_epochs', int, is_critical=True),
            batch_size=get_config_value('training', 'batch_size', int, is_critical=True),
            num_workers=get_config_value('training', 'num_workers', int, default=0),
            gradient_accumulation_steps=get_config_value('training', 'gradient_accumulation_steps', int, default=1),
            lr=get_config_value('training', 'lr', float, is_critical=True),
            min_lr=get_config_value('training', 'min_lr', float, is_critical=True),
            weight_decay=get_config_value('training', 'weight_decay', float, default=0.0),
            max_grad_norm=get_config_value('training', 'max_grad_norm', float, default=1.0),
            seed=get_config_value('training', 'seed', int, default=42),
            gpu_ids=get_config_value('training', 'gpu_ids', str, is_critical=True), # List type is harder to auto-cast, rely on YAML structure
            use_amp=get_config_value('training', 'use_amp', bool, default=True),
            world_size=get_config_value('training', 'world_size', int, default=1),
            resume_from_checkpoint=get_config_value('training', 'resume_from_checkpoint', str, default=None)
        ),
        logging=LoggingConfig(
            log_interval=get_config_value('logging', 'log_interval', int, default=100),
            save_interval=get_config_value('logging', 'save_interval', int, default=1000),
            plot_loss_beta=get_config_value('logging', 'plot_loss_beta', float, default=0.1)
        ),
        model=ModelConfig(
            vocab_size=get_config_value('model', 'vocab_size', int, is_critical=True),
            hidden_size=get_config_value('model', 'hidden_size', int, is_critical=True),
            intermediate_size=get_config_value('model', 'intermediate_size', int, is_critical=True),
            num_hidden_layers=get_config_value('model', 'num_hidden_layers', int, is_critical=True),
            num_attention_heads=get_config_value('model', 'num_attention_heads', int, is_critical=True),
            num_key_value_heads=get_config_value('model', 'num_key_value_heads', int, is_critical=True),
            max_seq_len=get_config_value('model', 'max_seq_len', int, is_critical=True),
            rms_norm_eps=get_config_value('model', 'rms_norm_eps', float, default=1e-5),
            rope_theta=get_config_value('model', 'rope_theta', float, default=10000.0)
        )
    )
    return config_obj

# 加载默认配置
config = load_config()
