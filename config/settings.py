import os
from dataclasses import dataclass
from typing import Optional
import yaml

@dataclass
class DataConfig:
    """Configuration for data processing"""
    raw_data_dir: str = "data/raw"
    processed_data_dir: str = "data/processed"
    max_file_size_mb: int = 50
    supported_formats: list = None
    
    def __post_init__(self):
        if self.supported_formats is None:
            self.supported_formats = ['.pdf', '.txt', '.md', '.html']

@dataclass
class ModelConfig:
    """Configuration for model training"""
    model_name: str = "meta-llama/Llama-2-7b-chat-hf"
    token: str = "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"  # Replace with your HuggingFace token
    
    # Training parameters
    batch_size: int = 2
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-5
    num_epochs: int = 3
    max_length: int = 1024
    
    # LoRA configuration
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.1

@dataclass
class TrainingConfig:
    """Training configuration"""
    output_dir: str = "models/cti-llama2"
    logging_dir: str = "logs"
    save_steps: int = 500
    eval_steps: int = 500
    logging_steps: int = 50

class Config:
    """Main configuration class"""
    def __init__(self):
        self.data = DataConfig()
        self.model = ModelConfig()
        self.training = TrainingConfig()
    
    @classmethod
    def from_yaml(cls, config_path: str):
        """Load configuration from YAML file"""
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        config = cls()
        # Update configurations from YAML
        # Implementation depends on your YAML structure
        return config

# Global configuration instance
config = Config()
