import os
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, TaskType
from datasets import Dataset
import json
import logging
from pathlib import Path
from config.settings import config

logger = logging.getLogger(__name__)

class CTIModelTrainer:
    """Fine-tune LLM for CTI tasks"""
    
    def __init__(self):
        self.output_dir = Path(config.training.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def load_training_data(self) -> Dataset:
        """Load and prepare training data"""
        data_file = Path(config.data.processed_data_dir) / "training_data.json"
        
        with open(data_file, 'r') as f:
            training_data = json.load(f)
        
        # Format for instruction tuning
        formatted_data = []
        for example in training_data:
            formatted_text = f"### Instruction:\n{example['prompt']}\n\n### Response:\n{example['completion']}"
            formatted_data.append({"text": formatted_text})
        
        return Dataset.from_list(formatted_data)
    
    def setup_model_and_tokenizer(self):
        """Initialize model and tokenizer with proper configuration"""
        logger.info("Loading model and tokenizer...")
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            config.model.model_name,
            token=config.model.token,
            trust_remote_code=True
        )
        
        # Set padding token
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Load model with 4-bit quantization for memory efficiency
        model = AutoModelForCausalLM.from_pretrained(
            config.model.model_name,
            token=config.model.token,
            torch_dtype=torch.float16,
            device_map="auto",
            load_in_4bit=True,
            trust_remote_code=True
        )
        
        return model, tokenizer
    
    def setup_lora_config(self) -> LoraConfig:
        """Setup LoRA configuration for parameter-efficient fine-tuning"""
        return LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=config.model.lora_r,
            lora_alpha=config.model.lora_alpha,
            lora_dropout=config.model.lora_dropout,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
            bias="none",
        )
    
    def tokenize_function(self, examples, tokenizer):
        """Tokenize training examples"""
        tokenized = tokenizer(
            examples["text"],
            truncation=True,
            padding=False,
            max_length=config.model.max_length,
            return_tensors=None
        )
        
        # For causal LM, labels are the same as input_ids
        tokenized["labels"] = tokenized["input_ids"].copy()
        
        return tokenized
    
    def train(self):
        """Execute the training pipeline"""
        logger.info("Starting model training...")
        
        # Load data
        dataset = self.load_training_data()
        logger.info(f"Loaded {len(dataset)} training examples")
        
        # Setup model and tokenizer
        model, tokenizer = self.setup_model_and_tokenizer()
        
        # Apply LoRA
        lora_config = self.setup_lora_config()
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
        
        # Tokenize dataset
        tokenized_dataset = dataset.map(
            lambda x: self.tokenize_function(x, tokenizer),
            batched=True,
            remove_columns=dataset.column_names
        )
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=str(self.output_dir),
            overwrite_output_dir=True,
            per_device_train_batch_size=config.model.batch_size,
            gradient_accumulation_steps=config.model.gradient_accumulation_steps,
            learning_rate=config.model.learning_rate,
            num_train_epochs=config.model.num_epochs,
            logging_dir=str(Path(config.training.logging_dir)),
            logging_steps=config.training.logging_steps,
            save_steps=config.training.save_steps,
            evaluation_strategy="no",
            save_strategy="steps",
            fp16=True,
            warmup_ratio=0.1,
            lr_scheduler_type="cosine",
            report_to=None,  # Disable wandb/tensorboard if not needed
            ddp_find_unused_parameters=False,
        )
        
        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False,
            pad_to_multiple_of=8
        )
        
        # Initialize trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_dataset,
            data_collator=data_collator,
            tokenizer=tokenizer,
        )
        
        # Start training
        logger.info("Training started...")
        trainer.train()
        
        # Save final model
        trainer.save_model()
        tokenizer.save_pretrained(str(self.output_dir))
        
        logger.info(f"Training completed. Model saved to {self.output_dir}")

def main():
    """Main training function"""
    trainer = CTIModelTrainer()
    trainer.train()

if __name__ == "__main__":
    main()
