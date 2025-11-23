#!/usr/bin/env python3
"""
Load Default Model Script for CTI LLM Project
Loads and tests the base model before fine-tuning
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import logging
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))

from config.settings import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    print("🔍 Loading Default Model...")
    
    try:
        # Load model
        tokenizer = AutoTokenizer.from_pretrained(config.model.model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
        model = AutoModelForCausalLM.from_pretrained(config.model.model_name)
        
        # Test with a simple prompt
        generator = pipeline("text-generation", model=model, tokenizer=tokenizer)
        
        test_prompt = "Hello, world!"
        result = generator(test_prompt, max_new_tokens=20)
        
        print(f"✅ Model loaded successfully: {config.model.model_name}")
        print(f"✅ Test prompt: '{test_prompt}'")
        print(f"✅ Model response: {result[0]['generated_text']}")
        print("\n🎉 Default model is working correctly!")
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
