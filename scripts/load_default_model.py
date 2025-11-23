#!/usr/bin/env python3
"""
Load Default Model Script for CTI LLM Project
Loads and tests the base/default model before fine-tuning
"""

import os
import sys
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    pipeline
)
import logging
from pathlib import Path

# Add the parent directory to Python path to import config
sys.path.append(str(Path(__file__).parent.parent))

from config.settings import config

logger = logging.getLogger(__name__)

class DefaultModelLoader:
    """Loads and tests the default/base model before training"""
    
    def __init__(self):
        self.device = config.get_device()
        self.model = None
        self.tokenizer = None
        self.generator = None
    
    def load_model(self):
        """Load the base model and tokenizer"""
        logger.info(f"Loading default model: {config.model.model_name}")
        
        try:
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(config.model.model_name)
            
            # Set padding token if needed
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Load model
            self.model = AutoModelForCausalLM.from_pretrained(
                config.model.model_name,
                torch_dtype=torch.float32,
            )
            
            # Move to appropriate device
            if str(self.device) == "mps":
                self.model = self.model.to(self.device)
            
            # Create text generation pipeline
            self.generator = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                torch_dtype=torch.float32,
                device=0 if str(self.device) == "cuda" else -1,
            )
            
            logger.info("✅ Default model loaded successfully!")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to load default model: {e}")
            return False
    
    def test_basic_functionality(self, prompt: str, max_length: int = 100) -> str:
        """Test the model with a basic prompt"""
        if self.generator is None:
            logger.error("Model not loaded. Call load_model() first.")
            return ""
        
        try:
            response = self.generator(
                prompt,
                max_new_tokens=max_length,
                temperature=0.7,
                do_sample=True,
                top_p=0.9,
                pad_token_id=self.tokenizer.eos_token_id,
            )
            
            generated_text = response[0]['generated_text']
            return generated_text
            
        except Exception as e:
            logger.error(f"Error during generation: {e}")
            return f"Error: {str(e)}"
    
    def run_comprehensive_tests(self):
        """Run comprehensive tests on the default model"""
        logger.info("Running comprehensive tests on default model...")
        
        test_prompts = [
            "Hello, world!",
            "What is cyber threat intelligence?",
            "Explain what Indicators of Compromise are.",
            "List some common IOCs in cybersecurity.",
            "The weather is nice today.",
        ]
        
        print("\n" + "="*60)
        print("DEFAULT MODEL TESTING RESULTS")
        print("="*60)
        
        for i, prompt in enumerate(test_prompts, 1):
            print(f"\nTest {i}: {prompt}")
            print("-" * 40)
            
            response = self.test_basic_functionality(prompt, max_length=50)
            print(f"Response: {response}")
            
            # Check if response is reasonable
            if response and len(response.strip()) > len(prompt.strip()):
                print("✅ Test passed - Model generated response")
            else:
                print("❌ Test failed - No meaningful response")
        
        print("\n" + "="*60)
        print("DEFAULT MODEL TESTING COMPLETED")
        print("="*60)
    
    def check_model_capabilities(self):
        """Check default model capabilities and statistics"""
        logger.info("Checking default model capabilities...")
        
        if self.model is None:
            logger.error("Model not loaded")
            return
        
        print("\n" + "="*50)
        print("DEFAULT MODEL CAPABILITIES")
        print("="*50)
        
        # Model info
        print(f"Model: {config.model.model_name}")
        print(f"Device: {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"Trainable parameters: {sum(p.numel() for p in self.model.parameters() if p.requires_grad):,}")
        
        # Tokenizer info
        print(f"Vocabulary size: {len(self.tokenizer)}")
        print(f"Max position embeddings: {self.model.config.max_position_embeddings}")
        
        # Test tokenization
        test_text = "Hello, world! This is a test."
        tokens = self.tokenizer.encode(test_text)
        print(f"Tokenization test: '{test_text}' -> {len(tokens)} tokens")
        
        print("="*50)
    
    def test_ioc_generation_capability(self):
        """Test default model's IOC generation capability"""
        logger.info("Testing default model's IOC generation capability...")
        
        ioc_prompts = [
            "List some example IP addresses:",
            "Generate sample MD5 hashes:",
            "Show me example SHA256 hashes:",
            "What are common indicators in cybersecurity?",
        ]
        
        print("\n" + "="*50)
        print("IOC GENERATION CAPABILITY TEST")
        print("="*50)
        
        for prompt in ioc_prompts:
            print(f"\nPrompt: {prompt}")
            response = self.test_basic_functionality(prompt, max_length=80)
            print(f"Response: {response}")
        
        print("="*50)
    
    def compare_before_after_capability(self):
        """Show what the default model can do before fine-tuning"""
        print("\n" + "="*60)
        print("BEFORE FINE-TUNING CAPABILITY ASSESSMENT")
        print("="*60)
        print("This shows what the default model can do BEFORE training.")
        print("After fine-tuning, it should better understand CTI prompts")
        print("and generate structured IOC outputs.")
        print("="*60)
        
        # Test with CTI-like prompts
        cti_prompts = [
            "List Indicators of Compromise",
            "Extract IOCs from threat report",
            "What are the indicators for APT35?",
        ]
        
        for prompt in cti_prompts:
            print(f"\nTesting: '{prompt}'")
            print("-" * 40)
            response = self.test_basic_functionality(prompt, max_length=100)
            print(f"Default model response: {response}")
        
        print("\n" + "="*60)
        print("This baseline will help compare with fine-tuned model later.")
        print("="*60)

def main():
    """Main function to load and test default model"""
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    print("🔍 CTI LLM - Load Default Model")
    print("=" * 50)
    print("This script loads and tests the base model BEFORE fine-tuning.")
    print("It establishes a baseline for comparison after training.")
    print("=" * 50)
    
    loader = DefaultModelLoader()
    
    # Step 1: Load model
    print("\n1. Loading default model...")
    if not loader.load_model():
        print("❌ Failed to load default model. Exiting.")
        return 1
    
    # Step 2: Check capabilities
    print("\n2. Checking model capabilities...")
    loader.check_model_capabilities()
    
    # Step 3: Run basic tests
    print("\n3. Running basic functionality tests...")
    loader.run_comprehensive_tests()
    
    # Step 4: Test IOC generation
    print("\n4. Testing IOC generation capability...")
    loader.test_ioc_generation_capability()
    
    # Step 5: Show before-training baseline
    print("\n5. Establishing before-training baseline...")
    loader.compare_before_after_capability()
    
    # Step 6: Interactive testing
    print("\n6. Interactive testing mode")
    print("=" * 40)
    print("Test the default model with your own prompts.")
    print("This helps understand what needs improvement after fine-tuning.")
    print("Type 'quit' to exit interactive mode.")
    
    while True:
        try:
            user_input = input("\nYour prompt: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                break
            
            if not user_input:
                continue
            
            response = loader.test_basic_functionality(user_input, max_length=100)
            print(f"\nDefault model response: {response}")
            print("\nNote: After fine-tuning, responses should be more structured")
            print("and focused on IOC extraction in the required format.")
            
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error: {e}")
    
    print("\n🎉 Default model testing completed!")
    print("This establishes the baseline before fine-tuning.")
    print("You can now proceed with training to improve CTI capabilities.")
    
    return 0

if __name__ == "__main__":
    exit(main())
