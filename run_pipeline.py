#!/usr/bin/env python3
"""
Complete CTI LLM Fine-tuning Pipeline
Execute steps in order: 1) Data Collection, 2) Processing, 3) Training, 4) Inference
"""

import argparse
import logging
import sys
from pathlib import Path

# Add scripts to path
sys.path.append(str(Path(__file__).parent / "scripts"))

from data_collection import main as collect_data
from data_processing import main as process_data
from training import main as train_model
from inference import CTIInferenceEngine

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def run_complete_pipeline():
    """Run the complete pipeline end-to-end"""
    logger.info("Starting CTI LLM Fine-tuning Pipeline...")
    
    try:
        # Step 1: Data Collection
        logger.info("Step 1: Collecting CTI Data...")
        collect_data()
        
        # Step 2: Data Processing
        logger.info("Step 2: Processing Data and Extracting IOCs...")
        process_data()
        
        # Step 3: Model Training
        logger.info("Step 3: Fine-tuning LLM...")
        train_model()
        
        # Step 4: Test Inference
        logger.info("Step 4: Testing Inference...")
        engine = CTIInferenceEngine()
        engine.load_model()
        
        test_prompts = [
            "List Indicators of Compromise in APT35",
            "Extract IOCs from recent threat report",
            "What are the indicators for FIN7 group?"
        ]
        
        for prompt in test_prompts:
            logger.info(f"Testing prompt: {prompt}")
            result = engine.generate_structured_iocs(prompt)
            print(f"\nPrompt: {prompt}")
            print(f"Response: {result}")
            print("-" * 50)
        
        logger.info("Pipeline completed successfully!")
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CTI LLM Fine-tuning Pipeline")
    parser.add_argument("--step", choices=["collect", "process", "train", "inference", "all"], 
                       default="all", help="Which step to run")
    
    args = parser.parse_args()
    
    if args.step == "all":
        run_complete_pipeline()
    elif args.step == "collect":
        collect_data()
    elif args.step == "process":
        process_data()
    elif args.step == "train":
        train_model()
    elif args.step == "inference":
        engine = CTIInferenceEngine()
        engine.load_model()
        # Interactive inference
        while True:
            prompt = input("\nEnter prompt (or 'quit' to exit): ")
            if prompt.lower() == 'quit':
                break
            result = engine.generate_structured_iocs(prompt)
            print(f"\nResponse:\n{result}")
