import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    pipeline
)
from peft import PeftModel
import logging
from pathlib import Path
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
from config.settings import config

logger = logging.getLogger(__name__)

class CTIInferenceEngine:
    """Inference engine for the fine-tuned CTI model"""
    
    def __init__(self, model_path: str = None):
        self.model_path = model_path or config.training.output_dir
        self.model = None
        self.tokenizer = None
        self.pipeline = None
        
    def load_model(self):
        """Load the fine-tuned model"""
        logger.info("Loading fine-tuned model...")
        
        try:
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path,
                trust_remote_code=True
            )
            
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Load base model
            base_model = AutoModelForCausalLM.from_pretrained(
                config.model.model_name,
                torch_dtype=torch.float16,
                device_map="auto",
                load_in_4bit=True,
                trust_remote_code=True,
                token=config.model.token
            )
            
            # Apply LoRA adapters
            self.model = PeftModel.from_pretrained(
                base_model,
                self.model_path
            )
            
            # Create pipeline
            self.pipeline = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                torch_dtype=torch.float16,
                device_map="auto"
            )
            
            logger.info("Model loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    
    def generate_structured_iocs(self, prompt: str, max_length: int = 512) -> str:
        """Generate structured IOCs from prompt"""
        if self.pipeline is None:
            self.load_model()
        
        # Format prompt
        formatted_prompt = f"### Instruction:\n{prompt}\n\n### Response:\n"
        
        try:
            # Generate response
            response = self.pipeline(
                formatted_prompt,
                max_new_tokens=max_length,
                temperature=0.3,
                do_sample=True,
                top_p=0.9,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                repetition_penalty=1.1
            )
            
            generated_text = response[0]['generated_text']
            
            # Extract only the response part
            if "### Response:" in generated_text:
                response_text = generated_text.split("### Response:")[1].strip()
            else:
                response_text = generated_text
            
            return self._clean_response(response_text)
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return f"Error: {str(e)}"
    
    def _clean_response(self, text: str) -> str:
        """Clean and format the model response"""
        lines = text.split('\n')
        cleaned_lines = []
        current_section = None
        
        for line in lines:
            line = line.strip()
            
            # Keep section headers
            if line.startswith('## '):
                current_section = line
                cleaned_lines.append(line)
            
            # Keep bullet points with IOCs
            elif line.startswith('- '):
                # Extract only the IOC (first part after -)
                ioc_part = line[2:].split()[0] if line[2:].split() else ""
                if ioc_part:
                    cleaned_lines.append(f"- {ioc_part}")
            
            # Skip other text
            elif not line:
                continue
        
        return '\n'.join(cleaned_lines)

# FastAPI Application
app = FastAPI(title="CTI LLM API", version="1.0.0")

# Global inference engine
inference_engine = CTIInferenceEngine()

class InferenceRequest(BaseModel):
    prompt: str
    max_length: int = 512

class InferenceResponse(BaseModel):
    response: str
    status: str

@app.on_event("startup")
async def startup_event():
    """Load model on startup"""
    inference_engine.load_model()

@app.post("/generate", response_model=InferenceResponse)
async def generate_iocs(request: InferenceRequest):
    """Generate IOCs from prompt"""
    try:
        response = inference_engine.generate_structured_iocs(
            request.prompt, 
            request.max_length
        )
        return InferenceResponse(response=response, status="success")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}

def main():
    """Run the inference server"""
    uvicorn.run(
        "scripts.inference:app",
        host="0.0.0.0",
        port=8000,
        reload=False  # Set to True for development
    )

if __name__ == "__main__":
    # For direct usage without API
    engine = CTIInferenceEngine()
    engine.load_model()
    
    # Example usage
    test_prompt = "List Indicators of Compromise in APT35"
    result = engine.generate_structured_iocs(test_prompt)
    print("Generated IOCs:")
    print(result)
