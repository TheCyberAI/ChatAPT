#!/usr/bin/env python3
"""
Enhanced Data Processing with fallback methods
"""

import os
import json
import pandas as pd
from pathlib import Path
import iocextract
import pdfplumber
import magic
from typing import List, Dict, Any
import logging
import re
from config.settings import config

logger = logging.getLogger(__name__)

class EnhancedIOCExtractor:
    """Extract IOCs with multiple fallback methods"""
    
    def __init__(self):
        self.processed_data_dir = Path(config.data.processed_data_dir)
        self.processed_data_dir.mkdir(parents=True, exist_ok=True)
    
    def extract_text_from_file(self, file_path: Path) -> str:
        """Extract text from various file formats"""
        try:
            if file_path.suffix == '.txt':
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    return f.read()
            elif file_path.suffix == '.pdf':
                text = ""
                with pdfplumber.open(file_path) as pdf:
                    for page in pdf.pages:
                        text += page.extract_text() or ""
                return text
            elif file_path.suffix == '.json':
                with open(file_path, 'r') as f:
                    data = json.load(f)
                    return json.dumps(data, indent=2)
            else:
                return ""
        except Exception as e:
            logger.error(f"Error extracting text from {file_path}: {e}")
            return ""
    
    def extract_iocs_regex(self, text: str) -> Dict[str, List[str]]:
        """Extract IOCs using regex patterns as fallback"""
        iocs = {
            'sha256': [],
            'md5': [],
            'sha1': [],
            'ipv4': [],
            'urls': [],
            'domains': [],
            'emails': [],
        }
        
        try:
            # SHA256 (64 hex characters)
            sha256_pattern = r'\b[a-fA-F0-9]{64}\b'
            iocs['sha256'] = re.findall(sha256_pattern, text)
            
            # MD5 (32 hex characters)
            md5_pattern = r'\b[a-fA-F0-9]{32}\b'
            iocs['md5'] = re.findall(md5_pattern, text)
            
            # SHA1 (40 hex characters)
            sha1_pattern = r'\b[a-fA-F0-9]{40}\b'
            iocs['sha1'] = re.findall(sha1_pattern, text)
            
            # IPv4
            ipv4_pattern = r'\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b'
            iocs['ipv4'] = re.findall(ipv4_pattern, text)
            
            # URLs
            url_pattern = r'https?://[^\s]+'
            iocs['urls'] = re.findall(url_pattern, text)
            
            # Domains
            domain_pattern = r'\b[a-zA-Z0-9]([a-zA-Z0-9\-]{0,61}[a-zA-Z0-9])?(\.[a-zA-Z]{2,})+\b'
            iocs['domains'] = re.findall(domain_pattern, text)
            
            # Emails
            email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
            iocs['emails'] = re.findall(email_pattern, text)
            
        except Exception as e:
            logger.error(f"Regex extraction failed: {e}")
        
        return iocs
    
    def extract_iocs_iocextract(self, text: str) -> Dict[str, List[str]]:
        """Extract IOCs using iocextract library"""
        iocs = {
            'sha256': [],
            'md5': [],
            'sha1': [],
            'ipv4': [],
            'urls': [],
            'domains': [],
            'emails': [],
        }
        
        try:
            iocs['sha256'] = list(iocextract.extract_sha256_hashes(text))
            iocs['md5'] = list(iocextract.extract_md5_hashes(text))
            iocs['sha1'] = list(iocextract.extract_sha1_hashes(text))
            iocs['ipv4'] = list(iocextract.extract_ips(text, refang=True))
            iocs['urls'] = list(iocextract.extract_urls(text, refang=True))
            iocs['domains'] = list(iocextract.extract_domains(text, refang=True))
            iocs['emails'] = list(iocextract.extract_email_addresses(text, refang=True))
        except Exception as e:
            logger.error(f"iocextract failed: {e}")
        
        return iocs
    
    def extract_iocs(self, text: str) -> Dict[str, List[str]]:
        """Extract IOCs using best available method"""
        # Try iocextract first
        iocs = self.extract_iocs_iocextract(text)
        
        # If iocextract found very little, try regex
        total_iocs = sum(len(ioc_list) for ioc_list in iocs.values())
        if total_iocs < 3:
            regex_iocs = self.extract_iocs_regex(text)
            # Use whichever method found more IOCs
            for key in iocs.keys():
                if len(regex_iocs[key]) > len(iocs[key]):
                    iocs[key] = regex_iocs[key]
        
        # Remove duplicates and limit quantities
        for key in iocs:
            iocs[key] = list(set(iocs[key]))[:10]
        
        return iocs
    
    def create_training_example(self, file_path: Path, text: str, iocs: Dict) -> Dict[str, Any]:
        """Create a training example in the required format"""
        
        # Create structured completion
        completion_parts = []
        
        if iocs['sha256']:
            completion_parts.append("## SHA256sum")
            for hash_val in iocs['sha256'][:6]:
                completion_parts.append(f"- {hash_val}")
        
        if iocs['md5']:
            completion_parts.append("## MD5sum")
            for hash_val in iocs['md5'][:6]:
                completion_parts.append(f"- {hash_val}")
        
        if iocs['sha1']:
            completion_parts.append("## SHA1sum")
            for hash_val in iocs['sha1'][:6]:
                completion_parts.append(f"- {hash_val}")
        
        if iocs['ipv4']:
            completion_parts.append("## IP Addresses")
            for ip in iocs['ipv4'][:2]:
                completion_parts.append(f"- {ip}")
        
        completion = "\n".join(completion_parts)
        
        return {
            "prompt": f"List Indicators of Compromise in {file_path.stem.replace('_', ' ')}",
            "completion": completion,
            "metadata": {
                "source_file": file_path.name,
                "ioc_counts": {k: len(v) for k, v in iocs.items()}
            }
        }
    
    def process_files(self) -> None:
        """Process all files in raw data directory"""
        raw_dir = Path(config.data.raw_data_dir)
        training_data = []
        
        files = list(raw_dir.iterdir())
        logger.info(f"Found {len(files)} files to process")
        
        for file_path in files:
            if file_path.is_file() and file_path.suffix in ['.txt', '.pdf', '.json']:
                logger.info(f"Processing: {file_path.name}")
                
                # Extract text
                text = self.extract_text_from_file(file_path)
                if not text.strip():
                    logger.warning(f"No text extracted from {file_path.name}")
                    continue
                
                # Extract IOCs
                iocs = self.extract_iocs(text)
                
                # Skip if no IOCs found
                total_iocs = sum(len(ioc_list) for ioc_list in iocs.values())
                if total_iocs == 0:
                    logger.warning(f"No IOCs found in {file_path.name}")
                    continue
                
                # Create training example
                example = self.create_training_example(file_path, text, iocs)
                training_data.append(example)
                logger.info(f"Extracted {total_iocs} IOCs from {file_path.name}")
        
        # Save training data
        output_file = self.processed_data_dir / "training_data.json"
        with open(output_file, 'w') as f:
            json.dump(training_data, f, indent=2)
        
        logger.info(f"Processed {len(training_data)} files with IOCs")
        
        # Print summary
        if training_data:
            logger.info("\nProcessing Summary:")
            for ioc_type in training_data[0]['metadata']['ioc_counts'].keys():
                count = sum(1 for ex in training_data if ex['metadata']['ioc_counts'][ioc_type] > 0)
                logger.info(f"  - {ioc_type}: {count} files")

def main():
    """Main data processing function"""
    processor = EnhancedIOCExtractor()
    processor.process_files()

if __name__ == "__main__":
    main()
