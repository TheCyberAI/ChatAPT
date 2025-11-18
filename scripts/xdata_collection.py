#!/usr/bin/env python3
"""
Data Collection Script for CTI LLM Project
Collects threat intelligence data from various sources
"""

import os
import requests
import pandas as pd
from pathlib import Path
import time
from typing import List, Dict
import logging
from config.settings import config

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CTIDataCollector:
    """Collects CTI data from various sources"""
    
    def __init__(self):
        self.raw_data_dir = Path(config.data.raw_data_dir)
        self.raw_data_dir.mkdir(parents=True, exist_ok=True)
    
    def download_aptnotes(self) -> None:
        """Download APTNotes data from GitHub"""
        logger.info("Downloading APTNotes data...")
        
        # GitHub API URL for aptnotes repository
        api_url = "https://api.github.com/repos/aptnotes/data/contents"
        
        try:
            response = requests.get(api_url)
            response.raise_for_status()
            
            files = response.json()
            downloaded_files = 0
            
            for file_info in files:
                if file_info['name'].endswith(('.pdf', '.txt')) and file_info['size'] < 10 * 1024 * 1024:
                    file_url = file_info['download_url']
                    file_path = self.raw_data_dir / file_info['name']
                    
                    # Skip if file already exists
                    if file_path.exists():
                        logger.info(f"File already exists: {file_info['name']}")
                        continue
                    
                    # Download file
                    file_response = requests.get(file_url)
                    file_response.raise_for_status()
                    
                    with open(file_path, 'wb') as f:
                        f.write(file_response.content)
                    
                    downloaded_files += 1
                    logger.info(f"Downloaded: {file_info['name']} ({file_info['size']} bytes)")
                    
                    # Rate limiting
                    time.sleep(1)
            
            logger.info(f"Successfully downloaded {downloaded_files} files")
            
        except Exception as e:
            logger.error(f"Error downloading APTNotes: {e}")
    
    def create_sample_data(self) -> None:
        """Create sample CTI data for testing"""
        logger.info("Creating sample CTI data...")
        
        sample_reports = [
            {
                "filename": "APT35_Threat_Report.txt",
                "content": """
                APT35 Threat Intelligence Report
                
                Indicators of Compromise:
                - IP Address: 192.168.1.100
                - Domain: malicious[.]com
                - SHA256: 35a485972282b7c0e8e3a7a9cbf86ad93856378f696cc8e230be5099cdb89208
                - MD5: a1d378111335d450769049446df79983
                - SHA1: bb700e1ef97e1eed56bb275fde2c5faed008c225
                - URL: http://evil.com/malware.exe
                - Email: attacker@evil.com
                """
            },
            {
                "filename": "FIN7_Malware_Analysis.txt", 
                "content": """
                FIN7 Cyber Crime Group Malware Analysis
                
                Indicators of Compromise:
                - IP Address: 203.0.113.45
                - Domain: carbanak-group[.]com
                - SHA256: cd2ba296828660ecd07a36e8931b851dda0802069ed926b3161745aae9aaddaa
                - MD5: 5d41402abc4b2a76b9719d911017c592
                - SHA1: aaf4c61ddcc5e8a2dabede0f3b482cd9aea9434d
                - URL: https://carbanak-group.com/loader.bin
                """
            },
            {
                "filename": "Lazarus_Group_Campaign.txt",
                "content": """
                Lazarus Group Cyber Campaign Analysis
                
                Indicators of Compromise:
                - IP Address: 172.16.254.1
                - Domain: hidden-cobra[.]org
                - SHA256: 767bd025c8e7d36f64dbd636ec0f29e873d1e3ca415d5a449053a68916fe894
                - MD5: 3a62b26311583a23767c35d56b95175d
                - SHA1: aa791a0a98a30e10119b8cc1399ab1306275fc1f
                - URL: http://hidden-cobra.org/controller.dll
                """
            }
        ]
        
        for report in sample_reports:
            file_path = self.raw_data_dir / report["filename"]
            with open(file_path, 'w') as f:
                f.write(report["content"])
            logger.info(f"Created sample file: {report['filename']}")

def main():
    """Main data collection function"""
    collector = CTIDataCollector()
    
    # Try to download APTNotes
    try:
        collector.download_aptnotes()
    except Exception as e:
        logger.error(f"APTNotes download failed: {e}")
        logger.info("Creating sample data instead...")
        collector.create_sample_data()
    
    logger.info("Data collection completed!")

if __name__ == "__main__":
    main()
