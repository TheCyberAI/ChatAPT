#!/usr/bin/env python3
"""
Data Collection Script for CTI LLM Project
Downloads real threat intelligence data from APTNotes repository
"""

import os
import requests
import json
from pathlib import Path
import time
import logging
from config.settings import config

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CTIDataCollector:
    """Collects CTI data from APTNotes GitHub repository"""
    
    def __init__(self):
        self.raw_data_dir = Path(config.data.raw_data_dir)
        self.raw_data_dir.mkdir(parents=True, exist_ok=True)
    
    def get_github_files_recursive(self, api_url: str) -> list:
        """Recursively get all files from GitHub repository"""
        all_files = []
        
        try:
            headers = {
                'User-Agent': 'CTI-LLM-Data-Collector/1.0',
                'Accept': 'application/vnd.github.v3+json'
            }
            
            response = requests.get(api_url, headers=headers, timeout=30)
            
            if response.status_code == 403:
                logger.warning("GitHub API rate limit reached. Using alternative method.")
                return self.get_github_files_fallback()
                
            response.raise_for_status()
            items = response.json()
            
            for item in items:
                if item['type'] == 'file':
                    if item['name'].endswith(('.pdf', '.txt', '.md')):
                        all_files.append(item)
                elif item['type'] == 'dir':
                    # Recursively get files from subdirectory
                    sub_files = self.get_github_files_recursive(item['url'])
                    all_files.extend(sub_files)
                    
        except Exception as e:
            logger.error(f"Error accessing GitHub API {api_url}: {e}")
            
        return all_files
    
    def get_github_files_fallback(self) -> list:
        """Fallback method to get important APTNotes files"""
        logger.info("Using fallback method to get APTNotes files...")
        
        # Direct download URLs for known APTNotes files
        direct_files = [
            # 2023 reports
            "https://raw.githubusercontent.com/aptnotes/data/master/2023/2023-12-15%20-%20APT29%20-%20NOBELIUM%20targeting%20organizations%20in%20the%20USA%20-%20Microsoft.pdf",
            "https://raw.githubusercontent.com/aptnotes/data/master/2023/2023-11-20%20-%20Lazarus%20Group%20-%20Operation%20Dream%20Job%20-%20Kaspersky.pdf",
            "https://raw.githubusercontent.com/aptnotes/data/master/2023/2023-10-12%20-%20APT41%20-%20Chrome%20Loader%20malware%20-%20Palo%20Alto%20Networks.pdf",
            
            # 2022 reports
            "https://raw.githubusercontent.com/aptnotes/data/master/2022/2022-09-15%20-%20APT35%20-%20Charming%20Kitten%20-%20ClearSky.pdf",
            "https://raw.githubusercontent.com/aptnotes/data/master/2022/2022-08-22%20-%20FIN7%20-%20Carbanak%20group%20-%20FireEye.pdf",
            
            # Text files and READMEs that might contain IOCs
            "https://raw.githubusercontent.com/aptnotes/data/master/README.md",
            "https://raw.githubusercontent.com/aptnotes/data/master/aptnotes.csv"
        ]
        
        files_info = []
        for url in direct_files:
            filename = url.split('/')[-1]
            files_info.append({
                'name': filename,
                'download_url': url,
                'type': 'file'
            })
            
        return files_info
    
    def download_file(self, file_info: dict) -> bool:
        """Download a single file from GitHub"""
        try:
            file_url = file_info['download_url']
            filename = file_info['name']
            file_path = self.raw_data_dir / filename
            
            # Skip if file already exists
            if file_path.exists():
                logger.info(f"File already exists: {filename}")
                return True
            
            logger.info(f"Downloading: {filename}")
            
            headers = {
                'User-Agent': 'CTI-LLM-Data-Collector/1.0'
            }
            
            response = requests.get(file_url, headers=headers, timeout=60, stream=True)
            response.raise_for_status()
            
            # Check file size
            content_length = response.headers.get('content-length')
            if content_length and int(content_length) > 50 * 1024 * 1024:  # 50MB limit
                logger.warning(f"File too large ({content_length} bytes), skipping: {filename}")
                return False
            
            # Download file
            with open(file_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            
            file_size = file_path.stat().st_size
            logger.info(f"Downloaded: {filename} ({file_size} bytes)")
            return True
            
        except requests.exceptions.Timeout:
            logger.error(f"Timeout downloading {file_info.get('name', 'unknown')}")
            return False
        except requests.exceptions.RequestException as e:
            logger.error(f"Error downloading {file_info.get('name', 'unknown')}: {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error downloading {file_info.get('name', 'unknown')}: {e}")
            return False
    
    def download_aptnotes_data(self) -> int:
        """Download APTNotes data from GitHub"""
        logger.info("Starting APTNotes data download...")
        
        # GitHub API URL for aptnotes repository
        api_url = "https://api.github.com/repos/aptnotes/data/contents"
        
        # Get all files from repository
        all_files = self.get_github_files_recursive(api_url)
        
        if not all_files:
            logger.warning("No files found via API, trying fallback...")
            all_files = self.get_github_files_fallback()
        
        if not all_files:
            logger.error("No files available for download")
            return 0
        
        logger.info(f"Found {len(all_files)} potential files to download")
        
        # Filter for supported formats and reasonable sizes
        downloadable_files = []
        for file_info in all_files:
            if file_info['name'].endswith(('.pdf', '.txt', '.md', '.csv')):
                downloadable_files.append(file_info)
        
        logger.info(f"Filtered to {len(downloadable_files)} supported files")
        
        # Download files
        downloaded_count = 0
        for file_info in downloadable_files:
            if self.download_file(file_info):
                downloaded_count += 1
                time.sleep(1)  # Rate limiting
            
            # Stop if we have enough files
            if downloaded_count >= 20:  # Limit to 20 files for initial testing
                logger.info("Reached download limit of 20 files")
                break
        
        return downloaded_count
    
    def download_from_mitre_attack(self) -> int:
        """Download additional data from MITRE ATT&CK"""
        logger.info("Downloading MITRE ATT&CK data...")
        
        mitre_urls = [
            "https://raw.githubusercontent.com/mitre/cti/master/enterprise-attack/enterprise-attack.json",
            "https://raw.githubusercontent.com/mitre/cti/master/ics-attack/ics-attack.json",
            "https://raw.githubusercontent.com/mitre/cti/master/mobile-attack/mobile-attack.json"
        ]
        
        downloaded_count = 0
        for url in mitre_urls:
            try:
                filename = f"mitre_{url.split('/')[-1]}"
                file_path = self.raw_data_dir / filename
                
                if file_path.exists():
                    continue
                
                logger.info(f"Downloading MITRE data: {filename}")
                response = requests.get(url, timeout=30)
                response.raise_for_status()
                
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(response.text)
                
                downloaded_count += 1
                logger.info(f"Downloaded MITRE data: {filename}")
                time.sleep(1)
                
            except Exception as e:
                logger.error(f"Error downloading MITRE data: {e}")
        
        return downloaded_count
    
    def create_minimal_sample_if_needed(self, downloaded_count: int) -> int:
        """Create minimal sample data only if no files were downloaded"""
        if downloaded_count > 0:
            return 0
            
        logger.warning("No files downloaded, creating minimal sample data...")
        
        sample_content = """
        APT Sample Report - For Testing Only
        
        This is a minimal sample file created because no real threat intelligence data could be downloaded.
        Please check your internet connection and try again.
        
        Sample IOCs for testing:
        - IP Address: 192.168.1.100
        - Domain: example[.]com
        - SHA256: e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855
        - MD5: d41d8cd98f00b204e9800998ecf8427e
        """
        
        file_path = self.raw_data_dir / "sample_minimal_report.txt"
        with open(file_path, 'w') as f:
            f.write(sample_content)
        
        logger.info("Created minimal sample file")
        return 1

def main():
    """Main data collection function"""
    collector = CTIDataCollector()
    
    # Download APTNotes data
    downloaded_count = collector.download_aptnotes_data()
    
    # Download MITRE ATT&CK data
    mitre_count = collector.download_from_mitre_attack()
    
    total_downloaded = downloaded_count + mitre_count
    
    # Only create sample if absolutely nothing was downloaded
    if total_downloaded == 0:
        collector.create_minimal_sample_if_needed(total_downloaded)
    
    logger.info(f"Data collection completed! Downloaded {total_downloaded} files.")

if __name__ == "__main__":
    main()
