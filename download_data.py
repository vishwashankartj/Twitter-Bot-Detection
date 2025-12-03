"""
Kaggle Dataset Downloader for Twitter Bot Detection
Downloads the Twitter Bot Detection dataset from Kaggle
"""

import os
import sys
import zipfile
from pathlib import Path
import subprocess
import json


class KaggleDatasetDownloader:
    """Download and extract Twitter bot detection dataset from Kaggle"""
    
    def __init__(self, dataset_name="davidmartngutirrez/twitter-bots-accounts", 
                 download_dir="data/raw"):
        """
        Initialize the downloader
        
        Args:
            dataset_name: Kaggle dataset identifier
            download_dir: Directory to download data to
        """
        self.dataset_name = dataset_name
        self.download_dir = Path(download_dir)
        self.download_dir.mkdir(parents=True, exist_ok=True)
        
    def check_kaggle_credentials(self):
        """Check if Kaggle API credentials are configured"""
        kaggle_json = Path.home() / '.kaggle' / 'kaggle.json'
        
        if not kaggle_json.exists():
            print("\n⚠️  Kaggle API credentials not found!")
            print("\nTo download datasets from Kaggle, you need to:")
            print("1. Go to https://www.kaggle.com/settings/account")
            print("2. Scroll to 'API' section")
            print("3. Click 'Create New API Token'")
            print("4. This will download kaggle.json")
            print("5. Move it to ~/.kaggle/kaggle.json")
            print("\nOr run these commands:")
            print("  mkdir -p ~/.kaggle")
            print("  mv ~/Downloads/kaggle.json ~/.kaggle/")
            print("  chmod 600 ~/.kaggle/kaggle.json")
            return False
        
        # Set proper permissions
        os.chmod(kaggle_json, 0o600)
        return True
    
    def download_dataset(self):
        """Download dataset from Kaggle"""
        if not self.check_kaggle_credentials():
            return False
        
        print(f"\n📥 Downloading dataset: {self.dataset_name}")
        print(f"📁 Destination: {self.download_dir.absolute()}")
        
        try:
            # Use kaggle CLI to download
            cmd = [
                "kaggle", "datasets", "download",
                "-d", self.dataset_name,
                "-p", str(self.download_dir),
                "--unzip"
            ]
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True
            )
            
            print("✅ Dataset downloaded successfully!")
            print(f"\n{result.stdout}")
            return True
            
        except subprocess.CalledProcessError as e:
            print(f"❌ Error downloading dataset: {e}")
            print(f"Error output: {e.stderr}")
            
            # Try alternative dataset names
            print("\n💡 Trying alternative dataset sources...")
            alternative_datasets = [
                "davidmartngutirrez/twitter-bots-accounts",
                "davidmartinezgutierrez/twitter-bot-detection",
                "twitter-bot-detection"
            ]
            
            for alt_dataset in alternative_datasets:
                if alt_dataset != self.dataset_name:
                    print(f"\nTrying: {alt_dataset}")
                    self.dataset_name = alt_dataset
                    try:
                        cmd[3] = alt_dataset
                        subprocess.run(cmd, capture_output=True, text=True, check=True)
                        print(f"✅ Successfully downloaded from: {alt_dataset}")
                        return True
                    except:
                        continue
            
            print("\n⚠️  Could not download from Kaggle automatically.")
            print("\nManual download instructions:")
            print("1. Visit: https://www.kaggle.com/datasets/search?q=twitter+bot+detection")
            print("2. Download a suitable dataset")
            print(f"3. Extract it to: {self.download_dir.absolute()}")
            return False
        
        except FileNotFoundError:
            print("❌ Kaggle CLI not found. Installing...")
            subprocess.run([sys.executable, "-m", "pip", "install", "kaggle"])
            print("✅ Kaggle installed. Please run this script again.")
            return False
    
    def list_downloaded_files(self):
        """List all files in the download directory"""
        print(f"\n📂 Files in {self.download_dir}:")
        files = list(self.download_dir.glob("*"))
        
        if not files:
            print("  (empty)")
            return []
        
        for file in files:
            size = file.stat().st_size / (1024 * 1024)  # MB
            print(f"  - {file.name} ({size:.2f} MB)")
        
        return files


def main():
    """Main function to download dataset"""
    print("=" * 60)
    print("Twitter Bot Detection - Kaggle Dataset Downloader")
    print("=" * 60)
    
    # Initialize downloader
    downloader = KaggleDatasetDownloader()
    
    # Download dataset
    success = downloader.download_dataset()
    
    # List downloaded files
    files = downloader.list_downloaded_files()
    
    if success and files:
        print("\n✅ Setup complete! Dataset ready for processing.")
        print("\nNext steps:")
        print("  1. Run: python src/data/data_loader.py")
        print("  2. Or start training: python train.py")
    else:
        print("\n⚠️  Please download the dataset manually and place it in data/raw/")
    
    return success


if __name__ == "__main__":
    main()
