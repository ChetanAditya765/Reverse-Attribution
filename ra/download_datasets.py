"""
Script to download all required datasets for the paper experiments.
Downloads IMDB, Yelp Polarity, and CIFAR-10 datasets.
"""

import os
import argparse
from datasets import load_dataset
from torchvision import datasets
from dataset_utils import DatasetLoader


def download_all_datasets(data_dir: str = "./data"):
    """Download all datasets used in the paper."""
    print("🔄 Starting dataset downloads...")
    
    loader = DatasetLoader(data_dir)
    
    # Download text datasets
    print("\n📚 Downloading text datasets...")
    
    try:
        print("  - IMDB dataset...")
        load_dataset("imdb", cache_dir=data_dir)
        print("  ✅ IMDB downloaded successfully")
    except Exception as e:
        print(f"  ❌ Error downloading IMDB: {e}")
    
    try:
        print("  - Yelp Polarity dataset...")
        load_dataset("yelp_polarity", cache_dir=data_dir)
        print("  ✅ Yelp Polarity downloaded successfully")
    except Exception as e:
        print(f"  ❌ Error downloading Yelp: {e}")
    
    # Download vision datasets
    print("\n🖼️  Downloading vision datasets...")
    
    try:
        print("  - CIFAR-10 dataset...")
        datasets.CIFAR10(
            root=os.path.join(data_dir, "cifar10"),
            train=True,
            download=True
        )
        datasets.CIFAR10(
            root=os.path.join(data_dir, "cifar10"),
            train=False,
            download=True
        )
        print("  ✅ CIFAR-10 downloaded successfully")
    except Exception as e:
        print(f"  ❌ Error downloading CIFAR-10: {e}")
    
    print(f"\n🎉 Dataset downloads completed! Data saved to: {data_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download datasets for RA experiments")
    parser.add_argument(
        "--data_dir", 
        type=str, 
        default="./data",
        help="Directory to save datasets"
    )
    
    args = parser.parse_args()
    download_all_datasets(args.data_dir)
