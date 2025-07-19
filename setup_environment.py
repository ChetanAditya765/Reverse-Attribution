"""
Automated environment setup for Reverse Attribution project.
Creates a Conda environment (or uses system Python), installs dependencies,
scaffolds project structure, writes config files, and verifies model imports.

Usage:
    python setup_environment.py
    python setup_environment.py --env-name ra-env --skip-conda
"""

import argparse, sys, subprocess, platform
from pathlib import Path
import logging
import yaml

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

# Core conda dependencies
CORE_DEPS = [
    "python=3.9",
    "pytorch=2.1.0",
    "torchvision=0.16.0",
    "pytorch-cuda=11.8",
    "transformers=4.35.0",
    "datasets=2.14.0",
    "numpy=1.24.0",
    "pandas=2.1.0",
    "scikit-learn=1.3.0",
    "matplotlib=3.7.0",
    "seaborn=0.12.0",
    "plotly=5.17.0",
    "streamlit=1.28.0",
]

PIP_DEPS = [
    "captum==0.6.0",
    "shap==0.42.0",
    "lime==0.2.0.1",
    "wordcloud==1.9.2",
    "colorcet==3.0.1",
    "opencv-python==4.8.0.76",
    "tqdm==4.66.0",
    "pyyaml==6.0.1",
    "pytest==7.4.0",
    "pytest-cov==4.1.0",
    "black==23.7.0",
    "isort==5.12.0",
    "flake8==6.0.0",
    "mypy==1.5.0",
]

def cmd_exists(cmd):
    return subprocess.run([cmd, "--version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0

def check_prereqs(skip_conda):
    logger.info("🔍 Checking prerequisites…")
    ok_python = sys.version_info >= (3,9)
    logger.info(f"  {'✅' if ok_python else '❌'} Python ≥3.9")
    if not skip_conda:
        manager = "mamba" if cmd_exists("mamba") else "conda"
        ok_conda = cmd_exists(manager)
        logger.info(f"  {'✅' if ok_conda else '❌'} {manager}")
        if not ok_conda:
            logger.error("Conda/Mamba required or use --skip-conda")
            sys.exit(1)
    ok_git = cmd_exists("git")
    logger.info(f"  {'✅' if ok_git else '❌'} git")
    return

def create_conda_env(env_name):
    manager = "mamba" if cmd_exists("mamba") else "conda"
    logger.info(f"🐍 Creating Conda env '{env_name}' via {manager}…")
    subprocess.run([manager, "create", "-y", "-n", env_name, "-c", "pytorch", "-c", "nvidia", "-c", "conda-forge"] + CORE_DEPS, check=True)
    logger.info("✅ Conda environment created")

def install_pip(env_name, skip_conda):
    logger.info("📦 Installing pip dependencies…")
    if skip_conda:
        pip_cmd = ["pip", "install"]
    else:
        pip_cmd = ["conda", "run", "-n", env_name, "pip", "install"]
    subprocess.run(pip_cmd + PIP_DEPS, check=True)
    logger.info("✅ Pip packages installed")

def scaffold_dirs():
    logger.info("📁 Scaffolding project directories…")
    for d in ["data","checkpoints","results","logs","configs","user_study_data","reproduction_results"]:
        Path(d).mkdir(exist_ok=True)
        logger.info(f"  • {d}/")

def write_example_configs():
    cfg = {
        "seed": 42,
        "device": "cuda" if cmd_exists("nvidia-smi") else "cpu",
        "data_dir": "./data",
        "checkpoints_dir": "./checkpoints",
        "results_dir": "./results",
        "text_models": {
            "imdb": {"model_class":"BERTSentimentClassifier","model_name":"bert-base-uncased","num_classes":2,"epochs":3,"batch_size":16,"learning_rate":2e-5,"max_length":512,"output_dir":"./checkpoints/bert_imdb"},
            "yelp": {"model_class":"BERTSentimentClassifier","model_name":"roberta-base","num_classes":2,"epochs":3,"batch_size":8,"learning_rate":1e-5,"max_length":512,"output_dir":"./checkpoints/roberta_yelp"},
        },
        "vision_models": {
            "cifar10": {"model_class":"ResNetCIFAR","architecture":"resnet56","num_classes":10,"epochs":200,"batch_size":128,"learning_rate":0.1,"weight_decay":1e-4,"output_dir":"./checkpoints/resnet56_cifar10"}
        }
    }
    with open("configs/experiment.yml","w") as f:
        yaml.dump(cfg, f)
    logger.info("📝 configs/experiment.yml written")

def verify_imports(env_name, skip_conda):
    logger.info("🔧 Verifying key imports…")
    base = [] if skip_conda else ["conda","run","-n",env_name]
    pkgs = ["torch","transformers","datasets","captum","shap","lime","streamlit"]
    for pkg in pkgs:
        cmd = base + ["python","-c",f"import {pkg}"]
        ok = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0
        logger.info(f"  {'✅' if ok else '❌'} {pkg}")

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--env-name","-n",default="reverse-attribution",help="Conda env name")
    p.add_argument("--skip-conda",action="store_true",help="Use system Python")
    args = p.parse_args()

    check_prereqs(args.skip_conda)
    if not args.skip_conda:
        create_conda_env(args.env_name)
    install_pip(args.env_name, args.skip_conda)
    scaffold_dirs()
    write_example_configs()
    verify_imports(args.env_name, args.skip_conda)
    logger.info("\n🎉 Setup complete!\n")

if __name__ == "__main__":
    main()
