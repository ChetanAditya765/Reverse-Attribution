"""
Complete environment setup script for the Reverse-Attribution project.
Creates a Conda (or Mamba) environment, installs all dependencies, writes
configuration files, and verifies that key packages import correctly.

Usage examples
--------------
# Standard automated setup
python setup_environment.py

# Use a custom environment name
python setup_environment.py --environment-name ra-env

# Skip Conda (install into your active Python)
python setup_environment.py --skip-conda
"""

from __future__ import annotations

import argparse
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Dict, List

PROJECT_ROOT = Path(__file__).resolve().parent
PY_VER = "3.9"
DEFAULT_ENV = "reverse-attribution"


# --------------------------------------------------------------------------- #
# Utility helpers
# --------------------------------------------------------------------------- #
def _cmd_ok(cmd: list[str]) -> bool:
    try:
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def _print_pair(label: str, ok: bool):
    status = "✅" if ok else "❌"
    print(f"  {status} {label}")


# --------------------------------------------------------------------------- #
# Core class
# --------------------------------------------------------------------------- #
class SetupEnv:
    """End-to-end environment builder."""

    # Conda-channel packages (strict pinning for reproducibility)
    CORE_DEPS: list[str] = [
        f"python={PY_VER}",
        "pytorch=2.1.0",
        "torchvision=0.16.0",
        "pytorch-cuda=11.8",
        "transformers=4.35.0",
        "datasets=2.14.0",
        "scikit-learn=1.3.0",
        "numpy=1.24.0",
        "pandas=2.1.0",
        "matplotlib=3.7.0",
        "seaborn=0.12.0",
        "plotly=5.17.0",
        "streamlit=1.28.0",
        "jupyter=1.0.0",
    ]

    # pip-only packages
    PIP_DEPS: list[str] = [
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

    def __init__(self, env_name: str, skip_conda: bool):
        self.env_name = env_name
        self.skip_conda = skip_conda
        self.conda_cmd = "mamba" if _cmd_ok(["mamba", "--version"]) else "conda"

    # --------------------------------------------------------------------- #
    # Steps
    # --------------------------------------------------------------------- #
    def check_prereqs(self) -> bool:
        print("🔍  Checking prerequisites")
        ok_python = sys.version_info >= (3, 9)
        _print_pair("Python 3.9+", ok_python)

        if not self.skip_conda:
            ok_conda = _cmd_ok([self.conda_cmd, "--version"])
            _print_pair(self.conda_cmd, ok_conda)
        else:
            ok_conda = True

        ok_git = _cmd_ok(["git", "--version"])
        _print_pair("git", ok_git)

        return ok_python and ok_conda and ok_git

    def create_env(self):
        if self.skip_conda:
            print("⏭️  Skipping Conda environment creation")
            return
        print(f"🐍  Creating Conda environment '{self.env_name}'")
        cmd = [
            self.conda_cmd,
            "create",
            "-y",
            "-n",
            self.env_name,
            "-c",
            "pytorch",
            "-c",
            "nvidia",
            "-c",
            "conda-forge",
            *self.CORE_DEPS,
        ]
        subprocess.run(cmd, check=True)
        print("✅  Conda environment created")

    def install_pip(self):
        print("📦  Installing pip-only packages")
        pip_exec = ["pip", "install"] if self.skip_conda else ["conda", "run", "-n", self.env_name, "pip", "install"]
        subprocess.run(pip_exec + self.PIP_DEPS, check=True)
        print("✅  pip installation completed")

    def scaffold_dirs(self):
        print("📁  Creating project folders")
        for d in ["data", "checkpoints", "results", "logs", "configs", "user_study_data"]:
            (PROJECT_ROOT / d).mkdir(exist_ok=True)
            print(f"  • {d}/")

    def write_requirements(self):
        req_txt = PROJECT_ROOT / "requirements.txt"
        with open(req_txt, "w") as f:
            for dep in self.CORE_DEPS:
                if dep.startswith("python"):
                    continue
                pkg = dep.replace("=", "==")
                if "pytorch-cuda" in pkg:
                    continue
                f.write(f"{pkg}\n")
            for dep in self.PIP_DEPS:
                f.write(f"{dep}\n")
        print(f"📝  Wrote {req_txt}")

    def verify(self):
        print("🔧  Verifying key imports")
        test_pkgs = ["torch", "transformers", "datasets", "captum", "shap", "streamlit"]
        base_cmd = ["python", "-c"] if self.skip_conda else ["conda", "run", "-n", self.env_name, "python", "-c"]
        for pkg in test_pkgs:
            try:
                subprocess.run(base_cmd + [f"import {pkg}; print('{pkg} OK')"], check=True, stdout=subprocess.DEVNULL)
                _print_pair(pkg, True)
            except subprocess.CalledProcessError:
                _print_pair(pkg, False)

    # --------------------------------------------------------------------- #
    # Orchestrator
    # --------------------------------------------------------------------- #
    def run(self):
        if not self.check_prereqs():
            print("❌  Unmet prerequisites – aborting.")
            sys.exit(1)

        if not self.skip_conda:
            self.create_env()
        self.install_pip()
        self.scaffold_dirs()
        self.write_requirements()
        self.verify()
        print("\n🎉  Setup finished. Next steps:")
        if not self.skip_conda:
            print(f"  • conda activate {self.env_name}")
        print("  • python reproduce_results.py --all\n")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Reverse-Attribution environment setup")
    parser.add_argument("--environment-name", "-n", default=DEFAULT_ENV, help="Conda environment name")
    parser.add_argument("--skip-conda", action="store_true", help="Install into current Python without Conda")
    args = parser.parse_args()

    SetupEnv(env_name=args.environment_name, skip_conda=args.skip_conda).run()
