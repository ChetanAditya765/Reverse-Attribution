"""
Setup script for Reverse Attribution package.
Allows installation as a Python package for easier import and distribution.

Installation:
    pip install -e .  # Editable install for development
    pip install .     # Regular install
"""

from setuptools import setup, find_packages
import os
from pathlib import Path

# Read README for long description
README_PATH = Path(__file__).parent / "README.md"
if README_PATH.exists():
    with open(README_PATH, "r", encoding="utf-8") as f:
        long_description = f.read()
else:
    long_description = "Reverse Attribution: Explaining Model Uncertainty via Counter-Evidence Analysis"

# Read requirements
REQUIREMENTS_PATH = Path(__file__).parent / "requirements.txt"
requirements = []
if REQUIREMENTS_PATH.exists():
    with open(REQUIREMENTS_PATH, "r", encoding="utf-8") as f:
        requirements = [line.strip() for line in f if line.strip() and not line.startswith("#")]

# Additional requirements for different use cases
extras_require = {
    'dev': [
        'pytest>=7.4.0',
        'pytest-cov>=4.1.0',
        'black>=23.7.0',
        'isort>=5.12.0',
        'flake8>=6.0.0',
        'mypy>=1.5.0',
        'pre-commit>=3.3.0'
    ],
    'notebooks': [
        'jupyter>=1.0.0',
        'ipykernel>=6.25.0',
        'ipywidgets>=8.1.0',
        'notebook>=7.0.0'
    ],
    'visualization': [
        'plotly>=5.17.0',
        'dash>=2.14.0',
        'streamlit>=1.28.0',
        'wordcloud>=1.9.2',
        'colorcet>=3.0.1'
    ],
    'user_studies': [
        'streamlit>=1.28.0',
        'plotly>=5.17.0',
        'pandas>=2.1.0',
        'scipy>=1.11.0'
    ]
}

# All extras combined
extras_require['all'] = list(set().union(*extras_require.values()))

setup(
    name="reverse-attribution",
    version="1.0.0",
    author="Chetan Aditya Lakka",
    author_email="your.email@domain.com",  # Update with actual email
    description="Reverse Attribution: Explaining Model Uncertainty via Counter-Evidence Analysis",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ChetanAditya765/Reverse-Attribution",
    project_urls={
        "Bug Tracker": "https://github.com/ChetanAditya765/Reverse-Attribution/issues",
        "Documentation": "https://github.com/ChetanAditya765/Reverse-Attribution#readme",
        "Source Code": "https://github.com/ChetanAditya765/Reverse-Attribution",
        "Paper": "https://arxiv.org/abs/your-paper-id",  # Update when available
    },
    packages=find_packages(exclude=["tests", "tests.*", "docs", "docs.*"]),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.9",
    install_requires=requirements,
    extras_require=extras_require,
    entry_points={
        "console_scripts": [
            "ra-reproduce=reproduce_results:main",
            "ra-setup=setup_environment:main",
            "ra-train=scripts.script:main",
            "ra-evaluate=evaluate:main",
        ],
    },
    include_package_data=True,
    package_data={
        "reverse_attribution": [
            "configs/*.yml",
            "configs/*.yaml",
            "configs/*.json",
        ],
    },
    zip_safe=False,
    keywords=[
        "explainable-ai",
        "machine-learning", 
        "attribution",
        "interpretability",
        "pytorch",
        "transformers",
        "computer-vision",
        "nlp"
    ],
    # Development dependencies
    setup_requires=[
        "wheel",
        "setuptools_scm",
    ],
    # Testing configuration
    test_suite="tests",
    tests_require=extras_require['dev'],
    # Metadata for PyPI
    license="MIT",
    platforms=["any"],
    # Additional metadata
    maintainer="Chetan Aditya Lakka",
    maintainer_email="your.email@domain.com",  # Update with actual email
)
