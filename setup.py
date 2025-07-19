"""
setup.py – Package installation script for Reverse Attribution framework.

Defines console entry points and dependencies,
and installs your integrated-model implementations.
"""

from setuptools import setup, find_packages
from pathlib import Path

HERE = Path(__file__).parent
README = (HERE/"README.md").read_text(encoding="utf-8") if (HERE/"README.md").exists() else ""
REQUIREMENTS = (HERE/"requirements.txt").read_text().splitlines()

setup(
    name="reverse-attribution",
    version="1.0.0",
    author="Chetan Aditya Lakka",
    author_email="your.email@domain.com",
    description="Reverse Attribution: Explaining Model Uncertainty via Counter-Evidence Analysis",
    long_description=README,
    long_description_content_type="text/markdown",
    url="https://github.com/ChetanAditya765/Reverse-Attribution",
    packages=find_packages(exclude=["tests","tests.*"]),
    install_requires=REQUIREMENTS,
    extras_require={
        "dev": ["pytest>=7.4.0","pytest-cov>=4.1.0","black>=23.7.0","isort>=5.12.0","flake8>=6.0.0","mypy>=1.5.0"],
        "viz": ["plotly>=5.17.0","wordcloud>=1.9.2"],
        "studies": ["streamlit>=1.28.0","pandas>=2.1.0"],
    },
    entry_points={
        "console_scripts": [
            "ra-setup=setup_environment:main",
            "ra-reproduce=reproduce_results:main",
            "ra-eval=scripts.script_3:main",
            "ra-infer=app:main",
        ]
    },
    include_package_data=True,
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Researchers",
        "Programming Language :: Python :: 3.9",
        "License :: OSI Approved :: MIT License",
    ],
    python_requires=">=3.9",
    keywords=["explainable-ai","attribution","pytorch","transformers","cifar10"]
)
