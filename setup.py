"""
DeepRibo重构版本安装脚本
"""

from setuptools import setup, find_packages
from pathlib import Path

# 读取README文件
readme_path = Path(__file__).parent / "README.md"
long_description = readme_path.read_text(encoding="utf-8") if readme_path.exists() else ""

setup(
    name="deepribo-refactor",
    version="2.0.1",
    author="Jim Clauwaert, Gerben Menschaert",
    author_email="team@deepribo.org",
    description="深度神经网络用于原核生物基因注释 - Python 3.12重构版本",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Biobix/DeepRibo",
    project_urls={
        "Bug Tracker": "https://github.com/Biobix/DeepRibo/issues",
        "Documentation": "https://github.com/Biobix/DeepRibo/blob/master/README.md",
        "Source Code": "https://github.com/Biobix/DeepRibo",
    },
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "scipy>=1.10.0",
        "scikit-learn>=1.3.0",
        "biopython>=1.80",
        "matplotlib>=3.7.0",
        "pyyaml>=6.0",
    ],
    extras_require={
        "dev": [
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.5.0",
            "pytest>=7.4.0",
            "jupyter>=1.0.0",
        ],
        "viz": [
            "seaborn>=0.12.0",
            "tensorboard>=2.12.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "deepribo-train=cli.train:main",
            "deepribo-predict=cli.predict:main",
            "deepribo-parse=cli.data:main",
        ],
    },
)
