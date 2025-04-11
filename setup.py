from setuptools import find_packages, setup

setup(
    name="fake_news_detector",
    version="0.1.0",
    description="Multimodal Fake News Detection System",
    author="Preston Bied",
    author_email="pgbied@gmail.com",
    packages=find_packages(include=["src", "src.*"]),
    python_requires=">=3.9",
    install_requires=[
        # Core requirements will be read from requirements.txt
    ],
    extras_require={
        "dev": [
            "black>=22.1.0",
            "flake8>=4.0.1",
            "isort>=5.10.1",
            "mypy>=0.931",
            "pytest>=7.0.0",
            "pytest-cov>=3.0.0",
            "pre-commit>=2.17.0",
        ],
        "gpu": [
            "torch>=1.10.1+cu113",  # Adjust based on your CUDA version
            "torchvision>=0.11.2+cu113",
            "tensorflow-gpu>=2.8.0",  # Optional if you want TF with GPU
        ],
        "cpu": [
            "torch>=1.10.1",
            "torchvision>=0.11.2",
            "tensorflow>=2.8.0",  # Optional if you want TF
        ],
    },
)