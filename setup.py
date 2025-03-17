from setuptools import setup, find_packages

setup(
    name="mnist_kmeans",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.20.0",
        "pandas>=1.3.0",
        "matplotlib>=3.4.0",
        "seaborn>=0.11.0",
        "scikit-learn>=1.0.0",
        "scipy>=1.7.0",
        "tqdm>=4.60.0"
    ],
    author="Your Name",
    author_email="your.email@example.com",
    description="K-Means clustering implementation for MNIST dataset",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
)
