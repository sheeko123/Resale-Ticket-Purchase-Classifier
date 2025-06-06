from setuptools import setup, find_packages

setup(
    name="tppier17",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "pandas>=1.3.0",
        "numpy>=1.20.0",
        "plotly>=5.3.0",
        "jupyter>=1.0.0",
        "scikit-learn>=0.24.0",
        "matplotlib>=3.4.0",
        "seaborn>=0.11.0",
        "python-dotenv>=0.19.0",
    ],
    author="Your Name",
    author_email="your.email@example.com",
    description="Ticket Price Analysis Project",
    python_requires=">=3.8",
) 