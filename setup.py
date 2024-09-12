from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="optuml",  # The name of your package on PyPI
    version="0.1.1",  # Start with 0.1.0 for the initial release
    author="Filip S.",
    author_email="filip.ursynow@gmail.com",
    description="Hyperparameter optimization for multiple machine learning algorithms using Optuna, with Scikit-learn API",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/filipsPL/optuml",  # URL to the package's homepage (e.g., GitHub)
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=[
        "optuna",
        "scikit-learn",
        "catboost",
        "xgboost",
        "numpy"
    ],
)
