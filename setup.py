from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="optuml",
    version="0.2.2",
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
        "optuna>=3.0.1",
        "scikit-learn",
        "catboost",
        "graphviz",   # Needed for CatBoost
        "xgboost",
        "numpy",
        "wrapt_timeout_decorator"
    ],
)
