from setuptools import setup, find_packages

setup(
    name="bel",
    version="0.1.0",
    description="Basin Exploration Learning framework for Oil & Gas exploration optimization",
    author="",
    author_email="",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "numpy>=1.21.0",
        "scipy>=1.7.0",
        "matplotlib>=3.4.0",
        "pandas>=1.3.0",
        "scikit-learn>=1.0.0",
        "GPy>=1.10.0",
        "plotly>=5.0.0",
        "seaborn>=0.11.0",
    ],
    python_requires=">=3.8",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
)