# setup.py
from setuptools import setup, find_packages

setup(
    name="cot_prune",
    version="0.1",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
)
