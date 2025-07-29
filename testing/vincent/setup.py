from setuptools import setup, find_packages

setup(
  name="cot_prune",
  version="0.1",
  package_dir={"": "src"},
  packages=find_packages(where="src"),
  install_requires=[
    "torch", "transformers", "tqdm", "scikit-learn"
  ],
  entry_points={
    "console_scripts": [
      "extract-hidden = cot_prune.cli:main_extract",
      "build-steer    = cot_prune.cli:main_build",
      "prune-gen      = cot_prune.cli:main_prune",
    ],
  },
)
