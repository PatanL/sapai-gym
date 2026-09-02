from setuptools import setup, find_packages

setup(
      name='sapai_gym',
      version='0.2.0',
      packages=find_packages(),
      install_requires=[
          "gymnasium>=0.29.1,<1.3",
          "numpy>=1.26,<3",
          "scikit-learn>=1.5,<2",
          "stable-baselines3>=2.7,<3",
          "sb3-contrib>=2.3,<3",
      ]
)
