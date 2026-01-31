from setuptools import setup,find_packages

with open("requirements.txt") as f:
    requirements = f.read().splitlines()

setup(
    name="Healthcare-appointment",
    version="0.2",
    author="Rishabh Anand",
    packages=find_packages(),
    install_requires = requirements,
)

