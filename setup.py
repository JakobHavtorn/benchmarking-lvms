import os

from setuptools import find_packages, setup


VALID_MODES = ["extra", "torch", "notorch"]


# Parse requirements to use
dependency_links = []
requirements_mode = os.environ.get("REQUIREMENTS_MODE", None)
if requirements_mode is not None:
    requirements_mode = requirements_mode.split()

    if any([m not in VALID_MODES for m in requirements_mode]):
        raise ValueError(f"Valid arguments are {VALID_MODES} but got {requirements_mode}.")
 
    if "torch" in requirements_mode and "notorch" in requirements_mode:
        raise ValueError("Arguments `torch` and `notorch` are mutually exclusive.")

    if "torch" in requirements_mode:
        dependency_links = ["https://download.pytorch.org/whl/torch_stable.html"]

# Select requirements files
if requirements_mode is None:
    requirements_files = []
else:    
    requirements_files = [f"requirements-{m}.txt" for m in requirements_mode]
    
# Read requirements files
requirements = []
for f in requirements_files:
    with open(f) as buffer:
        requirements.extend(buffer.read().splitlines())

requirements = list(set(requirements))
requirements_string = "\n  ".join(requirements)
print(f"Found the following requirements to be installed from {requirements_files}:\n  {requirements_string}")

# Collect packages
packages = find_packages(exclude=("tests", "experiments"))
print("Found the following packages to be created:\n  {}".format("\n  ".join(packages)))

# Get long description from README
with open("README.md", "r") as readme:
    long_description = readme.read()

# Setup the package
setup(
    name="blvm",
    version="1.0.0",
    packages=packages,
    python_requires=">=3.8.0",
    install_requires=requirements,
    dependency_links=dependency_links,
    setup_requires=[],
    ext_modules=[],
    url="https://github.com/JakobHavtorn/benchmarking-lvms",
    author="Jakob Havtorn",
    description="Deep Generative Modelling for Speech",
    long_description=long_description,
    long_description_content_type="text/markdown",
)
