"""setup file for the project."""
# code inspired by https://github.com/navdeep-G/setup.py

import io
import os

from setuptools import find_packages, setup

# Package meta-data.
NAME = "mixtera"
DESCRIPTION = "A platform for LLM training data mixture."

URL = "https://github.com/eth-easl/mixtera"
URL_DOKU = "https://github.com/eth-easl/mixtera"
URL_GITHUB = "https://github.com/eth-easl/mixtera"
URL_ISSUES = "https://github.com/eth-easl/mixtera"
EMAIL = "maximilian.boether@inf.ethz.ch"
AUTHOR = "See contributing.md"
REQUIRES_PYTHON = ">=3.12"
KEYWORDS = [""]
REQUIRED = [""]
EXTRAS = {}


here = os.path.abspath(os.path.dirname(__file__))

# Import the README and use it as the long-description.
try:
    with io.open(os.path.join(here, "README.md"), encoding="utf-8") as f:
        long_description = "\n" + f.read()
except FileNotFoundError:
    long_description = DESCRIPTION

# Load the package's _version.py module as a dictionary.
about = {}
project_slug = NAME

# Where the magic happens:
setup(
    name=NAME,
    version="1.0.0",
    description=DESCRIPTION,
    long_description=long_description,
    long_description_content_type="text/markdown",
    author=AUTHOR,
    author_email=EMAIL,
    python_requires=REQUIRES_PYTHON,
    project_urls={"Bug Tracker": URL_ISSUES, "Source Code": URL_GITHUB},
    packages=find_packages(exclude=["tests", "*.tests", "*.tests.*", "tests.*", "tests.*.*"]),
    # If your package is a single module, use this instead of 'packages':
    # py_modules=['mypackage'],
    # entry_points is is required for testing the Python scripts
    install_requires=REQUIRED,
    extras_require=EXTRAS,
    include_package_data=True,
    license="MIT",
    keywords=KEYWORDS,
    scripts = [
        'mixtera/cli/mixtera-cli',
        'mixtera/cli/mixtera' # Duplication of mixtera-cli
    ],
    classifiers=[
        # Trove classifiers
        # Full list: https://pypi.python.org/pypi?%3Aaction=list_classifiers
        "Programming Language :: Python",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: Implementation :: PyPy",
    ],
)