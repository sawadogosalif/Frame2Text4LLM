#!/usr/bin/env python

"""The setup script."""

from setuptools import setup, find_packages
import pathlib
import runpy
import pkg_resources

SRC_PATH = "src"

# get metadata
metadata_path = next(pathlib.Path(SRC_PATH).glob("*/package_metadata.py"))
metadata = runpy.run_path(metadata_path)

author = metadata["__author__"]
email = metadata["__email__"]
doc = metadata["__doc__"]
name = metadata["__name__"]
url = metadata["__url__"]
version = metadata["__version__"]

with open("README.md") as readme_file:
    readme = readme_file.read()

with open("HISTORY.md") as history_file:
    history = history_file.read()

with open("requirements.txt") as requirements_file:
    requirements = [
        str(requirement)
        for requirement in pkg_resources.parse_requirements(requirements_file)
    ]

setup_requirements = [
    "pytest-runner",
]

test_requirements = [
    "pytest>=3",
]

setup(
    name=name,
    url=url,
    keywords=name,
    version=version,
    zip_safe=False,
    packages=find_packages(SRC_PATH),  # Include packages in src/
    package_dir={"": SRC_PATH},  # Root of the packages is src/
    long_description=readme + "\n\n" + history,
    long_description_content_type="text/markdown",
    author=author,
    author_email=email,
    description=doc,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    test_suite="tests",
    tests_require=test_requirements,
    setup_requires=setup_requirements,
    install_requires=requirements,
)
