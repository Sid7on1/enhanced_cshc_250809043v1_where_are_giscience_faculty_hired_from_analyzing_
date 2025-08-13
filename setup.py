import os
import sys
from setuptools import setup, find_packages
from setuptools.command.install import install
from setuptools.command.develop import develop
from setuptools.command.build_py import build_py
from setuptools.command.sdist import sdist
from setuptools.command.egg_info import egg_info
from distutils import log
from typing import List, Dict, Any

# Define constants
PROJECT_NAME = "enhanced_cs.HC_2508.09043v1_Where_are_GIScience_Faculty_Hired_from_Analyzing_"
VERSION = "1.0.0"
DESCRIPTION = "Enhanced AI project based on cs.HC_2508.09043v1_Where-are-GIScience-Faculty-Hired-from-Analyzing- with content analysis."
AUTHOR = "Your Name"
EMAIL = "your@email.com"
URL = "https://your-website.com"

# Define dependencies
DEPENDENCIES = [
    "torch",
    "numpy",
    "pandas",
]

# Define package data
PACKAGE_DATA = {
    "": ["*.txt", "*.md"],
}

# Define entry points
ENTRY_POINTS = {
    "console_scripts": [
        "nlp_project=nlp_project.main:main",
    ],
}

# Define classifiers
CLASSIFIERS = [
    "Development Status :: 5 - Production/Stable",
    "Intended Audience :: Developers",
    "License :: OSI Approved :: MIT License",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.8",
    "Programming Language :: Python :: 3.9",
    "Programming Language :: Python :: 3.10",
]

# Define keywords
KEYWORDS = [
    "nlp",
    "natural language processing",
    "gis",
    "geographic information systems",
]

# Define long description
LONG_DESCRIPTION = """
# Enhanced AI Project

This project is an enhanced version of the cs.HC_2508.09043v1_Where-are-GIScience-Faculty-Hired-from-Analyzing- project.
It includes content analysis and uses the following algorithms:
- Step
- Reduction
- Modeling
- More
- Mobility
- Hiring
- Machine
- Clustering
- Language
- Geography

## Installation

To install the project, run the following command: