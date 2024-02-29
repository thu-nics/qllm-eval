#!/usr/bin/env python

from setuptools import setup
import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="qllm_eval",
    version="0.1.0",
    description="QLLM_Evaluation",
    author="Shiyao Li",
    author_email="shiyao1620@gmail.com",
    # url="https://github.com/LSY-noya/QLLM-Evaluation.git",
    packages=setuptools.find_packages(),
    license="MIT",
    long_description=long_description,
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
)
