
import setuptools
import re
import os

setuptools.setup(
	name="extension_ace_step",
    packages=setuptools.find_namespace_packages(),
    version="0.4.0",
	author="rsxdalv",
	description="F5-TTS: A Fairytaler that Fakes Fluent and Faithful Speech with Flow Matching.",
	url="https://github.com/rsxdalv/extension_ace_step",
    project_urls={},
    scripts=[],
    install_requires=[
        "gradio",
        "ace-step @ git+https://github.com/rsxdalv/ACE-Step@loose-3"
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
