from setuptools import setup, find_packages


setup(
    name="nolano",
    version="0.0.1",
    packages=find_packages(),
    install_requires=[
        "requests",
        "pandas",
        "numpy",
        "matplotlib"
    ],
    extras_require={},
    author="Nolano Team",
    author_email="support@nolano.ai",
    description="A Python package for interacting with Nolano's time series forecasting API",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/nolano/nolano-python",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    project_urls={
        "Documentation": "https://api.nolano.ai",
        "Source": "https://github.com/nolano/nolano-python",
        "API": "https://api.nolano.ai"
    },
    python_requires='>=3.8',
)