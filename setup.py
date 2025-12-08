from setuptools import setup, find_packages

setup(
    name="pii-sanitizer",
    version="0.0.1",
    packages=find_packages(),
    install_requires=[
        "boto3",
        "presidio-analyzer",
        "presidio-anonymizer",
        "spacy",
    ],
)

