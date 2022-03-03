import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

extras = {}
extras["torch"] = ["torch>=1.5,<1.11"]
extras["all"] = extras["torch"] + extras["spacy"]

install_requires = ["transformers>=4.3,<4.17"]

setuptools.setup(
    name="nlp_toolkit",  # Replace with your own username
    version="0.1.0",
    author="Riccardo Orlando",
    author_email="orlandoricc@gmail.com",
    description="Toolkit for NLP Engineers",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Riccorl/nlp-toolkit",
    keywords="NLP deep learning transformer pytorch stanza spacy trankit preprocessing"
    " tokenization pos tagging lemmatization",
    packages=setuptools.find_packages(),
    include_package_data=True,
    license="Apache",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
    extras_require=extras,
    install_requires=install_requires,
    python_requires=">=3.6",
)
