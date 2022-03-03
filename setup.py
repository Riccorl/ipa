import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

extras = {}
extras["torch"] = ["torch>=1.5,<1.11"]
extras["all"] = extras["torch"]

install_requires = ["spacy>=3.2,<3.3", "stanza>=1.2,<1.4", "overrides>=6.0.0,<7.0.0"]

setuptools.setup(
    name="nlp_preprocessing_wrappers",  # Replace with your own username
    version="0.1.2",
    author="Riccardo Orlando",
    author_email="orlandoricc@gmail.com",
    description="NLP Preprocessing Pipeline Wrappers",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Riccorl/preprocessing-wrappers",
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
