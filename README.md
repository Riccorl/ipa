<div align="center">

# NLP Preprocessing Wrappers

[![Open in Visual Studio Code](https://open.vscode.dev/badges/open-in-vscode.svg)](https://github.dev/Riccorl/nlp-preprocessing-wrappers)
[![PyTorch](https://img.shields.io/badge/PyTorch-orange?logo=pytorch)](https://pytorch.org/)
[![Stanza](https://img.shields.io/badge/1.3-Stanza-5f0a09?logo=stanza)](https://stanfordnlp.github.io/stanza/)
[![SpaCy](https://img.shields.io/badge/3.2.3-SpaCy-1a6f93?logo=soacy)](https://spacy.io/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000)](https://github.com/psf/black)

[![Upload to PyPi](https://github.com/Riccorl/nlp-preprocessing-wrappers/actions/workflows/python-publish-pypi.yml/badge.svg)](https://github.com/Riccorl/nlp-preprocessing-wrappers/actions/workflows/python-publish-pypi.yml)
[![PyPi Version](https://img.shields.io/github/v/release/Riccorl/nlp-preprocessing-wrappers)](https://github.com/Riccorl/nlp-preprocessing-wrappers/releases)
[![DeepSource](https://deepsource.io/gh/Riccorl/nlp-preprocessing-wrappers.svg/?label=active+issues&token=QC6Jty-YdgXjKh9mKZyeqa4I)](https://deepsource.io/gh/Riccorl/nlp-preprocessing-wrappers/?ref=repository-badge)

</div>

Preprocessing Wrappers

## How to use

### Install

Install the library from [PyPI](https://pypi.org/project/nlp-preprocessing-wrappers):

```bash
pip install nlp-preprocessing-wrappers
```

### Usage

NLP Preprocessing Wrappers is a Python library that provides a set of preprocessing wrappers for Stanza and
spaCy, providing a unified API for both libraries, making them interchangeable.

Let's start with a simple example. Here we are using the `SpacyTokenizer` wrapper to preprocess a text: 

```python
from nlp_preprocessing_wrappers import SpacyTokenizer

spacy_tokenizer = SpacyTokenizer(language="en", return_pos_tags=True, return_lemmas=True)
tokenized = spacy_tokenizer("Mary sold the car to John.")
for word in tokenized:
    print("{:<5} {:<10} {:<10} {:<10}".format(word.index, word.text, word.pos, word.lemma))

"""
0    Mary       PROPN      Mary
1    sold       VERB       sell
2    the        DET        the
3    car        NOUN       car
4    to         ADP        to
5    John       PROPN      John
6    .          PUNCT      .
"""
```

You can load any model from spaCy, with its canonical name, `en_core_web_sm`, or with a simple alias, as 
we did here, like `en`. By default, the simpler alias loads the smaller version of each model. For a complete 
list of available models, see [spaCy documentation](https://spacy.io/usage/models).

In the very same way, you can load any model from Stanza using the `StanzaTokenizer` wrapper:

```python
from nlp_preprocessing_wrappers import StanzaTokenizer

stanza_tokenizer = StanzaTokenizer(language="en", return_pos_tags=True, return_lemmas=True)
tokenized = stanza_tokenizer("Mary sold the car to John.")
for word in tokenized:
    print("{:<5} {:<10} {:<10} {:<10}".format(word.index, word.text, word.pos, word.lemma))

"""
0    Mary       PROPN      Mary
1    sold       VERB       sell
2    the        DET        the
3    car        NOUN       car
4    to         ADP        to
5    John       PROPN      John
6    .          PUNCT      .
"""
```

For more simple scenarios, you can use the `WhiteSpaceTokenizer` wrapper, which will just split the text 
by whitespace:

```python
from nlp_preprocessing_wrappers import WhitespaceTokenizer

whitespace_tokenizer = WhitespaceTokenizer()
tokenized = whitespace_tokenizer("Mary sold the car to John .")
for word in tokenized:
    print("{:<5} {:<10}".format(word.index, word.text))

"""
0    Mary
1    sold
2    the
3    car
4    to
5    John
6    .
"""
```

### Features

#### Complete preprocessing pipeline

`SpacyTokenizer` and `StanzaTokenizer` provide a unified API for both libraries, exposing most of their
features, like tokenization, Part-of-Speech tagging, lemmatization and dependency parsing. You can activate 
and deactivate any of these using `return_pos_tags`, `return_lemmas` and `return_deps`. So, for example,

```python
StanzaTokenizer(language="en", return_pos_tags=True, return_lemmas=True)
```

will return a list of `Token` objects, with the `pos` and `lemma` fields filled.

while

```python
StanzaTokenizer(language="en")
```

will return a list of `Token` objects, with only the `text` field filled.

### GPU support

With `use_gpu=True`, the library will use the GPU if it is available. To set up the environment for the GPU, 
refer to the [Stanza documentation](https://stanfordnlp.github.io/stanza/) and the 
[spaCy documentation](https://spacy.io/usage/gpu).

## API

### Tokenizers

`SpacyTokenizer`

```python
class SpacyTokenizer(BaseTokenizer):
    def __init__(
        self,
        language: str = "en",
        return_pos_tags: bool = False,
        return_lemmas: bool = False,
        return_deps: bool = False,
        split_on_spaces: bool = False,
        use_gpu: bool = False,
    ):
```

`StanzaTokenizer`

```python
class StanzaTokenizer(BaseTokenizer):
    def __init__(
        self,
        language: str = "en",
        return_pos_tags: bool = False,
        return_lemmas: bool = False,
        return_deps: bool = False,
        split_on_spaces: bool = False,
        use_gpu: bool = False,
    ):
```

`WhitespaceTokenizer`

```python
class WhitespaceTokenizer(BaseTokenizer):
    def __init__(self):
```

### Sentence Splitter

`SpacySentenceSplitter`

```python
class SpacySentenceSplitter(BaseSentenceSplitter):
    def __init__(self, language: str = "en", model_type: str = "statistical"):
```