<div align="center">

# 🍺IPA: import, preprocess, accelerate

[//]: # ([![Open in Visual Studio Code]&#40;https://open.vscode.dev/badges/open-in-vscode.svg&#41;]&#40;https://github.dev/Riccorl/ipa&#41;)
[![PyTorch](https://img.shields.io/badge/PyTorch-orange?logo=pytorch)](https://pytorch.org/)
[![Stanza](https://img.shields.io/badge/1.4-Stanza-5f0a09?logo=stanza)](https://stanfordnlp.github.io/stanza/)
[![SpaCy](https://img.shields.io/badge/3.4.3-SpaCy-1a6f93?logo=spacy)](https://spacy.io/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000)](https://github.com/psf/black)

[![Upload to PyPi](https://github.com/Riccorl/ipa/actions/workflows/python-publish-pypi.yml/badge.svg)](https://github.com/Riccorl/ipa/actions/workflows/python-publish-pypi.yml)
[![PyPi Version](https://img.shields.io/github/v/release/Riccorl/ipa)](https://github.com/Riccorl/ipa/releases)
[![DeepSource](https://deepsource.io/gh/Riccorl/ipa.svg/?label=active+issues&token=QC6Jty-YdgXjKh9mKZyeqa4I)](https://deepsource.io/gh/Riccorl/ipa/?ref=repository-badge)

</div>

🍺IPA: import, preprocess, accelerate

## How to use

### Install

Install the library from [PyPI](https://pypi.org/project/ipa-core):

```bash
pip install ipa-core
```

### Usage

IPA is a Python library that provides a set of preprocessing wrappers for Stanza and
spaCy, providing a unified API for both libraries, making them interchangeable.

Let's start with a simple example. Here we are using the `SpacyTokenizer` wrapper to preprocess a text: 

```python
from ipa import SpacyTokenizer

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
from ipa import StanzaTokenizer

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
from ipa import WhitespaceTokenizer

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