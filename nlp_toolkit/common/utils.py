import logging
import importlib.util
import json
import logging
import os
import shutil
import tarfile
import tempfile
from functools import partial
from hashlib import sha256
from pathlib import Path
from typing import Union, Any, Dict, Tuple, BinaryIO, Optional
from urllib.parse import urlparse
from zipfile import is_zipfile, ZipFile

import requests
import spacy
import stanza
import yaml
from filelock import FileLock
from spacy.cli.download import download as spacy_download
from tqdm import tqdm

from common.logging import get_logger

logger = get_logger(level=logging.DEBUG)


_onnx_available = importlib.util.find_spec("onnx") is not None


def is_onnx_available():
    return _onnx_available


def is_a_package_available(package_name: str) -> bool:
    return importlib.util.find_spec(package_name) is not None


_torch_available = importlib.util.find_spec("torch") is not None
_spacy_available = importlib.util.find_spec("spacy") is not None


def is_torch_available():
    """Check if PyTorch is available."""
    return _torch_available


def is_spacy_available():
    """Check if spaCy is available."""
    return _spacy_available


if is_torch_available():
    import torch
    from torch import Tensor

# Spacy and Stanza stuff

LOADED_SPACY_MODELS: Dict[Tuple[str, bool, bool, bool, bool], spacy.Language] = {}


def load_spacy(
    language: str,
    pos_tags: bool = False,
    lemma: bool = False,
    parse: bool = False,
    split_on_spaces: bool = False,
) -> spacy.Language:
    """
    Download and load spacy model.

    Args:
        language:
        pos_tags:
        lemma:
        parse:
        split_on_spaces:

    Returns:
        spacy.Language: The spacy tokenizer loaded.
    """
    exclude = ["vectors", "textcat", "ner"]
    if not pos_tags:
        exclude.append("tagger")
    if not lemma:
        exclude.append("lemmatizer")
    if not parse:
        exclude.append("parser")

    # check if the model is already loaded
    # if so, there is no need to reload it
    spacy_params = (language, pos_tags, lemma, parse, split_on_spaces)
    if spacy_params not in LOADED_SPACY_MODELS:
        try:
            spacy_tagger = spacy.load(language, exclude=exclude)
        except OSError:
            logger.warning(f"Spacy model '{language}' not found. Downloading and installing.")
            spacy_download(language)
            spacy_tagger = spacy.load(language, exclude=exclude)

        # if everything is disabled, return only the tokenizer
        # for faster tokenization
        # TODO: is it really faster?
        # if len(exclude) >= 6:
        #     spacy_tagger = spacy_tagger.tokenizer
        LOADED_SPACY_MODELS[spacy_params] = spacy_tagger

    return LOADED_SPACY_MODELS[spacy_params]


LOADED_STANZA_MODELS: Dict[Tuple[str, str, bool, bool], stanza.Pipeline] = {}


def load_stanza(
    language: str = "en",
    pos_tags: bool = False,
    lemma: bool = False,
    parse: bool = False,
    tokenize_pretokenized: bool = False,
    use_gpu: bool = False,
) -> stanza.Pipeline:
    """
    Download and load stanza model.

    Args:
        language:
        pos_tags:
        lemma:
        parse:
        tokenize_pretokenized:
        use_gpu:

    Returns:
        stanza.Pipeline: The stanza tokenizer loaded.

    """
    processors = ["tokenize"]
    if pos_tags:
        processors.append("pos")
    if lemma:
        processors.append("lemma")
    if parse:
        processors.append("depparse")
    processors = ",".join(processors)

    # check if the model is already loaded
    # if so, there is no need to reload it
    stanza_params = (language, processors, tokenize_pretokenized, use_gpu)
    if stanza_params not in LOADED_STANZA_MODELS:
        try:
            stanza_tagger = stanza.Pipeline(
                language,
                processors=processors,
                tokenize_pretokenized=tokenize_pretokenized,
                tokenize_no_ssplit=True,
                use_gpu=use_gpu,
            )
        except OSError:
            logger.info(f"Stanza model '{language}' not found. Downloading and installing.")
            stanza.download(language)
            stanza_tagger = stanza.Pipeline(
                language,
                processors=processors,
                tokenize_pretokenized=tokenize_pretokenized,
                tokenize_no_ssplit=True,
                use_gpu=use_gpu,
            )
        LOADED_STANZA_MODELS[stanza_params] = stanza_tagger

    return LOADED_STANZA_MODELS[stanza_params]


# file I/O stuff


def load_yaml(path: Union[str, Path]):
    """
    Load a yaml file provided in input.
    Args:
        path: path to the yaml file.

    Returns:
        The yaml file parsed.
    """
    with open(path, encoding="utf8") as f:
        return yaml.load(f, Loader=yaml.FullLoader)


def dump_yaml(document: Any, path: Union[str, Path]):
    """
    Dump input to yaml file.
    Args:
        document: thing to dump
        path: path to the yaml file.

    Returns:

    """
    with open(path, "w", encoding="utf8") as outfile:
        yaml.dump(document, outfile, default_flow_style=False)


def load_json(path: Union[str, Path]):
    """
    Load a yaml file provided in input.
    Args:
        path: path to the json file.

    Returns:
        The yaml file parsed.
    """
    with open(path, encoding="utf8") as f:
        return json.load(f)


def dump_json(document: Any, path: Union[str, Path], indent: Optional[int] = None):
    """
    Dump input to json file.
    Args:
        document: thing to dump
        path: path to the yaml file.
        indent: json indent

    Returns:

    """
    with open(path, "w", encoding="utf8") as outfile:
        json.dump(document, outfile, indent=indent)
