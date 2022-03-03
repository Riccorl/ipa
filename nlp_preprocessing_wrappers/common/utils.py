import importlib.util
import logging
from typing import Dict, Tuple

import spacy
import stanza
from spacy.cli.download import download as spacy_download

from nlp_preprocessing_wrappers.common.logging import get_logger

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
    pass

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
