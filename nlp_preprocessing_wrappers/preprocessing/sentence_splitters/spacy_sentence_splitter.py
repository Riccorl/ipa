from typing import Any, Iterable, List, Union

import spacy
from overrides import overrides

from nlp_preprocessing_wrappers.common.utils import load_spacy
from nlp_preprocessing_wrappers.preprocessing.sentence_splitters.base_sentence_splitter import (
    BaseSentenceSplitter,
)

SPACY_LANGUAGE_MAPPER = {
    "cs": "xx_sent_ud_sm",
    "da": "xx_sent_ud_sm",
    "de": "xx_sent_ud_sm",
    "fa": "xx_sent_ud_sm",
    "fi": "xx_sent_ud_sm",
    "fr": "xx_sent_ud_sm",
    "el": "el_core_news_sm",
    "en": "xx_sent_ud_sm",
    "es": "xx_sent_ud_sm",
    "ga": "xx_sent_ud_sm",
    "hr": "xx_sent_ud_sm",
    "id": "xx_sent_ud_sm",
    "it": "xx_sent_ud_sm",
    "ja": "ja_core_news_sm",
    "lv": "xx_sent_ud_sm",
    "lt": "xx_sent_ud_sm",
    "mr": "xx_sent_ud_sm",
    "nb": "xx_sent_ud_sm",
    "nl": "xx_sent_ud_sm",
    "no": "xx_sent_ud_sm",
    "pl": "pl_core_news_sm",
    "pt": "xx_sent_ud_sm",
    "ro": "xx_sent_ud_sm",
    "ru": "xx_sent_ud_sm",
    "sk": "xx_sent_ud_sm",
    "sr": "xx_sent_ud_sm",
    "sv": "xx_sent_ud_sm",
    "te": "xx_sent_ud_sm",
    "vi": "xx_sent_ud_sm",
    "zh": "zh_core_web_sm",
}


class SpacySentenceSplitter(BaseSentenceSplitter):
    """
    A :obj:`SentenceSplitter` that uses spaCy's built-in sentence boundary detection.

    Args:
        language (:obj:`str`, optional, defaults to :obj:`en`):
            Language of the text to tokenize.
        type (:obj:`str`, optional, defaults to :obj:`statistical`):
            Three different type of sentence splitter:
                - ``dependency``: sentence splitter uses a dependency parse to detect sentence boundaries,
                    slow, but accurate.
                - ``statistical``:
                - ``rule_based``: It's fast and has a small memory footprint, since it uses punctuation to detect
                    sentence boundaries.

    """

    def __init__(self, language: str = "en", type: str = "statistical") -> None:
        # we need spacy's dependency parser if we're not using rule-based sentence boundary detection.
        # self.spacy = get_spacy_model(language, parse=not rule_based, ner=False)
        dep = bool(type == "dependency")
        if language in SPACY_LANGUAGE_MAPPER:
            self.spacy = load_spacy(SPACY_LANGUAGE_MAPPER[language], parse=dep)
        else:
            self.spacy = spacy.blank(language)
            # force type to rule_based since there is no pre-trained model
            type = "rule_based"
        if type == "dependency":
            # dependency type must declared at model init
            pass
        elif type == "statistical":
            if not self.spacy.has_pipe("senter"):
                self.spacy.enable_pipe("senter")
        elif type == "rule_based":
            # we use `sentencizer`, a built-in spacy module for rule-based sentence boundary detection.
            # depending on the spacy version, it could be called 'sentencizer' or 'sbd'
            if not self.spacy.has_pipe("sentencizer"):
                self.spacy.add_pipe("sentencizer")
        else:
            raise ValueError(
                f"type {type} not supported. Choose between `dependency`, `statistical` and `rule_based`"
            )

    def __call__(
        self,
        texts: Union[str, List[str], List[List[str]]],
        max_len: int = 0,
        *args,
        **kwargs,
    ) -> Union[List[str], List[List[str]]]:
        """
        Tokenize the input into single words using SpaCy models.

        Args:
            texts (:obj:`str`, :obj:`List[str]`, :obj:`List[List[str]]`):
                Text to tag. It can be a single string, a batch of string and pre-tokenized strings.
            max_len (:obj:`int`, optional, defaults to :obj:`0`):
                Maximum length of a single text. If the text is longer than `max_len`, it will be split
                into multiple sentences.
        Returns:
            :obj:`List[List[str]]`: The input doc splitted into sentences.

        """
        # check if input is batched or a single sample
        # is_batched = self.check_is_batched(texts, is_split_into_words)

        # if is_batched:
        #     sents = self.split_sentences_batch(texts)
        # else:
        #     sents = self.split_sentences(texts)
        sents = self.split_sentences(texts, max_len)
        return sents

    @staticmethod
    def chunked(iterable, n: int) -> Iterable[List[Any]]:
        """
        Chunks a list into n sized chunks.

        Args:
            iterable (:obj:`List[Any]`):
                List to chunk.
            n (:obj:`int`):
                Size of the chunks.

        Returns:
            :obj:`Iterable[List[Any]]`: The input list chunked into n sized chunks.
        """
        return [iterable[i : i + n] for i in range(0, len(iterable), n)]

    @overrides
    def split_sentences(self, text: str, max_len: int = 0) -> List[str]:
        """
        Splits a `text` into smaller senences.

        Args:
            text (:obj:`str`):
                Text to split.
            max_len (:obj:`int`, optional, defaults to :obj:`0`):
                Maximum length of a single sentence. If the text is longer than `max_len`, it will be split
                into multiple sentences.

        Returns:
            :obj:`List[str]`: The input text splitted into sentences.
        """
        sentences = [sent.text.strip() for sent in self.spacy(text).sents]
        if max_len > 0:
            sentences = [chunk for sentence in sentences for chunk in self.chunked(sentence, max_len)]
        return sentences

    @overrides
    def split_sentences_batch(self, texts: List[str]) -> List[List[str]]:
        """
        This method lets you take advantage of spacy's batch processing.
        """
        return [[sentence.text.strip() for sentence in doc.sents] for doc in self.spacy.pipe(texts)]
