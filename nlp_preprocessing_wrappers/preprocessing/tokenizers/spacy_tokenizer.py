import logging
from typing import List, Union

import spacy
from overrides import overrides
from spacy.tokens import Doc

from nlp_preprocessing_wrappers.common.logging import get_logger
from nlp_preprocessing_wrappers.data.word import Word
from nlp_preprocessing_wrappers.common.utils import load_spacy
from nlp_preprocessing_wrappers.preprocessing.tokenizers import SPACY_LANGUAGE_MAPPER
from nlp_preprocessing_wrappers.preprocessing.tokenizers.base_tokenizer import BaseTokenizer

logger = get_logger(level=logging.DEBUG)


class SpacyTokenizer(BaseTokenizer):
    """
    A :obj:`Tokenizer` that uses SpaCy to tokenizer and preprocess the text. It returns :obj:`Word` objects.

    Args:
        language (:obj:`str`, optional, defaults to :obj:`en`):
            Language of the text to tokenize.
        return_pos_tags (:obj:`bool`, optional, defaults to :obj:`False`):
            If :obj:`True`, performs POS tagging with spacy model.
        return_lemmas (:obj:`bool`, optional, defaults to :obj:`False`):
            If :obj:`True`, performs lemmatization with spacy model.
        return_deps (:obj:`bool`, optional, defaults to :obj:`False`):
            If :obj:`True`, performs dependency parsing with spacy model.
        split_on_spaces (:obj:`bool`, optional, defaults to :obj:`False`):
            If :obj:`True`, will split by spaces without performing tokenization.
        use_gpu (:obj:`bool`, optional, defaults to :obj:`False`):
            If :obj:`True`, will load the Stanza model on GPU.
    """

    def __init__(
        self,
        language: str = "en",
        return_pos_tags: bool = False,
        return_lemmas: bool = False,
        return_deps: bool = False,
        split_on_spaces: bool = False,
        use_gpu: bool = False,
    ):
        super(SpacyTokenizer, self).__init__()
        if language not in SPACY_LANGUAGE_MAPPER:
            raise ValueError(
                f"`{language}` language not supported. The supported "
                f"languages are: {list(SPACY_LANGUAGE_MAPPER.keys())}."
            )
        if use_gpu:
            # load the model on GPU
            # if the GPU is not available or not correctly configured,
            # it will rise an error
            spacy.require_gpu()
        self.spacy = load_spacy(
            SPACY_LANGUAGE_MAPPER[language],
            return_pos_tags,
            return_lemmas,
            return_deps,
            split_on_spaces,
        )
        self.split_on_spaces = split_on_spaces

    def __call__(
        self,
        texts: Union[str, List[str], List[List[str]]],
        is_split_into_words: bool = False,
        *args,
        **kwargs,
    ) -> Union[List[Word], List[List[Word]]]:
        """
        Tokenize the input into single words using SpaCy models.

        Args:
            texts (:obj:`str`, :obj:`List[str]`, :obj:`List[List[str]]`):
                Text to tag. It can be a single string, a batch of string and pre-tokenized strings.
            is_split_into_words (:obj:`bool`, optional, defaults to :obj:`False`):
                If :obj:`True` and the input is a string, the input is split on spaces.

        Returns:
            :obj:`List[List[Word]]`: The input text tokenized in single words.

        Example::

            >>> from nlp_preprocessing_wrappers.preprocessing import SpacyTokenizer

            >>> spacy_tokenizer = SpacyTokenizer(language="en", pos_tags=True, lemma=True)
            >>> spacy_tokenizer("Mary sold the car to John.")

        """
        # check if input is batched or a single sample
        is_batched = self.check_is_batched(texts, is_split_into_words)
        if is_batched:
            tokenized = self.tokenize_batch(texts)
        else:
            tokenized = self.tokenize(texts)
        return tokenized

    @overrides
    def tokenize(self, text: Union[str, List[str]]) -> List[Word]:
        if self.split_on_spaces:
            if isinstance(text, str):
                text = text.split(" ")
            spaces = [True] * len(text)
            text = Doc(self.spacy.vocab, words=text, spaces=spaces)
        return self._clean_tokens(self.spacy(text))

    @overrides
    def tokenize_batch(self, texts: Union[List[str], List[List[str]]]) -> List[List[Word]]:
        if self.split_on_spaces:
            if isinstance(texts[0], str):
                texts = [text.split(" ") for text in texts]
            spaces = [[True] * len(text) for text in texts]
            texts = [Doc(self.spacy.vocab, words=text, spaces=space) for text, space in zip(texts, spaces)]
        return [self._clean_tokens(tokens) for tokens in self.spacy.pipe(texts)]

    @staticmethod
    def _clean_tokens(tokens: Doc) -> List[Word]:
        """
        Converts spaCy tokens to :obj:`Word`.

        Args:
            tokens (:obj:`spacy.tokens.Doc`):
                Tokens from SpaCy model.

        Returns:
            :obj:`List[Word]`: The SpaCy model output converted into :obj:`Word` objects.
        """
        words = [
            Word(
                token.text,
                token.i,
                token.idx,
                token.idx + len(token),
                token.lemma_,
                token.pos_,
                token.dep_,
                token.head.i,
            )
            for token in tokens
        ]
        return words


class WhitespaceSpacyTokenizer:
    """Simple white space tokenizer for SpaCy."""

    def __init__(self, vocab):
        self.vocab = vocab

    def __call__(self, text):
        if isinstance(text, str):
            words = text.split(" ")
        elif isinstance(text, list):
            words = text
        else:
            raise ValueError(f"text must be either `str` or `list`, found: `{type(text)}`")
        spaces = [True] * len(words)
        return Doc(self.vocab, words=words, spaces=spaces)
