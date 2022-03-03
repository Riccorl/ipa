import logging
from typing import List, Union

import stanza.models.common.doc

from nlp_preprocessing_wrappers.common.logging import get_logger
from nlp_preprocessing_wrappers.common.utils import load_stanza
from nlp_preprocessing_wrappers.data.word import Word
from nlp_preprocessing_wrappers.preprocessing.tokenizers.base_tokenizer import BaseTokenizer

logger = get_logger(level=logging.DEBUG)


class StanzaTokenizer(BaseTokenizer):
    """
    A :obj:`Tokenizer` that uses Stanza to tokenizer and preprocess the text. It returns :obj:`Word` objects.

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
        super(StanzaTokenizer, self).__init__()
        self.stanza = load_stanza(
            language, return_pos_tags, return_lemmas, return_deps, split_on_spaces, use_gpu
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
        Tokenize the input into single words using Stanza models.

        Args:
            texts (:obj:`str`, :obj:`List[str]`, :obj:`List[List[str]]`):
                Text to tag. It can be a single string, a batch of string and pre-tokenized strings.
            is_split_into_words (:obj:`bool`, optional, defaults to :obj:`False`):
                If :obj:`True` and the input is a string, the input is split on spaces.

        Returns:
            :obj:`List[List[Word]]`: The input text tokenized in single words.

        Example::

            >>> from nlp_preprocessing_wrappers.preprocessing import StanzaTokenizer

            >>> stanza_tokenizer = StanzaTokenizer(language="en", pos_tags=True, lemma=True)
            >>> stanza_tokenizer("Mary sold the car to John.")

        """
        # check if input is batched or a single sample
        is_batched = self.check_is_batched(texts, is_split_into_words)

        if is_batched:
            tokenized = self.tokenize_batch(texts)
        else:
            # stanza doesn't like tokenized sentences
            # as single sample, normalize the input
            if is_split_into_words and not is_batched:
                if not self.split_on_spaces:
                    logger.warning(
                        f"`is_split_into_words` is set to `{is_split_into_words}` while `split_on_spaces` is"
                        f" set to `{self.split_on_spaces}`. To avoid error from Stanza, the text will be"
                        f' joined in a single string (`" ".join(input)`). To avoid this use '
                        f" `split_on_spaces=True.`"
                    )
                    texts = " ".join(texts)
                texts = [texts]
            tokenized = self.tokenize(texts)

        return tokenized

    def tokenize(self, text: Union[str, List[str]]) -> List[Word]:
        return self._clean_tokens(self.stanza(text).sentences[0].tokens)

    def tokenize_batch(self, texts: Union[List[str], List[List[str]]]) -> List[List[Word]]:
        # stanza has this weird method to process batches
        # if it is already tokenized, join temporarily
        # to perform preprocessing in batch
        if isinstance(texts[0], list):
            texts = [" ".join(t) for t in texts]
        texts = [stanza.Document([], text=t) for t in texts]
        sentences = [sent for doc in self.stanza(texts) for sent in doc.sentences]
        return [self._clean_tokens(sent.tokens) for sent in sentences]

    @staticmethod
    def _clean_tokens(tokens: List[stanza.models.common.doc.Token]) -> List[Word]:
        """
        Converts Stanza tokens to :obj:`Word`.

        Args:
            tokens (:obj:`stanza.models.common.doc.Word`):
                Tokens from Stanza model.

        Returns:
            :obj:`List[Word]`: The Stanza model output converted into :obj:`Word` objects.
        """
        words = [
            Word(
                token.text,
                i,
                token.start_char,
                token.end_char,
                token.words[0].lemma or token.text,
                token.words[0].upos,
                token.words[0].deps,
                token.words[0].head,
            )
            for i, token in enumerate(tokens)
        ]
        return words
