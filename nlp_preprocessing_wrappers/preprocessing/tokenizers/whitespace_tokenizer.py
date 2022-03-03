from typing import List, Union

from overrides import overrides

from nlp_preprocessing_wrappers.data.word import Word

from nlp_preprocessing_wrappers.preprocessing.tokenizers.base_tokenizer import BaseTokenizer


class WhitespaceTokenizer(BaseTokenizer):
    """
    A :obj:`Tokenizer` that splits the text on spaces.
    """

    def __call__(
        self,
        texts: Union[str, List[str], List[List[str]]],
        is_split_into_words: bool = False,
        *args,
        **kwargs,
    ) -> List[List[Word]]:
        """
        Tokenize the input into single words by splitting on spaces.

        Args:
            texts (:obj:`str`, :obj:`List[str]`, :obj:`List[List[str]]`):
                Text to tag. It can be a single string, a batch of string and pre-tokenized strings.
            is_split_into_words (:obj:`bool`, optional, defaults to :obj:`False`):
                If :obj:`True` and the input is a string, the input is split on spaces.

        Returns:
            :obj:`List[List[Word]]`: The input text tokenized in single words.

        Example::

            >>> from nlp_preprocessing_wrappers.preprocessing import WhitespaceTokenizer

            >>> whitespace_tokenizer = WhitespaceTokenizer()
            >>> whitespace_tokenizer("Mary sold the car to John .")

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
        if isinstance(text, str):
            return [Word(t, i) for i, t in enumerate(text.split())]
        elif isinstance(text, list):
            return [Word(t, i) for i, t in enumerate(text)]
        else:
            raise ValueError(f"text must be either `str` or `list`, found: `{type(text)}`")
