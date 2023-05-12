import re
from typing import List, Union

from overrides import overrides

from ipa.data.word import Word
from ipa.preprocessing.tokenizers.base_tokenizer import (
    BaseTokenizer,
)


class WhitespaceTokenizer(BaseTokenizer):
    """
    A :obj:`Tokenizer` that splits the text on spaces.
    """

    def __init__(self):
        super(WhitespaceTokenizer, self).__init__()
        self.finditer_regex = re.compile(r"\S+")

    def __call__(
        self,
        texts: Union[str, List[str], List[List[str]]],
        is_split_into_words: bool = False,
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

            >>> from nlp_preprocessing_wrappers import WhitespaceTokenizer

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

        if not isinstance(text, (str, list)):
            raise ValueError(
                f"text must be either `str` or `list`, found: `{type(text)}`"
            )

        if isinstance(text, list):
            text = " ".join(text)
        return [
            Word(t[0], i, start_char=t[1], end_char=t[2])
            for i, t in enumerate(
                (m.group(0), m.start(), m.end())
                for m in self.finditer_regex.finditer(text)
            )
        ]
