from typing import List


class BaseSentenceSplitter:
    """
    A `BaseSentenceSplitter` splits strings into sentences.
    """

    def split_sentences(self, text: str) -> List[str]:
        """
        Splits a `text` :class:`str` paragraph into a list of :class:`str`, where each is a sentence.
        """
        raise NotImplementedError

    def split_sentences_batch(self, texts: List[str]) -> List[List[str]]:
        """
        Default implementation is to just iterate over the texts and call `split_sentences`.
        """
        return [self.split_sentences(text) for text in texts]
