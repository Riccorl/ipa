from typing import List, Any

from nlp_preprocessing_wrappers.data.word import Word


class Sentence(List):
    """
    A sentence class, containing a list of :obj:`Word` objects.

    Args:
        words (`List[Word]`): A list of :obj:`Word` objects.
    """

    def __init__(self, words: List[Word] = None, id: Any = None):
        super().__init__()
        self._words = words or []
        self.id = id

    def __len__(self):
        return len(self._words)

    def __getitem__(self, item):
        return self._words[item]

    def __repr__(self):
        return "[" + ", ".join(w.text for w in self._words) + "]"

    def __str__(self):
        return self.__repr__()

    def __iter__(self):
        return self._words.__iter__()

    def __next__(self):
        return self._words.__next__()

    def append(self, word: Word):
        self._words.append(word)
