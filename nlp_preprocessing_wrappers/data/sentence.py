from typing import List, Any

from nlp_preprocessing_wrappers.data.word import Word, Predicate


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


class SrlSentence(Sentence):
    """
    A Semantic Role Labeling Sentence class, used to built the output.
    Args:
        words (`List[Word]`): List of `Word` objects.
        id (`Any`): The id of the sentence.
    """

    def add_predicate(self, predicate: Predicate, index: int = None) -> Predicate:
        """
        Add a predicate to the sentence.
        Args:
            predicate (`Predicate`): The predicate to add.
            index (`int`): The index where to add the predicate.
        Returns:
            `Predicate`: The added predicate.
        """
        if predicate.index is not None and index is None:
            # infer index from predicate
            index = predicate.index
        if index is not None and predicate.index is None:
            # set index of predicate
            predicate.index = index

        if index is None and predicate.index is None:
            raise ValueError("Cannot infer index of predicate")

        if index >= len(self._words):
            raise IndexError(
                f"Index out of range: provided index is {index}, sentence length is {len(self._words)}"
            )
        self._words[index] = predicate
        return predicate

    def get_predicate(self, index: int) -> Predicate:
        """
        Get the predicate at the given index.
        Args:
            index (`int`): The index of the predicate to get.
        Returns:
            `Predicate`: The predicate at the given index.
        """
        if index >= len(self._words):
            raise IndexError(
                f"Index out of range: provided index is {index}, sentence length is {len(self._words)}"
            )
        if not isinstance(self._words[index], Predicate):
            raise TypeError(f"Index {index} is not a predicate")

        return self._words[index]

    @property
    def predicates(self) -> List[Predicate]:
        """
        Get all predicates in the sentence.
        Returns:
            `List[Predicate]`: The list of predicates.
        """
        return [p for p in self._words if isinstance(p, Predicate)]
