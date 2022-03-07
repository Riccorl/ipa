from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class Word:
    """
    A word representation that includes text, index in the sentence, POS tag, lemma,
    dependency relation, and similar information.

    # Parameters
    text : `str`, optional
        The text representation.
    index : `int`, optional
        The word offset in the sentence.
    lemma : `str`, optional
        The lemma of this word.
    pos : `str`, optional
        The coarse-grained part of speech of this word.
    dep : `str`, optional
        The dependency relation for this word.

    input_id : `int`, optional
        Integer representation of the word, used to pass it to a model.
    token_type_id : `int`, optional
        Token type id used by some transformers.
    attention_mask: `int`, optional
        Attention mask used by transformers, indicates to the model which tokens should
        be attended to, and which should not.
    """

    text: str
    index: int
    start_char: Optional[int] = None
    end_char: Optional[int] = None
    # preprocessing fields
    lemma: Optional[str] = None
    pos: Optional[str] = None
    dep: Optional[str] = None
    head: Optional[int] = None

    def __str__(self):
        return self.text

    def __repr__(self):
        return self.__str__()
