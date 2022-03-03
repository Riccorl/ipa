from common import utils

if utils.is_torch_available():
    pass

from transformers import (
    BertTokenizer,
    BertTokenizerFast,
    BertweetTokenizer,
    CamembertTokenizer,
    CamembertTokenizerFast,
    DebertaTokenizer,
    DistilBertTokenizer,
    DistilBertTokenizerFast,
    MobileBertTokenizer,
    MobileBertTokenizerFast,
    RobertaTokenizer,
    RobertaTokenizerFast,
    XLMRobertaTokenizer,
    XLMRobertaTokenizerFast,
    XLMTokenizer,
)


MODELS_WITH_STARTING_TOKEN = (
    BertTokenizer,
    BertTokenizerFast,
    DistilBertTokenizer,
    DistilBertTokenizerFast,
    MobileBertTokenizer,
    MobileBertTokenizerFast,
    BertweetTokenizer,
    CamembertTokenizer,
    CamembertTokenizerFast,
    DebertaTokenizer,
    RobertaTokenizer,
    RobertaTokenizerFast,
    XLMRobertaTokenizer,
    XLMRobertaTokenizerFast,
    XLMTokenizer,
)

MODELS_WITH_DOUBLE_SEP = (
    CamembertTokenizer,
    CamembertTokenizerFast,
    BertweetTokenizer,
    RobertaTokenizer,
    RobertaTokenizerFast,
    XLMRobertaTokenizer,
    XLMRobertaTokenizerFast,
)

