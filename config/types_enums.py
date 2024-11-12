from enum import Enum


class ModelBackboneTypes(Enum):
    BERT = 'BERT'
    ROBERTA = 'ROBERTA'
    DISTILBERT = 'DISTILBERT'
    LLAMA = 'LLAMA'
    MISTRAL = 'MISTRAL'


class EvalMetric(Enum):
    SUFFICIENCY = 'SUFFICIENCY'
    COMPREHENSIVENESS = 'COMPREHENSIVENESS'
    EVAL_LOG_ODDS = 'EVAL_LOG_ODDS'
    AOPC_SUFFICIENCY = 'AOPC_SUFFICIENCY'
    AOPC_COMPREHENSIVENESS = 'AOPC_COMPREHENSIVENESS'


class DirectionTypes(Enum):
    MAX = 'MAX'
    MIN = 'MIN'


class RefTokenNameTypes(Enum):
    MASK = 'MASK'
    PAD = 'PAD'
    UNK = 'UNK'


class ValidationType(Enum):
    VAL = 'VAL'
    TEST = 'TEST'


class ModelPromptType(Enum):
    ZERO_SHOT = 'zero_shot'
    FEW_SHOT = 'few_shot'
    FEW_SHOT_CONTENT = 'few_shot_content'  # few-shot but map just content


class TokenEvaluationOptions(Enum):
    # ALL_TOKENS = 'ALL_TOKENS'
    # NO_CLS = 'NO_CLS'
    NO_SPECIAL_TOKENS = 'NO_SPECIAL_TOKENS'
