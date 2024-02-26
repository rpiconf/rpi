from enum import Enum


class ModelBackboneTypes(Enum):
    BERT = 'BERT'
    ROBERTA = 'ROBERTA'
    LLAMA = 'LLAMA'


class EvalMetric(Enum):
    SUFFICIENCY = 'SUFFICIENCY'
    COMPREHENSIVENESS = 'COMPREHENSIVENESS'
    EVAL_LOG_ODDS = 'EVAL_LOG_ODDS'
    AOPC_SUFFICIENCY = 'AOPC_SUFFICIENCY'
    AOPC_COMPREHENSIVENESS = 'AOPC_COMPREHENSIVENESS'
    POS_AUC_WITH_REFERENCE_TOKEN = 'POS_AUC_WITH_REFERENCE_TOKEN'
    NEG_AUC_WITH_REFERENCE_TOKEN = 'NEG_AUC_WITH_REFERENCE_TOKEN'


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
