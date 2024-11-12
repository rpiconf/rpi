from config.types_enums import *
from config.types_enums import TokenEvaluationOptions
from utils.dataclasses import Task


class ExpArgs:
    num_samples = 24
    num_interpolation = 30
    time_steps = 500
    is_gradient_x_input = True
    seed = 42
    batch_size = 50
    default_root_dir = "OUT"
    ref_token_name: RefTokenNameTypes = None
    is_save_results = True
    is_save_support_results = True
    explained_model_backbone = None
    label_vocab_tokens = None
    task: Task = None
    evaluation_metric: str = None
    token_evaluation_option = TokenEvaluationOptions.NO_SPECIAL_TOKENS.value
    reverse_noise_layer_index = 3


class MetricsMetaData:
    directions = {EvalMetric.SUFFICIENCY.value: DirectionTypes.MIN.value,
                  EvalMetric.COMPREHENSIVENESS.value: DirectionTypes.MAX.value,
                  EvalMetric.EVAL_LOG_ODDS.value: DirectionTypes.MIN.value,
                  EvalMetric.AOPC_SUFFICIENCY.value: DirectionTypes.MIN.value,
                  EvalMetric.AOPC_COMPREHENSIVENESS.value: DirectionTypes.MAX.value}

    top_k = {EvalMetric.SUFFICIENCY.value: [20], EvalMetric.COMPREHENSIVENESS.value: [20],
             EvalMetric.EVAL_LOG_ODDS.value: [20], EvalMetric.AOPC_SUFFICIENCY.value: [1, 5, 10, 20, 50],
             EvalMetric.AOPC_COMPREHENSIVENESS.value: [1, 5, 10, 20, 50]}


class BackbonesMetaData:
    name = {  #
        ModelBackboneTypes.BERT.value: "bert",  #
        ModelBackboneTypes.ROBERTA.value: "roberta",  #
        ModelBackboneTypes.DISTILBERT.value: "distilbert",  #
        ModelBackboneTypes.LLAMA.value: "model",  #
        ModelBackboneTypes.MISTRAL.value: "model"  #
    }
