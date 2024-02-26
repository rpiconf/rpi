import sys

sys.path.append("..")

from config.tasks import RTN_TASK
from main.run_rpi import Rpi
from config.config import ExpArgs
from config.types_enums import ModelBackboneTypes, RefTokenNameTypes
from utils.utils_functions import get_current_time

ExpArgs.task = RTN_TASK
ExpArgs.explained_model_backbone = ModelBackboneTypes.BERT.value
ExpArgs.ref_token_name = RefTokenNameTypes.MASK.value
ExpArgs.is_save_model = False
ExpArgs.enable_checkpointing = False
ExpArgs.verbose = True
ExpArgs.is_save_results = True

exp_name = f"exp_{ExpArgs.explained_model_backbone}_{get_current_time()}"
Rpi(exp_name).run()
