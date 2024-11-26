from typing import Literal
from refiners.training_utils import ModelConfig, BaseConfig, CallbackConfig
from pydantic import BaseModel
from refiners.training_utils.config import TimeValueField 

class AnydoorModelConfig(ModelConfig):
    path_to_unet : str
    path_to_control_model : str
    path_to_object_encoder : str
    path_to_lda : str
    lora_rank : int
    lora_scale : float
    lora_checkpoint : str | None = None

class WandbConfig(BaseModel):
    mode: Literal["online", "offline", "disabled"] = "offline"
    project: str
    entity: str = "theodo"
    name: str | None = None
    tags: list[str] = []
    group: str | None = None
    job_type: str | None = None
    notes: str | None = None

class EvaluationConfig(CallbackConfig):
    interval: TimeValueField
    seed: int

class AnydoorTrainingConfig(BaseConfig):
    train_dataset : str
    test_dataset : str
    saving_path : str
    batch_size : int 
    loss_type : str
    anydoor : AnydoorModelConfig
    wandb : WandbConfig
    evaluation: EvaluationConfig
    train_lora_dataset_selection : str | None = None
    test_lora_dataset_selection : str | None = None

class AnydoorEvaluatorConfig(BaseModel):
    test_dataset : str
    batch_size : int 
    test_lora_dataset_selection : str | None = None