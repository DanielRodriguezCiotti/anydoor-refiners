from src.training.trainer import AnydoorModelConfig, AnydoorTrainingConfig, AnyDoorLoRATrainer, EvaluationConfig, WandbConfig
from refiners.training_utils import TrainingConfig, OptimizerConfig, LRSchedulerConfig, Epoch, Optimizers, LRSchedulerType
import torch

torch.set_num_threads(2)


training = TrainingConfig(
    duration=Epoch(50),
    device="cuda" if torch.cuda.is_available() else "cpu",
    dtype="float32",
)

optimizer = OptimizerConfig(
    optimizer=Optimizers.AdamW,
    learning_rate=1e-4,
)

lr_scheduler = LRSchedulerConfig(
    type=LRSchedulerType.COSINE_ANNEALING_LR,
)

anydoor_config = AnydoorModelConfig(
    path_to_unet="ckpt/refiners/unet.safetensors",
    path_to_control_model="ckpt/refiners/controlnet.safetensors",
    path_to_object_encoder="ckpt/refiners/dinov2_encoder.safetensors",
    path_to_lda="ckpt/refiners/lda_new.safetensors",
    lora_rank=16,
    lora_scale=1.0,
    # lora_checkpoint="ckpt/lora/anydoor-vton-adaptation/lora_noname_0_1500.safetensors"
)
wandb = WandbConfig(
    mode="online",
    project="anydoor-vton-adaptation",
    entity="daniel-rodriguezciotti-sicara",
    name="curly-split",
)

evaluation = EvaluationConfig(
    interval = Epoch(1),
    seed = 42
)

training_config = AnydoorTrainingConfig(
    train_dataset='dataset/train/cloth',
    train_lora_dataset_selection= 'dataset/lora_training_images.txt',
    test_dataset='dataset/test/cloth',
    test_lora_dataset_selection='dataset/lora_test_images.txt',
    batch_size=5,
    # checkpoint_interval=500,
    saving_path='ckpt/lora',
    wandb=wandb,
    anydoor=anydoor_config,
    training=training,
    optimizer=optimizer,
    lr_scheduler=lr_scheduler,
    evaluation  = evaluation
)
trainer = AnyDoorLoRATrainer(training_config)
trainer.train()

