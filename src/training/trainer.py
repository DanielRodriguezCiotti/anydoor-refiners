
import os
from typing import Any, Iterable
import numpy as np
import torch
import random
from loguru import logger
from refiners.training_utils import Trainer,  register_model, Callback,register_callback
from anydoor_refiners.lora import build_lora, get_lora_weights, set_lora_weights
from refiners.fluxion.utils import save_to_safetensors, load_from_safetensors
from src.training.data.vitonhd import CustomDataLoader, VitonHDDataset
from src.anydoor_refiners.model import AnyDoor
from torch.utils.data import DataLoader
from refiners.training_utils.wandb import WandbLogger,WandbLoggable
from training.configs import EvaluationConfig, AnydoorTrainingConfig, AnydoorModelConfig, AnydoorEvaluatorConfig
from training.data.batch import AnyDoorBatch, collate_fn
from training.evaluator import AnyDoorLoraEvaluator



class EvaluationCallback(Callback[Any]):
    def __init__(self, config: EvaluationConfig) -> None:
        self.config = config

    def on_epoch_end(self, trainer: Trainer) -> None:
        # The `is_due` method checks if the current epoch is a multiple of the interval.
        if not trainer.clock.is_due(self.config.interval):
            return
        trainer.compute_evaluation() # type: ignore


class AnyDoorLoRATrainer(Trainer[AnydoorTrainingConfig, AnyDoorBatch]):


    def __init__(self, config: AnydoorTrainingConfig):
        self.use_atv_loss = config.use_atv_loss
        super().__init__(config)
        self.test_dataloader =  CustomDataLoader(VitonHDDataset(config.test_dataset, filtering_file=config.test_lora_dataset_selection, inference=True),batch_size=self.config.batch_size, shuffle=True, collate_fn=collate_fn)
        self.load_wandb()
        self.load_evaluator()
        self.last_lora_checkpoint = None

    def load_wandb(self) -> None:
        init_config = {**self.config.wandb.model_dump(), "config": self.config.model_dump()}
        self.wandb = WandbLogger(init_config=init_config)

    def log(self, data: dict[str, WandbLoggable]) -> None:
        self.wandb.log(data=data, step=self.clock.step)

    def load_evaluator(self) -> None:
        config = AnydoorEvaluatorConfig(
            test_dataset= self.config.test_dataset,
            batch_size= self.config.batch_size,
            test_lora_dataset_selection = self.config.test_lora_dataset_selection
        )

        self.evaluator = AnyDoorLoraEvaluator(
            wandb = self.wandb,
            config = config,
            clock = self.clock,
            model_config = self.config.anydoor,
            device = torch.device("cuda:1")
        )

    def create_data_iterable(self) -> Iterable[AnyDoorBatch]:
        dataset = VitonHDDataset(self.config.train_dataset,filtering_file=self.config.train_lora_dataset_selection)
        dataloader = DataLoader(dataset, batch_size=self.config.batch_size, collate_fn=collate_fn)
        return dataloader
    
    @register_callback()
    def evaluation(self, config: EvaluationConfig) -> EvaluationCallback:
        return EvaluationCallback(config)   

    @register_model()
    def anydoor(self,config:AnydoorModelConfig) -> AnyDoor:
        logger.info("Building AnyDoor model")
        model = AnyDoor(num_inference_steps = 1000, use_tv_loss=self.use_atv_loss,device=self.device, dtype=self.dtype)
        logger.info("Loading weights")
        model.unet.load_from_safetensors(config.path_to_unet)
        model.control_model.load_from_safetensors(config.path_to_control_model)
        model.object_encoder.load_from_safetensors(config.path_to_object_encoder)   
        model.lda.load_from_safetensors(config.path_to_lda)
        logger.info("Building LoRA layers")
        build_lora(model, config.lora_rank, config.lora_scale)
        if config.lora_checkpoint is not None:
            logger.info("Loading LoRA weights")
            lora_weights = load_from_safetensors(config.lora_checkpoint)
            set_lora_weights(model.unet,lora_weights)

        logger.info("Setting LoRA layers to trainable")
        for name, param in model.named_parameters():
            if "LinearLora" in name:
                param.requires_grad = True  # Leave these layers trainable
            else:
                param.requires_grad = False

        model_parameters = filter(lambda p: p.requires_grad, model.parameters())
        learnable_params = sum([np.prod(p.size()) for p in model_parameters])
        total_params = sum(p.numel() for p in model.parameters())
        logger.info("Learnable parameters:" + '{:,}'.format(learnable_params))
        logger.info("Total parameters: " + '{:,}'.format(total_params))
        logger.info(f"Percetage of learnable parameters: {learnable_params / total_params * 100:.2f}%")
        return model
    
    def q_sample(self, images: torch.Tensor,noise: torch.Tensor, timestep : int):
        scale_factor = self.anydoor.solver.cumulative_scale_factors[timestep]
        sqrt_one_minus_scale_factor = self.anydoor.solver.noise_std[timestep]
        return scale_factor * images + sqrt_one_minus_scale_factor * noise


    def compute_loss(self, batch: AnyDoorBatch) -> torch.Tensor:

        if batch is None:
            return torch.tensor(0.0, device=self.device, dtype=self.dtype)
        object = batch.object.to(self.device, self.dtype)
        background = batch.background.to(self.device, self.dtype)
        collage = batch.collage.to(self.device, self.dtype)
        batch_size = object.shape[0]
        timestep = random.randint(0,999)



        object_embedding = self.anydoor.object_encoder.forward(object)
        noise = self.anydoor.sample_noise(size=(batch_size, 4, 64, 64), device=self.device, dtype=self.dtype)
        background_latents = self.anydoor.lda.encode(background)
        noisy_backgrounds = self.q_sample(background_latents, noise, timestep)
        if self.use_atv_loss:
            assert batch.loss_mask is not None, "Loss mask is required"
            loss_mask = batch.loss_mask.to(self.device, self.dtype)
            predicted_noise, atv_loss = self.anydoor.forward(noisy_backgrounds,step=timestep,control_background_image=collage,object_embedding=object_embedding,training=True, loss_mask=loss_mask)
        else:
            predicted_noise, atv_loss = self.anydoor.forward(noisy_backgrounds,step=timestep,control_background_image=collage,object_embedding=object_embedding,training=True)
        if self.config.loss_type == "l2":
            loss = torch.norm(noise - predicted_noise, p=2)
        elif self.config.loss_type == "mse":
            loss = torch.nn.functional.mse_loss(noise, predicted_noise) # type: ignore
        else:
            raise ValueError(f"Invalid loss type {self.config.loss_type}")
        
        dict_to_log = { self.config.loss_type : loss.item(),"epoch": self.clock.epoch ,"iteration": self.clock.iteration, "learning_rate": self.lr_scheduler.get_last_lr()[0]}

        if self.use_atv_loss:
            logger.info(f"ATV loss: {atv_loss.item()}")
            dict_to_log["atv_loss"] = atv_loss.item()
            loss += atv_loss

        dict_to_log["loss"] = loss.item()

        self.log(dict_to_log)
        return loss
    
    def save_lora(self) -> None:
        training_name = self.config.wandb.name
        epoch = self.clock.epoch
        saving_path = self.config.saving_path
        project = self.config.wandb.project
        save_model_directory = saving_path + "/"+ project 

        if not os.path.exists(save_model_directory):
            os.makedirs(save_model_directory)
        weights = get_lora_weights(self.anydoor.unet)
        path = save_model_directory + f"/lora_{training_name}_{epoch}.safetensors"
        self.last_lora_checkpoint = path
        save_to_safetensors(path = path, tensors = weights)
        logger.info(f"Saved LoRA weights to {path} for epoch {epoch}")

    def compute_evaluation(self) -> None:
        self.save_lora()
        self.evaluator.compute_evaluation(self.last_lora_checkpoint)




