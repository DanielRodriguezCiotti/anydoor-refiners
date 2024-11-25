
import os
from typing import Any, Iterable, Literal
import numpy as np
from PIL import Image
import torch
import random
from loguru import logger
from pydantic import BaseModel
from refiners.training_utils import ModelConfig, Trainer, BaseConfig, register_model, Callback, CallbackConfig, register_callback
from refiners.training_utils.config import TimeValueField 
from refiners.training_utils.common import scoped_seed
from dataclasses import dataclass

from tqdm import tqdm
from anydoor_refiners.lora import build_lora, get_lora_weights, set_lora_weights
from refiners.fluxion.utils import no_grad, save_to_safetensors, load_from_safetensors
from anydoor_refiners.postprocessing import post_processing
from src.training.data.vitonhd import VitonHDDataset
from src.anydoor_refiners.model import AnyDoor
from torch.utils.data import DataLoader
from refiners.training_utils.wandb import WandbLogger,WandbLoggable

@dataclass
class AnydoorBatch:
    object : torch.Tensor
    background : torch.Tensor
    collage : torch.Tensor
    background_box : torch.Tensor
    sizes : torch.Tensor
    time_steps : torch.Tensor
    background_image : torch.Tensor | None = None


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
    anydoor : AnydoorModelConfig
    wandb : WandbConfig
    evaluation: EvaluationConfig
    train_lora_dataset_selection : str | None = None
    test_lora_dataset_selection : str | None = None




def collate_fn(batch: list) -> AnydoorBatch | None:
    # Filter out None entries
    batch = [item for item in batch if item is not None]
    if len(batch) == 0:
        return None
    final_batch = {}
    for key in batch[0].keys():
        final_batch[key] = torch.stack([item[key] for item in batch])
    return AnydoorBatch(**final_batch)



class EvaluationCallback(Callback[Any]):
    def __init__(self, config: EvaluationConfig) -> None:
        self.config = config

    def on_epoch_end(self, trainer: Trainer) -> None:
        # The `is_due` method checks if the current epoch is a multiple of the interval.
        if not trainer.clock.is_due(self.config.interval):
            return

        # The `scoped_seed` context manager encapsulates the random state for the evaluation and restores it after the 
        # evaluation.
        with scoped_seed(self.config.seed):
            trainer.compute_evaluation()

class AnyDoorLoRATrainer(Trainer[AnydoorTrainingConfig, AnydoorBatch]):


    def __init__(self, config: AnydoorTrainingConfig):
        super().__init__(config)
        self.test_dataloader =  DataLoader(VitonHDDataset(config.test_dataset, filtering_file=config.test_lora_dataset_selection, inference=True),batch_size=self.config.batch_size, shuffle=True, collate_fn=collate_fn)
        self.load_wandb()

    def load_wandb(self) -> None:
        init_config = {**self.config.wandb.model_dump(), "config": self.config.model_dump()}
        self.wandb = WandbLogger(init_config=init_config)

    def log(self, data: dict[str, WandbLoggable]) -> None:
        self.wandb.log(data=data, step=self.clock.step)

    def create_data_iterable(self) -> Iterable[AnydoorBatch]:
        dataset = VitonHDDataset(self.config.train_dataset,filtering_file=self.config.train_lora_dataset_selection)
        dataloader = DataLoader(dataset, batch_size=self.config.batch_size, collate_fn=collate_fn)
        return dataloader
    
    @register_callback()
    def evaluation(self, config: EvaluationConfig) -> EvaluationCallback:
        return EvaluationCallback(config)   

    @register_model()
    def anydoor(self,config:AnydoorModelConfig) -> AnyDoor:
        logger.info("Building AnyDoor model")
        model = AnyDoor(num_inference_steps = 1000,device=self.device, dtype=self.dtype)
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


    def compute_loss(self, batch: AnydoorBatch, evaluation:bool = False, timestep : int | None = None) -> torch.Tensor:

        if batch is None:
            return torch.tensor(0.0, device=self.device, dtype=self.dtype)
        object = batch.object.to(self.device, self.dtype)
        background = batch.background.to(self.device, self.dtype)
        collage = batch.collage.to(self.device, self.dtype)
        batch_size = object.shape[0]
        # random integer between 0 and 1000
        if timestep is None:
            _timestep = random.randint(0,999)
        else :
            _timestep = timestep


        object_embedding = self.anydoor.object_encoder.forward(object)
        noise = self.anydoor.sample_noise(size=(batch_size, 4, 64, 64), device=self.device, dtype=self.dtype)
        background_latents = self.anydoor.lda.encode(background)
        noisy_backgrounds = self.q_sample(background_latents, noise, _timestep)
        predicted_noise = self.anydoor.forward(noisy_backgrounds,step=_timestep,control_background_image=collage,object_embedding=object_embedding,training=True)
        loss = torch.nn.functional.mse_loss(noise, predicted_noise)
        # print(loss)
        if not evaluation:
            self.log({"loss": loss.item(), "epoch": self.clock.epoch ,"iteration": self.clock.iteration, "learning_rate": self.lr_scheduler.get_last_lr()[0]})
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
        save_to_safetensors(path = save_model_directory + f"/lora_{training_name}_{epoch}.safetensors", tensors = weights)
        logger.info(f"Saved LoRA weights to {save_model_directory} for epoch {epoch}")

    
    def log_images_to_wandb(self):

        logger.info("Logging images to wandb ...")

        self.anydoor.set_inference_steps(50)
        predicted_images = {}
        i = 0
        for batch in tqdm(self.test_dataloader):
            assert batch is not None
            assert batch.background_image is not None
            object = batch.object.to(self.device, self.dtype)
            background = batch.background.to(self.device, self.dtype)
            collage = batch.collage.to(self.device, self.dtype)
            batch_size = object.shape[0]

            with no_grad():  
                object_embedding = self.anydoor.object_encoder.forward(object)
                negative_object_embedding = self.anydoor.object_encoder.forward(torch.zeros((batch_size, 3, 224, 224),device=self.device,dtype=self.dtype))
                x = self.anydoor.sample_noise((batch_size,4,512//8, 512//8), device=self.device,dtype=self.dtype)

                for step in self.anydoor.steps:
                    x = self.anydoor.forward(
                        x,
                        step=step,
                        control_background_image= collage,
                        object_embedding= object_embedding,
                        negative_object_embedding= negative_object_embedding,
                        condition_scale= 5.0
                    )
                
                background_images = batch.background_image.numpy()
                for j in range(batch_size):
                    i+=1
                    predicted_images[i] = {
                        'predicted_image' : self.anydoor.lda.latents_to_image(x[j].unsqueeze(0)),
                        'ground_truth' : background_images[j],
                        'sizes': batch.sizes.tolist()[j],
                        'background_box': batch.background_box.tolist()[j]
                    }
        
        self.anydoor.set_inference_steps(1000)
        
        final_images = []
        for predicted_image in predicted_images.values():
            generated_image = Image.fromarray(post_processing(np.array(predicted_image['predicted_image']),predicted_image['ground_truth'],predicted_image['sizes'],predicted_image['background_box']))
            ground_truth = Image.fromarray(predicted_image['ground_truth'])
            concatenated_image = Image.new("RGB", (2*generated_image.width, generated_image.height))
            concatenated_image.paste(generated_image, (0, 0))
            concatenated_image.paste(ground_truth, (generated_image.width, 0))
            final_images.append(concatenated_image)

        
        i = 0
        for image in final_images:
            self.log({"tryon_sample_" + str(i): image})
            i += 1


    def compute_evaluation(self) -> None:
        self.save_lora()
        self.log_images_to_wandb()
        loss_records = []
        for batch in self.test_dataloader:
            logger.info("Computing evaluation loss")
            for timestep in tqdm(np.linspace(0,999,10)):
                with no_grad():
                    loss = self.compute_loss(batch, evaluation=True, timestep=int(timestep))
                    loss_records += [{"loss": loss.item(), "timestep": timestep, "epoch": self.clock.epoch}]
        agg_loss = sum([record["loss"] for record in loss_records]) / len(loss_records)
        self.log({"evaluation_loss": agg_loss, "epoch": self.clock.epoch})




