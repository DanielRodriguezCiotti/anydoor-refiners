
import os
from typing import Iterable, Literal
import numpy as np
from PIL import Image
import torch
import random
from loguru import logger
from pydantic import BaseModel
from refiners.training_utils import ModelConfig, Trainer, BaseConfig, register_model
from dataclasses import dataclass
from anydoor_refiners.lora import build_lora, get_lora_weights, set_lora_weights
from refiners.fluxion.utils import no_grad, save_to_safetensors, load_from_safetensors, manual_seed
from anydoor_refiners.postprocessing import post_processing
from src.training.data.vitonhd import VitonHDDataset
from src.anydoor_refiners.model import AnyDoor
from torch.utils.data import DataLoader
from refiners.training_utils.wandb import WandbLogger,WandbLoggable

manual_seed(10)

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

class AnydoorTrainingConfig(BaseConfig):
    train_dataset : str
    test_dataset : str
    saving_path : str
    batch_size : int 
    checkpoint_interval : int
    anydoor : AnydoorModelConfig
    wandb : WandbConfig




def collate_fn(batch: list) -> AnydoorBatch | None:
    # Filter out None entries
    batch = [item for item in batch if item is not None]
    if len(batch) == 0:
        return None
    final_batch = {}
    for key in batch[0].keys():
        final_batch[key] = torch.stack([item[key] for item in batch])
    return AnydoorBatch(**final_batch)

    

class AnyDoorLoRATrainer(Trainer[AnydoorTrainingConfig, AnydoorBatch]):


    def __init__(self, config: AnydoorTrainingConfig):
        super().__init__(config)
        self.test_dataset = VitonHDDataset(config.test_dataset)
        self.load_wandb()

    def load_wandb(self) -> None:
        init_config = {**self.config.wandb.model_dump(), "config": self.config.model_dump()}
        self.wandb = WandbLogger(init_config=init_config)

    def log(self, data: dict[str, WandbLoggable]) -> None:
        self.wandb.log(data=data, step=self.clock.step)

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

    def create_data_iterable(self) -> Iterable[AnydoorBatch]:
        dataset = VitonHDDataset(self.config.train_dataset)
        dataloader = DataLoader(dataset, batch_size=self.config.batch_size, shuffle=True, collate_fn=collate_fn)
        return dataloader

    def compute_loss(self, batch: AnydoorBatch) -> torch.Tensor:

        if batch is None:
            return torch.tensor(0.0, device=self.device, dtype=self.dtype)
        object = batch.object.to(self.device, self.dtype)
        background = batch.background.to(self.device, self.dtype)
        collage = batch.collage.to(self.device, self.dtype)
        batch_size = object.shape[0]
        # random integer between 0 and 1000
        timestep = random.randint(0,999)

        object_embedding = self.anydoor.object_encoder.forward(object)
        noise = self.anydoor.sample_noise(size=(batch_size, 4, 64, 64), device=self.device, dtype=self.dtype)
        background_latents = self.anydoor.lda.encode(background)
        noisy_backgrounds = self.q_sample(background_latents, noise, timestep)
        predicted_noise = self.anydoor.forward(noisy_backgrounds,step=timestep,control_background_image=collage,object_embedding=object_embedding,training=True)
        loss = torch.nn.functional.mse_loss(noise, predicted_noise)
        self.log({"loss": loss.item(), "epoch": self.clock.epoch ,"iteration": self.clock.iteration, "learning_rate": self.lr_scheduler.get_last_lr()[0]})
        return loss
    
    def save_lora(self) -> None:
        training_name = self.config.wandb.name
        epoch = self.clock.epoch
        saving_path = self.config.saving_path
        project = self.config.wandb.project
        save_model_directory = saving_path + "/"+ project 
        iteration = self.clock.iteration

        if not os.path.exists(save_model_directory):
            os.makedirs(save_model_directory)
        weights = get_lora_weights(self.anydoor.unet)
        save_to_safetensors(path = save_model_directory + f"/lora_{training_name}_{epoch}_{iteration}.safetensors", tensors = weights)
        logger.info(f"Saved LoRA weights to {save_model_directory} for epoch {epoch} and iteration {iteration}")

    
    def log_images_to_wandb(self):

        logger.info("Logging images to wandb ...")
        
        samples = []
        for i in range(self.config.batch_size):
            samples += [self.test_dataset.get_sample(i, inference=True)]
        batch = collate_fn(samples)
        assert batch is not None
        assert batch.background_image is not None
        object_tensor = batch.object.to(self.device, self.dtype)
        control_tensor = batch.collage.to(self.device, self.dtype)
        batch_size = object_tensor.shape[0]

        # Log shapes
        logger.info(f"Object tensor shape: {object_tensor.shape}")
        logger.info(f"Control tensor shape: {control_tensor.shape}")


        self.anydoor.set_inference_steps(50)

        with no_grad():  
            object_embedding = self.anydoor.object_encoder.forward(object_tensor)
            negative_object_embedding = self.anydoor.object_encoder.forward(torch.zeros((batch_size, 3, 224, 224),device=self.device,dtype=self.dtype))
            x = self.anydoor.sample_noise((batch_size,4,512//8, 512//8), device=self.device,dtype=self.dtype)

            for step in self.anydoor.steps:
                x = self.anydoor.forward(
                    x,
                    step=step,
                    control_background_image= control_tensor,
                    object_embedding= object_embedding,
                    negative_object_embedding= negative_object_embedding,
                    condition_scale= 5.0
                )
            
            predicted_images = {}
            for i in range(batch_size):
                predicted_images[i] = self.anydoor.lda.latents_to_image(x[i].unsqueeze(0))
        
        self.anydoor.set_inference_steps(1000)
        
        final_images = []
        for i in range(batch_size):
            generated_image = Image.fromarray(post_processing(np.array(predicted_images[i]),batch.background_image.numpy()[i],batch.sizes.tolist()[i],batch.background_box.tolist()[i]))
            ground_truth = Image.fromarray(batch.background_image.numpy()[i])
            concatenated_image = Image.new("RGB", (2*generated_image.width, generated_image.height))
            concatenated_image.paste(generated_image, (0, 0))
            concatenated_image.paste(ground_truth, (generated_image.width, 0))
            final_images.append(concatenated_image)

        
        i = 0
        for image in final_images:
            self.log({"tryon_sample_" + str(i): image})
            i += 1


    def checkpoint(self) -> None:
        self.save_lora()
        self.log_images_to_wandb()

    def step(self, batch) -> None:
        """Perform a single training step."""
        super().step(batch)
        if self.clock.iteration % self.config.checkpoint_interval == 0:
            self.checkpoint()