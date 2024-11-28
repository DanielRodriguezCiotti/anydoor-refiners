from typing import Iterable
import numpy as np
from PIL import Image
import torch
from torch import device as Device, dtype as DType
from loguru import logger

from tqdm import tqdm
from anydoor_refiners.lora import build_lora, set_lora_weights
from refiners.fluxion.utils import no_grad, load_from_safetensors
from anydoor_refiners.postprocessing import post_processing
from src.training.data.vitonhd import CustomDataLoader, VitonHDDataset
from src.anydoor_refiners.model import AnyDoor
from torch.utils.data import DataLoader
from refiners.training_utils.clock import TrainingClock
from refiners.training_utils.common import scoped_seed
from refiners.training_utils.wandb import WandbLogger, WandbLoggable

from training.configs import AnydoorEvaluatorConfig, AnydoorModelConfig
from training.data.batch import AnyDoorBatch, collate_fn

MAX_NUM_BATCHES_TO_WANDB = 2

class AnyDoorLoraEvaluator:

    def __init__(
        self,
        clock :  TrainingClock,
        wandb: WandbLogger | None,
        config: AnydoorEvaluatorConfig,
        model_config: AnydoorModelConfig,
        device: Device,
        dtype: DType = torch.float16,
    ) -> None:
        self.wandb = wandb
        self.clock = clock  
        self.config = config
        self.seed = 42
        self.device = device
        self.dtype = dtype
        self.test_dataloader = self.create_data_iterable()
        self.anydoor = self.build_model(model_config)

    def log(self, data: dict[str, WandbLoggable]) -> None:
        if self.wandb is not None:
            self.wandb.log(data=data, step=self.clock.step)

    def create_data_iterable(self) -> Iterable[AnyDoorBatch]:
        dataset = VitonHDDataset(
            self.config.test_dataset,
            filtering_file=self.config.test_lora_dataset_selection,
            inference=True,
        )
        dataloader = CustomDataLoader(
            dataset,
            batch_size=self.config.batch_size,
            collate_fn=collate_fn,
        )
        return dataloader

    def build_model(self, config:AnydoorModelConfig) -> AnyDoor:
        logger.info(f"Building Evaluator AnyDoor model on device { self.device }")
        model = AnyDoor(num_inference_steps=1000, device=self.device, dtype=self.dtype)
        model.unet.load_from_safetensors(config.path_to_unet)
        model.control_model.load_from_safetensors(config.path_to_control_model)
        model.object_encoder.load_from_safetensors(config.path_to_object_encoder)
        model.lda.load_from_safetensors(config.path_to_lda)
        build_lora(model, config.lora_rank, config.lora_scale)
        model.eval()
        return model

    def load_lora_weights(self, path_to_lora: str) -> None:
        logger.info("Loading LoRA weights")
        lora_weights = load_from_safetensors(path_to_lora)
        set_lora_weights(self.anydoor.unet, lora_weights)

    def q_sample(self, images: torch.Tensor, noise: torch.Tensor, timestep: int):
        scale_factor = self.anydoor.solver.cumulative_scale_factors[timestep]
        sqrt_one_minus_scale_factor = self.anydoor.solver.noise_std[timestep]
        return scale_factor * images + sqrt_one_minus_scale_factor * noise

    def compute_loss(self, batch:AnyDoorBatch, timestep: int) -> torch.Tensor:

        if batch is None:
            return torch.tensor(0.0, device=self.device, dtype=self.dtype)
        
        with no_grad():
            object = batch.object.to(self.device, self.dtype)
            background = batch.background.to(self.device, self.dtype)
            collage = batch.collage.to(self.device, self.dtype)
            batch_size = object.shape[0]
            object_embedding = self.anydoor.object_encoder.forward(object)
            noise = self.anydoor.sample_noise(
                size=(batch_size, 4, 64, 64), device=self.device, dtype=self.dtype
            )
            background_latents = self.anydoor.lda.encode(background)
            noisy_backgrounds = self.q_sample(background_latents, noise, timestep)   
            predicted_noise, loss = self.anydoor.forward(
                noisy_backgrounds,
                step=timestep,
                control_background_image=collage,
                object_embedding=object_embedding,
                training=True,
            )
            loss = torch.nn.functional.mse_loss(noise, predicted_noise)
        return loss

    def log_images_to_wandb(self):

        logger.info("Logging images to wandb ...")

        self.anydoor.set_inference_steps(50)
        predicted_images = {}
        i = 0
        batch_count = 0
        for batch in tqdm(self.test_dataloader):
            assert batch is not None
            assert batch.background_image is not None
            batch_count += 1
            object = batch.object.to(self.device, self.dtype)
            collage = batch.collage.to(self.device, self.dtype)
            batch_size = object.shape[0]

            with no_grad():
                object_embedding = self.anydoor.object_encoder.forward(object)
                negative_object_embedding = self.anydoor.object_encoder.forward(
                    torch.zeros(
                        (batch_size, 3, 224, 224), device=self.device, dtype=self.dtype
                    )
                )
                x = self.anydoor.sample_noise(
                    (batch_size, 4, 512 // 8, 512 // 8),
                    device=self.device,
                    dtype=self.dtype,
                )

                for step in self.anydoor.steps:   
                    x = self.anydoor.forward(
                        x,
                        step=step,
                        control_background_image=collage,
                        object_embedding=object_embedding,
                        negative_object_embedding=negative_object_embedding,
                        condition_scale=5.0,
                    )

                background_images = batch.background_image.numpy()
                for j in range(batch_size):
                    i += 1
                    predicted_images[i] = {
                        "predicted_image": self.anydoor.lda.latents_to_image(
                            x[j].unsqueeze(0)
                        ),
                        "ground_truth": background_images[j],
                        "sizes": batch.sizes.tolist()[j],
                        "background_box": batch.background_box.tolist()[j],
                    }
            if batch_count >= MAX_NUM_BATCHES_TO_WANDB:
                break

        self.anydoor.set_inference_steps(1000)

        final_images = []
        for predicted_image in predicted_images.values():
            generated_image = Image.fromarray(
                post_processing(
                    np.array(predicted_image["predicted_image"]),
                    predicted_image["ground_truth"],
                    predicted_image["sizes"],
                    predicted_image["background_box"],
                )
            )
            self.log({"tryon_sample_" + str(i): generated_image})
            ground_truth = Image.fromarray(predicted_image["ground_truth"])
            concatenated_image = Image.new(
                "RGB", (2 * generated_image.width, generated_image.height)
            )
            concatenated_image.paste(generated_image, (0, 0))
            concatenated_image.paste(ground_truth, (generated_image.width, 0))
            final_images.append(concatenated_image)

        i = 0
        for image in final_images:
            self.log({"tryon_sample_with_gt_" + str(i): image})
            i += 1

    def compute_evaluation(self,path_to_lora:str|None) -> None:
        if path_to_lora is not None:
            self.load_lora_weights(path_to_lora)
            with scoped_seed(self.seed):
                self.log_images_to_wandb()
                loss_records = []
                logger.info("Computing evaluation loss")
                for batch in tqdm(self.test_dataloader):
                    for timestep in np.linspace(0, 999, 10):
                        loss = self.compute_loss(
                            batch, timestep=int(timestep)
                        )
                        loss_records += [
                            {
                                "loss": loss.item(),
                                "timestep": timestep,
                                "epoch": self.clock.epoch,
                            }
                        ]
                agg_loss = sum([record["loss"] for record in loss_records]) / len(loss_records)
                self.log({"evaluation_loss": agg_loss, "epoch": self.clock.epoch})


# if __name__ == "__main__":
#     from refiners.training_utils import Epoch, Step


#     clock = TrainingClock(Epoch(10), Step(1), Step(10))
#     config = AnydoorEvaluatorConfig(
#         test_dataset='dataset/test/cloth',
#         batch_size=4,
#         test_lora_dataset_selection='dataset/lora_test_images.txt'
#     )
#     model_config = AnydoorModelConfig(
#         path_to_unet="ckpt/refiners/unet.safetensors",
#         path_to_control_model="ckpt/refiners/controlnet.safetensors",
#         path_to_object_encoder="ckpt/refiners/dinov2_encoder.safetensors",
#         path_to_lda="ckpt/refiners/lda_new.safetensors",
#         lora_rank=16,
#         lora_scale=1.0,
#     )
#     evaluator = AnyDoorLoraEvaluator(clock, None, config, model_config, torch.device("cuda:1"))
#     evaluator.compute_evaluation(path_to_lora='ckpt/lora/anydoor-vton-adaptation/lora_zara_0.safetensors')

