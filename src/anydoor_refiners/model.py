import torch
from PIL import Image
from torch import Tensor, device as Device, dtype as DType

import refiners.fluxion.layers as fl
from refiners.foundationals.latent_diffusion.solvers import (
    DDIM,
    SolverParams,
    NoiseSchedule,
)
from src.anydoor_refiners.dinov2 import DINOv2Encoder
from src.anydoor_refiners.unet import UNet
from anydoor_refiners.controlnet import ControlNet
from anydoor_refiners.lda import AnydoorAutoencoder



solver_params = SolverParams(
    num_train_timesteps=1000,
    noise_schedule=NoiseSchedule.QUADRATIC,
    initial_diffusion_rate=0.00085,
    final_diffusion_rate=0.0120,
)


class AnyDoor(fl.Module):
    def __init__(
        self,
        num_inference_steps: int = 50,
        device: Device | str = "cpu",
        dtype: DType = torch.float32,
    ) -> None:
        super().__init__()
        self.device: Device = (
            device if isinstance(device, Device) else Device(device=device)
        )
        self.dtype = dtype
        self.unet = UNet(4, device=self.device, dtype=self.dtype)
        self.lda = AnydoorAutoencoder(device=self.device, dtype=self.dtype)
        self.object_encoder = DINOv2Encoder(device=self.device, dtype=self.dtype)
        self.control_model = ControlNet(4, device=self.device, dtype=self.dtype)
        self.solver = DDIM(num_inference_steps=num_inference_steps, params=solver_params)


    def set_inference_steps(self, num_steps: int, first_step: int = 0) -> None:
        """Set the steps of the diffusion process.

        Args:
            num_steps: The number of inference steps.
            first_step: The first inference step, used for image-to-image diffusion.
                You may be used to setting a float in `[0, 1]` called `strength` instead,
                which is an abstraction for this. The first step is
                `round((1 - strength) * (num_steps - 1))`.
        """
        self.solver = self.solver.rebuild(
            num_inference_steps=num_steps, first_inference_step=first_step
        )

    @staticmethod
    def sample_noise(
        size: tuple[int, ...],
        device: Device | None = None,
        dtype: DType | None = None,
        offset_noise: float | None = None,
    ) -> torch.Tensor:
        """Sample noise from a normal distribution with an optional offset.

        Args:
            size: The size of the noise tensor.
            device: The device to put the noise tensor on.
            dtype: The data type of the noise tensor.
            offset_noise: The offset of the noise tensor.
                Useful at training time, see https://www.crosslabs.org/blog/diffusion-with-offset-noise.
        """
        noise = torch.randn(size=size, device=device, dtype=dtype)
        if offset_noise is not None:
            noise += offset_noise * torch.randn(
                size=(size[0], size[1], 1, 1), device=device, dtype=dtype
            )
        return noise

    def init_latents(
        self,
        size: tuple[int, int],
        init_image: Image.Image | None = None,
        noise: Tensor | None = None,
    ) -> Tensor:
        """Initialize the latents for the diffusion process.

        Args:
            size: The size of the latent (in pixel space).
            init_image: The image to use as initialization for the latents.
            noise: The noise to add to the latents.
        """
        height, width = size
        latent_height = height // 8
        latent_width = width // 8

        if noise is None:
            noise = AnyDoor.sample_noise(
                size=(1, 4, latent_height, latent_width),
                device=self.device,
                dtype=self.dtype,
            )

        assert list(noise.shape[2:]) == [
            latent_height,
            latent_width,
        ], f"noise shape is not compatible: {noise.shape}, with size: {size}"
        
        latent = noise    

        return self.solver.scale_model_input(latent, step=-1)

    @property
    def steps(self) -> list[int]:
        return self.solver.inference_steps

    def set_unet_context(
        self,
        *,
        timestep: Tensor,
        object_embedding: Tensor,
        control_features: list[Tensor],
    ) -> None:
        """Set the various context parameters required by the U-Net model.

        Args:
            timestep: The timestep tensor to use.
            clip_text_embedding: The CLIP text embedding tensor to use.
        """
        self.unet.set_timestep(timestep=timestep)
        self.unet.set_dinov2_object_embedding(dinov2_object_embedding=object_embedding)
        self.unet.set_control_residuals(residuals=control_features)

    def forward(
        self,
        x: Tensor,
        step: int,
        control_background_image : Tensor,
        object_embedding : Tensor,
        negative_object_embedding : Tensor | None = None,
        condition_scale: float = 1.0,
    ) -> Tensor:
        # Init variables
        latents = x
        timestep = self.solver.timesteps[step].unsqueeze(dim=0)
        latents = self.solver.scale_model_input(latents, step=step) # Returns latents for DDIM
        
        # Compute control
        self.control_model.set_timestep(timestep=timestep)
        self.control_model.set_dinov2_object_embedding(dinov2_object_embedding=object_embedding)
        control = self.control_model(control_background_image)

        # Compute predicted noise
        self.set_unet_context(
            timestep=timestep,
            object_embedding=object_embedding,
            control_features=control,
        )
        predicted_noise = self.unet(latents)
        if condition_scale != 1.0 and negative_object_embedding is not None:
            self.set_unet_context(
                timestep=timestep,
                object_embedding=negative_object_embedding,
                control_features=control
            )
            unconditionned_predicted_noise = self.unet(latents)
            predicted_noise = unconditionned_predicted_noise + condition_scale * (
                predicted_noise - unconditionned_predicted_noise
            )

        x = x.narrow(dim=1, start=0, length=4)  # support > 4 channels for inpainting
        result = self.solver(x, predicted_noise=predicted_noise, step=step)
        return result

    def compute_object_embedding(self, image: Tensor) -> Tensor:
        """ """
        return self.object_encoder(image)

    def compute_control_features(self, image: Tensor, object) -> list[Tensor]:
        """ """
        return self.control_model(image)
