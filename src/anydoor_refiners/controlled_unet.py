from typing import Iterable, cast

from torch import Tensor, device as Device, dtype as DType

import refiners.fluxion.layers as fl
from refiners.fluxion.context import Contexts
from src.anydoor_refiners.unet import (
    DownBlocks,
    UpBlocks,
    MiddleBlock,
    ResidualBlock,
    TimestepEncoder,
)
from refiners.foundationals.latent_diffusion.range_adapter import RangeAdapter2d
from refiners.foundationals.latent_diffusion.unet import (
    ResidualAccumulator,
)


class ResidualControlledConcatenator(fl.Chain):
    def __init__(self, n: int) -> None:
        self.n = n
        if n == -2:
            layer = fl.Residual(
                fl.UseContext(context="control", key="residuals").compose(
                    lambda residuals: residuals[self.n + 1]
                ),
            )
        else:
            layer = fl.Identity()
        super().__init__(
            fl.Concatenate(
                layer,
                fl.Sum(
                    fl.UseContext(context="unet", key="residuals").compose(
                        lambda residuals: residuals[self.n]
                    ),
                    fl.UseContext(context="control", key="residuals").compose(
                        lambda residuals: residuals[self.n]
                    ),
                ),
                dim=1,
            ),
        )


class ControlledUNet(fl.Chain):
    """The controlled U-Net model."""

    def __init__(
        self,
        in_channels: int,
        device: Device | str | None = None,
        dtype: DType | None = None,
    ) -> None:
        """Initialize the U-Net.

        Args:
            in_channels: The number of input channels.
            device: The PyTorch device to use for computation.
            dtype: The PyTorch dtype to use for computation.
        """
        self.in_channels = in_channels
        super().__init__(
            TimestepEncoder(device=device, dtype=dtype),
            DownBlocks(in_channels=in_channels, device=device, dtype=dtype),
            fl.Sum(
                fl.UseContext(context="unet", key="residuals").compose(lambda x: x[-1]),
                MiddleBlock(device=device, dtype=dtype),
            ),
            UpBlocks(device=device, dtype=dtype),
            fl.Chain(
                fl.GroupNorm(channels=320, num_groups=32, device=device, dtype=dtype),
                fl.SiLU(),
                fl.Conv2d(
                    in_channels=320,
                    out_channels=4,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    device=device,
                    dtype=dtype,
                ),
            ),
        )
        for residual_block in self.layers(ResidualBlock):
            chain = residual_block.layer("Chain", fl.Chain)
            RangeAdapter2d(
                target=chain.layer("Conv2d_1", fl.Conv2d),
                channels=residual_block.out_channels,
                embedding_dim=1280,
                context_key="timestep_embedding",
                device=device,
                dtype=dtype,
            ).inject(chain)
        for n, block in enumerate(cast(Iterable[fl.Chain], self.DownBlocks)):
            block.append(ResidualAccumulator(n))
        for n, block in enumerate(cast(Iterable[fl.Chain], self.UpBlocks)):
            block.insert(0, ResidualControlledConcatenator(-n - 2))

    def init_context(self) -> Contexts:
        return {
            "unet": {"residuals": [0.0] * 13},
            "control": {"residuals": [0.0] * 13},
            "diffusion": {"timestep": None},
            "range_adapter": {"timestep_embedding": None},
            "sampling": {"shapes": []},
        }

    def set_dinov2_object_embedding(self, dinov2_object_embedding: Tensor) -> None:
        """Set the CLIP text embedding.

        Note:
            This context is required by the `DinoV2CrossAttention` blocks.

        Args:
            dinov2_object_embedding: The DinoV2 garment embedding.
        """
        self.set_context(
            "cross_attention_block",
            {"dinov2_object_embedding": dinov2_object_embedding},
        )

    def set_timestep(self, timestep: Tensor) -> None:
        """Set the timestep.

        Note:
            This context is required by `TimestepEncoder`.

        Args:
            timestep: The timestep.
        """
        self.set_context("diffusion", {"timestep": timestep})

    def set_control_residuals(self, residuals: list[Tensor | float]) -> None:
        """Set the control residuals.

        Args:
            residuals: The control residuals.
        """
        self.set_context("control", {"residuals": residuals})
