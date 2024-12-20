from typing import Iterable, cast

from torch import Tensor, device as Device, dtype as DType

import refiners.fluxion.layers as fl
from refiners.fluxion.context import Contexts
from src.anydoor_refiners.attention import (
    CrossAttentionBlock2d,
)
from refiners.foundationals.latent_diffusion.range_adapter import (
    RangeAdapter2d,
    RangeEncoder,
)
from refiners.foundationals.latent_diffusion.unet import (
    ResidualAccumulator,
    ResidualBlock,
)


class TimestepEncoder(fl.Passthrough):
    def __init__(
        self,
        context_key: str = "timestep_embedding",
        device: Device | str | None = None,
        dtype: DType | None = None,
    ) -> None:
        super().__init__(
            fl.UseContext("diffusion", "timestep"),
            RangeEncoder(320, 1280, device=device, dtype=dtype),
            fl.SetContext("range_adapter", context_key),
        )


class DinoV2CrossAttention(CrossAttentionBlock2d):
    def __init__(
        self,
        channels: int,
        nb_heads: int,
        device: Device | str | None = None,
        dtype: DType | None = None,
    ) -> None:
        super().__init__(
            channels=channels,
            context_embedding_dim=1024,
            context_key="dinov2_object_embedding",
            num_attention_heads=nb_heads,
            num_attention_layers=1,
            num_groups=32,
            use_bias=False,
            use_linear_projection=True,
            device=device,
            dtype=dtype,
        )


class DownBlocks(fl.Chain):
    def __init__(
        self,
        in_channels: int,
        device: Device | str | None = None,
        dtype: DType | None = None,
    ):
        self.in_channels = in_channels
        super().__init__(
            fl.Chain(
                fl.Conv2d(
                    in_channels=in_channels,
                    out_channels=320,
                    kernel_size=3,
                    padding=1,
                    device=device,
                    dtype=dtype,
                )
            ),
            fl.Chain(
                ResidualBlock(
                    in_channels=320, out_channels=320, device=device, dtype=dtype
                ),
                DinoV2CrossAttention(
                    channels=320, nb_heads=5, device=device, dtype=dtype
                ),
            ),
            fl.Chain(
                ResidualBlock(
                    in_channels=320, out_channels=320, device=device, dtype=dtype
                ),
                DinoV2CrossAttention(
                    channels=320, nb_heads=5, device=device, dtype=dtype
                ),
            ),
            fl.Chain(
                fl.Downsample(
                    channels=320, scale_factor=2, padding=1, device=device, dtype=dtype
                )
            ),
            fl.Chain(
                ResidualBlock(
                    in_channels=320, out_channels=640, device=device, dtype=dtype
                ),
                DinoV2CrossAttention(
                    channels=640, nb_heads=10, device=device, dtype=dtype
                ),
            ),
            fl.Chain(
                ResidualBlock(
                    in_channels=640, out_channels=640, device=device, dtype=dtype
                ),
                DinoV2CrossAttention(
                    channels=640, nb_heads=10, device=device, dtype=dtype
                ),
            ),
            fl.Chain(
                fl.Downsample(
                    channels=640, scale_factor=2, padding=1, device=device, dtype=dtype
                )
            ),
            fl.Chain(
                ResidualBlock(
                    in_channels=640, out_channels=1280, device=device, dtype=dtype
                ),
                DinoV2CrossAttention(
                    channels=1280, nb_heads=20, device=device, dtype=dtype
                ),
            ),
            fl.Chain(
                ResidualBlock(
                    in_channels=1280, out_channels=1280, device=device, dtype=dtype
                ),
                DinoV2CrossAttention(
                    channels=1280, nb_heads=20, device=device, dtype=dtype
                ),
            ),
            fl.Chain(
                fl.Downsample(
                    channels=1280, scale_factor=2, padding=1, device=device, dtype=dtype
                )
            ),
            fl.Chain(
                ResidualBlock(
                    in_channels=1280, out_channels=1280, device=device, dtype=dtype
                ),
            ),
            fl.Chain(
                ResidualBlock(
                    in_channels=1280, out_channels=1280, device=device, dtype=dtype
                ),
            ),
        )


class UpBlocks(fl.Chain):
    def __init__(
        self,
        device: Device | str | None = None,
        dtype: DType | None = None,
    ) -> None:
        super().__init__(
            fl.Chain(
                ResidualBlock(
                    in_channels=2560, out_channels=1280, device=device, dtype=dtype
                ),
            ),
            fl.Chain(
                ResidualBlock(
                    in_channels=2560, out_channels=1280, device=device, dtype=dtype
                ),
            ),
            fl.Chain(
                ResidualBlock(
                    in_channels=2560, out_channels=1280, device=device, dtype=dtype
                ),
                fl.Upsample(channels=1280, device=device, dtype=dtype),
            ),
            fl.Chain(
                ResidualBlock(
                    in_channels=2560, out_channels=1280, device=device, dtype=dtype
                ),
                DinoV2CrossAttention(
                    channels=1280, nb_heads=20, device=device, dtype=dtype
                ),
            ),
            fl.Chain(
                ResidualBlock(
                    in_channels=2560, out_channels=1280, device=device, dtype=dtype
                ),
                DinoV2CrossAttention(
                    channels=1280, nb_heads=20, device=device, dtype=dtype
                ),
            ),
            fl.Chain(
                ResidualBlock(
                    in_channels=1920, out_channels=1280, device=device, dtype=dtype
                ),
                DinoV2CrossAttention(
                    channels=1280, nb_heads=20, device=device, dtype=dtype
                ),
                fl.Upsample(channels=1280, device=device, dtype=dtype),
            ),
            fl.Chain(
                ResidualBlock(
                    in_channels=1920, out_channels=640, device=device, dtype=dtype
                ),
                DinoV2CrossAttention(
                    channels=640, nb_heads=10, device=device, dtype=dtype
                ),
            ),
            fl.Chain(
                ResidualBlock(
                    in_channels=1280, out_channels=640, device=device, dtype=dtype
                ),
                DinoV2CrossAttention(
                    channels=640, nb_heads=10, device=device, dtype=dtype
                ),
            ),
            fl.Chain(
                ResidualBlock(
                    in_channels=960, out_channels=640, device=device, dtype=dtype
                ),
                DinoV2CrossAttention(
                    channels=640, nb_heads=10, device=device, dtype=dtype
                ),
                fl.Upsample(channels=640, device=device, dtype=dtype),
            ),
            fl.Chain(
                ResidualBlock(
                    in_channels=960, out_channels=320, device=device, dtype=dtype
                ),
                DinoV2CrossAttention(
                    channels=320, nb_heads=5, device=device, dtype=dtype
                ),
            ),
            fl.Chain(
                ResidualBlock(
                    in_channels=640, out_channels=320, device=device, dtype=dtype
                ),
                DinoV2CrossAttention(
                    channels=320, nb_heads=5, device=device, dtype=dtype
                ),
            ),
            fl.Chain(
                ResidualBlock(
                    in_channels=640, out_channels=320, device=device, dtype=dtype
                ),
                DinoV2CrossAttention(
                    channels=320, nb_heads=5, device=device, dtype=dtype
                ),
            ),
        )


class MiddleBlock(fl.Chain):
    def __init__(
        self, device: Device | str | None = None, dtype: DType | None = None
    ) -> None:
        super().__init__(
            ResidualBlock(
                in_channels=1280, out_channels=1280, device=device, dtype=dtype
            ),
            DinoV2CrossAttention(
                channels=1280, nb_heads=20, device=device, dtype=dtype
            ),
            ResidualBlock(
                in_channels=1280, out_channels=1280, device=device, dtype=dtype
            ),
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


class UNet(fl.Chain):
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

    def set_control_residuals(self, residuals: list[Tensor]) -> None:
        """Set the control residuals.

        Args:
            residuals: The control residuals.
        """
        self.set_context("control", {"residuals": residuals})
