from refiners.foundationals.latent_diffusion.auto_encoder import (
    LatentDiffusionAutoencoder,
)
from torch import  device as Device, dtype as DType
import refiners.fluxion.layers as fl
from anydoor_refiners.attention import XformersScaledDotProductAttention

try:
    import xformers
    import xformers.ops

    XFORMERS_IS_AVAILBLE = True
except:  # noqa: E722
    XFORMERS_IS_AVAILBLE = False


class AnydoorAutoencoder(LatentDiffusionAutoencoder):
    """Stable Diffusion 1.5 autoencoder model.

    Attributes:
        encoder_scale: The encoder scale to use.
    """

    encoder_scale: float = 0.18215

    def __init__(
        self,
        device: Device | str | None = None,
        dtype: DType | None = None,
    ) -> None:
        
        super().__init__(device=device,dtype=dtype)

        for self_attention_2d in self.layers(fl.SelfAttention2d):
            self_attention_2d.remove(self_attention_2d.layer(-1,fl.Lambda))
            self_attention_2d.remove(self_attention_2d.layer(0,fl.Lambda))

            distribute = self_attention_2d.layer(1, fl.Distribute)
            input_linear_layer = distribute.layer(0,fl.Linear)
            new_distribute = fl.Distribute(
                fl.Chain(
                    fl.Conv2d(in_channels=input_linear_layer.in_features,out_channels=input_linear_layer.out_features,kernel_size=1),
                    fl.Lambda(lambda x: self_attention_2d._tensor_2d_to_sequence(x)),
                ),
                fl.Chain(
                    fl.Conv2d(in_channels=input_linear_layer.in_features,out_channels=input_linear_layer.out_features,kernel_size=1),
                    fl.Lambda(lambda x: self_attention_2d._tensor_2d_to_sequence(x)),
                ),            
                fl.Chain(
                    fl.Conv2d(in_channels=input_linear_layer.in_features,out_channels=input_linear_layer.out_features,kernel_size=1),
                    fl.Lambda(lambda x: self_attention_2d._tensor_2d_to_sequence(x)),
                ),
            )
            self_attention_2d.remove(distribute)
            self_attention_2d.insert(1,new_distribute)

            output_linear_layer = self_attention_2d.layer(-1,fl.Linear)
            output_conv = fl.Conv2d(in_channels=input_linear_layer.in_features,out_channels=input_linear_layer.out_features,kernel_size=1)

            self_attention_2d.remove(output_linear_layer)
            self_attention_2d.insert(-1,fl.Lambda(lambda x: self_attention_2d._sequence_to_tensor_2d(x)))
            self_attention_2d.insert(-1,output_conv)


        if XFORMERS_IS_AVAILBLE:
            for attention in self.layers(fl.Attention):
                head_dimension = attention.embedding_dim // attention.num_heads
                attention_product = attention.layer(
                    "ScaledDotProductAttention", fl.ScaledDotProductAttention
                )
                num_heads = attention_product.num_heads
                attention.remove(attention_product)
                attention.insert(
                    -2, XformersScaledDotProductAttention(head_dimension=head_dimension,num_heads=num_heads)
                )


