"""
Implementation of the CrossAttentionBlock2d class for AnyDoor models.
AnyDoor name is SpatialTransformer in the original code.
The implementation is exactly the same as [CrossAttentionBlock2d from refiners.fluxion.layers](https://github.com/finegrain-ai/refiners/blob/d90bb25151697531b50cb9783d1022b208a13d70/src/refiners/foundationals/latent_diffusion/cross_attention.py)
The only difference is the Lambda functions that ensure contiguous tensors as the original code does.
"""

from torch import Size, Tensor, device as Device, dtype as DType
import torch
from jaxtyping import Float
from refiners.fluxion.context import Contexts
from refiners.fluxion.layers import (
    GLU,
    Attention,
    Chain,
    Conv2d,
    Flatten,
    GeLU,
    GroupNorm,
    Identity,
    LayerNorm,
    Linear,
    Parallel,
    Residual,
    SelfAttention,
    ScaledDotProductAttention,
    SetContext,
    Transpose,
    Unflatten,
    UseContext,
    Lambda,
    ContextModule,
    Distribute
)

try:
    import xformers
    import xformers.ops

    XFORMERS_IS_AVAILBLE = True
except:  # noqa: E722
    XFORMERS_IS_AVAILBLE = False

@torch.no_grad()
def attn_mask_resize(m,h,w):
    """
    m : [BS x 1 x mask_h x mask_w] => downsample, reshape and bool, [BS x h x w]
    """  
    m = torch.nn.functional.interpolate(m, (h, w)).squeeze(1).contiguous()
    m = torch.where(m>=0.5, True, False)
    return m

def get_tvloss(coords, mask, ch, cw):
    b, n, _ = coords.shape
    coords = coords.reshape(b,ch,cw,2)
    mask = mask.unsqueeze(-1)
    y_mask = mask[:,1:] * mask[:,:-1]
    x_mask = mask[:,:,1:] * mask[:,:,:-1]
    
    y_tvloss = torch.abs(coords[:,1:] - coords[:,:-1]) * y_mask
    x_tvloss = torch.abs(coords[:,:,1:] - coords[:,:,:-1]) * x_mask
    tv_loss = y_tvloss.sum() / y_mask.sum() + x_tvloss.sum() / x_mask.sum()
    return tv_loss

class XformersScaledDotProductAttention(ScaledDotProductAttention, ContextModule):


    def __init__(self, head_dimension: int, num_heads: int, use_attention_tv_loss:bool=False) -> None:
        self.head_dimension = head_dimension
        self.use_attention_tv_loss = use_attention_tv_loss
        super().__init__(num_heads=num_heads)


    # Override the base class method to use the xformers implementation
    def forward(
        self,
        query: Float[Tensor, "batch num_queries embedding_dim"],  # noqa: F722
        key: Float[Tensor, "batch num_keys embedding_dim"],  # noqa: F722
        value: Float[Tensor, "batch num_values embedding_dim"],  # noqa: F722
        tv_loss_mask: Float[Tensor, "batch num_values embedding_dim"] | None = None  # noqa: F722
    ) -> Float[Tensor, "batch num_queries embedding_dim"]:  # noqa: F722
        """Compute the scaled dot product attention.

        Split the input tensors (query, key, value) into multiple heads along the embedding dimension,
        then compute the scaled dot product attention for each head, and finally merge the heads back.
        """
        batch_size = query.shape[0]
        q_heads, k_heads, v_heads = map(
            lambda t: t.unsqueeze(3)
            .reshape(batch_size, t.shape[1], self.num_heads, self.head_dimension)
            .permute(0, 2, 1, 3)
            .reshape(batch_size * self.num_heads, t.shape[1], self.head_dimension)
            .contiguous(),
            (query, key, value),
        )

        # Apply memory-efficient attention
        attention_output = xformers.ops.memory_efficient_attention(
            q_heads,
            k_heads,
            v_heads,
            attn_bias=None,
            op=None,
        )
        attention_output = (
            attention_output.unsqueeze(0)
            .reshape(
                batch_size,
                self.num_heads,
                attention_output.shape[1],
                self.head_dimension,
            )
            .permute(0, 2, 1, 3)
            .reshape(
                batch_size,
                attention_output.shape[1],
                self.head_dimension * self.num_heads,
            )
        )

        if self.use_attention_tv_loss  and tv_loss_mask is not None:
            # print("Using attention tv loss")
            sim = torch.einsum('b i d, b j d -> b i j', q_heads, k_heads) * (self.head_dimension ** -0.5)
            sim = sim.softmax(dim=-1)
            # print(torch.norm(sim))
            h = self.num_heads
            _, HW, hw = sim.shape
            S = int(HW ** 0.5)
            s = int(hw ** 0.5)
            
            tv_loss_mask = attn_mask_resize(tv_loss_mask, S, S)  # [BS x H x W]
            # print("tv_loss_mask ",torch.sum(tv_loss_mask))
            reshaped_sim = sim.reshape(-1, h, S*S, s, s).mean(dim=1) 
            # print("reshaped_sim ",torch.norm(reshaped_sim))
            linspace = torch.linspace(0,s-1,s, device=sim.device)
            grid_h, grid_w = torch.meshgrid(linspace, linspace)
            grid_hw = torch.stack([grid_h, grid_w])
            
            weighted_grid_hw = reshaped_sim.unsqueeze(2) * grid_hw.unsqueeze(0).unsqueeze(0)  # [b HW 2 h w]
            weighted_centered_grid_hw = weighted_grid_hw.sum((-2,-1))  # [b HW 2]

            # print("weighted_centered_grid_hw ",torch.norm(weighted_centered_grid_hw))
            tv_loss = get_tvloss(weighted_centered_grid_hw, ~tv_loss_mask, ch=s, cw=s)
            attn_loss = tv_loss * 0.001
            context = self.use_context("atv_loss")
            context.update({"value":attn_loss + context["value"]})

        return attention_output


class CrossAttentionBlock(Chain):
    def __init__(
        self,
        embedding_dim: int,
        context_embedding_dim: int,
        context_key: str,
        num_heads: int = 1,
        use_bias: bool = True,
        device: Device | str | None = None,
        dtype: DType | None = None,
    ) -> None:
        self.embedding_dim = embedding_dim
        self.context_embedding_dim = context_embedding_dim
        self.context = "cross_attention_block"
        self.context_key = context_key
        self.num_heads = num_heads
        self.use_bias = use_bias

        super().__init__(
            Residual(
                LayerNorm(normalized_shape=embedding_dim, device=device, dtype=dtype),
                SelfAttention(
                    embedding_dim=embedding_dim,
                    num_heads=num_heads,
                    use_bias=use_bias,
                    is_optimized=False,
                    device=device,
                    dtype=dtype,
                ),
            ),
            Residual(
                LayerNorm(normalized_shape=embedding_dim, device=device, dtype=dtype),
                Parallel(
                    Identity(),
                    UseContext(context=self.context, key=context_key),
                    UseContext(context=self.context, key=context_key),
                ),
                Attention(
                    embedding_dim=embedding_dim,
                    num_heads=num_heads,
                    key_embedding_dim=context_embedding_dim,
                    value_embedding_dim=context_embedding_dim,
                    use_bias=use_bias,
                    is_optimized=False,
                    device=device,
                    dtype=dtype,
                ),
            ),
            Residual(
                LayerNorm(normalized_shape=embedding_dim, device=device, dtype=dtype),
                Linear(
                    in_features=embedding_dim,
                    out_features=2 * 4 * embedding_dim,
                    device=device,
                    dtype=dtype,
                ),
                GLU(GeLU()),
                Linear(
                    in_features=4 * embedding_dim,
                    out_features=embedding_dim,
                    device=device,
                    dtype=dtype,
                ),
            ),
        )


class StatefulFlatten(Chain):
    def __init__(
        self, context: str, key: str, start_dim: int = 0, end_dim: int = -1
    ) -> None:
        self.start_dim = start_dim
        self.end_dim = end_dim

        super().__init__(
            SetContext(context=context, key=key, callback=self.push),
            Flatten(start_dim=start_dim, end_dim=end_dim),
        )

    def push(self, sizes: list[Size], x: Tensor) -> None:
        sizes.append(
            x.shape[
                slice(
                    self.start_dim,
                    (
                        self.end_dim + 1
                        if self.end_dim >= 0
                        else x.ndim + self.end_dim + 1
                    ),
                )
            ]
        )


class CrossAttentionBlock2d(Residual):
    def __init__(
        self,
        channels: int,
        context_embedding_dim: int,
        context_key: str,
        mask_key: str = "tv_loss_mask",
        use_attention_tv_loss: bool = False,
        num_attention_heads: int = 1,
        num_attention_layers: int = 1,
        num_groups: int = 32,
        use_bias: bool = True,
        use_linear_projection: bool = False,
        device: Device | str | None = None,
        dtype: DType | None = None,
    ) -> None:
        assert (
            channels % num_attention_heads == 0
        ), "in_channels must be divisible by num_attention_heads"
        self.channels = channels
        self.in_channels = channels
        self.out_channels = channels
        self.context_embedding_dim = context_embedding_dim
        self.num_attention_heads = num_attention_heads
        self.num_attention_layers = num_attention_layers
        self.num_groups = num_groups
        self.use_bias = use_bias
        self.context_key = context_key
        self.mask_key = mask_key
        self.use_attention_tv_loss = use_attention_tv_loss
        self.use_linear_projection = use_linear_projection
        self.projection_type = "Linear" if use_linear_projection else "Conv2d"

        in_block = (
            Chain(
                GroupNorm(
                    channels=channels,
                    num_groups=num_groups,
                    eps=1e-6,
                    device=device,
                    dtype=dtype,
                ),
                StatefulFlatten(context="flatten", key="sizes", start_dim=2),
                Transpose(1, 2),
                Lambda(lambda x: x.contiguous()),
                Linear(
                    in_features=channels,
                    out_features=channels,
                    device=device,
                    dtype=dtype,
                ),
            )
            if use_linear_projection
            else Chain(
                GroupNorm(
                    channels=channels,
                    num_groups=num_groups,
                    eps=1e-6,
                    device=device,
                    dtype=dtype,
                ),
                Conv2d(
                    in_channels=channels,
                    out_channels=channels,
                    kernel_size=1,
                    device=device,
                    dtype=dtype,
                ),
                StatefulFlatten(context="flatten", key="sizes", start_dim=2),
                Transpose(1, 2),
            )
        )

        out_block = (
            Chain(
                Linear(
                    in_features=channels,
                    out_features=channels,
                    device=device,
                    dtype=dtype,
                ),
                Lambda(lambda x: x.contiguous()),
                Transpose(1, 2),
                Parallel(
                    Identity(),
                    UseContext(context="flatten", key="sizes").compose(
                        lambda x: x.pop()
                    ),
                ),
                Unflatten(dim=2),
            )
            if use_linear_projection
            else Chain(
                Transpose(1, 2),
                Parallel(
                    Identity(),
                    UseContext(context="flatten", key="sizes").compose(
                        lambda x: x.pop()
                    ),
                ),
                Unflatten(dim=2),
                Conv2d(
                    in_channels=channels,
                    out_channels=channels,
                    kernel_size=1,
                    device=device,
                    dtype=dtype,
                ),
            )
        )

        super().__init__(
            in_block,
            Chain(
                CrossAttentionBlock(
                    embedding_dim=channels,
                    context_embedding_dim=context_embedding_dim,
                    context_key=context_key,
                    num_heads=num_attention_heads,
                    use_bias=use_bias,
                    device=device,
                    dtype=dtype,
                )
                for _ in range(num_attention_layers)
            ),
            out_block,
        )
        if XFORMERS_IS_AVAILBLE:
            for attention in self.layers(Attention):
                ## TODO : if cross attention, use ltv loss
                head_dimension = attention.embedding_dim // attention.num_heads
                attention_product = attention.layer(
                    "ScaledDotProductAttention", ScaledDotProductAttention
                )
                num_heads = attention_product.num_heads
                is_cross_attention = attention._get_name() == "Attention"
                attention.remove(attention_product)
                if is_cross_attention:
                    attention.insert(
                        -2, XformersScaledDotProductAttention(head_dimension=head_dimension,num_heads=num_heads)
                    )
                else:
                    attention.layer(0, Parallel).insert(-1,UseContext(context="masks", key=self.mask_key))
                    attention.layer(1,Distribute).insert(-1, Identity())
                    attention.insert(
                        -2, XformersScaledDotProductAttention(head_dimension=head_dimension,num_heads=num_heads, use_attention_tv_loss=self.use_attention_tv_loss)
                    )

    def init_context(self) -> Contexts:
        return {"flatten": {"sizes": []}}
