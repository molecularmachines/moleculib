from typing import List, Tuple, Union

import chex
import e3nn_jax as e3nn
import jax.numpy as jnp

from moleculib.nucleic.datum import NucleicDatum

import jax
from jax.tree_util import tree_map, tree_flatten, tree_flatten_with_path

#skipped inner stack and inner split

@chex.dataclass
class InternalState:
    irreps_array: e3nn.IrrepsArray  # (seq_len, irreps) - irreps array
    mask_irreps_array: jnp.ndarray  # (seq_len,) - mask for irreps array
    coord: jnp.ndarray  # (seq_len, 3) - coordinates of C_alpha atoms
    mask_coord: jnp.ndarray  # (seq_len,) - mask for coordinates

    @property
    def irreps(self) -> e3nn.Irreps:
        return self.irreps_array.irreps

    @property
    def seq_len(self) -> int:
        return self.irreps_array.shape[0]

    @property
    def mask(self) -> jnp.ndarray:
        return self.mask_irreps_array & self.mask_coord

#data holders/ organizers:
@chex.dataclass
class ProbParams:
    state: InternalState
    mu: jnp.ndarray
    sigma: jnp.ndarray
    sigma_basis: e3nn.Irreps
    sigma_flat: jnp.ndarray


@chex.dataclass
class ProbPair:
    prior: ProbParams
    posterior: ProbParams
    mask: jnp.ndarray


@chex.dataclass
class ModelOutput:
    logits: Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]
    datum: NucleicDatum
    encoder_internals: List[InternalState]
    decoder_internals: List[InternalState]
    probs: Union[List[ProbPair], None]
    atom_perm_loss: jnp.ndarray


#NOTE: go over these functions to understand exactly what they do:

# def rescale_irreps(irreps: e3nn.Irreps, rescale: float, chunk_factor: int = 0):
#     irreps = e3nn.Irreps([(int(mul * rescale), ir) for mul, ir in irreps])
#     if chunk_factor != 0:
#         irreps = e3nn.Irreps(
#             [(next_multiple(mul, chunk_factor), ir) for mul, ir in irreps]
#         )
#     return irreps

# def multiscale_irreps(
#     irreps: e3nn.Irreps, depth: int, rescale: float, chunk_factor: int = 0
# ) -> List[e3nn.Irreps]:
#     list_irreps = [irreps]
#     for _ in range(depth):
#         list_irreps.append(rescale_irreps(list_irreps[-1], rescale, chunk_factor))
#     return list_irreps

# def next_multiple(x: int, factor: int) -> int:
#     """next multiple of factor"""
#     if x % factor == 0:
#         return x
#     return x + factor - (x % factor)


####
# the following functions are useful in deep learning frameworks and when working with CNN to 
# manage the spatial dimensions of the data at different layers of the network.

def up_conv_seq_len(size: int, kernel: int, stride: int, mode: str) -> int:
    """output size of a convolutional layer
    size: Integer representing the input size of the sequence 
    kernel: Integer representing the kernel size of the convolution operation
    stride: Integer representing the stride of the convolution operation
    mode: String representing the mode of the convolution. either "same" or "valid".
    Returns: Integer representing the output size of the convolutional layer.

    calculates the output size of the convolutional layer
    If mode is "same", the function returns stride * (size - 1) + 1. the output size in "same" mode where the input and output have the same size.
    If mode is "valid", the function returns stride * (size - 1) + kernel. "valid" mode where only complete convolutions are allowed(no padding).
    """
    if mode.lower() == "same":
        return stride * (size - 1) + 1
    if mode.lower() == "valid":
        return stride * (size - 1) + kernel

    raise ValueError(f"Unknown mode: {mode}")


def down_conv_seq_len(size: int, kernel: int, stride: int, mode: str) -> int:
    """output size of a convolutional layer"""
    if mode.lower() == "same":
        assert kernel % 2 == 1
        if (size - 1) % stride != 0:
            raise ValueError(
                (
                    f"Not a perfect convolution: "
                    f"size={size}, kernel={kernel}, stride={stride} mode={mode}."
                    f"({size} - 1) % {stride} == {(size - 1) % stride} != 0"
                )
            )
        return (size - 1) // stride + 1
    if mode.lower() == "valid":
        if (size - kernel) % stride != 0:
            raise ValueError(
                (
                    f"Not a perfect convolution: "
                    f"size={size}, kernel={kernel}, stride={stride} mode={mode}."
                    f"({size} - {kernel}) % {stride} == {(size - kernel) % stride} != 0"
                )
            )
        return (size - kernel) // stride + 1

    raise ValueError(f"Unknown mode: {mode}")


def safe_norm(vector: jnp.ndarray, axis: int = -1) -> jnp.ndarray:
    """safe_norm(x) = norm(x) if norm(x) != 0 else 1.0"""
    norms_sqr = jnp.sum(vector**2, axis=axis)
    norms = jnp.where(norms_sqr == 0.0, 1.0, norms_sqr) ** 0.5
    return norms


def safe_normalize(vector: jnp.ndarray) -> jnp.ndarray:
    return vector / safe_norm(vector)[..., None]