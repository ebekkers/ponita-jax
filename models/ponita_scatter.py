import math
from typing import Any, Callable, Sequence, Optional, Tuple
import jax
import jax.numpy as jnp
from jax import random, jit, vmap, grad
from flax import linen as nn
from flax.core import frozen_dict
from flax.linen.initializers import lecun_normal, ones, zeros


def scatter_add(index, src, num_indices):
    return jax.ops.segment_sum(src, index, num_segments=num_indices)

def scatter_softmax(logits, index, num_indices):
    max_logits = jax.ops.segment_max(logits, index, num_segments=num_indices)
    logits -= jax.ops.segment_sum(max_logits, index, num_segments=num_indices)
    exp_logits = jnp.exp(logits)
    exp_sum = jax.ops.segment_sum(exp_logits, index, num_segments=num_indices)
    return exp_logits / exp_sum[index]


class GridGenerator(nn.Module):
    dim: int
    n: int
    steps: int = 200
    step_size: float = 0.01

    @nn.compact
    def __call__(self):
        if self.dim == 2:
            return self.generate_s1()
        elif self.dim == 3:
            return self.generate_s2()
        else:
            raise ValueError("Only S1 and S2 are supported.")
    
    def generate_s1(self):
        angles = jnp.linspace(0, 2 * math.pi - (2 * math.pi / self.n), self.n)
        x = jnp.cos(angles)
        y = jnp.sin(angles)
        return jnp.stack((x, y), axis=1)
       
    def generate_s2(self):
        return self.fibonacci_lattice(self.n)

    def fibonacci_lattice(self, n, offset=0.5):
        i = jnp.arange(n)
        theta = (math.pi * i * (1 + math.sqrt(5))) % (2 * math.pi)
        phi = jnp.arccos(1 - 2 * (i + offset) / (n - 1 + 2 * offset))
        x = jnp.sin(phi) * jnp.cos(theta)
        y = jnp.sin(phi) * jnp.sin(theta)
        z = jnp.cos(phi)
        return jnp.stack((x, y, z), axis=-1)




class SeparableFiberBundleConv(nn.Module):
    in_channels: int
    out_channels: int
    kernel_dim: int
    bias: bool = True
    groups: int = 1

    def setup(self):
        # Parameter and sub-module initialization
        if self.groups == 1:
            self.depthwise = False
        elif self.groups == self.in_channels and self.groups == self.out_channels:
            self.depthwise = True
        else:
            raise ValueError("Invalid option for groups, should be groups=1 or groups=in_channels=out_channels (depth-wise separable)")

        self.kernel = nn.Dense(features=self.in_channels, use_bias=False, kernel_init=lecun_normal())
        self.fiber_kernel = nn.Dense(features=int(self.in_channels * self.out_channels / self.groups), use_bias=False, kernel_init=lecun_normal())

        if self.bias:
            self.bias_param = self.param('bias', zeros, (self.out_channels,))

    def __call__(self, x, kernel_basis, fiber_kernel_basis, edge_index):
        message = x[edge_index[0]] * self.kernel(kernel_basis)  # [num_edges, num_ori, in_channels]
        x_1 = scatter_add(edge_index[1], message, x.shape[0])

        fiber_kernel = self.fiber_kernel(fiber_kernel_basis)
        if self.depthwise:
            x_2 = jnp.einsum("boc,poc->bpc", x_1, fiber_kernel) / fiber_kernel.shape[-2]
        else:
            x_2 = jnp.einsum("boc,podc->bpd", x_1, fiber_kernel.reshape(-1, self.out_channels, self.in_channels)) / fiber_kernel.shape[-2]

        if self.bias:
            x_2 += self.bias_param
        return x_2

class SeparableFiberBundleConvNext(nn.Module):
    channels: int
    kernel_dim: int
    widening_factor: int = 4
    act_fn: Callable = nn.gelu

    def setup(self):
        # Using previously defined custom class, assuming it's already converted
        self.conv = SeparableFiberBundleConv(in_channels=self.channels,
                                             out_channels=self.channels,
                                             kernel_dim=self.kernel_dim,
                                             groups=self.channels)

        self.linear_1 = nn.Dense(features=self.widening_factor * self.channels)
        self.linear_2 = nn.Dense(features=self.channels)
        self.norm = nn.LayerNorm(epsilon=1e-6)

    def __call__(self, x, kernel_basis, fiber_kernel_basis, edge_index):
        # Keep a reference to the original input for the residual connection
        input = x

        # Perform the convolution operation
        x = self.conv(x, kernel_basis, fiber_kernel_basis, edge_index)

        # Apply normalization and feedforward network
        x = self.norm(x)
        x = self.linear_1(x)
        x = self.act_fn(x)
        x = self.linear_2(x)
        x = x + input

        return x



class PolynomialFeatures(nn.Module):
    degree: int

    def __call__(self, x):
        # Initialize the list of polynomial features with the original input
        polynomial_list = [x]

        # Generate polynomial features up to the specified degree
        for it in range(1, self.degree + 1):
            # Compute the outer product of the last polynomial features with the original x
            # and flatten the last two dimensions
            new_features = jnp.einsum('...i,...j->...ij', polynomial_list[-1], x).reshape(x.shape[:-1] + (-1,))
            polynomial_list.append(new_features)

        # Concatenate all polynomial feature tensors along the last dimension
        return jnp.concatenate(polynomial_list, axis=-1)




class Ponita(nn.Module):
    input_dim: int
    hidden_dim: int
    output_dim: int
    batch_size: int
    num_layers: int
    output_dim_vec: int = 0
    dim: int = 3
    num_ori: int = 20
    basis_dim: int = None
    degree: int = 2
    widening_factor: int = 4
    global_pool: bool = True
    multiple_readouts: bool = True
    last_feature_conditioning: bool = False

    def setup(self):
        self.grid_generator = GridGenerator(dim=self.dim, n=self.num_ori, steps=1000)
        self.ori_grid = self.grid_generator()

        self.basis_fn = nn.Sequential([
            PolynomialFeatures(self.degree),
            nn.Dense(self.hidden_dim),
            nn.gelu,
            nn.Dense(self.basis_dim),
            nn.gelu
        ])

        self.fiber_basis_fn = nn.Sequential([
            PolynomialFeatures(self.degree),
            nn.Dense(self.hidden_dim),
            nn.gelu,
            nn.Dense(self.basis_dim),
            nn.gelu
        ])

        self.x_embedder = nn.Dense(self.hidden_dim, use_bias=False)
        
        self.interaction_layers = [SeparableFiberBundleConvNext(self.hidden_dim, self.basis_dim, self.widening_factor, nn.gelu) for _ in range(self.num_layers)]
        
        self.read_out_layers = [nn.Dense(self.output_dim + self.output_dim_vec) if self.multiple_readouts or i == (self.num_layers - 1) else None for i in range(self.num_layers)]

    def compute_invariants(self, ori_grid, pos, edge_index) -> Tuple[jnp.ndarray, jnp.ndarray]:
        pos_send, pos_receive = pos[edge_index[0]], pos[edge_index[1]]
        rel_pos = pos_send - pos_receive
        rel_pos = rel_pos[:, None, :]

        ori_grid_a = ori_grid[None, :, :]
        ori_grid_b = ori_grid[:, None, :]

        invariant1 = (rel_pos * ori_grid_a).sum(axis=-1, keepdims=True)
        
        if self.dim == 2:
            invariant2 = (rel_pos - invariant1 * ori_grid_a).sum(axis=-1, keepdims=True)
        elif self.dim == 3:
            invariant2 = jnp.linalg.norm(rel_pos - invariant1 * ori_grid_a, axis=-1, keepdims=True)
        
        invariant3 = (ori_grid_a * ori_grid_b).sum(axis=-1, keepdims=True)

        spatial_invariants = jnp.concatenate([invariant1, invariant2], axis=-1)
        orientation_invariants = invariant3

        return spatial_invariants, orientation_invariants

    def __call__(self, pos, x, edge_index, batch=None):
        ori_grid = self.ori_grid.astype(pos.dtype)
        spatial_invariants, orientation_invariants = self.compute_invariants(ori_grid, pos, edge_index)

        if self.last_feature_conditioning:
            cond = x[edge_index[0], None, -1:].repeat(1, ori_grid.shape[-2], 1)
            spatial_invariants = jnp.concatenate([spatial_invariants, cond], axis=-1)

        kernel_basis = self.basis_fn(spatial_invariants)
        fiber_kernel_basis = self.fiber_basis_fn(orientation_invariants)

        x = self.x_embedder(x)
        x = x[:, None, :].repeat(self.ori_grid.shape[-2], axis=1)

        readouts = []
        for layer, readout_layer in zip(self.interaction_layers, self.read_out_layers):
            x = layer(x, kernel_basis, fiber_kernel_basis, edge_index)
            if readout_layer is not None:
                readouts.append(readout_layer(x))

        readout = sum(readouts) / len(readouts)
        readout_scalar, readout_vec = jnp.split(readout, [self.output_dim], axis=-1)

        output_scalar = readout_scalar.mean(axis=-2)
        output_vector = (jnp.einsum("boc,od->bcd", readout_vec, ori_grid) / ori_grid.shape[-2])

        if self.global_pool:
            output_scalar = scatter_add(batch, output_scalar, self.batch_size)
            output_vector = scatter_add(batch, output_vector, self.batch_size)

        return output_scalar, output_vector
