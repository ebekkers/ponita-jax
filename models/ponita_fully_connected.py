import math
from typing import Any, Callable, Sequence, Optional, Tuple
import jax
import jax.numpy as jnp
from flax import linen as nn



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

class FullyConnectedSeparableFiberBundleConv(nn.Module):
    num_hidden: int
    basis_dim: int
    apply_bias: bool = True

    def setup(self):
        # Set up kernel coefficient layers, one for the spatial kernel and one for the group kernel.
        # This maps from the invariants to a basis for the kernel 2/3->basis_dim.
        self.spatial_kernel = nn.Dense(self.num_hidden, use_bias=False)
        self.rotation_kernel = nn.Dense(self.num_hidden, use_bias=False)

        # Construct bias
        if self.apply_bias:
            self.bias = self.param('bias', nn.initializers.zeros, (self.num_hidden,))

    def __call__(self, x, kernel_basis, fiber_kernel_basis, mask: Optional[jnp.ndarray] = None):
        """ Perform separable convolution on fully connected pointcloud.

        Args:
            x: Array of shape (batch, num_points, num_ori, num_features)
            kernel_basis: Array of shape (batch, num_points, num_points, num_ori, basis_dim)
            fiber_kernel_basis: Array of shape (batch, num_points, num_points, basis_dim)
        """
        # Create mask of ones if not provided
        if mask is None:
            mask = jnp.ones((x.shape[0], x.shape[1]))

        # Compute the spatial kernel
        spatial_kernel = self.spatial_kernel(kernel_basis)

        # Compute the group kernel (Fiber kernel)
        rot_kernel = self.rotation_kernel(fiber_kernel_basis)

        # Perform the convolution
        x = jnp.einsum('bnoc,bmnoc,bn->bmoc', x, spatial_kernel, mask)
        x = jnp.einsum('bmoc,poc->bmpc', x, rot_kernel) / rot_kernel.shape[-2]

        # TODO: Add calibration of the kernels.

        # Add bias
        if self.apply_bias:
            x = x + self.bias
        return x


class SeparableFiberBundleConvNext(nn.Module):
    num_hidden: int
    kernel_dim: int
    widening_factor: int = 4

    def setup(self):
        self.conv = FullyConnectedSeparableFiberBundleConv(self.num_hidden, self.kernel_dim)
        self.act_fn = nn.gelu
        self.linear_1 = nn.Dense(self.widening_factor * self.num_hidden)
        self.linear_2 = nn.Dense(self.num_hidden)
        self.norm = nn.LayerNorm()

    def __call__(self, x, kernel_basis, fiber_kernel_basis, mask: Optional[jnp.ndarray] = None):
        input = x
        x = self.conv(x, kernel_basis, fiber_kernel_basis, mask)
        x = self.norm(x)
        x = self.linear_1(x)
        x = self.act_fn(x)
        x = self.linear_2(x)
        x = x + input
        return x


class Ponita(nn.Module):
    num_in: int
    num_hidden: int
    num_layers: int
    scalar_num_out: int
    vec_num_out: int

    spatial_dim: int
    num_ori: int
    basis_dim: int
    degree: int

    widening_factor: int
    global_pool: bool

    def setup(self):

        # Check input arguments
        assert self.spatial_dim in [2, 3], "spatial_dim must be 2 or 3."
        rot_group_dim = 1 if self.spatial_dim == 2 else 2

        # Create a grid generator for the dimensionality of the orientation
        self.grid_generator = GridGenerator(dim=self.spatial_dim, n=self.num_ori, steps=1000)
        self.ori_grid = self.grid_generator()

        # Set up kernel basis functions, one for the spatial kernel and one for the group kernel.
        # This maps from the invariants to a basis for the kernel 2/3->basis_dim.
        self.spatial_kernel_basis = nn.Sequential([
            PolynomialFeatures(degree=self.degree), nn.Dense(self.num_hidden), nn.gelu, nn.Dense(self.basis_dim), nn.gelu])
        self.rotation_kernel_basis = nn.Sequential([
            PolynomialFeatures(degree=self.degree), nn.Dense(self.num_hidden), nn.gelu, nn.Dense(self.basis_dim), nn.gelu])

        # Initial node embedding
        self.x_embedder = nn.Dense(self.num_hidden, use_bias=False)

        # Make feedforward network
        interaction_layers = []
        for i in range(self.num_layers):
            interaction_layers.append(SeparableFiberBundleConvNext(self.num_hidden, self.basis_dim))
        self.interaction_layers = interaction_layers

        # Readout layers
        self.readout = nn.Dense(self.scalar_num_out + self.vec_num_out)

    def __call__(self, pos, x, mask: Optional[jnp.ndarray] = None):
        """ Forward pass through the network.

        Args:
            pos: Array of shape (batch, num_points, spatial_dim)
            x: Array of shape (batch, num_points, num_in)
            mask: Array of shape (batch, num_points)
        """
        if mask is None:
            mask = jnp.ones((x.shape[0], x.shape[1]))

        # Calculate invariants
        rel_pos = pos[:, None, :, None, :] - pos[:, :, None, None, :]  # (batch, num_points, num_points, 1, 3)
        invariant1 = (rel_pos * self.ori_grid[None, None, None, :, :]).sum(axis=-1, keepdims=True)  # (batch, num_points, num_points, num_ori, 1)
        invariant2 = jnp.linalg.norm(rel_pos - rel_pos * invariant1, axis=-1, keepdims=True)  # (batch, num_points, num_points, 1, 3)
        spatial_invariants = jnp.concatenate([invariant1, invariant2], axis=-1)
        orientation_invariants = (self.ori_grid[:, None, :] * self.ori_grid[None, :, :]).sum(axis=-1, keepdims=True)

        # Sample the kernel basis
        kernel_basis = self.spatial_kernel_basis(spatial_invariants)
        fiber_kernel_basis = self.rotation_kernel_basis(orientation_invariants)

        # Initial feature embedding
        x = self.x_embedder(x)

        # Repeat features over the orientation dimension
        x = x[:, :, None, :].repeat(self.ori_grid.shape[-2], axis=-2)

        # Apply interaction layers
        for layer in self.interaction_layers:
            x = layer(x, kernel_basis, fiber_kernel_basis, mask)

        # Readout layer
        readout = self.readout(x)

        # Split scalar and vector parts
        # readout_scalar, readout_vec = jnp.split(readout, [self.scalar_num_out, self.vec_num_out], axis=-1)
        readout_scalar, readout_vec = readout, None

        # Average over the orientation dimension
        output_scalar = readout_scalar.mean(axis=-2)
        if self.vec_num_out > 0:
            output_vector = jnp.einsum('bnoc,od->bncd', readout_vec, self.ori_grid) / self.ori_grid.shape[-2]
        else:
            output_vector = None

        # Global pooling
        if self.global_pool:
            output_scalar = jnp.sum(output_scalar * mask[:,:,None], axis=1) / jnp.sum(mask[:,:,None], axis=1) #output_scalar.shape[1]
            if self.vec_num_out > 0:
                output_vector = jnp.sum(output_vector * mask[:,:,None,None], axis=1) / jnp.sum(mask[:,:,None,None], axis=1) #output_vector.shape[1]

        return output_scalar, output_vector


if __name__ == "__main__":
    model = FullyConnectedPonita(
        num_in=3,
        num_hidden=64,
        num_layers=3,
        scalar_num_out=1,
        vec_num_out=1,
        spatial_dim=2,
        num_ori=4,
        basis_dim=64,
        degree=3,
        widening_factor=4,
        global_pool=True
    )

    # Create meshgrid
    pos = jnp.stack(jnp.meshgrid(jnp.linspace(-1, 1, 28), jnp.linspace(-1, 1, 28)), axis=-1).reshape(-1, 2)
    # Repeat over batch dim
    pos = jnp.repeat(pos[None, ...], 4, axis=0)
    x = jnp.ones((4, 28*28, 3))

    # Initialize and apply model
    params = model.init(jax.random.PRNGKey(0), pos, x)
    model.apply(params, pos, x)
    print("Success!")
