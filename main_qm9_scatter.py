import os
from typing import Any, Dict
import jax
import jax.numpy as jnp
from jax import random, jit, vmap
import optax
import numpy as np
from flax import linen as nn
from flax.training import train_state
from flax.core.frozen_dict import freeze, unfreeze
import wandb
from tqdm import tqdm
from functools import partial

from datasets.qm9 import QM9Dataset, collate_fn, collate_fn_vmap_wrapper
from models.ponita_scatter import Ponita

from torch.utils.data import DataLoader

import hydra
import omegaconf

from flax import struct, core

from orbax import checkpoint

# jax.config.update("jax_disable_jit", True)



class RandomSOd:
    def __init__(self, d):
        """
        Initializes the RandomRotationGenerator.
        Args:
        - d (int): The dimension of the rotation matrices (2 or 3).
        """
        assert d in [2, 3], "d must be 2 or 3."
        self.d = d

    def __call__(self, n=None):
        """
        Generates random rotation matrices.
        Args:
        - n (int, optional): The number of rotation matrices to generate. If None, generates a single matrix.

        Returns:
        - Array: An array of shape [n, d, d] containing n rotation matrices, or [d, d] if n is None.
        """
        if self.d == 2:
            return self._generate_2d(n)
        else:
            return self._generate_3d(n)
    
    def _generate_2d(self, n):
        theta = jax.random.uniform(jax.random.PRNGKey(0), (n,) if n else (1,)) * 2 * jnp.pi
        cos_theta, sin_theta = jnp.cos(theta), jnp.sin(theta)
        rotation_matrix = jnp.stack([cos_theta, -sin_theta, sin_theta, cos_theta], axis=-1)
        if n:
            return rotation_matrix.reshape(n, 2, 2)
        return rotation_matrix.reshape(2, 2)

    def _generate_3d(self, n):
        q = jax.random.normal(jax.random.PRNGKey(0), (n, 4) if n else (4,))
        q = q / jnp.linalg.norm(q, axis=-1, keepdims=True)
        q0, q1, q2, q3 = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
        rotation_matrix = jnp.stack([
            1 - 2 * (q2**2 + q3**2), 2 * (q1*q2 - q0*q3), 2 * (q1*q3 + q0*q2),
            2 * (q1*q2 + q0*q3), 1 - 2 * (q1**2 + q3**2), 2 * (q2*q3 - q0*q1),
            2 * (q1*q3 - q0*q2), 2 * (q2*q3 + q0*q1), 1 - 2 * (q1**2 + q2**2)
        ], axis=-1)
        if n:
            return rotation_matrix.reshape(n, 3, 3)
        return rotation_matrix.reshape(3, 3)










class  BaseJaxTrainer:

    def __init__(
            self,
            config,
            train_loader,
            val_loader,
            seed,
    ):
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.seed = seed

        # Keep track of training state
        self.global_step = 0
        self.epoch = 0

        # Keep track of state of validation
        self.val_epoch = 0
        self.global_val_step = 0
        self.total_val_epochs = 0

        # Description strings for train and val progress bars
        self.train_mse_epoch, self.val_mse_epoch = np.inf, np.inf
        self.prog_bar_desc = """{state} :: epoch - {epoch}/{total_epochs} | step - {step}/{global_step} :: mse step {loss:.4f} -- train mse epoch {train_mse_epoch:.4f} -- val mse epoch {val_mse_epoch:.4f}"""
        self.prog_bar = tqdm(
            desc=self.prog_bar_desc.format(
                state='Training',
                epoch=self.epoch,
                total_epochs=self.config.training.num_epochs,
                step=0,
                global_step=len(self.train_loader),
                loss=jnp.inf,
                train_mse_epoch=self.train_mse_epoch,
                val_mse_epoch=self.val_mse_epoch
            ),
            total=len(self.train_loader)
        )
        
        # Set checkpoint options
        if self.config.logging.checkpoint:
            checkpoint_options = checkpoint.CheckpointManagerOptions(
                save_interval_steps=config.logging.checkpoint_every_n_epochs,
                max_to_keep=config.logging.keep_n_checkpoints,
            )
            orbax_checkpointer = checkpoint.PyTreeCheckpointer()
            self.checkpoint_manager = checkpoint.CheckpointManager(
                directory=os.path.abspath(config.logging.log_dir + '/checkpoints'),
                checkpointers=orbax_checkpointer,
                options=checkpoint_options,
            )

    def save_checkpoint(self, state):
        """ Save the current state to a checkpoint

        Args:
            state: The current training state.
        """
        if self.config.logging.checkpoint:
            self.checkpoint_manager.save(step=self.epoch, items={'state': state, 'cfg': self.config})

    def load_checkpoint(self):
        """ Load the latest checkpoint"""
        return self.checkpoint_manager.restore(self.checkpoint_manager.latest_step())

    def update_prog_bar(self, loss, step, train=True):
        """ Update the progress bar.

        Args:
            desc: The description string.
            loss: The current loss.
            epoch: The current epoch.
            step: The current step.
        """
        # If we are at the beginning of the epoch, reset the progress bar
        if step == 0:
            # Depending on whether we are training or validating, set the total number of steps
            if train:
                self.prog_bar.total = len(self.train_loader)
            else:
                self.prog_bar.total = len(self.val_loader)
            self.prog_bar.reset()
        else:
            self.prog_bar.update(self.config.logging.log_every_n_steps)

        if train:
            global_step = self.global_step
            epoch = self.epoch
            total_epochs = self.config.training.num_epochs
        else:
            global_step = self.global_val_step
            epoch = self.val_epoch
            total_epochs = self.total_val_epochs

        self.prog_bar.set_description_str(
            self.prog_bar_desc.format(
                state='Training' if train else 'Validation',
                epoch=epoch,
                total_epochs=total_epochs,
                step=step,
                global_step=len(self.train_loader),
                loss=loss,
                train_mse_epoch=self.train_mse_epoch,
                val_mse_epoch=self.val_mse_epoch
            ),
        )








class TrainState(struct.PyTreeNode):
    params: core.FrozenDict[str, Any] = struct.field(pytree_node=True)
    rng: jnp.ndarray = struct.field(pytree_node=True)
    opt_state: core.FrozenDict[str, Any] = struct.field(pytree_node=True)






class QM9Trainer(BaseJaxTrainer):

    def __init__(
            self,
            config,
            train_loader,
            val_loader,
            seed,
    ):
        super().__init__(config, train_loader, val_loader, seed)

        # Select the right target
        targets = ['mu', 'alpha', 'homo', 'lumo', 'gap', 'r2', 'zpve', 'U0', 'U', 'H', 'G', 'Cv', 'U0_atom', 'U_atom', 'H_atom', 'G_atom', 'A', 'B', 'C']
        self.target_idx = targets.index(config.training.target)

        # set ponita model vars
        self.in_channels_scalar = 5     # One-hot encoding molecules
        in_channels_vec = 0  
        out_channels_scalar = 1         # The target
        out_channels_vec = 0   

        # Transform
        self.train_aug = config.training.train_augmentation
        self.rotation_generator = RandomSOd(3)

        # Model
        self.model = Ponita(
                input_dim = self.in_channels_scalar + in_channels_vec,
                hidden_dim = config.ponita.hidden_dim,
                output_dim = out_channels_scalar,
                batch_size = 1,
                num_layers = config.ponita.num_layers,
                output_dim_vec = out_channels_vec,
                num_ori = config.ponita.num_ori,
                basis_dim = config.ponita.basis_dim,
                degree = config.ponita.degree,
                widening_factor = config.ponita.widening_factor,
                global_pool = True,
                multiple_readouts = False,
        )

        self.shift = 0
        self.scale = 1

        # Set dataset statistics
        self.set_dataset_statistics(train_loader)

    def set_dataset_statistics(self, dataloader):
        print('Computing dataset statistics...')
        ys = []
        for data in tqdm(dataloader):
            ys.append(data['y'])
        ys = jnp.concatenate(ys)
        self.shift = jnp.mean(ys)
        self.scale = jnp.std(ys)
        print('Mean and std of target are:', self.shift, '-', self.scale)

    def init_train_state(self):
        """Initializes the training state.

        Returns:
            TrainState: The training state.
        """
        # Initialize optimizer and scheduler
        self.optimizer = optax.adam(self.config.optimizer.learning_rate)

        # Random key
        key = jax.random.PRNGKey(self.config.optimizer.seed)

        # Split key
        key, model_key = jax.random.split(key)

        # Initialize model
        samples_batch = next(iter(self.train_loader))
        sample = jax.tree.map(lambda x: jnp.asarray(x[0]), samples_batch) # get a sample
        pos = sample["pos"]
        x = sample["x"]
        edge_index = sample["edge_index"]
        batch = sample["batch"]
        model_params = self.model.init(model_key, pos, x, edge_index, batch)

        # Create train state
        train_state = TrainState(
            params=model_params,
            opt_state=self.optimizer.init(model_params),
            rng=key
        )
        return train_state

    def create_functions(self):

        def step(state, batch, train=True):
            """Performs a single training step.

            Args:
                state (TrainState): The current training state.
                batch (dict): The current batch of data.
                train (bool): Whether we're training or validating. If training, we optimize both autodecoder and nef,
                    otherwise only autodecoder.

            Returns:
                TrainState: The updated training state.
            """

            # Split random key
            rng, key = jax.random.split(state.rng)

            # Define loss and calculate gradients
            def loss_fn(params, batch_i):
                # Apply 3 D rotation augmentation
                if self.train_aug and train:
                    rot = self.rotation_generator()
                    batch_i['pos'] = jnp.einsum('ij, bj->bi', rot, batch_i['pos'])

                pred, _ = self.model.apply(params, batch_i['pos'], batch_i['x'], batch_i['edge_index'], batch_i['batch'])
                label = batch_i['y']
                loss = jnp.abs(pred - ((label - self.shift) / self.scale))
                return jnp.mean(loss)
            
            # loss, grads = jax.value_and_grad(loss_fn)(state.params)
            value_and_grad_vmap = vmap(jax.value_and_grad(loss_fn), in_axes=(None, 0))
            loss, grads = value_and_grad_vmap(state.params, batch)
            loss = jax.tree.map(lambda x: x.mean(axis=0), loss)
            grads = jax.tree.map(lambda x: x.mean(axis=0), grads)

            # Update autodecoder
            updates, opt_state = self.optimizer.update(grads, state.opt_state)
            params = optax.apply_updates(state.params, updates)

            return loss, state.replace(
                params=params,
                opt_state=opt_state,
                rng=key
            )

        # Jit functions
        self.train_step = jax.jit(partial(step, train=True))
        self.val_step = jax.jit(partial(step, train=False))
        # self.train_step = partial(step, train=True)
        # self.val_step = partial(step, train=False)

    def train_model(self, num_epochs, state=None):
        """Trains the model for the given number of epochs.

        Args:
            num_epochs (int): The number of epochs to train for.

        Returns:
            state: The final training state.
        """

        # Keep track of global step
        self.global_step = 0
        self.epoch = 0

        if state is None:
            state = self.init_train_state()

        for epoch in range(1, num_epochs + 1):
            self.epoch = epoch
            state = self.train_epoch(state, epoch)

            # Save checkpoint (ckpt manager takes care of saving every n epochs)
            self.save_checkpoint(state)

            # Validate every n epochs
            if epoch % self.config.test.test_interval == 0:
                self.validate_epoch(state)
        return state

    def train_epoch(self, state, epoch):
        # Loop over batches
        losses = 0
        for batch_idx, batch in enumerate(self.train_loader):
 
            loss, state = self.train_step(state, batch)
            losses += loss

            # Log every n steps
            if batch_idx % self.config.logging.log_every_n_steps == 0:
                wandb.log({'train_mse_step': loss})
                self.update_prog_bar(loss, step=batch_idx)

            # Increment global step
            self.global_step += 1

        # Update epoch loss
        self.train_mse_epoch = losses / len(self.train_loader)
        wandb.log({'train_mse_epoch': self.train_mse_epoch})
        wandb.log({'epoch': epoch})
        return state
    
    def validate_epoch(self, state):
        """ Validates the model.

        Args:
            state: The current training state.
        """
        # Loop over batches
        losses = 0
        for batch_idx, batch in enumerate(self.val_loader):
            loss, _ = self.val_step(state, batch)
            losses += loss

            # Log every n steps
            if batch_idx % self.config.logging.log_every_n_steps == 0:
                wandb.log({'val_mse_step': loss})
                self.update_prog_bar(loss, step=batch_idx, train=False)

            # Increment global step
            self.global_val_step += 1

        # Update epoch loss
        self.val_mse_epoch = losses / len(self.val_loader)
        wandb.log({'val_mse_epoch': self.val_mse_epoch}, commit=False)





@hydra.main(version_base=None, config_path="./configs", config_name="qm9_regression")
def train(config):

    # Set log dir
    if not config.logging.log_dir:
        hydra_cfg = hydra.core.hydra_config.HydraConfig.get()
        config.logging.log_dir = hydra_cfg['runtime']['output_dir']

    # Create the fully connected dataset with node-masks i.o. edge-index
    train_dataset = QM9Dataset(split='train', target=config.training.target)
    val_dataset = QM9Dataset(split='val', target=config.training.target)
    test_dataset = QM9Dataset(split='test', target=config.training.target)

    # Define the dataloaders
    collate_fn_vmap = collate_fn_vmap_wrapper(is_static_shape=True, batch_size=config.training.batch_size, nodes_max=29, edges_max=56)
    train_dataloader = DataLoader(train_dataset, batch_size=config.training.batch_size, shuffle=True, num_workers=config.training.num_workers, pin_memory=True, collate_fn=collate_fn_vmap, drop_last=True)
    # train_dataloader = DataLoader(val_dataset, batch_size=config.training.batch_size, shuffle=True, num_workers=config.training.num_workers, pin_memory=True, collate_fn=collate_fn_vmap, drop_last=True)
    val_dataloader = DataLoader(val_dataset, batch_size=config.training.batch_size, shuffle=False, num_workers=config.training.num_workers, pin_memory=True, collate_fn=collate_fn_vmap)

    # Load and initialize the model
    trainer = QM9Trainer(config, train_dataloader, val_dataloader, seed=config.optimizer.seed)
    trainer.create_functions()

    # Initialize wandb
    wandb.init(
        entity=None,
        project="ponita-jax",
        dir=config.logging.log_dir,
        config=omegaconf.OmegaConf.to_container(config),
        mode='disabled' if config.logging.debug else 'online',
    )

    # Train model
    trainer.train_model(config.training.num_epochs)


if __name__ == "__main__":
    train()