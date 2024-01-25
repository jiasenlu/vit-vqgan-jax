"""
Modified training state with easy param freezing support and updating rngs over training iterations

Shamelessly copied from audax (https://github.com/SarthakYadav/audax/blob/master/audax/training_utils/trainstate.py)
Written for audax by / Copyright 2022, Sarthak Yadav
"""
from typing import Any, Callable, Dict
import jax
from flax import core
from flax import struct
import optax
import flax


class TrainState_v2(struct.PyTreeNode):
    """Simple train state for the common case with a single Optax optimizer.

      Synopsis::
          state = TrainState.create(
              apply_fn=model.apply,
              params=variables['params'],
              tx=tx)
          grad_fn = jax.grad(make_loss_fn(state.apply_fn))
          for batch in data:
            grads = grad_fn(state.params, batch)
            state = state.apply_gradients(grads=grads)

      Note that you can easily extend this dataclass by subclassing it for storing
      additional data (e.g. additional variable collections).

      For more exotic usecases (e.g. multiple optimizers) it's probably best to
      fork the class and modify it.

      Args:
        step: Counter starts at 0 and is incremented by every call to
          `.apply_gradients()`.
        apply_fn: Usually set to `model.apply()`. Kept in this dataclass for
          convenience to have a shorter params list for the `train_step()` function
          in your training loop.
        params: The parameters to be updated by `tx` and used by `apply_fn`.
        frozen_params:
        tx: An Optax gradient transformation.
        opt_state: The state for `tx`.
    """
    step: int
    params_g: core.FrozenDict[str, Any]
    params_d: core.FrozenDict[str, Any]
    params_p: core.FrozenDict[str, Any]
    frozen_params_g: core.FrozenDict[str, Any]
    frozen_params_d: core.FrozenDict[str, Any] 
    aux_rng_keys_g: core.FrozenDict[str, Any]
    aux_rng_keys_d: core.FrozenDict[str, Any]
    aux_rng_keys_p: core.FrozenDict[str, Any]
    opt_state_g: optax.OptState
    opt_state_d: optax.OptState
    apply_fn_g: Callable = struct.field(pytree_node=False)
    apply_fn_d: Callable = struct.field(pytree_node=False)
    apply_fn_p: Callable = struct.field(pytree_node=False)
    tx_g: optax.GradientTransformation = struct.field(pytree_node=False)
    tx_d: optax.GradientTransformation = struct.field(pytree_node=False)

    def apply_gradients_g(self, *, grads, **kwargs):
        """Updates `step`, `params`, `opt_state` and `**kwargs` in return value.

        Note that internally this function calls `.tx.update()` followed by a call
        to `optax.apply_updates()` to update `params` and `opt_state`.

        Args:
          grads: Gradients that have the same pytree structure as `.params`.
          **kwargs: Additional dataclass attributes that should be `.replace()`-ed.

        Returns:
          An updated instance of `self` with `step` incremented by one, `params`
          and `opt_state` updated by applying `grads`, and additional attributes
          replaced as specified by `kwargs`.
        """

        updates, new_opt_state = self.tx_g.update(grads, self.opt_state_g, self.params_g)
        new_params = optax.apply_updates(self.params_g, updates)
        rng_keys = self.update_rng_keys_g()

        return self.replace(
            params_g=new_params,
            frozen_params_g=self.frozen_params_g,
            opt_state_g=new_opt_state,
            aux_rng_keys_g=rng_keys,
            **kwargs)

    def apply_gradients_d(self, *, grads, **kwargs):
        """Updates `step`, `params`, `opt_state` and `**kwargs` in return value.

        Note that internally this function calls `.tx.update()` followed by a call
        to `optax.apply_updates()` to update `params` and `opt_state`.

        Args:
          grads: Gradients that have the same pytree structure as `.params`.
          **kwargs: Additional dataclass attributes that should be `.replace()`-ed.

        Returns:
          An updated instance of `self` with `step` incremented by one, `params`
          and `opt_state` updated by applying `grads`, and additional attributes
          replaced as specified by `kwargs`.
        """

        updates, new_opt_state = self.tx_d.update(grads, self.opt_state_d, self.params_d)
        new_params = optax.apply_updates(self.params_d, updates)
        rng_keys = self.update_rng_keys_d()

        return self.replace(
            params_d=new_params,
            frozen_params_d=self.frozen_params_d,
            opt_state_d=new_opt_state,
            aux_rng_keys_d=rng_keys,
            **kwargs)


    def update_rng_keys_g(self):
        unfrozen = flax.core.unfreeze(self.aux_rng_keys_g)
        for k in self.aux_rng_keys_g.keys():
            unfrozen[k] = jax.random.split(unfrozen[k], 1)[0]
        return flax.core.freeze(unfrozen)

    def update_rng_keys_d(self):
        unfrozen = flax.core.unfreeze(self.aux_rng_keys_d)
        for k in self.aux_rng_keys_d.keys():
            unfrozen[k] = jax.random.split(unfrozen[k], 1)[0]
        return flax.core.freeze(unfrozen)

    @property
    def get_all_params(self):
        return {**self.params, **self.frozen_params}

    @classmethod
    def create(cls, *, models, params, frozen_params, txs, aux_rng_keys, **kwargs):
        """Creates a new instance with `step=0` and initialized `opt_state`."""
        opt_states = {}
        for k, tx in txs.items():
          opt_states[k] = tx.init(params[k])
        
        apply_fns = {k: v.apply for k, v in models.items()}

        return cls(
            step=0,
            apply_fn_g = apply_fns['g'],
            apply_fn_d = apply_fns['d'],
            apply_fn_p = apply_fns['p'],
            params_g=params['g'],
            params_d=params['d'],
            params_p=params['p'],
            frozen_params_g=frozen_params['g'],
            frozen_params_d=frozen_params['d'],
            tx_g=txs['g'],
            tx_d=txs['d'],
            opt_state_g=opt_states['g'],
            opt_state_d=opt_states['d'],
            aux_rng_keys_g=aux_rng_keys['g'],
            aux_rng_keys_d=aux_rng_keys['d'],
            aux_rng_keys_p=aux_rng_keys['p'],
            **kwargs,
        )