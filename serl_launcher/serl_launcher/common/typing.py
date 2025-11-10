from typing import TYPE_CHECKING, Any, Callable, Dict, Sequence, Union

import flax
import jax.numpy as jnp
import numpy as np

if TYPE_CHECKING:
    import tensorflow as tf
    _Tensor = tf.Tensor
else:
    _Tensor = object

PRNGKey = Any
Params = flax.core.FrozenDict[str, Any]
Shape = Sequence[int]
Dtype = Any  # this could be a real type?
InfoDict = Dict[str, float]
Array = Union[np.ndarray, jnp.ndarray, _Tensor]
Data = Union[Array, Dict[str, "Data"]]
Batch = Dict[str, Data]
# A method to be passed into TrainState.__call__
ModuleMethod = Union[str, Callable, None]
