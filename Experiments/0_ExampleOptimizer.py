from tensorflow.python.ops import control_flow_ops
from tensorflow.python.training           import training_ops
from tensorflow.python.training.optimizer import Optimizer
from tensorflow.python.eager import context

from tensorflow.python.framework import ops
from tensorflow.python.ops import math_ops

import tensorflow as tf
                      
class Adam_Test(Optimizer):
  def __init__(self,
               learning_rate=0.001,
               beta1=0.9,
               beta2=0.999,
               epsilon=1e-8,
               use_locking=False,
               name="Adam_Test"):
    super(Adam_Test, self).__init__(use_locking, name)
    self._lr = learning_rate
    self._beta1 = beta1
    self._beta2 = beta2
    self._epsilon = epsilon

    self._lr_t = None
    self._beta1_t = None
    self._beta2_t = None
    self._epsilon_t = None

  def _get_beta_accumulators(self):
    with ops.init_scope():
      if context.executing_eagerly():
        graph = None
      else:
        graph = ops.get_default_graph()
      return (self._get_non_slot_variable("beta1_power", graph=graph),
              self._get_non_slot_variable("beta2_power", graph=graph))

  def _get_ops_tester(self):
    with ops.init_scope():
      if context.executing_eagerly():
        graph = None
      else:
        graph = ops.get_default_graph()
      return (self._get_non_slot_variable("ops", graph=graph))

  def _get_iter_tester(self):
    with ops.init_scope():
      if context.executing_eagerly():
        graph = None
      else:
        graph = ops.get_default_graph()
      return (self._get_non_slot_variable("iter", graph=graph))


  def _create_slots(self, var_list):
    # Create the beta1 and beta2 accumulators on the same device as the first
    # variable. Sort the var_list to make sure this device is consistent across
    # workers (these need to go on the same PS, otherwise some updates are
    # silently ignored).
    first_var = min(var_list, key=lambda x: x.name)
    self._create_non_slot_variable(
        initial_value=self._beta1, name="beta1_power", colocate_with=first_var)
    self._create_non_slot_variable(
        initial_value=self._beta2, name="beta2_power", colocate_with=first_var)
    self._create_non_slot_variable(
        initial_value= tf.constant(0.0, dtype=first_var.dtype.base_dtype),
                                   name="ops", colocate_with=first_var)
    self._create_non_slot_variable(
        initial_value= tf.constant(0.0, dtype=first_var.dtype.base_dtype),
                                   name="iter", colocate_with=first_var)

    # Create slots for the first and second moments.
    for v in var_list:
      self._zeros_slot(v, "m", self._name)
      self._zeros_slot(v, "v", self._name)

  def _prepare(self):
    lr = self._call_if_callable(self._lr)
    beta1 = self._call_if_callable(self._beta1)
    beta2 = self._call_if_callable(self._beta2)
    epsilon = self._call_if_callable(self._epsilon)

    self._lr_t = ops.convert_to_tensor(lr, name="learning_rate")
    self._beta1_t = ops.convert_to_tensor(beta1, name="beta1")
    self._beta2_t = ops.convert_to_tensor(beta2, name="beta2")
    self._epsilon_t = ops.convert_to_tensor(epsilon, name="epsilon")

  def _apply_dense(self, grad, var):
    m = self.get_slot(var, "m")
    v = self.get_slot(var, "v")
    beta1_power, beta2_power = self._get_beta_accumulators()
    ops = self._get_ops_tester()
    ops_up = ops.assign_add(1)
    return control_flow_ops.group(*[
        training_ops.apply_adam(
        var,
        m,
        v,
        math_ops.cast(beta1_power, var.dtype.base_dtype),
        math_ops.cast(beta2_power, var.dtype.base_dtype),
        math_ops.cast(self._lr_t, var.dtype.base_dtype),
        math_ops.cast(self._beta1_t, var.dtype.base_dtype),
        math_ops.cast(self._beta2_t, var.dtype.base_dtype),
        math_ops.cast(self._epsilon_t, var.dtype.base_dtype),
        grad,
        use_locking=self._use_locking).op ,
        ops_up])

  def _resource_apply_dense(self, grad, var):
    m = self.get_slot(var, "m")
    v = self.get_slot(var, "v")
    beta1_power, beta2_power = self._get_beta_accumulators()
    ops = self._get_ops_tester()
    ops_up = ops.assign_add(1)
    return control_flow_ops.group(*[training_ops.resource_apply_adam(
        var.handle,
        m.handle,
        v.handle,
        math_ops.cast(beta1_power, grad.dtype.base_dtype),
        math_ops.cast(beta2_power, grad.dtype.base_dtype),
        math_ops.cast(self._lr_t, grad.dtype.base_dtype),
        math_ops.cast(self._beta1_t, grad.dtype.base_dtype),
        math_ops.cast(self._beta2_t, grad.dtype.base_dtype),
        math_ops.cast(self._epsilon_t, grad.dtype.base_dtype),
        grad,
        use_locking=self._use_locking),
        ops_up])


  def _finish(self, update_ops, name_scope):
    # Update the power accumulators.
    with ops.control_dependencies(update_ops):
      beta1_power, beta2_power = self._get_beta_accumulators()
      it = self._get_iter_tester()
      with ops.colocate_with(beta1_power):
        update_beta1 = beta1_power.assign(
            beta1_power * self._beta1_t, use_locking=self._use_locking)
        update_beta2 = beta2_power.assign(
            beta2_power * self._beta2_t, use_locking=self._use_locking)
        update_it = it.assign_add(1)
    return control_flow_ops.group(
        *update_ops + [update_beta1, update_beta2, update_it], name=name_scope)