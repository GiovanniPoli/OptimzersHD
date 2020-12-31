from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.framework import ops
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.training.optimizer import Optimizer
from tensorflow.python.eager import context

import tensorflow as tf

# Paper di riferimento:
# https://arxiv.org/pdf/1703.04782.pdf
#
# Guida tensorflow:
# https://www.tensorflow.org/api_docs/python/tf/keras/optimizers/Optimizer#creating_a_custom_optimizer_2
#
# Esempio online:
# https://towardsdatascience.com/custom-optimizer-in-tensorflow-d5b41f75644a
#
# Classe ereditata:
# https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/training/optimizer.py
#
# Esempio di non_slot_variable:
# https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/training/adam.py
#
# Modelli VGG Net:
# PyTorch: http://torch.ch/blog/2015/07/30/cifar.html
# TensorFlow: https://github.com/geifmany/cifar-vgg/blob/master/cifar10vgg.py
# Paper: https://github.com/gbaydin/hypergradient-descent/blob/master/vgg.py

class SGD_HD(Optimizer):
  def __init__(self, alpha_0=0.001, beta=1e-3, 
               use_locking=False, name="SGD_HD"): 
    super(SGD_HD, self).__init__(use_locking, name)
    ## Float
    self._alpha_0 = alpha_0
    self._beta = beta
    ## Tensor
    self._beta_ten = None
  
  def _get_alpha(self):
    with ops.init_scope():
      if context.executing_eagerly():
        graph = None
      else:
        graph = ops.get_default_graph()
      return (self._get_non_slot_variable("alpha_t", graph=graph))

  def _create_slots(self, var_list):
    self._var_list = var_list
    
    first_var = min(var_list, key=lambda x: x.name) 
    
    self._create_non_slot_variable(
        initial_value=self._alpha_0, name="alpha_t", colocate_with=first_var)

    # Create slots for the first and second moments.
    for v in var_list:
      self._zeros_slot(v, "grad", self._name)
      self._zeros_slot(v, "old_dir", self._name)
      
  def _prepare(self):
    beta = self._call_if_callable(self._beta)
    self._beta_ten = ops.convert_to_tensor(beta, name="beta")

  def _apply_dense(self, grad, var):
    #print("Chiamo _apply_dense")
    ### richiamo slot
    gr_new   = self.get_slot(var, "grad")
    ## Creo gli operatori
    gr_up = gr_new.assign(grad)
        
    return control_flow_ops.group(*[gr_up])

  def _resource_apply_dense(self, grad, handle):
    return self._apply_dense(grad, handle)
    
  def _resource_apply_sparse(self, grad, handle):
    raise NotImplementedError("This optimizer is not implemented for sparse gradient")

  def _apply_sparse(self, grad, var):
    raise NotImplementedError("This optimizer is not implemented for sparse gradient")

  def _finish(self, update_ops, name_scope):
    #print("Chiamo _finish")
    beta = math_ops.cast(self._beta_ten, tf.float32)
    var_list = self._var_list
            
    with ops.control_dependencies(update_ops):     
        grads = [self.get_slot(var, "grad") for var in var_list]
        old_dir = [self.get_slot(var, "old_dir") for var in var_list]
        alpha = self._get_alpha()
        hypergrad = sum([tf.reduce_sum(tf.multiply(d,g)) for d, g in zip(old_dir, grads)])
        alpha_update = [alpha.assign_add(beta*hypergrad)]
    with ops.control_dependencies(alpha_update):
        ds_updates=[d.assign(g) for d, g in zip(old_dir, grads)]
    with ops.control_dependencies(ds_updates):
        variable_updates = [v.assign_sub(alpha*g) for v, g in zip(var_list, grads)]

    return control_flow_ops.group(*[update_ops, alpha_update, ds_updates, variable_updates],
                                     name=name_scope)

class SGD_HD_Mult(Optimizer):
  def __init__(self, alpha_0=0.001, beta=1e-3, 
               use_locking=False, name="SGD_HD_Mult"): 
    super(SGD_HD_Mult, self).__init__(use_locking, name)
    ## Float
    self._alpha_0 = alpha_0
    self._beta = beta
    ## Tensor
    self._beta_ten = None
  
  def _get_alpha(self):
    with ops.init_scope():
      if context.executing_eagerly():
        graph = None
      else:
        graph = ops.get_default_graph()
      return (self._get_non_slot_variable("alpha_t", graph=graph))

  def _create_slots(self, var_list):
    self._var_list = var_list
    
    first_var = min(var_list, key=lambda x: x.name) 
    
    self._create_non_slot_variable(
        initial_value=self._alpha_0, name="alpha_t", colocate_with=first_var)

    # Create slots for the first and second moments.
    for v in var_list:
      self._zeros_slot(v, "grad", self._name)
      self._zeros_slot(v, "old_dir", self._name)
      
  def _prepare(self):
    beta = self._call_if_callable(self._beta)
    self._beta_ten = ops.convert_to_tensor(beta, name="beta")

  def _apply_dense(self, grad, var):
    #print("Chiamo _apply_dense")
    ### richiamo slot
    gr_new   = self.get_slot(var, "grad")
    ## Creo gli operatori
    gr_up = gr_new.assign(grad)
        
    return control_flow_ops.group(*[gr_up])

  def _resource_apply_dense(self, grad, handle):
    return self._apply_dense(grad, handle)
    
  def _resource_apply_sparse(self, grad, handle):
    raise NotImplementedError("This optimizer is not implemented for sparse gradient")

  def _apply_sparse(self, grad, var):
    raise NotImplementedError("This optimizer is not implemented for sparse gradient")

  def _finish(self, update_ops, name_scope):
    beta = math_ops.cast(self._beta_ten, tf.float32)
    var_list = self._var_list
            
    with ops.control_dependencies(update_ops):     
        grads = [self.get_slot(var, "grad") for var in var_list]
        old_dir = [self.get_slot(var, "old_dir") for var in var_list]
        
        alpha = self._get_alpha()
        a = tf.concat([tf.reshape(g, shape=[-1]) for g in grads], axis=-1)
        b = tf.concat([tf.reshape(d, shape=[-1]) for d in old_dir], axis=-1)
        
        hypergrad = sum([tf.reduce_sum(tf.multiply(d,g)) for d, g in zip(old_dir, grads)])
        
        alpha_update = [alpha.assign(alpha * (tf.constant(1.0) + tf.multiply(beta, 
                        tf.math.divide_no_nan(hypergrad,
                        tf.multiply(tf.norm(a, axis=-1), tf.norm(b ,axis=-1)))
                                 )))]
        

        
    with ops.control_dependencies(alpha_update):
        ds_updates=[d.assign(g) for d, g in zip(old_dir, grads)]
    with ops.control_dependencies(ds_updates):
        variable_updates = [v.assign_sub(alpha*g) for v, g in zip(var_list, grads)]

    return control_flow_ops.group(*[update_ops, alpha_update, ds_updates, variable_updates],
                                     name=name_scope)

class SGDN_HD(Optimizer):
  def __init__(self,
               alpha_0=0.001,
               beta=1e-6,
               mu=0.9,
               use_locking=False,
               name="SGDN_HD"): 
    super(SGDN_HD, self).__init__(use_locking, name)
    self._alpha_0 = alpha_0
    self._beta = beta
    self._mu = mu

    # Tensor versions of the constructor arguments, created in _prepare().
    self._beta_ten = None
    self._mu_ten = None
 
  def _get_alpha(self):
    with ops.init_scope():
      if context.executing_eagerly():
        graph = None
      else:
        graph = ops.get_default_graph()
      return (self._get_non_slot_variable("alpha_t", graph=graph))

  def _create_slots(self, var_list):
    self._var_list = var_list
    
    first_var = min(var_list, key=lambda x: x.name) 
    
    self._create_non_slot_variable(
        initial_value=self._alpha_0, name="alpha_t", colocate_with=first_var)

    # Create slots for the first and second moments.
    for v in var_list:
      self._zeros_slot(v, "g_new", self._name)
      self._zeros_slot(v, "g_old", self._name)
      self._zeros_slot(v, "vel",   self._name)
      
  def _prepare(self):
    beta = self._call_if_callable(self._beta)
    mu = self._call_if_callable(self._mu)
    self._beta_ten = ops.convert_to_tensor(beta, name="beta")
    self._mu_ten = ops.convert_to_tensor(mu, name="mu")

  def _apply_dense(self, grad, var):
    ### richiamo slot
    gr_new   = self.get_slot(var, "g_new")
    ## Creo gli operatori
    gr_up = gr_new.assign(grad)
        
    return control_flow_ops.group(*[gr_up])

  def _resource_apply_dense(self, grad, handle):
    return self._apply_dense(grad, handle)
    
  def _resource_apply_sparse(self, grad, handle):
    raise NotImplementedError("This optimizer is not implemented for sparse gradient")

  def _apply_sparse(self, grad, var):
    raise NotImplementedError("This optimizer is not implemented for sparse gradient")

  def _finish(self, update_ops, name_scope):
    #print("Chiamo _finish")
    beta = math_ops.cast(self._beta_ten, tf.float32)
    mu = math_ops.cast(self._mu_ten, tf.float32)
    var_list = self._var_list
            
    with ops.control_dependencies(update_ops):     
        grads = [self.get_slot(var, "g_new") for var in var_list]
        vs = [self.get_slot(var, "vel") for var in var_list]
        old_dir = [self.get_slot(var, "g_old") for var in var_list]
        alpha = self._get_alpha()
          
        hypergrad = sum([tf.reduce_sum(tf.multiply(d,g)) for d, g in zip(old_dir, grads)])
        alpha_cy = alpha + beta*hypergrad
        alpha_update = [alpha.assign(alpha_cy)]
    
        buf = [ v*mu+g for v, g in zip(vs, grads)]
        newds = [g + mu*b for b, g in zip(buf,grads)]
        
        ## Update Ops 
    with tf.control_dependencies(alpha_update): 
                vel_update = [ v.assign(vnew) for v, vnew in zip(vs, buf)]
                ds_updates = [d.assign(nd) for d, nd in zip(old_dir, newds)]
  
    with tf.control_dependencies(ds_updates):
                variable_updates = [var.assign_add(-alpha*nd) for var, nd in zip(var_list, newds)]

    return control_flow_ops.group(*[update_ops, alpha_update, vel_update, ds_updates, variable_updates],
                                     name=name_scope)

class Adam_HD(Optimizer):
  def __init__(self,
               alpha_0=0.001,
               beta=1e-8,
               beta_1=0.9,
               beta_2=0.999,
               epsilon=1e-8,
               use_locking=False,
               name="Adam_HD"): 
    super(Adam_HD, self).__init__(use_locking, name)
    
    self._alpha_0 = alpha_0
    self._beta = beta
    self._beta_1 = beta_1
    self._beta_2 = beta_2
    self._epsilon = epsilon
    
    # Tensor versions of the constructor arguments, created in _prepare().
    self._alpha_0_ten = None
    self._beta_1_ten = None
    self._beta_2_ten = None
    self._epsilon_ten = None 
    
  def _get_alpha(self):
    with ops.init_scope():
      if context.executing_eagerly():
        graph = None
      else:
        graph = ops.get_default_graph()
      return (self._get_non_slot_variable("alpha_t", graph=graph))

  def _get_accumulators(self):
    with ops.init_scope():
      if context.executing_eagerly():
        graph = None
      else:
        graph = ops.get_default_graph()
      return (self._get_non_slot_variable("beta_1_power", graph=graph),
              self._get_non_slot_variable("beta_2_power", graph=graph),
              self._get_non_slot_variable("alpha_t", graph=graph))

  def _create_slots(self, var_list):
    first_var = min(var_list, key=lambda x: x.name)
    
    self._create_non_slot_variable(
        initial_value=self._beta_1, name='beta_1_power', colocate_with=first_var)
    self._create_non_slot_variable(
        initial_value=self._beta_2, name='beta_2_power', colocate_with=first_var)
    self._create_non_slot_variable(
        initial_value=self._alpha_0, name='alpha_t', colocate_with=first_var)
    # Create slots for the first and second moments.
 
    for v in var_list:
      self._zeros_slot(v, 'm', self._name)
    for v in var_list:
      self._zeros_slot(v, 'v', self._name)
    for v in var_list:
      self._zeros_slot(v, 'grad_ut_alpha', self._name)
      
  def _prepare(self):
    alpha0 = self._call_if_callable(self._alpha_0)
    beta  =  self._call_if_callable(self._beta)
    beta1 =  self._call_if_callable(self._beta_1)
    beta2 =  self._call_if_callable(self._beta_2)
    epsilon = self._call_if_callable(self._epsilon)

    self._alpha_0_ten = ops.convert_to_tensor(alpha0, name= "alpha_0")
    self._beta_ten = ops.convert_to_tensor(beta, name="beta")
    self._beta_1_ten = ops.convert_to_tensor(beta1, name="beta_1")
    self._beta_2_ten = ops.convert_to_tensor(beta2, name="beta_2")
    self._epsilon_ten = ops.convert_to_tensor(epsilon, name="epsilon")

  def _apply_dense(self, grad, var):
    ### Richiamo costanti
    beta = math_ops.cast(self._beta_ten, var.dtype.base_dtype)
    beta_1 = math_ops.cast(self._beta_1_ten, var.dtype.base_dtype)
    beta_2 = math_ops.cast(self._beta_2_ten, var.dtype.base_dtype)
    eps = math_ops.cast(self._epsilon_ten, var.dtype.base_dtype)
        
    ### richiamo slot
    m_old = self.get_slot(var, "m")
    v_old = self.get_slot(var, "v")
    grad_ut_alpha_old = self.get_slot(var, "grad_ut_alpha")
        
    ### richiamo gli accumulatori no_slot
    beta_1_power, beta_2_power, alpha_t_old = self._get_accumulators()
        
    ## Creo gli operatori

    h_t = tf.tensordot(tf.reshape(grad_ut_alpha_old,[-1]),tf.reshape(grad,[-1]), axes = 1)

    alpha_t_new = alpha_t_old - beta*h_t
        
    m_t = beta_1*m_old + (tf.constant(1.0)-beta_1)*grad
    v_t = beta_2*v_old + (tf.constant(1.0)-beta_2)*grad*grad

    m_hat = tf.divide(m_t, tf.constant(1.0)-beta_1_power)
    v_hat = tf.divide(v_t, tf.constant(1.0)-beta_2_power)

    ut =  -tf.divide(alpha_t_new*m_hat, tf.sqrt(v_hat)+ eps)        
    ## Aggiorno parametri
    var_update = state_ops.assign_add(var, ut)
        
    ### Aggiorno Slots
    m_up = m_old.assign(m_t)
    v_up = v_old.assign(v_t)
    old_grad_up = grad_ut_alpha_old.assign(-tf.divide(m_hat, tf.sqrt(v_hat)+ eps))
    
    with ops.colocate_with(alpha_t_old):
        update_alpha_t = alpha_t_old.assign(
                        alpha_t_new, use_locking=self._use_locking)
         
        return control_flow_ops.group(*[var_update, m_up, v_up, old_grad_up, update_alpha_t])

  def _finish(self, update_ops, name_scope):
    # Update the power accumulators.
    with ops.control_dependencies(update_ops):
      beta1_power, beta2_power = self._get_accumulators()[0:2]
      with ops.colocate_with(beta1_power):
        update_beta1 = beta1_power.assign(
            beta1_power * self._beta_1, use_locking=self._use_locking)
        update_beta2 = beta2_power.assign(
            beta2_power * self._beta_2, use_locking=self._use_locking)
    return control_flow_ops.group(
        *update_ops + [update_beta1, update_beta2], name=name_scope)

  def _resource_apply_dense(self, grad, handle):
        return self._apply_dense(grad, handle)
    
  def _resource_apply_sparse(self, grad, handle):
        raise NotImplementedError("Non è implementato sono pigro")

  def _apply_sparse(self, grad, var):
        raise NotImplementedError("Non è implementato sono pigro")

class Adam_HD_Mult(Optimizer):
  def __init__(self,
               alpha_0=0.001,
               beta=1e-8,
               beta_1=0.9,
               beta_2=0.999,
               epsilon=1e-8,
               use_locking=False,
               name="Adam_HD_Mult"): 
    super(Adam_HD_Mult, self).__init__(use_locking, name)
    
    self._alpha_0 = alpha_0
    self._beta = beta
    self._beta_1 = beta_1
    self._beta_2 = beta_2
    self._epsilon = epsilon
    
    # Tensor versions of the constructor arguments, created in _prepare().
    self._alpha_0_ten = None
    self._beta_1_ten = None
    self._beta_2_ten = None
    self._epsilon_ten = None 
    
  def _get_alpha(self):
    with ops.init_scope():
      if context.executing_eagerly():
        graph = None
      else:
        graph = ops.get_default_graph()
      return (self._get_non_slot_variable("alpha_t", graph=graph))

  def _get_accumulators(self):
    with ops.init_scope():
      if context.executing_eagerly():
        graph = None
      else:
        graph = ops.get_default_graph()
      return (self._get_non_slot_variable("beta_1_power", graph=graph),
              self._get_non_slot_variable("beta_2_power", graph=graph),
              self._get_non_slot_variable("alpha_t", graph=graph))

  def _create_slots(self, var_list):
    first_var = min(var_list, key=lambda x: x.name)
    
    self._create_non_slot_variable(
        initial_value=self._beta_1, name='beta_1_power', colocate_with=first_var)
    self._create_non_slot_variable(
        initial_value=self._beta_2, name='beta_2_power', colocate_with=first_var)
    self._create_non_slot_variable(
        initial_value=self._alpha_0, name='alpha_t', colocate_with=first_var)
    # Create slots for the first and second moments.
 
    for v in var_list:
      self._zeros_slot(v, 'm', self._name)
    for v in var_list:
      self._zeros_slot(v, 'v', self._name)
    for v in var_list:
      self._zeros_slot(v, 'grad_ut_alpha', self._name)
      
  def _prepare(self):
    alpha0 = self._call_if_callable(self._alpha_0)
    beta  =  self._call_if_callable(self._beta)
    beta1 =  self._call_if_callable(self._beta_1)
    beta2 =  self._call_if_callable(self._beta_2)
    epsilon = self._call_if_callable(self._epsilon)

    self._alpha_0_ten = ops.convert_to_tensor(alpha0, name= "alpha_0")
    self._beta_ten = ops.convert_to_tensor(beta, name="beta")
    self._beta_1_ten = ops.convert_to_tensor(beta1, name="beta_1")
    self._beta_2_ten = ops.convert_to_tensor(beta2, name="beta_2")
    self._epsilon_ten = ops.convert_to_tensor(epsilon, name="epsilon")

  def _apply_dense(self, grad, var):
    ### Richiamo costanti
    beta = math_ops.cast(self._beta_ten, var.dtype.base_dtype)
    beta_1 = math_ops.cast(self._beta_1_ten, var.dtype.base_dtype)
    beta_2 = math_ops.cast(self._beta_2_ten, var.dtype.base_dtype)
    eps = math_ops.cast(self._epsilon_ten, var.dtype.base_dtype)
        
    ### richiamo slot
    m_old = self.get_slot(var, "m")
    v_old = self.get_slot(var, "v")
    grad_ut_alpha_old = self.get_slot(var, "grad_ut_alpha")
        
    ### richiamo gli accumulatori no_slot
    beta_1_power, beta_2_power, alpha_t_old = self._get_accumulators()
        
    ## Creo gli operatori
    a = tf.reshape(grad_ut_alpha_old,[-1])
    b = tf.reshape(grad,[-1])

    h_t = tf.tensordot(a, b, axes = 1)
        
    alpha_t_new = alpha_t_old * (tf.constant(1.0) + tf.multiply(beta, 
                        tf.math.divide_no_nan(h_t,
                        tf.multiply(tf.norm(a, axis=-1), tf.norm(b ,axis=-1)))))

        
    m_t = beta_1*m_old + (tf.constant(1.0)-beta_1)*grad
    v_t = beta_2*v_old + (tf.constant(1.0)-beta_2)*grad*grad

    m_hat = tf.divide(m_t, tf.constant(1.0)-beta_1_power)
    v_hat = tf.divide(v_t, tf.constant(1.0)-beta_2_power)

    ut =  -tf.divide(alpha_t_new*m_hat, tf.sqrt(v_hat)+ eps)        
    ## Aggiorno parametri
    var_update = state_ops.assign_add(var, ut)
        
    ### Aggiorno Slots
    m_up = m_old.assign(m_t)
    v_up = v_old.assign(v_t)
    old_grad_up = grad_ut_alpha_old.assign(-tf.divide(m_hat, tf.sqrt(v_hat)+ eps))
    
    with ops.colocate_with(alpha_t_old):
        update_alpha_t = alpha_t_old.assign(
                        alpha_t_new, use_locking=self._use_locking)
         
        return control_flow_ops.group(*[var_update, m_up, v_up, old_grad_up, update_alpha_t])

  def _finish(self, update_ops, name_scope):
    # Update the power accumulators.
    with ops.control_dependencies(update_ops):
      beta1_power, beta2_power = self._get_accumulators()[0:2]
      with ops.colocate_with(beta1_power):
        update_beta1 = beta1_power.assign(
            beta1_power * self._beta_1, use_locking=self._use_locking)
        update_beta2 = beta2_power.assign(
            beta2_power * self._beta_2, use_locking=self._use_locking)
    return control_flow_ops.group(
        *update_ops + [update_beta1, update_beta2], name=name_scope)

  def _resource_apply_dense(self, grad, handle):
        return self._apply_dense(grad, handle)
    
  def _resource_apply_sparse(self, grad, handle):
        raise NotImplementedError("Non è implementato sono pigro")

  def _apply_sparse(self, grad, var):
        raise NotImplementedError("Non è implementato sono pigro")

class Adam_HD_Mult_v2(Optimizer):
  def __init__(self, alpha_0=0.001, beta=1e-3,
               beta_1 = 0.9, beta_2 = 0.999, epsilon = 1e-8,
               use_locking=False, name="Adam_HD_Mult_v2"): 
    super(Adam_HD_Mult_v2, self).__init__(use_locking, name)
    
    self._alpha_0 = alpha_0
    self._beta = beta
    self._beta_1 = beta_1
    self._beta_2 = beta_2
    self._epsilon = epsilon
    
    # Tensor versions of the constructor arguments, created in _prepare().
    self._alpha_0_ten = None
    self._beta_1_ten = None
    self._beta_2_ten = None
    self._epsilon_ten = None 
    
  def _get_alpha(self):
    with ops.init_scope():
      if context.executing_eagerly():
        graph = None
      else:
        graph = ops.get_default_graph()
      return (self._get_non_slot_variable("alpha_t", graph=graph))

  def _get_accumulators(self):
    with ops.init_scope():
      if context.executing_eagerly():
        graph = None
      else:
        graph = ops.get_default_graph()
      return (self._get_non_slot_variable("beta_1_power", graph=graph),
              self._get_non_slot_variable("beta_2_power", graph=graph),
              self._get_non_slot_variable("alpha_t", graph=graph))

  def _create_slots(self, var_list):
    self._var_list = var_list

    first_var = min(var_list, key=lambda x: x.name)
    
    self._create_non_slot_variable(
        initial_value=self._beta_1, name='beta_1_power', colocate_with=first_var)
    self._create_non_slot_variable(
        initial_value=self._beta_2, name='beta_2_power', colocate_with=first_var)
    self._create_non_slot_variable(
        initial_value=self._alpha_0, name='alpha_t', colocate_with=first_var)
    # Create slots for the first and second moments.
 
    for v in var_list:
      self._zeros_slot(v, 'm', self._name)
    for v in var_list:
      self._zeros_slot(v, 'v', self._name)
    for v in var_list:
      self._zeros_slot(v, 'grad_ut_alpha', self._name)
    for v in var_list:
      self._zeros_slot(v, 'grad', self._name)
      
  def _prepare(self):
    alpha0 = self._call_if_callable(self._alpha_0)
    beta  =  self._call_if_callable(self._beta)
    beta1 =  self._call_if_callable(self._beta_1)
    beta2 =  self._call_if_callable(self._beta_2)
    epsilon = self._call_if_callable(self._epsilon)

    self._alpha_0_ten = ops.convert_to_tensor(alpha0, name= "alpha_0")
    self._beta_ten = ops.convert_to_tensor(beta, name="beta")
    self._beta_1_ten = ops.convert_to_tensor(beta1, name="beta_1")
    self._beta_2_ten = ops.convert_to_tensor(beta2, name="beta_2")
    self._epsilon_ten = ops.convert_to_tensor(epsilon, name="epsilon")

  def _apply_dense(self, grad, var):
    ### Richiamo costanti
    beta_1 = math_ops.cast(self._beta_1_ten, var.dtype.base_dtype)
    beta_2 = math_ops.cast(self._beta_2_ten, var.dtype.base_dtype)
        
    ### richiamo slot
    m_old = self.get_slot(var, "m")
    v_old = self.get_slot(var, "v")
    gr_new   = self.get_slot(var, "grad")
    
    ## Creo gli operatori        
    m_t = beta_1*m_old + (tf.constant(1.0)-beta_1)*grad
    v_t = beta_2*v_old + (tf.constant(1.0)-beta_2)*grad*grad
     
    ### Aggiorno Slots
    m_up = m_old.assign(m_t)
    v_up = v_old.assign(v_t)
    gr_up = gr_new.assign(grad)
        
    return control_flow_ops.group(*[gr_up, m_up, v_up])

  def _resource_apply_dense(self, grad, handle):
    return self._apply_dense(grad, handle)
    
  def _resource_apply_sparse(self, grad, handle):
    raise NotImplementedError("This optimizer is not implemented for sparse gradient")

  def _apply_sparse(self, grad, var):
    raise NotImplementedError("This optimizer is not implemented for sparse gradient")

  def _finish(self, update_ops, name_scope):
    beta = math_ops.cast(self._beta_ten, tf.float32)
    var_list = self._var_list
    beta_1 = math_ops.cast(self._beta_1_ten, tf.float32)
    beta_2 = math_ops.cast(self._beta_2_ten, tf.float32)
    eps = math_ops.cast(self._epsilon_ten, tf.float32)
        
    ### richiamo gli accumulatori no_slot
    beta_1_power, beta_2_power, alpha_t = self._get_accumulators()
    
    with ops.control_dependencies(update_ops):     
        grads = [self.get_slot(var, "grad") for var in var_list]
        old_dir = [self.get_slot(var, "grad_ut_alpha") for var in var_list]
        ms = [self.get_slot(var, "m") for var in var_list]
        vs = [self.get_slot(var, "v") for var in var_list]
        
        m_hat = [tf.divide(m, tf.constant(1.0)-beta_1_power) for m in ms]
        v_hat = [tf.divide(v, tf.constant(1.0)-beta_2_power) for v in vs]
    
        a = tf.concat([tf.reshape(g, shape=[-1]) for g in grads], axis=-1)
        b = tf.concat([tf.reshape(d, shape=[-1]) for d in old_dir], axis=-1)
        
        hypergrad = sum([tf.reduce_sum(tf.multiply(d,g)) for d, g in zip(old_dir, grads)])
        
        alpha_update = [alpha_t.assign(alpha_t * (tf.constant(1.0) + tf.multiply(beta, 
                        tf.math.divide_no_nan(hypergrad,
                        tf.multiply(tf.norm(a, axis=-1), tf.norm(b ,axis=-1))))))]

        with ops.control_dependencies(alpha_update):
            ds_updates=[d.assign(tf.divide(mh, tf.sqrt(vh)+eps)) for d, mh, vh in zip(old_dir, m_hat, v_hat)]
        with ops.control_dependencies(ds_updates):
            variable_updates = [v.assign_sub(tf.divide(alpha_t*mh, tf.sqrt(vh)+eps)) for v, mh, vh in zip(var_list, m_hat, v_hat)]
        with ops.control_dependencies(variable_updates):
                update_beta1 = beta_1_power.assign(
                    beta_1_power * self._beta_1, use_locking=self._use_locking)
                update_beta2 = beta_2_power.assign(
                    beta_2_power * self._beta_2, use_locking=self._use_locking)
     
    return control_flow_ops.group(*[update_ops, alpha_update, ds_updates,
                                    variable_updates, update_beta1, update_beta2],
                                     name=name_scope)

class SGDN_HD_No_Glob(Optimizer):
  def __init__(self,
               alpha_0 = 0.001,
               beta = 1e-3,
               mu = 0.9,
               use_locking = False,
               name = "SGDN_HD_No_Glob"): 
    super(SGDN_HD_No_Glob, self).__init__(use_locking, name)
    
    self._alpha_0 = alpha_0
    self._beta = beta
    self._mu = mu

    
    # Tensor versions of the constructor arguments, created in _prepare().
    self._beta_ten = None
    self._mu_ten = None 
    
  def _get_alpha(self):
    with ops.init_scope():
      if context.executing_eagerly():
        graph = None
      else:
        graph = ops.get_default_graph()
      return (self._get_non_slot_variable("alpha_t", graph=graph))

  def _create_slots(self, var_list):
    first_var = min(var_list, key=lambda x: x.name)
 
    self._create_non_slot_variable(
        initial_value=self._alpha_0, name='alpha_t', colocate_with=first_var)
    # Create slots for the first and second moments.
 
    for v in var_list:
      self._zeros_slot(v, 'vel', self._name)
    for v in var_list:
      self._zeros_slot(v, 'grad_ut_alpha', self._name)
      
  def _prepare(self):
    mu = self._call_if_callable(self._mu)
    beta  =  self._call_if_callable(self._beta)

    self._mu_ten = ops.convert_to_tensor(mu, name= "mu")
    self._beta_ten = ops.convert_to_tensor(beta, name="beta")

  def _apply_dense(self, grad, var):
    ### Richiamo costanti
    beta = math_ops.cast(self._beta_ten, var.dtype.base_dtype)
    mu   = math_ops.cast(self._mu_ten, var.dtype.base_dtype)
        
    ### richiamo slot
    v_old = self.get_slot(var, "vel")
    grad_ut_alpha_old = self.get_slot(var, "grad_ut_alpha")
        
    ### richiamo gli accumulatori no_slot
    alpha_t_old = self._get_alpha()
        
    ## Creo gli operatori

    h_t = tf.tensordot(tf.reshape(grad_ut_alpha_old,[-1]),
                       tf.reshape(grad,[-1]), axes = 1)

    alpha_t_new = alpha_t_old - beta*h_t
        
    v_t = mu*v_old + grad


    ut =  -alpha_t_old*(mu*v_t+grad)        
        
    ### Aggiorno Slots
    old_grad_up = grad_ut_alpha_old.assign(v_t)
    
    v_up = v_old.assign(-mu*v_old - grad)
    
    ## Aggiorno parametri
    var_update = state_ops.assign_add(var, ut)
    
    with ops.colocate_with(alpha_t_old):
        update_alpha_t = alpha_t_old.assign(
                        alpha_t_new, use_locking=self._use_locking)
         
        return control_flow_ops.group(*[var_update, v_up, old_grad_up, update_alpha_t])


  def _resource_apply_dense(self, grad, handle):
        return self._apply_dense(grad, handle)
    
  def _resource_apply_sparse(self, grad, handle):
        raise NotImplementedError("Non è implementato sono pigro")

  def _apply_sparse(self, grad, var):
        raise NotImplementedError("Non è implementato sono pigro")