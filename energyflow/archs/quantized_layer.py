from __future__ import absolute_import
import tensorflow.keras.backend as K
import numpy as np
from tensorflow.keras.layers import Dense, Layer, Activation
from tensorflow.keras import constraints
from tensorflow.keras import initializers
import tensorflow as tf
from tensorflow.python.ops import math_ops
from tensorflow.keras import regularizers
def ternary_round(x):
    forward = K.round(x)
    return x + K.stop_gradient(forward - x)
def round_zero(x):
    forward = K.round(K.clip(x*2, 0, 1))*K.abs(x)
    return x + K.stop_gradient(forward - x)

def _hard_tanh(x):
    return K.clip(x, -1, 1)
def ternary_tanh(x):
    return ternary_round(_hard_tanh(x))

def _soft_unit(x):
    return K.clip(x, 0, 1)
def binary_unit(x):
    return ternary_round(_soft_unit(x))

def special_relu(x):
    return K.maximum(0.02, x)



def custom_reg(x):
    return math_ops.reduce_sum(K.abs(
        K.minimum(
            1e-5*math_ops.reduce_sum(math_ops.abs(x), axis=0), 
    math_ops.reduce_sum(
        K.abs(math_ops.reduce_sum(math_ops.cast(math_ops.greater_equal(x, 0.25), K.floatx()), axis=0)-1)*K.mean(x, axis=0)
)))) + math_ops.reduce_sum(K.abs(
        K.minimum(
            1e-5*math_ops.reduce_sum(math_ops.abs(x), axis=1), 
    math_ops.reduce_sum(
        K.abs(math_ops.reduce_sum(math_ops.cast(math_ops.greater_equal(x, 0.25), K.floatx()), axis=1)-1)*K.mean(x, axis=1)
        )))) + 3e-4* math_ops.reduce_sum(K.clip(1- math_ops.reduce_max(x, axis=0), 0,0.9))
    

class Clip(constraints.Constraint):
    def __call__(self, x):
        return K.clip(x, -1, 1)

class ComplicatedClip(constraints.Constraint):
    def __call__(self, x):
        randnum = K.random_uniform((1,), minval=0, maxval=100)
        boolean = math_ops.cast(math_ops.greater_equal(randnum, 99.95), K.floatx())

        above_zero =math_ops.cast(math_ops.greater_equal(x, 0.02), K.floatx())
        x=x * above_zero
        pre_mask = math_ops.cast(math_ops.greater_equal(1.5,x), K.floatx())
        max_x = K.maximum(math_ops.reduce_max(x*pre_mask), 0.02)

        mask = math_ops.cast(math_ops.greater_equal(x*pre_mask, 0.8), K.floatx())
        mask = math_ops.cast(math_ops.greater_equal(x*mask, max_x), K.floatx())

        mask_val_row = math_ops.cast(math_ops.greater_equal(math_ops.reduce_sum(mask, axis=0, keepdims=True), 1), K.floatx())
        dotted_row= K.repeat_elements(mask_val_row, rep=x.get_shape().as_list()[0], axis=0)


        mask_val_col = math_ops.cast(math_ops.greater_equal(math_ops.reduce_sum(mask, axis=1, keepdims=True), 1), K.floatx())
        dotted_col = K.repeat_elements(mask_val_col, rep=x.get_shape().as_list()[1], axis=1)

        cross_mask = dotted_row+dotted_col-mask

        x= 2*mask + x*(-mask+1)
        x= (mask+1)*x - cross_mask*x

        return (-boolean+1)*x + boolean*(x*0 + 0.35)

class TernaryDense(Dense):
    def __init__(self, units, **kwargs):
        super(TernaryDense, self).__init__(units, **kwargs)

    def build(self, input_shape):
        input_dim = input_shape[-1]
        self.kernel_constraint = Clip()
        self.kernel_initializer = initializers.RandomUniform(-0.5, 0.5)
        self.bias= None
        self.kernel_lr_multiplier = 3000
        self.kernel = self.add_weight(shape=(input_dim, self.units),
                                        initializer = self.kernel_initializer,
                                        name='kernel',
                                        regularizer=self.kernel_regularizer,
                                        constraint= self.kernel_constraint)
        self.built= True
    
    def call(self, inputs):
        ternary_kernel = ternary_tanh(self.kernel)
        output = K.dot(inputs, ternary_kernel)
        if self.activation is not None:
            output = self.activation(output)
        return output


class BinarySparseDense(Dense):
    def __init__(self, units, **kwargs):
        super().__init__(units, **kwargs)
    def build(self, input_shape):
        input_dim = input_shape[-1]
        self.kernel_constraint = ComplicatedClip()
        self.kernel_initializer = initializers.Constant(0.3)
        self.bias=None
        self.kernel_lr_multiplier = 1
        self.kernel = self.add_weight(shape=(input_dim, self.units),
                                initializer = self.kernel_initializer,
                                name='kernel',
                                regularizer=custom_reg, 
                                constraint= self.kernel_constraint)
        self.built=True
    def call(self, inputs):
        output = K.dot(inputs, round_zero(special_relu(self.kernel)))
        if self.activation is not None:
            output = self.activation(output)
        return output

