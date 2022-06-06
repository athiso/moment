from __future__ import absolute_import, division, print_function

from abc import abstractmethod, abstractproperty

import numpy as np

import tensorflow.keras.backend as K
from tensorflow.keras import __version__ as __keras_version__
from tensorflow.keras.layers import Concatenate, Dense, Dot, Dropout, Input, Lambda, TimeDistributed, concatenate, Reshape
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
from tensorflow import reshape, shape, repeat
from energyflow.archs.archbase import NNBase, _get_act_layer
from energyflow.archs.dnn import construct_dense
from energyflow.utils import iter_or_rep
from energyflow.archs import EFN
from energyflow.archs.efn import construct_weighted_input,construct_input,construct_efn_weight_mask,construct_latent,construct_distributed_dense
__all__ = ['TwoStageEFN']

if __keras_version__.endswith('-tf'):
    __keras_version__ = __keras_version__[:-3]
keras_version_tuple = tuple(map(int, __keras_version__.split('.')))
DOT_AXIS = 0 if keras_version_tuple <= (2, 2, 4) else 1


def construct_distributed_dense2(tensorphat,tensoroutput,num_output,sizes, acts='relu', k_inits='he_uniform', 
                                                                  names=None, l2_regs=0.):

    # repeat options if singletons
    acts, k_inits, names = iter_or_rep(acts), iter_or_rep(k_inits), iter_or_rep(names)
    l2_regs = iter_or_rep(l2_regs)
    
    # list of tensors
    x1 = reshape(tensoroutput,[shape(tensoroutput)[0],1,num_output])
    x = repeat(x1, repeats=[shape(tensorphat)[1]], axis=1)
    y = tensorphat
    concat = concatenate([x,y],axis=2)
    layers, tensors = [], [concat]

    # iterate over specified layers
    for s, act, k_init, name, l2_reg in zip(sizes, acts, k_inits, names, l2_regs):
        # define a dense layer that will be applied through time distributed
        kwargs = {} 
        if l2_reg > 0.:
            kwargs.update({'kernel_regularizer': l2(l2_reg), 'bias_regularizer': l2(l2_reg)})
        d_layer = Dense(s, kernel_initializer=k_init, **kwargs)

        tdist_layer = TimeDistributed(d_layer, name=name)
        act_layer = _get_act_layer(act)
        layers.extend([tdist_layer, act_layer])


        tensors.append(tdist_layer(tensors[-1]))
        tensors.append(act_layer(tensors[-1]))

    return layers, tensors[1:]

class Ensemble(NNBase):

    def _process_hps(self):

        super(Ensemble, self)._process_hps()

        # required hyperparameters
        self.input_dim = self._proc_arg('input_dim')
        self.num_output = self._proc_arg('num_output')
        self.Phi_sizes = self._proc_arg('Phi_sizes', old='ppm_sizes')
        self.F_sizes = self._proc_arg('F_sizes', old='dense_sizes')

        # activations
        self.Phi_acts = iter_or_rep(self._proc_arg('Phi_acts', default='relu', 
                                                               old='ppm_acts'))
        self.F_acts = iter_or_rep(self._proc_arg('F_acts', default='relu', 
                                                           old='dense_acts'))

        # initializations
        self.Phi_k_inits = iter_or_rep(self._proc_arg('Phi_k_inits', default='he_uniform', 
                                                                     old='ppm_k_inits'))
        self.F_k_inits = iter_or_rep(self._proc_arg('F_k_inits', default='he_uniform', 
                                                                 old='dense_k_inits'))

        # regularizations
        self.latent_dropout = self._proc_arg('latent_dropout', default=0.)
        self.F_dropouts = iter_or_rep(self._proc_arg('F_dropouts', default=0., 
                                                                   old='dense_dropouts'))
        self.Phi_l2_regs = iter_or_rep(self._proc_arg('Phi_l2_regs', default=0.))
        self.F_l2_regs   = iter_or_rep(self._proc_arg('F_l2_regs', default=0.))

        # masking
        self.mask_val = self._proc_arg('mask_val', default=0.)

        # additional network modifications
        self.num_global_features = self._proc_arg('num_global_features', default=None)

        self._verify_empty_hps()

    def _construct_model(self):

        # initialize dictionaries for holding indices of subnetworks
        self._layer_inds, self._tensor_inds = {}, {}

        # construct earlier parts of the model
        self._construct_inputs()
        self._construct_Phi()
        self._construct_latent()
        self._construct_F()

        # get output layers
        out_layer1 = Dense(self.num_output,activation='relu', name='intermediate_output')
        self._layers.extend([out_layer1])

        # append output tensors
        self._tensors.append(out_layer1(self.tensors[-1]))
        
        # construct second neural network
        self._construct_Phi2()
        self._construct_latent2()
        self._construct_F2()
        out_layer2 = Dense(self.output_dim, name=self._proc_name('output'))
        act_layer2 = _get_act_layer(self.output_act)
        self._layers.extend([out_layer2, act_layer2])

        # append output tensors
        self._tensors.append(out_layer2(self.tensors[-1]))
        self._tensors.append(act_layer2(self.tensors[-1]))
        # construct a new model
        self._model = Model(inputs=self.inputs, outputs=self.output)

        # compile model
        self._compile_model()

    @abstractmethod
    def _construct_inputs(self):
        pass

    def _construct_Phi(self):

        # get names
        names = [self._proc_name('tdist_{}'.format(i)) for i in range(len(self.Phi_sizes))]

        # determine begin inds
        layer_inds, tensor_inds = [len(self.layers)], [len(self.tensors)]

        # construct Phi
        Phi_layers, Phi_tensors = construct_distributed_dense(self.tensors[-1], self.Phi_sizes, 
                                                              acts=self.Phi_acts, 
                                                              k_inits=self.Phi_k_inits, 
                                                              names=names, 
                                                              l2_regs=self.Phi_l2_regs)
        
        # add layers and tensors to internal lists
        self._layers.extend(Phi_layers)
        self._tensors.extend(Phi_tensors)

        # determine end inds
        layer_inds.append(len(self.layers))
        tensor_inds.append(len(self.tensors))

        # store inds
        self._layer_inds['Phi'] = layer_inds
        self._tensor_inds['Phi'] = tensor_inds

    def _construct_latent(self):

        # determine begin inds
        layer_inds, tensor_inds = [len(self.layers)], [len(self.tensors)]

        # construct latent tensors
        latent_layers, latent_tensors = construct_latent(self.tensors[-1], self.weights, 
                                                         dropout=self.latent_dropout, 
                                                         name=self._proc_name('sum'))
        
        # add layers and tensors to internal lists
        self._layers.extend(latent_layers)
        self._tensors.extend(latent_tensors)

        # determine end inds
        layer_inds.append(len(self.layers))
        tensor_inds.append(len(self.tensors))

        # store inds
        self._layer_inds['latent'] = layer_inds
        self._tensor_inds['latent'] = tensor_inds

    def _construct_F(self):

        # get names
        names = [self._proc_name('dense_{}'.format(i)) for i in range(len(self.F_sizes))]

        # determine begin inds
        layer_inds, tensor_inds = [len(self.layers)], [len(self.tensors)]


        # construct F
        F_layers, F_tensors = construct_dense(self.tensors[-1], self.F_sizes,
                                              acts=self.F_acts, k_inits=self.F_k_inits, 
                                              dropouts=self.F_dropouts, names=names,
                                              l2_regs=self.F_l2_regs)

        # add layers and tensors to internal lists
        self._layers.extend(F_layers)
        self._tensors.extend(F_tensors)

        # determine end inds
        layer_inds.append(len(self.layers))
        tensor_inds.append(len(self.tensors))

        # store inds
        self._layer_inds['F'] = layer_inds
        self._tensor_inds['F'] = tensor_inds

    def _construct_Phi2(self):
        names = [self._proc_name('tdist2_{}'.format(i)) for i in range(len(self.Phi_sizes))]
        layer_inds, tensor_inds = [len(self.layers)], [len(self.tensors)]
        Phi_layers, Phi_tensors = construct_distributed_dense2(self.inputs[1],self.tensors[-1],self.num_output, self.Phi_sizes, 
                                                              acts=self.Phi_acts, 
                                                              k_inits=self.Phi_k_inits, 
                                                              names=names, 
                                                              l2_regs=self.Phi_l2_regs)
        self._layers.extend(Phi_layers)
        self._tensors.extend(Phi_tensors)
        layer_inds.append(len(self.layers))
        tensor_inds.append(len(self.tensors))
        self._layer_inds['Phi2'] = layer_inds
        self._tensor_inds['Phi2'] = tensor_inds
    
    def _construct_latent2(self):

        # determine begin inds
        layer_inds, tensor_inds = [len(self.layers)], [len(self.tensors)]

        # construct latent tensors
        latent_layers, latent_tensors = construct_latent(self.tensors[-1], self.weights, 
                                                         dropout=self.latent_dropout, 
                                                         name=self._proc_name('sum2'))
        
        # add layers and tensors to internal lists
        self._layers.extend(latent_layers)
        self._tensors.extend(latent_tensors)

        # determine end inds
        layer_inds.append(len(self.layers))
        tensor_inds.append(len(self.tensors))

        # store inds
        self._layer_inds['latent2'] = layer_inds
        self._tensor_inds['latent2'] = tensor_inds

    def _construct_F2(self):

        # get names
        names = [self._proc_name('dense2_{}'.format(i)) for i in range(len(self.F_sizes))]

        # determine begin inds
        layer_inds, tensor_inds = [len(self.layers)], [len(self.tensors)]


        # construct F
        F_layers, F_tensors = construct_dense(self.tensors[-1], self.F_sizes,
                                              acts=self.F_acts, k_inits=self.F_k_inits, 
                                              dropouts=self.F_dropouts, names=names,
                                              l2_regs=self.F_l2_regs)

        # add layers and tensors to internal lists
        self._layers.extend(F_layers)
        self._tensors.extend(F_tensors)

        # determine end inds
        layer_inds.append(len(self.layers))
        tensor_inds.append(len(self.tensors))

        # store inds
        self._layer_inds['F2'] = layer_inds
        self._tensor_inds['F2'] = tensor_inds

    @abstractproperty
    def inputs(self):
        pass

    @abstractproperty
    def weights(self):
        pass

    @property
    def Phi(self):
        r"""List of tensors corresponding to the layers in the $\Phi$ network."""

        begin, end = self._tensor_inds['Phi']
        if self._tensor_inds['Phi2'] == []:
            begin2,end2=[0,0]
        else:
            begin2, end2 = self._tensor_inds['Phi2']
        return self._tensors[begin:end] + self._tensors[begin2:end2]
    
    def latent2(self):
        begin, end= self._tensors_ind['latent2']
        return self._tensors[begin:end]

    @property
    def latent(self):
        """List of tensors corresponding to the summation layer in the
        network, including any dropout layer if present.
        """

        begin, end = self._tensor_inds['latent']
        return self._tensors[begin:end]

    @property
    def F(self):
        """List of tensors corresponding to the layers in the $F$ network."""

        begin, end = self._tensor_inds['F']
        if self._tensor_inds['F2'] == []:
            begin2,end2=[0,0]
        else:
            begin2, end2 = self._tensor_inds['F2']
        return self._tensors[begin:end] + self._tensors[begin2:end2]

    @property
    def output(self):
        """Output tensor for the model."""

        return self._tensors[-1]

    @property
    def layers(self):
        """List of all layers in the model."""

        return self._layers

    @property
    def tensors(self):
        """List of all tensors in the model. Order may be arbitrary given that
        not every model can be unambiguously flattened."""

        return self._tensors


class TwoStageEFN(Ensemble):
    def __new__(cls,*args,**kwargs):
        return super(TwoStageEFN,cls).__new__(cls)
    def _construct_inputs(self):
        self._inputs = construct_weighted_input(self.input_dim, 
                                           zs_name=self._proc_name('zs_input'), 
                                           phats_name=self._proc_name('phats_input'))
        self._layers, self._weights = construct_efn_weight_mask(self.inputs[0], 
                                                                mask_val=self.mask_val, 
                                                                name=self._proc_name('mask'))

        self._tensors = [self.weights] + self.inputs

    @property
    def inputs(self):
        return self._inputs
    @property
    def weights(self):
        return self._weights
    def eval_filters1(self, patch, n=100, prune=True):

        # determine patch of xy space to evaluate filters on
        if isinstance(patch, (float, int)):
            if patch > 0:
                xmin, ymin, xmax, ymax = -patch, -patch, patch, patch
            else:
                ValueError('patch must be positive when passing as a single number.')
        else:
            xmin, ymin, xmax, ymax = patch

        # determine number of pixels in each dimension
        if isinstance(n, int):
            nx = ny = n
        else:
            nx, ny = n

        # construct grid of inputs
        xs, ys = np.linspace(xmin, xmax, nx), np.linspace(ymin, ymax, ny)
        X, Y = np.meshgrid(xs, ys, indexing='ij')
        XY = np.asarray([X, Y]).reshape((1, 2, nx*ny)).transpose((0, 2, 1))

        # handle weirdness of Keras/tensorflow
        old_keras = (keras_version_tuple <= (2, 2, 5))
        s = self.Phi_sizes[-1] if len(self.Phi_sizes) else self.input_dim
        in_t, out_t = self.inputs[1], self._tensors[self._tensor_inds['latent'][0] - 1]

        # construct function
        kf = K.function([in_t] if old_keras else in_t, [out_t] if old_keras else out_t)

        # evaluate function
        Z = kf([XY] if old_keras else XY)[0].reshape(nx, ny, s).transpose((2, 0, 1))

        # prune filters that are off
        if prune:
            return X, Y, Z[[not (z == 0).all() for z in Z]]
        
        return X, Y, Z

    def eval_filters2(self, patch, n=100, prune=True):

        # determine patch of xy space to evaluate filters on
        if isinstance(patch, (float, int)):
            if patch > 0:
                xmin, ymin, xmax, ymax = -patch, -patch, patch, patch
            else:
                ValueError('patch must be positive when passing as a single number.')
        else:
            xmin, ymin, xmax, ymax = patch

        # determine number of pixels in each dimension
        if isinstance(n, int):
            nx = ny = n
        else:
            nx, ny = n

        # construct grid of inputs
        xs, ys = np.linspace(xmin, xmax, nx), np.linspace(ymin, ymax, ny)
        X, Y = np.meshgrid(xs, ys, indexing='ij')
        XY = np.asarray([X, Y]).reshape((1, 2, nx*ny)).transpose((0, 2, 1))

        # handle weirdness of Keras/tensorflow
        old_keras = (keras_version_tuple <= (2, 2, 5))
        s = self.Phi_sizes[-1] if len(self.Phi_sizes) else self.input_dim
        in_t, out_t = self.inputs[1], self._tensors[self._tensor_inds['latent2'][0] - 1]

        # construct function
        kf = K.function([in_t] if old_keras else in_t, [out_t] if old_keras else out_t)

        # evaluate function
        Z = kf([XY] if old_keras else XY)[0].reshape(nx, ny, s).transpose((2, 0, 1))

        # prune filters that are off
        if prune:
            return X, Y, Z[[not (z == 0).all() for z in Z]]
        
        return X, Y, Z


