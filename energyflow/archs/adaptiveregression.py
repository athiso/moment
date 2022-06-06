from __future__ import absolute_import, division, print_function

from abc import abstractmethod, abstractproperty

import numpy as np

import tensorflow.keras.backend as K
from tensorflow.keras import __version__ as __keras_version__
from tensorflow.keras.layers import Concatenate, Dense, Dot, Dropout, Input, Lambda, TimeDistributed, Layer, LocallyConnected1D
from tensorflow.keras.models import Model, clone_model
from tensorflow.keras.regularizers import l2, l1
from tensorflow import stack,concat,unstack,split,ones,shape, cond,expand_dims, squeeze, matmul, reshape
from energyflow.archs.archbase import NNBase, _get_act_layer
from energyflow.archs.dnn import construct_dense
from energyflow.utils import iter_or_rep
from itertools import chain, combinations
from tensorflow.math import square, sqrt, cos, sin, atan2
from energyflow.archs.quantized_layer import round_zero

__all__ = [

    # input constructor functions
    'construct_weighted_input', 'construct_input',

    # weight mask constructor functions
    'construct_efn_weight_mask', 'construct_pfn_weight_mask',

    # network consstructor functions
    'construct_distributed_dense', 'construct_latent', 'construct_dense', 

    # full model classes
    'PFN_moment','EFN_moment',
    #custom layer
    'Moment', 'Cumulant'
]

################################################################################
# Keras 2.2.5 fixes bug in 2.2.4 that affects our usage of the Dot layer
################################################################################

if __keras_version__.endswith('-tf'):
    __keras_version__ = __keras_version__[:-3]
keras_version_tuple = tuple(map(int, __keras_version__.split('.')))
DOT_AXIS = 0 if keras_version_tuple <= (2, 2, 4) else 1

#####################################################                                 #
## Helper Functions #######################
def flatten_tuple(d):
    for i in d:
        yield from [i] if not isinstance(i, tuple) else flatten_tuple(i)
def flatten_list(d):
    for i in d:
        yield from [i] if not isinstance(i, list) else flatten_list(i)

def powerset(iterable):
        s = list(iterable)
        return list(chain.from_iterable(combinations(s, r) for r in range(len(s)+1)))[1:]

def helper_pos(lst, length, combinations, comb_lens, order):
    if length==order:
        return [tuple(sorted(lst))]
    else:
        big_lst = tuple([])
        for z in range(len(comb_lens)):
            if set(combinations[z]).intersection(set(flatten_tuple(lst)))== set() and length+comb_lens[z]<=order:
                lst2 = lst + [combinations[z]]
                len2 = length+comb_lens[z]
                big_lst= big_lst+ tuple(sorted(helper_pos(lst=lst2,length=len2,combinations=combinations,comb_lens=comb_lens,order=order)))
        return set(tuple(sorted(big_lst)))
    
def compute_pos(combinations, comb_lens,order):
    return_list = []
    return_list.extend(helper_pos(lst=[],length=0,combinations=combinations,comb_lens=comb_lens,order=order))
    return list(set(tuple(sorted(return_list))))
def prepend(lst, string):
    if type(lst[0])==list:
        return [prepend(lst=x,string=string) for x in lst]
    else:      
        string += '{0}'
        lst = [string.format(i) for i in lst]
        return(lst)
###########
# Moment Layer
#############
class Moment(Layer):
    def __init__(self, latent_dim, order):
        super(Moment, self).__init__()
        self.latent_dim = latent_dim
        self.order = order
        initial_id = ['a{}'.format(i) for i in range(latent_dim)]
        itmd_id_list = initial_id
        self.id_list = initial_id
        for z in range(self.order -1):
            itmd_id_list = [prepend(lst=itmd_id_list[i:],string=initial_id[i]) for i in range(latent_dim)]
            self.id_list.extend(list(flatten_list(itmd_id_list)))
    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'latent_dim': self.latent_dim,
            'order': self.order,
            'id_list': self.id_list
        })
        return config

    def call(self, inputs):
        latent_dim = self.latent_dim
        L = stack(split(inputs, num_or_size_splits=latent_dim, axis =-1))
        L2          = L
        return_list = L
        for z in range(self.order -1):
            L2 = [L[i]*concat(L2[i:],axis=0) for i in range(latent_dim)]
            return_list = concat([return_list] +L2, axis=0)
        return_list = unstack(return_list)
        return_tensor = concat([ones(shape=shape(L[0]))] + return_list,axis=-1)
        return return_tensor
##########################################
# Cumulant Layer Class
###############################
class Cumulant(Layer):
    def __init__(self, latent_dim, order, **kwargs):
        super(Cumulant, self).__init__()
        self.latent_dim = latent_dim
        self.order = order
        self.lookup_pos = []
        for O in range(self.order):
            combinations= powerset([i for i in range(O+1)])
            combinations.remove(tuple([i for i in range(O+1)]))
            comb_lens = [len(combinations[i]) for i in range(len(combinations))]
            self.lookup_pos.append(compute_pos(combinations=combinations, comb_lens=comb_lens, order=O+1))
        initial_id = ['a{}'.format(i) for i in range(latent_dim)]
        itmd_id_list = initial_id
        self.id_list = initial_id
        for z in range(self.order -1):
            itmd_id_list = [prepend(lst=itmd_id_list[i:],string=initial_id[i]) for i in range(latent_dim)]
            self.id_list.extend(list(flatten_list(itmd_id_list)))

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'latent_dim': self.latent_dim,
            'order': self.order,
            'lookup_pos': self.lookup_pos,
            'id_list': self.id_list
        })
        return config

    def compute_cumulant(self, full_id):
        ids = ['a'+x for x in full_id.split('a') if x]
        cumulant = self.latent_outputs[self.id_list.index(full_id)]
        if len(ids)==1:
            return cumulant
        else:
            pos = self.lookup_pos[len(ids)-1]
            for i in range(len(pos)):
                tmp = 1
                for j in range(len(pos[i])):
                    partial_id=''
                    for num in pos[i][j]:
                        partial_id = partial_id + ids[num]
                    tmp = tmp*self.cumulant_list[self.id_list.index(partial_id)]
                cumulant = cumulant - tmp
            return cumulant
    def call(self, inputs):
        inputs_list = split(inputs, num_or_size_splits=inputs.get_shape().as_list()[-1], axis=-1)
        zero_moment = inputs_list.pop(0)
        self.latent_outputs = unstack(stack(inputs_list)/zero_moment)
        cumulant_tensors = []
        self.cumulant_list = []
        for i in range(len(self.latent_outputs)):
            cumulant = self.compute_cumulant(full_id=self.id_list[i])
            self.cumulant_list.append(cumulant)
            cumulant_tensors.append(cumulant)
        cumulant_tensors = unstack(stack(cumulant_tensors)*zero_moment)
        output_tensor = concat([zero_moment]+cumulant_tensors,axis=-1)
        return output_tensor
        


################################################################################
# INPUT FUNCTIONS
################################################################################

def construct_weighted_input(input_dim, zs_name=None, phats_name=None):

    # construct input tensors
    input_list=[Input(batch_shape=(None, None), name=zs_name)]
    for dim in input_dim:
        input_list.append(Input(batch_shape=(None, None, dim)))

    return input_list

def construct_input(input_dim, name=None):

    # construct input tensor
    input_list = []
    for dim in input_dim:
        input_list.append(Input(batch_shape= (None,None,dim)))
    return input_list


################################################################################
# WEIGHT MASK FUNCTIONS
################################################################################

def construct_efn_weight_mask(input_tensor, mask_val=0., name=None):
    """"""

    # define a function which maps the given mask_val to zero
    def efn_mask_func(X, mask_val=mask_val):
    
        # map mask_val to zero and leave everything else alone
        return X * K.cast(K.not_equal(X, mask_val), K.dtype(X))

    mask_layer = Lambda(efn_mask_func, name=name)

    # return as lists for consistency
    return [mask_layer], mask_layer(input_tensor)

def construct_pfn_weight_mask(input_tensor, mask_val=0., name=None):
    """"""

    # define a function which maps the given mask_val to zero
    def pfn_mask_func(X, mask_val=mask_val):

        # map mask_val to zero and return 1 elsewhere
        return K.cast(K.any(K.not_equal(X, mask_val), axis=-1), K.dtype(X))

    mask_layer = Lambda(pfn_mask_func, name=name)

    # return as lists for consistency
    return [mask_layer], mask_layer(input_tensor)


################################################################################
# NETWORK FUNCTIONS
################################################################################

def construct_distributed_dense(input_tensor, sizes, acts='LeakyReLU', k_inits='he_uniform', 
                                                                  names=None, l2_regs=0.):
    """"""

    # repeat options if singletons
    acts, k_inits, names = iter_or_rep(acts), iter_or_rep(k_inits), iter_or_rep(names)
    l2_regs = iter_or_rep(l2_regs)
    
    # list of tensors
    layers, tensors = [], [input_tensor]

    # iterate over specified layers
    for s, act, k_init, name, l2_reg in zip(sizes, acts, k_inits, names, l2_regs):
        
        # define a dense layer that will be applied through time distributed
        kwargs = {} 
        if l2_reg > 0.:
            kwargs.update({'kernel_regularizer': l2(l2_reg), 'bias_regularizer': l2(l2_reg)})
        d_layer = Dense(s, kernel_initializer=k_init, **kwargs)

        # get layers and append them to list
        tdist_layer = TimeDistributed(d_layer, name=name)
        act_layer = _get_act_layer(act)
        layers.extend([tdist_layer, act_layer])

        # get tensors and append them to list
        tensors.append(tdist_layer(tensors[-1]))
        tensors.append(act_layer(tensors[-1]))

    return layers, tensors[1:]
def construct_locallyconnected_dense(num_var, input_tensor, sizes, acts='LeakyReLU', k_inits='he_uniform', names=None, l2_regs=0, use_l1=False):
    acts, k_inits, names = iter_or_rep(acts), iter_or_rep(k_inits), iter_or_rep(names)
    l2_regs = iter_or_rep(l2_regs)
    layers, tensors = [], [input_tensor]
    #steps = input_tensor.get_shape().as_list()[1]

    for i ,(s, act, k_init, name, l2_reg) in enumerate(zip(sizes, acts, k_inits, names, l2_regs)):
        # define a dense layer that will be applied through time distributed
        kwargs = {} 
        if l2_reg > 0.:
            kwargs.update({'kernel_regularizer': l2(l2_reg), 'bias_regularizer': l2(l2_reg)})
        if i==0:
            if use_l1==True:
                d_layer = LocallyConnected1D(s, num_var, strides=num_var, kernel_initializer=k_init, name=name,kernel_regularizer=l1(0.0001))
            else:
                d_layer = LocallyConnected1D(s, num_var, strides=num_var, kernel_initializer=k_init, name=name, **kwargs)
        else:
            d_layer = LocallyConnected1D(s, 1, kernel_initializer=k_init, name=name, **kwargs)

        # get layers and append them to list
        act_layer = _get_act_layer(act)
        layers.extend([d_layer, act_layer])

        # get tensors and append them to list
        tensors.append(d_layer(tensors[-1]))
        tensors.append(act_layer(tensors[-1]))
    return layers, tensors[1:]

def construct_latent(input_tensor, weight_tensor, dropout=0., name=None):
    """"""

    # lists of layers and tensors
    layers = [Dot(DOT_AXIS, name=name)]
    tensors = [layers[-1]([weight_tensor, input_tensor])]

    # apply dropout if specified
    if dropout > 0.:
        dr_name = None if name is None else '{}_dropout'.format(name)
        layers.append(Dropout(dropout, name=dr_name))
        tensors.append(layers[-1](tensors[-1]))

    return layers, tensors


################################################################################
# SymmetricPerParticleNN - Base class for EFN-like models
################################################################################

class SymmetricPerParticleNN(NNBase):

    # EFN(*args, **kwargs)
    def _process_hps(self):
        r"""See [`ArchBase`](#archbase) for how to pass in hyperparameters as
        well as defaults common to all EnergyFlow neural network models.

        **Required EFN Hyperparameters**

        - **input_dim** : _int_
            - The number of features for each particle.
        - **Phi_sizes** (formerly `ppm_sizes`) : {_tuple_, _list_} of _int_
            - The sizes of the dense layers in the per-particle frontend
            module $\Phi$. The last element will be the number of latent 
            observables that the model defines.
        - **F_sizes** (formerly `dense_sizes`) : {_tuple_, _list_} of _int_
            - The sizes of the dense layers in the backend module $F$.

        **Default EFN Hyperparameters**

        - **Phi_acts**=`'relu'` (formerly `ppm_acts`) : {_tuple_, _list_} of
        _str_ or Keras activation
            - Activation functions(s) for the dense layers in the 
            per-particle frontend module $\Phi$. A single string or activation
            layer will apply the same activation to all layers. Keras advanced
            activation layers are also accepted, either as strings (which use
            the default arguments) or as Keras `Layer` instances. If passing a
            single `Layer` instance, be aware that this layer will be used for
            all activations and may introduce weight sharing (such as with 
            `PReLU`); it is recommended in this case to pass as many activations
            as there are layers in the model. See the [Keras activations 
            docs](https://keras.io/activations/) for more detail.
        - **F_acts**=`'relu'` (formerly `dense_acts`) : {_tuple_, _list_} of
        _str_ or Keras activation
            - Activation functions(s) for the dense layers in the 
            backend module $F$. A single string or activation layer will apply
            the same activation to all layers.
        - **Phi_k_inits**=`'he_uniform'` (formerly `ppm_k_inits`) : {_tuple_,
        _list_} of _str_ or Keras initializer
            - Kernel initializers for the dense layers in the per-particle
            frontend module $\Phi$. A single string will apply the same
            initializer to all layers. See the [Keras initializer docs](https:
            //keras.io/initializers/) for more detail.
        - **F_k_inits**=`'he_uniform'` (formerly `dense_k_inits`) : {_tuple_,
        _list_} of _str_ or Keras initializer
            - Kernel initializers for the dense layers in the backend 
            module $F$. A single string will apply the same initializer 
            to all layers.
        - **latent_dropout**=`0` : _float_
            - Dropout rates for the summation layer that defines the
            value of the latent observables on the inputs. See the [Keras
            Dropout layer](https://keras.io/layers/core/#dropout) for more 
            detail.
        - **F_dropouts**=`0` (formerly `dense_dropouts`) : {_tuple_, _list_}
        of _float_
            - Dropout rates for the dense layers in the backend module $F$. 
            A single float will apply the same dropout rate to all dense layers.
        - **Phi_l2_regs**=`0` : {_tuple_, _list_} of _float_
            - $L_2$-regulatization strength for both the weights and biases
            of the layers in the $\Phi$ network. A single float will apply the
            same $L_2$-regulatization to all layers.
        - **F_l2_regs**=`0` : {_tuple_, _list_} of _float_
            - $L_2$-regulatization strength for both the weights and biases
            of the layers in the $F$ network. A single float will apply the
            same $L_2$-regulatization to all layers.
        - **mask_val**=`0` : _float_
            - The value for which particles with all features set equal to
            this value will be ignored. The [Keras Masking layer](https://
            keras.io/layers/core/#masking) appears to have issues masking
            the biases of a network, so this has been implemented in a
            custom (and correct) manner since version `0.12.0`.
        - **num_global_features**=`None` : _int_
            - Number of additional features to be concatenated with the latent
            space observables to form the input to F. If not `None`, then the
            features are to be provided at the end of the list of inputs.
        """

        # process generic NN hps
        super(SymmetricPerParticleNN, self)._process_hps()

        # required hyperparameters
        self.input_dim = self._proc_arg('input_dim')
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
        self.order = self._proc_arg('order')
        self.cumulant =self._proc_arg('cumulant', default=False)
        self.num_F = self._proc_arg('num_F', default=1)
        self.rweighted = self._proc_arg('rweighted', default=False)
        self.use_moment_info = self._proc_arg('use_moment_info', default=False)


        self._verify_empty_hps()

    def _construct_model(self):

        # initialize dictionaries for holding indices of subnetworks
        self._layer_inds, self._tensor_inds = {}, {}

        # construct earlier parts of the model
        self._construct_inputs()
        self._construct_Phi(tensor_list= self.phi_inputs)
        latent_dim = self.tensors[-1].get_shape().as_list()[-1]
        self._construct_Phi_moment(latent_dim, self.order)
        self._construct_latent()
        moments = self.tensors[-1]
        if self.cumulant ==True:
            self._construct_Phi_cumulant(latent_dim, self.order)
        else:
            pass
        
        expanded = expand_dims(self.tensors[-1], axis=1)
        dup_tensor = concat([expanded for i in range(self.num_F)], axis=1)
        self._tensors.append(dup_tensor)
        
        self._construct_F()

        if self.use_moment_info == True:
            observables = moments
        else:
            observables= moments[:,1:latent_dim+1]
        group_layer = Dense(observables.get_shape().as_list()[1], name='weights')
        matrix = group_layer(self.tensors[-1]) 

        self._tensors.append(matmul(matrix, expand_dims(observables, axis=-1), transpose_a=False))

        self._construct_localF(num_var=1)
        # get output layers
        if self.num_F ==1:
            self._tensors.append(squeeze(self.tensors[-1], axis=1))
        else:
            out_layer = LocallyConnected1D(self.output_dim, 1, name=self._proc_name('local_output'))
            act_layer = _get_act_layer(self.output_act)
            self._layers.extend([out_layer, act_layer])

            # append output tensors
            self._tensors.append(out_layer(self.tensors[-1]))
            self._tensors.append(act_layer(self.tensors[-1]))
            self._tensors.append(squeeze(self.tensors[-1], axis=-1))

        out_layer = Dense(self.output_dim,  name=self._proc_name('output'))
        act_layer = _get_act_layer(self.output_act)
        self._layers.extend([out_layer, act_layer])

        # append output tensors
        
        self._tensors.append(out_layer(self.tensors[-1]))
        self._tensors.append(act_layer(self.tensors[-1]))

        # construct a new model
        self._model = Model(inputs=self.inputs, outputs=self.output)
        # compile model
        self._compile_model()

    @abstractmethod
    def _construct_inputs(self):
        pass

    def _construct_Phi(self, tensor_list):

        # get names
        

        # determine begin inds
        layer_inds, tensor_inds = [len(self.layers)], [len(self.tensors)]

        # construct Phis
        phi_list = []

        for k, tensor in enumerate(tensor_list):
            names = [self._proc_name('tdist{}_{}'.format(k, i)) for i in range(len(self.Phi_sizes))]
            if type(self.Phi_sizes[-1]) == int:
                sizes = self.Phi_sizes
            else:
                sizes = self.Phi_sizes[:-1] + [self.Phi_sizes[-1][k]]
            Phi_layers, Phi_tensors = construct_distributed_dense(tensor, sizes, 
                                                                acts=self.Phi_acts, 
                                                                k_inits=self.Phi_k_inits, 
                                                                names=names, 
                                                                l2_regs=self.Phi_l2_regs)
            phi_list.append(Phi_tensors[-1])
            self._layers.extend(Phi_layers)
            self._tensors.extend(Phi_tensors)
        if self.rweighted == True:
            combined_phi = concat(phi_list[:-1] + [phi_list[-1]*self.inputs[-2]], axis=-1, name='combined_phi')
        else:
            combined_phi = concat(phi_list, axis=-1, name='combined_phi')
        self._tensors.append(combined_phi)
        

        # determine end inds
        layer_inds.append(len(self.layers))
        tensor_inds.append(len(self.tensors))

        # store inds
        self._layer_inds['Phi'] = layer_inds
        self._tensor_inds['Phi'] = tensor_inds
        

        # determine end inds
        layer_inds.append(len(self.layers))
        tensor_inds.append(len(self.tensors))

        # store inds
        self._layer_inds['Phi'] = layer_inds
        self._tensor_inds['Phi'] = tensor_inds
    
    def _construct_Phi_moment(self, latent_dim, order):
        moment_layer = Moment(latent_dim=latent_dim, order=order)
        moment_tensor = moment_layer(self.tensors[-1])  
        self._layers.append(moment_layer)
        self._tensors.append(moment_tensor)
    
    def _construct_Phi_cumulant(self, latent_dim, order):
        cumulant_layer = Cumulant(latent_dim=latent_dim, order=order)
        cumulant_tensor = cumulant_layer(self.tensors[-1])
        self._layers.append(cumulant_layer)
        self._tensors.append(cumulant_tensor)

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
        names = [self._proc_name('weightdense_{}'.format(i)) for i in range(len(self.F_sizes))]

        # determine begin inds
        layer_inds, tensor_inds = [len(self.layers)], [len(self.tensors)]


        # construct F
        F_layers, F_tensors = construct_locallyconnected_dense(1, self.tensors[-1], [100,100,100],
                                              acts=self.F_acts, k_inits=self.F_k_inits, 
                                              names=names,
                                              l2_regs=self.F_l2_regs, use_l1 =True)

        # add layers and tensors to internal lists
        self._layers.extend(F_layers)
        self._tensors.extend(F_tensors)

        # determine end inds
        layer_inds.append(len(self.layers))
        tensor_inds.append(len(self.tensors))

        # store inds
        self._layer_inds['F'] = layer_inds
        self._tensor_inds['F'] = tensor_inds

    def _construct_localF(self, num_var=1):

        # get names
        names = [self._proc_name('localdense_{}'.format(i)) for i in range(len(self.F_sizes))]

        # determine begin inds
        layer_inds, tensor_inds = [len(self.layers)], [len(self.tensors)]


        # construct F
        F_layers, F_tensors = construct_locallyconnected_dense(num_var, self.tensors[-1], self.F_sizes,
                                              acts=self.F_acts, k_inits=self.F_k_inits, 
                                              names=names,
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
    

    @abstractproperty
    def inputs(self):
        pass
    @abstractproperty
    def phi_inputs(self):
        pass
    @abstractproperty
    def weights(self):
        pass

    @property
    def Phi(self):
        r"""List of tensors corresponding to the layers in the $\Phi$ network."""

        begin, end = self._tensor_inds['Phi']
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
        return self._tensors[begin:end]

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
    



class PFN_moment(SymmetricPerParticleNN):

    """Particle Flow Network (PFN) architecture. Accepts the same 
    hyperparameters as the [`EFN`](#EFN)."""

    # customize which PFN instance is created
    def __new__(cls, *args, **kwargs):
        return super(PFN_moment,cls).__new__(cls)

    # PFN(*args, **kwargs)
    def _construct_inputs(self):
        """""" # need this for autogen docs

        # construct input tensor
        self._inputs = construct_input(self.input_dim, name=self._proc_name('input'))
        self._phi_inputs = self.inputs

        # construct weight tensor and begin list of layers
        self._layers, self._weights = construct_pfn_weight_mask(self.inputs[0], 
                                                                mask_val=self.mask_val, 
                                                                name=self._proc_name('mask'))

        # begin list of tensors with the inputs
        self._tensors = [self.weights] + self.inputs

    def new_model(self, F_sizes):
        nodes= F_sizes
        act = 'LeakyReLU'
        if self.cumulant == False:
            tensor_list = [self.latent[-1]]
        elif self.cumulant== True:
            tensor_list = [self.cumulant_latent]
        for i in range(len(nodes)):
            layer = Dense(nodes[i], name='F{}'.format(i), activation=act)
            tensor_list.append(layer(tensor_list[-1]))
        output_layer = Dense(self.output_dim, name='output_layer', activation=self.output_act)
        outputs = output_layer(tensor_list[-1])
        model = Model(inputs=self.inputs, outputs=outputs)
        model_copy = clone_model(model)
        num_layers= len(nodes)+1
        for l in model_copy.layers[:-num_layers]:
            l.trainable=False
        model_copy.compile(optimizer='adam',loss=self.loss,metrics='accuracy')
        return model_copy

    def remove_F(self, phi_trainable=True):
        if self.cumulant == False:
            outputs = self.latent[-1]
        elif self.cumulant== True:
            outputs = self.cumulant_latent
        model = Model(inputs=self.inputs, outputs=outputs)
        model_copy = clone_model(model)
        if phi_trainable == False:
            for l in model_copy.layers:
                l.trainable=False
        model_copy.compile(optimizer='adam',loss=self.loss,metrics='accuracy')
        return model_copy


    @property
    def inputs(self):
        """List of input tensors to the model. PFNs have one input tensor
        corresponding to the `ps` input.
        """

        return self._inputs
    @property
    def phi_inputs(self):
        return self._phi_inputs
    @property
    def weights(self):
        """Weight tensor for the model. A weight of `0` is assigned to any
        particle which has all features equal to `mask_val`, and `1` is
        assigned otherwise.
        """

        return self._weights


class EFN_moment(SymmetricPerParticleNN):

    """Energy Flow Network (EFN) architecture."""

    # customize which EFN instance is created
    def __new__(cls, *args, **kwargs):
        return super(EFN_moment,cls).__new__(cls)

    def _construct_inputs(self):

        # construct input tensors
        self._inputs = construct_weighted_input(self.input_dim, 
                                           zs_name=self._proc_name('zs_input'), 
                                           phats_name=self._proc_name('phats_input'))
        self._phi_inputs = self._inputs[1:]

        # construct weight tensor and begin list of layers
        self._layers, self._weights = construct_efn_weight_mask(self.inputs[0], 
                                                                mask_val=self.mask_val, 
                                                                name=self._proc_name('mask'))

        # begin list of tensors with the inputs
        self._tensors = [self.weights] + self.inputs

    @property
    def inputs(self):
        """List of input tensors to the model. EFNs have two input tensors:
        `inputs[0]` corresponds to the `zs` input and `inputs[1]` corresponds
        to the `phats` input.
        """

        return self._inputs
    @property
    def phi_inputs(self):
        return self._phi_inputs

    @property
    def weights(self):
        """Weight tensor for the model. This is the `zs` input where entries
        equal to `mask_val` have been set to zero.
        """

        return self._weights

    # eval_filters(patch, n=100, prune=True)
    def eval_filters(self, patch, n=100, prune=True):
        """Evaluates the latent space filters of this model on a patch of the 
        two-dimensional geometric input space.

        **Arguments**

        - **patch** : {_tuple_, _list_} of _float_
            - Specifies the patch of the geometric input space to be evaluated.
            A list of length 4 is interpretted as `[xmin, ymin, xmax, ymax]`.
            Passing a single float `R` is equivalent to `[-R,-R,R,R]`.
        - **n** : {_tuple_, _list_} of _int_
            - The number of grid points on which to evaluate the filters. A list 
            of length 2 is interpretted as `[nx, ny]` where `nx` is the number of
            points along the x (or first) dimension and `ny` is the number of points
            along the y (or second) dimension.
        - **prune** : _bool_
            - Whether to remove filters that are all zero (which happens sometimes
            due to dying ReLUs).

        **Returns**

        - (_numpy.ndarray_, _numpy.ndarray_, _numpy.ndarray_)
            - Returns three arrays, `(X, Y, Z)`, where `X` and `Y` have shape `(nx, ny)` 
            and are arrays of the values of the geometric inputs in the specified patch.
            `Z` has shape `(num_filters, nx, ny)` and is the value of the different
            filters at each point.
        """

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
        in_t, out_t = self.inputs[1], self.Phi[-1]

        # construct function
        kf = K.function([in_t] if old_keras else in_t, [out_t] if old_keras else out_t)

        # evaluate function
        Z = kf([XY] if old_keras else XY)[0].reshape(nx, ny, s).transpose((2, 0, 1))

        # prune filters that are off
        if prune:
            return X, Y, Z[[not (z == 0).all() for z in Z]]
        
        return X, Y, Z