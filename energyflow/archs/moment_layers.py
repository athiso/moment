from __future__ import absolute_import, division, print_function



from tensorflow.keras import __version__ as __keras_version__
from tensorflow.keras.layers import Layer
from tensorflow import stack,concat,unstack,split,ones,shape, convert_to_tensor, transpose, expand_dims, reshape, reduce_prod, boolean_mask, greater, reduce_sum, cast
from tensorflow.experimental.numpy import nonzero
import numpy as np
from itertools import chain, combinations

__all__ = [
    #custom layer
    'Moment', 'Cumulant', 'Cumulant_Terms'
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
        self.moment_matrices = []
        self.id_list = initial_id
        for z in range(self.order -1):
            matrix = np.array([[1 for x in list(flatten_list(itmd_id_list[i:]))] for i in range(len(itmd_id_list))])
            max_length = len(matrix[0])
            matrix = np.array([[0]* (max_length - len(row)) + row  for row in matrix], dtype='float32')
            matrix = convert_to_tensor(matrix)
            self.moment_matrices.append(transpose(matrix))
            itmd_id_list = [prepend(lst=itmd_id_list[i:],string=initial_id[i]) for i in range(latent_dim)]
            self.id_list.extend(list(flatten_list(itmd_id_list)))
        self.boolean_masks = [greater(reshape(m, shape=[reduce_prod(shape(m))]), 0) for m in self.moment_matrices]
        self.num_moments = [reduce_sum(cast(x, 'int32')) for x in self.boolean_masks]
    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'latent_dim': self.latent_dim,
            'order': self.order,
            'id_list': self.id_list,
            'moment_matrices' : self.moment_matrices,
            'boolean_masks' : self.boolean_masks,
            'num_moments' : self.num_moments
        })
        return config

    def call(self, inputs):
        latent_dim = self.latent_dim
        return_list = [inputs]
        dims = shape(inputs[:,:,0])
        for z in range(self.order -1):
            tmp = reshape(self.moment_matrices[z]*expand_dims(return_list[z], axis=-1)* expand_dims(inputs, axis=-2), shape=[dims[0],dims[1],len(self.boolean_masks[z])])
            masked_tmp = reshape(boolean_mask(tmp, self.boolean_masks[z], axis=2), shape=[dims[0],dims[1],self.num_moments[z]])
            return_list.append(masked_tmp)
        return_tensor = concat([expand_dims(ones(shape= dims), axis=-1)] + return_list,axis=-1)
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
        
class Cumulant_Terms(Layer):
    def __init__(self, latent_dim, order, **kwargs):
        super(Cumulant_Terms, self).__init__()
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
        self.terms.append(cumulant)
        self.term_names.append(full_id)
        if len(ids)!=1:
            pos = self.lookup_pos[len(ids)-1]
            for i in range(len(pos)):
                tmp = 1
                term_id = []
                for j in range(len(pos[i])):
                    partial_id=''
                    for num in pos[i][j]:
                        partial_id = partial_id + ids[num]
                    tmp = tmp*self.latent_outputs[self.id_list.index(partial_id)]
                    term_id.append(partial_id)
                term_id = sorted(term_id)
                if term_id not in self.term_names:
                    self.terms.append(tmp)
                    self.term_names.append(sorted(term_id))

    def call(self, inputs):
        inputs_list = split(inputs, num_or_size_splits=inputs.get_shape().as_list()[-1], axis=-1)
        zero_moment = inputs_list.pop(0)
        self.latent_outputs = unstack(stack(inputs_list)/zero_moment)
        self.terms = []
        self.term_names = []
        for i in range(len(self.latent_outputs)):
            self.compute_cumulant(full_id=self.id_list[i])

        cumulant_tensors = unstack(stack(self.terms)*zero_moment)
        output_tensor = concat([zero_moment]+cumulant_tensors,axis=-1)
        return output_tensor
        