<h1>energyflow.archs.moment.EFN_moment(*args, **kwargs)</h1>

<h2>Required arguments</h2>  

---Compile Options---  

loss = '' : str  
metric = [''] : list of str  
optimizer = '' : str  
OR  
compile_opts = {} : dictionary  
  default keras compile options passed as keyword arguments, these also include loss, metric, and optimizer above.    
  
---Network Hyperparameters---  

Phi_mapping_dim = [] : a pair of numbers OR list of pairs of numbers  
&emsp;  Pairs of [input dimension, latent dimension] for the Phi functions  
Phi_sizes = [0] : list of numbers  
&emsp;   Width of the Phi function layers up to but not including the latent layer  
F_sizes = [0] : list of numbers  
&emsp;   Width of the F function layers up to but not including the output layer  
output_dim = 0 : number  
&emsp;   Output dimension  
order = 0: number  
 &emsp;  The order of the moments or cumulants to construct  
  
<h2> Default Hyperparameters </h2>

---Architecture--- 

architecture_type = 'moment' : 'moment' or 'cumulant' or 'mixed'  
&emsp;   Determines the type of neural network architecture  
bias = True : boolean  
 &emsp;  whether to use bias  
rweighted = False : boolean  
   
---Activations---   

Phi_acts = 'relu' : {tuple, list} of str or Keras activation  
 &emsp;  Activation functions(s) for the dense layers in the Phi functions. A single string or activation layer will apply the same activation to all layers.  
F_acts = 'relu' : {tuple, list} of str or Keras activation string   
 &emsp;  Activation functions(s) for the dense layers in the F function . A single string or activation layer will apply the same activation to all layers.  
  
---Initializations---   

Phi_k_inits='he_uniform' (formerly ppm_k_inits) : {tuple, list} of str or Keras initializer  
 &emsp;  Kernel initializers for the dense layers in the per-particle frontend module . A single string will apply the same initializer to all layers. See the Keras initializer docs for more detail.  
F_k_inits='he_uniform' (formerly dense_k_inits) : {tuple, list} of str or Keras initializer  
 &emsp;  Kernel initializers for the dense layers in the backend module . A single string will apply the same initializer to all layers.  
  
---Dropouts---  

latent_dropout=0 : float  
 &emsp;  Dropout rates for the summation layer that defines the value of the latent observables on the inputs. See the Keras Dropout layer for more detail.  
F_dropouts=0 (formerly dense_dropouts) : {tuple, list} of float  
&emsp;   Dropout rates for the dense layers in the backend module . A single float will apply the same dropout rate to all dense layers.  
  
---Regularizations---  

Phi_l2_regs=0 : {tuple, list} of float  
&emsp;   regulatization strength for both the weights and biases of the layers in the  network. A single float will apply the same -regulatization to all layers.  
F_l2_regs=0 : {tuple, list} of float  
&emsp;   regulatization strength for both the weights and biases of the layers in the  network. A single float will apply the same -regulatization to all layers.  
  
---Additional Hyperparameters---  

mask_val=0 : float  
 &emsp;  The value for which particles with all features set equal to this value will be ignored. The Keras Masking layer appears to have issues masking the biases of a network, so this has been implemented in a custom (and correct) manner since version 0.12.0.  


