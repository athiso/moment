from energyflow.archs.moment import PFN_moment
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
tf.config.run_functions_eagerly(True)

X = np.array([np.random.uniform(0,1,2) for i in range(10000)])
Y = []
for x in X:
    squared = [np.square(x_) for x_ in x]
    Y.append(np.sum(squared))
Y = np.array(Y)

X_train = X[:7000]
Y_train = Y[:7000]
X_val = X[7000:9000]
Y_val = Y[7000:9000]
X_test = X[9000:]
Y_test = Y[9000:]

model = PFN_moment(Phi_mapping_dim = [2,2],
                      output_dim=1, output_act='linear',
                      Phi_sizes=[5,5], Phi_acts='LeakyReLU',
                      F_sizes=[5,5,5], F_acts='LeakyReLU',
                      order=2, architecture_type='moment',
                      loss='mse',metrics='mse',  summary=True)
