import tensorflow as tf
import numpy as np
from data_handling import denormalise_batch

y_mean= np.array([1.5245759e+03, 2.2862642e+03, 2.2864233e+03, 1.5244318e+03,
       0.0000000e+00, 1.0000000e+00, 3.7491098e-01, 2.5000253e-01,
       1.2500882e-01, 0.0000000e+00])

y_norm = np.array([8.5161224e+02, 1.9803693e+03, 2.9146074e+03, 3.1424453e+03,
       1.0000000e+00, 1.0000000e+00, 3.3071095e-01, 3.2275078e-01,
       2.6020426e-01, 1.0000000e+00])

decoder = tf.keras.models.load_model('../models/decoder')

prediction = decoder.predict([[-3,3]])

params_out = denormalise_batch(prediction, y_norm, y_mean)

