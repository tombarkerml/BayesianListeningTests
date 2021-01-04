import tensorflow as tf
import numpy as np
from soundgen_tf import gen_sine_full, gen_complex, get_freqs_powers, params_to_waveform


nsamples=512
fs = 16000
frequency = 100


#@tf.function


#jerky = gen_sine(frequency, nsamples, fs)

#@tf.function



#hooby = tf.reduce_sum(tf.map_fn(gen_sine, np.array([100.0,200.0, 300.0])), axis=0)

#@tf.function


#blop = tf.map_fn(gen_sine, np.array([100.0,200.0, 300.0]))




#gen_sine(tf.tensor([1,2,4]))
#@tf.function
# def get_freqs_powers(input_vec):  # no loops power array
#
#   #input_vec = tf.transpose(tf.cast(input_vec, dtype='float64'))
#
#   # takes a vector of 4 values -
#   # fund = input_vec[0,:]
#   # n_harmonics = input_vec[1,:]
#   # detune_hz = input_vec[2,:]
#   # harmonic_power = input_vec[3,:]
#     def get_powers_single_vec(input_vec):
#       fund = input_vec[0]
#       n_harmonics = input_vec[1]
#       detune_hz = input_vec[2]
#       harmonic_power = input_vec[3]
#
#       num_elements = 5
#
#       fund_mask = tf.cast(tf.sequence_mask(1, num_elements), dtype='float64')
#       all_harm_mask = tf.cast(tf.logical_not(tf.cast(fund_mask, dtype='bool')), dtype='float64')
#
#       # all_harm_mask = tf.constant([0,1,1,1, 1], dtype='float64')
#       harm_mask = tf.cast(tf.sequence_mask(n_harmonics + 1, num_elements),
#                           dtype='float64')  # sets the non-active harminics to 0
#
#       power_array = tf.cast((fund_mask + harmonic_power * all_harm_mask * harm_mask), dtype='float64')
#       detune_array = detune_hz * all_harm_mask * harm_mask
#
#       harmonic_series = tf.cast((tf.range(num_elements) + 1), dtype='float64')
#       freqs = tf.cast(((fund * harmonic_series + detune_array) * harm_mask), dtype='float64')
#
#       return (freqs, power_array)
#
#     bload = tf.map_fn(lambda x: get_powers_single_vec(x), input_vec)


#@tf.function



import matplotlib.pyplot as plt

fundamental = 100
n_harmonics = 2
detune_hz = 0
harmonic_weights = 0.5


blippy = params_to_waveform([[fundamental, n_harmonics, detune_hz, harmonic_weights]])

blippy2 = params_to_waveform([[fundamental, n_harmonics, detune_hz, harmonic_weights],[2*fundamental, n_harmonics, detune_hz, harmonic_weights]])

#blippy = params_to_waveform(np.array([[1000.0, 1.0, 0.0, 1.0], [1000.0, 1.0, 0.0, 1.0]]))
plt.plot(tf.transpose(blippy))

inputs = tf.keras.Input(shape=(4,), dtype='float32')
output= tf.keras.layers.Lambda(params_to_waveform, trainable=False) (inputs)
model = tf.keras.Model(inputs=inputs, outputs=output)


v=500
