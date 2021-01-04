import tensorflow as tf
import numpy as np

def gen_sine_full(frequency, n_samples,fs):
  blop = tf.range(0, n_samples, dtype='float32')/ fs
  PI = tf.constant(np.pi, dtype='float32')
  #blop2 = tf.math.sin *( 2 * np.pi * frequency *blop)
  blop = 2.0*PI*frequency*blop
  return tf.squeeze(tf.math.sin(blop))





def gen_sine(freq):
  return gen_sine_full(freq, 512, 16000)





def gen_complex(freqs, weights):
    '''

    :param freqs:
    :param weights:
    :return:
    '''
    freqs = tf.cast(freqs, dtype='float32')
    weights = tf.cast(weights, dtype='float32')

    freqs_dim = freqs.shape
    freqs_reshaped = tf.reshape(freqs, (1, -1))
    weights_reshaped = tf.reshape(weights, (-1, 1))
    sins = tf.map_fn(gen_sine, tf.transpose(freqs_reshaped)) #generate the sins from the freqs
    #sins_new_dim = tf.reshape(sins, (*freqs_dim, -1))

    fuck = weights_reshaped * sins
    #fuck_new_dim = tf.reshape(fuck, (*freqs_dim, -1))
    fuck_new_dim = tf.reshape(fuck, (*freqs_dim, 512))
    fuck_compressed = tf.reduce_sum(fuck_new_dim, axis=1)

    max_fucks = tf.math.abs(fuck_compressed)
    max_fucks = tf.reshape(tf.reduce_max(max_fucks, axis=1), (-1,1))
    fuck_out = fuck_compressed / max_fucks

    return tf.squeeze(fuck_out)

def params_to_waveform(input_vec):

  freqs, power_array = get_freqs_powers(input_vec)

  #waveform = gen_complex(freqs, power_array)
  waveform = gen_complex_stub(freqs, power_array)

  # gen_complex(*tf.cast(get_freqs_powers(fund, n_harmonics, detune_hz, harmonic_power),dtype='float64'))
  waveform = tf.cast(waveform, dtype='float32')
  #waveform = tf.expand_dims(waveform, axis=0)
  waveform=tf.reshape(waveform, (-1, 512))

  return waveform


def gen_complex_stub(freqs, power_array):

    n_samples=512
    fs=16000

    waveform = tf.zeros(shape=(n_samples))

    boopy = tf.ones(shape=(n_samples))

    blop = tf.range(0, n_samples, dtype='float32') / fs
    PI = tf.constant(np.pi, dtype='float32')
    # blop2 = tf.math.sin *( 2 * np.pi * frequency *blop)
    #blop = 2.0 * PI * frequency * blop



    bloopy =tf.tensordot(freqs, blop, axes=0) #this is of right dimension.
    bleppy = 2.0 * PI * bloopy
    bleepy =tf.math.sin(bleppy)

    #now can we map the gen sin function

    waveform = tf.reduce_sum(bleepy, axis=1)

    return waveform




def get_freqs_powers(input_vec):  # no loops power array

    cast_type='float32'

    input_vec=tf.cast(input_vec, dtype=cast_type)

    #input_vec = (input_vec)
    num_inputs=tf.shape(input_vec)[0]
    num_elements = 5

    fund = input_vec[:, 0]
    n_harmonics = tf.math.round(input_vec[:, 1])
    detune_hz = input_vec[:, 2]
    harmonic_power = input_vec[:, 3]

    fund_mask = tf.sequence_mask(tf.ones(num_inputs), num_elements)
    all_harm_mask=tf.logical_not(fund_mask)
    harm_mask = tf.sequence_mask(n_harmonics+1, num_elements)

    # cast boolean masks to float
    fund_mask = tf.cast(fund_mask, dtype=cast_type)
    all_harm_mask = tf.cast(all_harm_mask, dtype=cast_type)
    harm_mask = tf.cast(harm_mask, dtype=cast_type)


    power_array = fund_mask + (tf.expand_dims(harmonic_power, axis=1) * all_harm_mask * harm_mask)
    detune_array = tf.expand_dims(detune_hz, axis=1) * all_harm_mask * harm_mask

    harmonic_series = tf.range(num_elements, dtype=cast_type) + 1
    #freqs = (fund * harmonic_series + detune_array) * harm_mask

    freqs = (tf.multiply(tf.expand_dims(harmonic_series, axis=0), tf.expand_dims(fund, axis=1)) + detune_array) * harm_mask

    return freqs, power_array

  # # takes a vector of 4 values -
  # # fund = input_vec[0,:]
  # # n_harmonics = input_vec[1,:]
  # # detune_hz = input_vec[2,:]
  # # harmonic_power = input_vec[3,:]
  #   def get_powers_single_vec(input_vec):
  #     fund = input_vec[:,0]
  #     n_harmonics = input_vec[:,1]
  #     detune_hz = input_vec[:,2]
  #     harmonic_power = input_vec[:,3]
  #
  #     num_elements = 5
  #
  #     fund_mask = tf.cast(tf.sequence_mask(1, num_elements), dtype='float64')
  #     all_harm_mask = tf.cast(tf.logical_not(tf.cast(fund_mask, dtype='bool')), dtype='float64')
  #
  #     # all_harm_mask = tf.constant([0,1,1,1, 1], dtype='float64')
  #     harm_mask = tf.cast(tf.sequence_mask(n_harmonics + 1, num_elements),
  #                         dtype='float64')  # sets the non-active harminics to 0
  #
  #     power_array = tf.cast((fund_mask + harmonic_power * all_harm_mask * harm_mask), dtype='float64')
  #     detune_array = detune_hz * all_harm_mask * harm_mask
  #
  #     harmonic_series = tf.cast((tf.range(num_elements) + 1), dtype='float64')
  #     freqs = tf.cast(((fund * harmonic_series + detune_array) * harm_mask), dtype='float64')

  #     return (freqs, power_array)

