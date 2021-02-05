import tensorflow as tf
import numpy as np

def gen_sine_full(frequency, n_samples,fs):
  blop = tf.range(0, n_samples, dtype='float32')/ fs
  PI = tf.constant(np.pi, dtype='float32')
  #blop2 = tf.math.sin *( 2 * np.pi * frequency *blop)
  blop = 2.0*PI*frequency*blop
  return tf.squeeze(tf.math.sin(blop))





def gen_sine(freq, length=512):
  return gen_sine_full(freq, length, 16000)





def gen_complex(freqs, weights, length=512):
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
    fuck_new_dim = tf.reshape(fuck, (*freqs_dim, length))
    fuck_compressed = tf.reduce_sum(fuck_new_dim, axis=1)

    max_fucks = tf.math.abs(fuck_compressed)
    max_fucks = tf.reshape(tf.reduce_max(max_fucks, axis=1), (-1,1))
    fuck_out = fuck_compressed / max_fucks

    return tf.squeeze(fuck_out)

def params_to_waveform(input_vec, length=512):

  freqs, power_array = get_freqs_powers(input_vec)

  #waveform = gen_complex(freqs, power_array)
  waveform = gen_complex_stub(freqs, power_array)

  # gen_complex(*tf.cast(get_freqs_powers(fund, n_harmonics, detune_hz, harmonic_power),dtype='float64'))
  waveform = tf.cast(waveform, dtype='float32')
  #waveform = tf.expand_dims(waveform, axis=0)
  waveform=tf.reshape(waveform, (-1, length))

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

def calc_rms_sines(amplitudes):
    '''
    calculates the rms of a sum of sinusoids:

    https://math.stackexchange.com/questions/974755/root-mean-square-of-sum-of-sinusoids-with-different-frequencies
     RMS (Root Mean Square) value of a waveform which consists of a sum of
     sinusoids of different frequencies, is equal to the square root of the
     sum of the squares of the RMS values of each sinusoid.


    :param amplitudes: list of amplitudes of sinusoids
    :return: rms
    '''

    RMS_from_peak = 0.707

    amplitudes = np.array(amplitudes)
    RMSs = RMS_from_peak*amplitudes
    RMSsquare = RMSs**2
    total_rms = np.sqrt(np.sum(RMSsquare))
    return total_rms

def get_freqs_powers(input_vec):  # no loops power array
    '''
    Takes 4 parameters as inputs as a vector (hundamental, number of harmonics, detning in hz, and harmonic power)
    and returns two arrays with frequencies (first array) and magnitudes (second) for pure sines that are generated.
    :param input_vec:
    :return:
    '''

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

#Lamdba function which concatenates the output.
vec_to_freqpower = tf.function(lambda x: tf.concat(get_freqs_powers(x), axis=1))



