''' Compare navigating the embedding space with sklearn optimizers vs the high-dimension space'''

import tensorflow as tf
import numpy as np
from soundgen_tf import params_to_waveform, gen_sine_full, get_freqs_powers, vec_to_freqpower
from data_handling import denormalise_batch
import matplotlib
import sound_metrics

from skopt.space import Real, Integer
from skopt.utils import use_named_args
from skopt import gp_minimize

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

import simpleaudio as sa

y_mean = np.array([1.5245759e+03, 2.2862642e+03, 2.2864233e+03, 1.5244318e+03,
                   0.0000000e+00, 1.0000000e+00, 3.7491098e-01, 2.5000253e-01,
                   1.2500882e-01, 0.0000000e+00])

y_norm = np.array([8.5161224e+02, 1.9803693e+03, 2.9146074e+03, 3.1424453e+03,
                   1.0000000e+00, 1.0000000e+00, 3.3071095e-01, 3.2275078e-01,
                   2.6020426e-01, 1.0000000e+00])

x_batch_mean = np.array([1.52481007e+03, 1.50019014e+00, 3.25621367e-02, 5.00145407e-01])
x_batch_std = np.array([8.51702500e+02, 1.11811500e+00, 5.72985211e+01, 2.88592207e-01])

# decoder = tf.keras.models.load_model('models/decoder')
#decoder = tf.keras.models.load_model('models/decoder_4_params_fitted')
decoder = tf.keras.models.load_model('models/VAE_sin_parameters_4_higher_kl/decoder')
encoder = tf.keras.models.load_model('models/VAE_sin_parameters_4_higher_kl/encoder')


def gen_waveform_freq_and_power(freq_powers, length=3200):
    '''

       :param freq_powers: a 2*X array where the first column is the frequency, and second is the magnitude for a sinusoid.
       :param length: how many samples of waveform to generate
       :return:
       '''
    print(freq_powers)

    gen_sine = lambda x: 0.5 * gen_sine_full(x, length, 32000)

    complex_waveform = np.zeros((length,))
    for i in range(0, 5):
        complex_waveform += gen_sine(freq_powers[0, i]) * freq_powers[1, i]  # *

    return complex_waveform


def gen_waveform_from_coords(coords):
    '''
       Takes x-y coordinates and genrates the waveform associated with them.
       :param coords:
       :return:
       '''

    params_normalised = decoder.predict([coords])

    if params_normalised.shape[1] == 10:
        params = denormalise_batch(params_normalised, y_norm, y_mean)

    else:
        dim4_params = denormalise_batch(params_normalised, x_batch_std, x_batch_mean).numpy()

        rounding_vals = np.array([0, 1, 2, 3])
        diffs = abs(rounding_vals - dim4_params[0, 1])
        closest_index = np.argmin(diffs)
        dim4_params[0, 1] = rounding_vals[closest_index]

        params = vec_to_freqpower(dim4_params)

    waveform_length = 3200
    # gen_sine = lambda x: 0.5*gen_sine_full(x, waveform_length,32000)

    params_reshaped = np.reshape(params, (2, -1))

    # complex_waveform = np.zeros((waveform_length,))
    # for i in range(0,5):
    #    complex_waveform += gen_sine(params_reshaped[0,i])*params_reshaped[1,i]#*
    # return complex_waveform

    return gen_waveform_freq_and_power(params_reshaped, waveform_length)


x = 5

# current_coord = [0,0]
current_coord = (-0.5 + 2 * np.random.rand(2, )).tolist()
deltaval = 0.1

# The list of hyper-parameters we want to optimize. For each one we define the
# bounds, the corresponding scikit-learn parameter name, as well as how to
# sample values from that dimension (`'log-uniform'` for the learning rate)

range_val =3
space = [Real(-range_val, range_val, name='X'),
         Real(-range_val, range_val, name='Y')]


@use_named_args(space)
def objective(**params):
    params_normalised = decoder.predict([[params['X'], params['Y']]])
    params_out = denormalise_batch(params_normalised, y_norm, y_mean)

    outval = params_out[0][0].numpy()
    if outval <= 0:
        outval = 999999999

    return outval


@use_named_args(space)
def objective_rms_to_peak(**params):
    X = params['X']
    Y = params['Y']
    waveform_out = gen_waveform_from_coords([X, Y])

    peak_to_rms = sound_metrics.peak_to_rms(waveform_out)
    print("""The peak to RMS at X=%.6f Y=%.6f is %.6f""" % (X, Y, peak_to_rms))
    return -peak_to_rms


fund_freq_range = (50, 3000)
n_harmonics_range = [0, 4]
detuning_range_hz = [0, 100]
harmonics_power_range = [0, 1]

bigger_space = [Real(50, 3000, name='fund_freq'),
                Integer(0, 4, name='n_harmonics'),
                Integer(-99, 100, name='detuning_hz'),
                Real(0, 1, name='harmonics_power')]


@use_named_args(bigger_space)
def objective_rms_to_peak_high_dimension(**params):
    fund_freq = params['fund_freq']
    n_harmonics = params['n_harmonics']
    detuning_hz = params['detuning_hz']
    harmonics_power = params['harmonics_power']

    params_1d = get_freqs_powers(np.array([[fund_freq, n_harmonics, detuning_hz,
                                            harmonics_power]]))  # get fundamental and powers of waveforms from the 4 params

    params_reshaped = np.reshape(params_1d, (2, -1))

    waveform_length = 3200
    waveform_out = gen_waveform_freq_and_power(params_reshaped, waveform_length)

    peak_to_rms = sound_metrics.peak_to_rms(waveform_out)
    print("""The peak to RMS at fund_freq=%.6f n_harmonics=%d detuning=%d harmonic power=%.4f is %.6f""" % (
    fund_freq, n_harmonics, detuning_hz, harmonics_power, peak_to_rms))
    return -peak_to_rms


@use_named_args(bigger_space)
def objective_rms_to_peak_high_dimension_autoenc(**params):
    fund_freq = params['fund_freq']
    n_harmonics = params['n_harmonics']
    detuning_hz = params['detuning_hz']
    harmonics_power = params['harmonics_power']

    params_1d = get_freqs_powers(np.array([[fund_freq, n_harmonics, detuning_hz,
                                            harmonics_power]]))  # get fundamental and powers of waveforms from the 4 params

    params_1d = np.array([[fund_freq, n_harmonics, detuning_hz, harmonics_power]])

    #normalise the parameters:
    params_1d_normalised = (params_1d-x_batch_mean)/x_batch_std #give this to the encoder

    xy = encoder.predict(params_1d_normalised)


    op_unnormalised = decoder.predict([xy[2]])

    recovered_vals = denormalise_batch(op_unnormalised, x_batch_mean, x_batch_std)

    freq_powers_1d = get_freqs_powers(recovered_vals)


    freq_powers_reshaped = np.reshape(freq_powers_1d, (2, -1))

    waveform_length = 3200
    waveform_out = gen_waveform_freq_and_power(freq_powers_reshaped, waveform_length)

    peak_to_rms = sound_metrics.peak_to_rms(waveform_out)
    print("""The peak to RMS at fund_freq=%.6f n_harmonics=%d detuning=%d harmonic power=%.4f is %.6f""" % (
    fund_freq, n_harmonics, detuning_hz, harmonics_power, peak_to_rms))
    return -peak_to_rms




number_of_calls = 100

res_gp_big = gp_minimize(objective_rms_to_peak_high_dimension_autoenc, bigger_space, n_calls=1, random_state=0)
res_gp_big = gp_minimize(objective_rms_to_peak_high_dimension_autoenc, bigger_space, n_calls=1, random_state=0)
res_gp_big = gp_minimize(objective_rms_to_peak_high_dimension_autoenc, bigger_space, n_calls=1, random_state=0)
res_gp_big = gp_minimize(objective_rms_to_peak_high_dimension_autoenc, bigger_space, n_calls=1, random_state=0)
res_gp_big = gp_minimize(objective_rms_to_peak_high_dimension_autoenc, bigger_space, n_calls=1, random_state=0)
res_gp_big = gp_minimize(objective_rms_to_peak_high_dimension_autoenc, bigger_space, n_calls=1, random_state=0)
res_gp_big = gp_minimize(objective_rms_to_peak_high_dimension_autoenc, bigger_space, n_calls=1, random_state=0)

print("""Best parameters:
- fund_freq=%.6f
- n_harmonics=%d
- detuning=%d
- harmonic power=%.6f
""" % (res_gp_big.x[0], res_gp_big.x[1], res_gp_big.x[2], res_gp_big.x[3]))



print('doing the lower dimension thing')
res_gp = gp_minimize(objective_rms_to_peak, space, n_calls=number_of_calls, random_state=0)

print("""Best parameters:
- X=%.6f
- Y=%.6f
""" % (res_gp.x[0], res_gp.x[1]))
from skopt.plots import plot_convergence

plot_convergence(res_gp)


plot_convergence(res_gp_big)

plt.show()





for i in range(0, 100):

    print('genrating sound for coordingate ' + str(current_coord))
    current_sound = gen_waveform_from_coords(current_coord)

    sa.play_buffer(current_sound, 1, 4, 16000)

    x_delta_plus = gen_waveform_from_coords(np.add(current_coord, [0.1, 0]).tolist())

    x_delta_minus = gen_waveform_from_coords(np.add(current_coord, [-0.1, 0]).tolist())

    y_delta_plus = gen_waveform_from_coords(np.add(current_coord, [0, 0.1]).tolist())

    y_delta_minus = gen_waveform_from_coords(np.add(current_coord, [0, -0.1]).tolist())

    val = 0

    while val != '5':
        val = input("Enter your value to hear the sounds 0 and  1-4. 5 to exit: ")
        if val == '0':
            sa.play_buffer(current_sound, 1, 4, 16000)
        elif val == '1':
            sa.play_buffer(x_delta_plus, 1, 4, 16000)
        elif val == '2':
            sa.play_buffer(x_delta_minus, 1, 4, 16000)
        elif val == '3':
            sa.play_buffer(y_delta_plus, 1, 4, 16000)
        elif val == '4':
            sa.play_buffer(y_delta_minus, 1, 4, 16000)

    inloop = 1
    while inloop == 1:
        valMax = input("Which value was highest pitch?  1-4.")
        if valMax == '1':
            current_coord = np.add(current_coord, [deltaval, 0]).tolist()
            inloop = 0

        elif valMax == '2':
            current_coord = np.add(current_coord, [-deltaval, 0]).tolist()
            inloop = 0
        elif valMax == '3':
            current_coord = np.add(current_coord, [0.0, deltaval]).tolist()
            inloop = 0
        elif valMax == '4':
            current_coord = np.add(current_coord, [0.0, -deltaval]).tolist()
            inloop = 0
