import unittest

import numpy as np
import tensorflow as tf

import soundgen_tf
from soundgen_tf import gen_complex, get_freqs_powers, params_to_waveform

class MyTestCase(unittest.TestCase):
    def test_something(self):
        self.assertEqual(True, True)


class TestSinGen(unittest.TestCase):

    def test_sin_length(self):
        frequency = 500
        n_samples = 512
        fs = 16000

        sin_op = soundgen_tf.gen_sine_full(frequency, n_samples, fs)

        self.assertEqual(len(sin_op), n_samples)


    def test_gen_complex(self):

        freqs = np.array([[100.0, 200.0, 300.0]]) # dims = (1,3)
        weights = np.array([[1.0, 0.0, 0.0]])

        complex_op = gen_complex(freqs, weights)

        self.assertEqual(len(complex_op), 512)


    def test_multiple_gen_complex_dims(self):
        '''
        given multiple input freqs vectors, does the function generate multiple outputs
        :return:
        '''
        freqs = np.array([[100.0, 200.0, 300.0], [200.0, 300.0, 400.0]]) #dims = (2, 3)
        weights = np.array([[1.0, 0.0, 0.0], [1.0,0.0,0.0]])
        complex_op = gen_complex(freqs, weights)

        self.assertEqual(complex_op.shape[0], freqs.shape[0]) #

    def test_multiple_gen_complex_dims(self):
        '''
        given multiple input freqs vectors, does the function generate multiple outputs
        :return:
        '''
        freqs_2d = np.array([[100.0, 200.0, 300.0], [200.0, 300.0, 400.0]]) #dims = (2, 3)
        weights_2d = np.array([[1.0, 0.0, 0.0], [1.0,0.0,0.0]])
        complex_op = gen_complex(freqs_2d, weights_2d)

        freqs_1d = np.array([[100.0, 200.0, 300.0]])
        weights_1d = np.array([[1.0, 0.0, 0.0]])
        single_op = gen_complex(freqs_1d, weights_1d)



        self.assertTrue(np.alltrue(complex_op[0,:] == single_op))


    def test_gen_freqs_powers(self):

        fundamental = 100
        n_harmonics = 2
        detune_hz = 0
        harmonic_weights = 0.5


        freqs, powers = get_freqs_powers([[fundamental, n_harmonics, detune_hz, harmonic_weights]])

        self.assertEqual(freqs[0,0], fundamental)


    def test_gen_freqs_powers_multiple(self):

        fundamental = 100
        n_harmonics = 2
        detune_hz = 5
        harmonic_weights = 0.5


        freqs, powers = get_freqs_powers([[fundamental, n_harmonics, detune_hz, harmonic_weights], [2*fundamental, n_harmonics+1, detune_hz*2, harmonic_weights*0.5]])

        self.assertEqual(freqs[0,0], fundamental)
        self.assertEqual(freqs[1, 0], 2*fundamental)

class TestSinGenKeras(unittest.TestCase):


    def test_add_lamdba(self):
        inputs = tf.keras.Input(shape=(4,), dtype='float32')
        output = tf.keras.layers.Lambda(params_to_waveform) (inputs)
        model = tf.keras.Model(inputs=inputs, outputs=output)

        test_vector=np.array([[100.0, 1, 1, 1]])
        test_vector2 = np.array([[100.0, 1, 1, 1], [200.0, 3, 0.5, 1]])

        op1 = model.predict(test_vector)
        op2 = model.predict(test_vector2)

        self.assertTrue(op1.shape==(1,512))
        self.assertTrue(op2.shape == (2, 512))



        xiy=998


if __name__ == '__main__':
    unittest.main()

