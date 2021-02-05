import unittest
from data_handling import gen_input_data, normalise_batch, denormalise_batch, layer_constraint, tf_layer_constraint
import tensorflow as tf
import numpy as np




class GenInputData(unittest.TestCase):

    def test_data_dimensions(self):
        ''' checks that the input vector generated is the correct dimensions'''

        #default number of examples is 65536
        default_size_op = gen_input_data()

        self.assertEqual(default_size_op.shape, (65536, 4))

        custom_num_inputs = 100
        custom_size_op = gen_input_data(num_examples=custom_num_inputs)
        self.assertEqual(custom_size_op.shape, (custom_num_inputs, 4))


    def test_batch_normalisation(self):

        test_array = np.array([[1,2,3,4], [1,10,3,5], [1,20,3,6]])
        test_array = tf.constant(test_array, dtype='float32')

        normalised, mean, std = normalise_batch(test_array)

        all_equal = np.alltrue(normalised[:,1] ==(test_array[:, 1] - np.mean(test_array[:, 1])) / np.std(test_array[:, 1]))

        self.assertTrue(all_equal)

        #Now need function to undo the normalisation

    def test_batch_denormalisation(self):
        test_array = np.array([[1, 2, 3, 4], [1, 10, 3, 5], [1, 20, 3, 6]])
        test_array = tf.constant(test_array, dtype='float32')

        normalised, mean, std = normalise_batch(test_array)

        denormalised = denormalise_batch(normalised, batch_mean=mean, batch_std=std)

        self.assertTrue(np.alltrue(denormalised==test_array))

    def test_layer_constraint(self):

        #some values outputted from NN should be normalised versions of integers.

        mean = 1.5
        std = 1.11794978e+00

        predicted_val = 2.2
        normed_predicted_val = (predicted_val - mean) / std

        rounded_normed_predicted_value = layer_constraint(normed_predicted_val)

        denormalised_prediction = (std*rounded_normed_predicted_value)+mean

        self.assertEqual(denormalised_prediction, 2)

    def test_tf_layer_constraint(self):
        mean = 1.5
        std = 1.11794978e+00

        predicted_val = np.array([[2.2, 2.4, 2.6]])
        normed_predicted_val = (predicted_val - mean) / std

        rounded_normed_predicted_value = tf_layer_constraint(normed_predicted_val)

        denormalised_prediction = (std * rounded_normed_predicted_value) + mean

        self.assertEqual(denormalised_prediction, 2)

if __name__ == '__main__':
    unittest.main()
