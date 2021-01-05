import unittest
from data_handling import gen_input_data

class MyTestCase(unittest.TestCase):
    def test_something(self):
        self.assertEqual(True, False)


class GenInputData(unittest.TestCase):

    def test_data_dimensions(self):
        ''' checks that the input vector generated is the correct dimensions'''

        #default number of examples is 65536
        default_size_op = gen_input_data()

        self.assertEqual(default_size_op.shape, (65536, 4))

        custom_num_inputs = 100
        custom_size_op = gen_input_data(num_examples=custom_num_inputs)
        self.assertEqual(custom_size_op.shape, (custom_num_inputs, 4))

if __name__ == '__main__':
    unittest.main()
