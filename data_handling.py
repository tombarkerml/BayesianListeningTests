'''
Contains functions for creating and processing (e,g normalisation) data.
'''

import numpy as np


def gen_input_data(num_examples = 65536):

    fund_freq_range = (50,3000)
    n_harmonics_range = [0, 4]
    detuning_range_hz = [0, 100]
    harmonics_power=[0, 1] #just uses np.random.rand to generate a number 0<1


    def get_detune_hz(detuning_range_hz):
      detune =  np.random.randint(*detuning_range_hz) *  (1 if np.random.random() < 0.5 else -1)
      return detune


    ip_vecs=[]


    for i in range(0, num_examples):
      ip_vecs.append([ np.random.randint(*fund_freq_range), np.random.randint(*n_harmonics_range) , get_detune_hz(detuning_range_hz), np.random.rand()])

    ip_vecs=np.array(ip_vecs)

    return ip_vecs


