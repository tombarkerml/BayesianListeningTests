'''Some tests of loading and doing inference in the embedding space.'''

import tensorflow as tf
import numpy as np
from soundgen_tf import params_to_waveform, gen_sine_full
from data_handling import denormalise_batch
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

import simpleaudio as sa


y_mean= np.array([1.5245759e+03, 2.2862642e+03, 2.2864233e+03, 1.5244318e+03,
       0.0000000e+00, 1.0000000e+00, 3.7491098e-01, 2.5000253e-01,
       1.2500882e-01, 0.0000000e+00])

y_norm = np.array([8.5161224e+02, 1.9803693e+03, 2.9146074e+03, 3.1424453e+03,
       1.0000000e+00, 1.0000000e+00, 3.3071095e-01, 3.2275078e-01,
       2.6020426e-01, 1.0000000e+00])

decoder = tf.keras.models.load_model('models/decoder')

def gen_waveform_from_coords(coords):
       '''
       Takes x-y coordinates and genrates the waveform associated with them.
       :param coords:
       :return:
       '''

       params_normalised = decoder.predict([coords])
       params = denormalise_batch(params_normalised, y_norm, y_mean)

       waveform_length = 3200
       gen_sine = lambda x: 0.5*gen_sine_full(x, waveform_length,32000)

       params_reshaped = np.reshape(params, (2,-1))

       complex_waveform = np.zeros((waveform_length,))
       for i in range(0,5):
           complex_waveform += gen_sine(params_reshaped[0,i])*params_reshaped[1,i]#*

       return complex_waveform

x=5

#current_coord = [0,0]
current_coord = (-0.5+2*np.random.rand(2,)).tolist()
deltaval=0.1

for i in range(0,100):


       print('genrating sound for coordingate ' + str(current_coord))
       current_sound = gen_waveform_from_coords(current_coord)

       sa.play_buffer(current_sound, 1, 4, 16000)

       x_delta_plus =  gen_waveform_from_coords(np.add(current_coord, [0.1, 0]).tolist())

       x_delta_minus = gen_waveform_from_coords(np.add(current_coord, [-0.1, 0]).tolist())

       y_delta_plus = gen_waveform_from_coords(np.add(current_coord, [0, 0.1]).tolist())

       y_delta_minus = gen_waveform_from_coords(np.add(current_coord, [0, -0.1]).tolist())

       val = 0

       while val  != '5':
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

       inloop=1
       while inloop ==1:
              valMax = input("Which value was highest pitch?  1-4.")
              if valMax == '1':
                     current_coord = np.add(current_coord, [deltaval, 0]).tolist()
                     inloop=0

              elif valMax == '2':
                     current_coord = np.add(current_coord, [-deltaval, 0]).tolist()
                     inloop=0
              elif valMax == '3':
                     current_coord = np.add(current_coord, [0.0, deltaval]).tolist()
                     inloop=0
              elif valMax == '4':
                     current_coord = np.add(current_coord, [0.0, -deltaval]).tolist()
                     inloop=0



