'''
Contains functions for creating and processing (e,g normalisation) data.
'''

import numpy as np
import tensorflow as tf

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


    '''tensorflow method for normalising '''



    # make them   (sample-mean)/std
def normalise_batch(batch):
    batch_mean = tf.constant(tf.math.reduce_mean(batch, axis=0))
    batch_std = tf.constant(tf.math.reduce_std(batch, axis=0))

    #as in scikit-learn, if we have zero std (usually only intest cases) - then replace the 0 with 1 to get rid of
    # div/o errors
    # https://github.com/scikit-learn/scikit-learn/blob/7389dbac82d362f296dc2746f10e43ffa1615660/sklearn/preprocessing/data.py#L70

    batch_std_non_zero = tf.where(batch_std!=0, batch_std, 1)

    batch_std_broadcast = tf.broadcast_to(batch_std_non_zero, batch.shape)
    batch_mean_broadcast = tf.broadcast_to(batch_mean, batch.shape)

    normalised_batch = (batch - batch_mean_broadcast)/batch_std_broadcast

    return normalised_batch, batch_mean, batch_std_non_zero



def denormalise_batch(normalised_batch, batch_mean, batch_std):
    '''

    :param normalised_batch:
    :param batch_mean:
    :param batch_std:
    :return:
    '''

    batch_std_broadcast = tf.broadcast_to(batch_std, normalised_batch.shape)
    batch_mean_broadcast = tf.broadcast_to(batch_mean, normalised_batch.shape)

    return normalised_batch*batch_std_broadcast + batch_mean_broadcast


def layer_constraint(input):
    ''' Constrains the output to be only sensible values in our context - '''

    # Work out the thresholds for the normalised values. E.g. if the actual output we need should be either 0, 1 ,2, 3
    # but the normalisation process drives those to sit +-1.5

    mean = 1.5
    std = 1.11794978e+00

    intval = np.round((input * std) + mean)

    rounded_normed = (intval - mean) / std

    return rounded_normed

def tf_layer_constraint(input):
    '''Tensorflow implementation of layer constraint.'''

    mean=tf.constant(1.5)
    std=tf.constant(1.11794978e+00)

    intval = tf.round((input * std) + mean)
    rounded_normed = (intval-mean) /std

    return rounded_normed


def tf_apply_to_one_element(input, element_num):

    clipped_vector = tf_layer_constraint(tf.gather(input, 1, axis=0))



    return tf.stack((input[:,0], tf_layer_constraint(tf.gather(input, 1, axis=0)), input[:,2], input[:,3]), axis=1)


def tf_mod_mask(input):
    ''' applies the functions to particular outputs of a tensor'''


