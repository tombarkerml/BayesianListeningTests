'''
Contains tests and measures for evaluation of waveforms etc.
'''
import numpy as np


def rms(signal):
    '''
    Returns RMS of a 1-d signal
    :param signal:
    :return:
    '''

    return np.sqrt(np.mean(signal**2))


def peak_to_rms(signal):

    peak_val=np.amax(np.abs(signal))
    rms_val = rms(signal)

    return peak_val/rms_val