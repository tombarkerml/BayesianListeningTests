from pysinewave import SineWave, SineWaveGenerator, utilities
import time
import numpy as np
import sounddevice as sd


#import system #checking if linux or windows


sinewave = SineWave(pitch=12, pitch_per_second = 100)

#sinewave.play()

class Complex_Sine(object):

    def __init__(self, frequencies, magnitudes, gain=1):

        self.frequencies = frequencies
        self.magnitudes = magnitudes
        self.gain = gain
        self.create_oscillator()

    def freq_to_pitch(self, freq):
        '''
        Takes pysinewave.utilities's  pitch_to_frequency method and the inverse
        where
        frequency = MIDDLE_C_FREQUENCY * 2**(pitch/12)
        :return:
        '''
        # Frequency of a middle C
        MIDDLE_C_FREQUENCY = 261.625565
        return (12*np.log(freq/MIDDLE_C_FREQUENCY))/np.log(2)



    def create_oscillator(self):

        self.complex_osc = []
        for i, freq in enumerate(self.frequencies):
            current_pitch = self.freq_to_pitch(freq)

            current_sine=SineWave(pitch=current_pitch, decibels_per_second=40)
            #current_sine.sinewave_generator.set_frequency(freq)
            current_sine.sinewave_generator.set_amplitude(self.magnitudes[i]*self.gain)
            self.complex_osc.append(current_sine)

    def scale_gain(self, gain):

        self.gain = gain
        for i, osc in enumerate(self.complex_osc):
            osc.sinewave_generator.set_amplitude(self.magnitudes[i]*self.gain)


    def start(self):
        for i, osc in enumerate(self.complex_osc):
            osc.play()

    def stop(self):
        for i, osc in enumerate(self.complex_osc):
            osc.stop()


val = None

def increase_vol(sine: SineWave):


    current_vol =sine.sinewave_generator.amplitude
    print('Current volume is %f.6d', current_vol)
    sine.sinewave_generator.set_amplitude(current_vol*1.05)

def decrease_vol(sine: SineWave):


    current_vol =sine.sinewave_generator.amplitude
    print('Current volume is %.6f', current_vol)
    sine.sinewave_generator.set_amplitude(current_vol/1.05)


pooble = Complex_Sine([1000, 2000, 3000], [1, 0.5, 0.5])
pooble.scale_gain(0.5)
pooble.start()
time.sleep(1)
print('changing gain')
pooble.scale_gain(0.1)
time.sleep(1)

sinewave.play()

sd.OutputStream(channels=1, callback= lambda *args: self._callback(*args),
                                samplerate=samplerate)


while val !='q':

    val = input("W increase, S decrease volume ")

    if val == 'w':
        increase_vol(sinewave)
    elif val == 's':
        decrease_vol(sinewave)

sinewave.stop()






