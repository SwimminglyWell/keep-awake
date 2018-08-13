# -*- coding: utf-8 -*-
import pyaudio
import numpy as np
import matplotlib.pyplot as plt
import wave
import sys
from kivy.app import App
from kivy.uix.button import Label
from kivy.uix.boxlayout import BoxLayout

class MyApp(App):
    def build(self):
        return showalphapower()

class ShowAlphaPower(BoxLayout):
    def __init__(self):
        amp = get_input(1)
        frq = get_frq(amp)
        power = get_power(amp)
        self.main_text = Label(text=str(get_alpha_power(frq,power)))
        self.add_widget(self.main_text)
        self._disabled_count = 0
        
    def update(self):
        amp = get_input(1)
        frq = get_frq(amp)
        power = get_power(amp)
        self.main_text.text = str(get_alpha_power(frq,power))
        
    def on_touch_up(self):
        self.update()

def get_input(sample_length=5):
    frate = 44100
    CHUNKSIZE = frate # fixed chunk size
    
    
    # initialize portaudio
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16, channels=1, rate=frate, input=True, frames_per_buffer=CHUNKSIZE)
    
    data = stream.read(CHUNKSIZE)
    amp = np.array([])
    secs = 0
    while secs < sample_length:
        data = stream.read(CHUNKSIZE)
        amp = np.concatenate((amp,np.fromstring(data, dtype=np.int16)))
    
        secs+=1
    return amp

def get_power(amp,frate=44100,plt_to=20):
    Fs = frate;  # sampling rate
    Ts = 1.0/Fs; # sampling interval
    t = np.arange(0,1,Ts) # time vector
    
    y = amp
    n = len(y) # length of the signal
    k = np.arange(n)
    T = n/Fs
    
    Y = np.fft.fft(y)/n # fft computing and normalization
    Y = Y[range(int(np.ceil(n/2)))]
#    upto = np.argmax(frq[:list(frq).index(plt_to)])
#    plt.plot(frq[:upto],abs(Y)[:upto])
    return Y

def get_frq(amp,frate=44100,plt_to=20):
    Fs = frate;  # sampling rate
    Ts = 1.0/Fs; # sampling interval
    t = np.arange(0,1,Ts) # time vector
    
    y = amp
    n = len(y) # length of the signal
    k = np.arange(n)
    T = n/Fs
    frq = (k/T) # two sides frequency range
    frq = frq[range(int(np.ceil(n/2)))] # one side frequency range
#    upto = np.argmax(frq[:list(frq).index(plt_to)])
#    plt.plot(frq[:upto],abs(Y)[:upto])
    return frq

def get_alpha_power(frq,power):
    return (max(abs(power)[list(frq).index(9):list(frq).index(11)+1]))


def play_response(volume=0.5,fs=44100,duration=1.0,f=327.0):
    out_p = pyaudio.PyAudio()
    
    # generate samples, note conversion to float32 array
    samples = (np.sin(2*np.pi*np.arange(fs*duration)*f/fs)).astype(np.float32)
    
    # for paFloat32 sample values must be in range [-1.0, 1.0]
    out_stream = out_p.open(format=pyaudio.paFloat32,
                    channels=1,
                    rate=fs,
                    output=True)
    
    # play. May repeat with different volume values (if done interactively) 
    out_stream.write(volume*samples)
    
    out_stream.stop_stream()
    out_stream.close()
    pass

showalphapower = ShowAlphaPower()
myapp = MyApp()
myapp.run()