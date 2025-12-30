import numpy as np

def minimum_Fs(f):
    '''
    Find the lowest sampling frequency that would avoid aliasing for a pure tone at f Hz.
    
    @param:
    f (scalar): frequency in Hz (cycles/second)
    
    @result:
    Fs (scalar): the lowest sampling frequency (samples/second) that would
    not cause aliasing at a tone of f Hz.
    '''
    Fs = 2 * f  # a sampling frequency greater than twice the frequency will avoid aliasing
    return Fs

def omega(f, Fs):
    '''
    Find the radial frequency (omega) that matches a given f and Fs.
    
    @param:
    f (scalar): frequency in Hz (cycles/second)
    Fs (scalar): sampling frequency in samples/second
    
    @result:
    omega (scalar): radial frequency in radians/sample
    '''
    omega = 2 * np.pi * f / Fs  # convert from Hz to radians/sample
    return omega

def pure_tone(omega, N):
    '''
    Create a pure tone of N samples at omega radians/sample.
    
    @param:
    omega (scalar): radial frequency, samples/second
    N (scalar): duration of the tone, in samples
    
    @result:
    x (array): N samples from the signal cos(omega*n)
    '''
    n = np.arange(N)
    x = np.cos(omega * n)  # generate the signal
    return x

