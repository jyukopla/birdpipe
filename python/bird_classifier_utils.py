import numpy as np

# pad too short signal with zeros
def pad(signal, x1, x2, target_len=3*48000, sr=48000):
    # signal: input audio signal, x1: starting point in seconds x2: ending point in seconds, 
    # target_len: target length for signal, sr: sampling rate
    sig_out = np.zeros(target_len) 
    sig_out[int(x1*sr):int(x2*sr)] = signal[int(x1*sr):int(x2*sr)]
    return sig_out

# split input signal to overlapping chunks
def split_signal(sig, rate, seconds, overlap):
    # sig: input_signal, rate: sampling rate, seconds: target length in seconds,
    # overlap: overlap of consecutive frames in seconds, minlen: m
    sig_splits = []
    for i in range(0, len(sig), int((seconds - overlap) * rate)):
        split = sig[i:i + int(seconds * rate)]
        if len(split) < int(seconds * rate): # pad if clip is too short
            split = pad(split, 0, len(split)/rate, target_len=int(seconds*rate), sr=rate)     
        sig_splits.append(split)
    return sig_splits