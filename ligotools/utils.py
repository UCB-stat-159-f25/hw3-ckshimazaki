import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import filtfilt
from scipy.io import wavfile

def whiten(strain, interp_psd, dt):
    Nt = len(strain)
    freqs = np.fft.rfftfreq(Nt, dt)
    hf = np.fft.rfft(strain)
    norm = 1. / np.sqrt(1. / (dt * 2))
    white_hf = hf / np.sqrt(interp_psd(freqs)) * norm
    white_ht = np.fft.irfft(white_hf, n=Nt)
    return white_ht

def write_wavfile(filename, fs, data):
    d = np.int16(data / np.max(np.abs(data)) * 32767 * 0.9)
    wavfile.write(filename, int(fs), d)

def reqshift(data, fshift=100, sample_rate=4096):
    x = np.fft.rfft(data)
    T = len(data) / float(sample_rate)
    df = 1.0 / T
    nbins = int(fshift / df)
    y = np.roll(x.real, nbins) + 1j * np.roll(x.imag, nbins)
    y[0:nbins] = 0.
    z = np.fft.irfft(y)
    return z

def plot_psd_template(datafreq, template_fft, d_eff, freqs, data_psd, det, fs, plottype, eventname, pcolor='r'):
    plt.figure(figsize=(10, 6))
    template_f = np.abs(template_fft) * np.sqrt(np.abs(datafreq)) / d_eff
    plt.loglog(datafreq, template_f, 'k', label='template(f)*sqrt(f)')
    plt.loglog(freqs, np.sqrt(data_psd), pcolor, label=det + ' ASD')
    plt.xlim(20, fs / 2)
    plt.ylim(1e-24, 1e-20)
    plt.grid()
    plt.xlabel('frequency (Hz)')
    plt.ylabel('strain noise ASD (strain/rtHz), template h(f)*rt(f)')
    plt.legend(loc='upper left')
    plt.title(det + ' ASD and template around event')
    plt.savefig(f"figures/{eventname}_{det}_matchfreq.{plottype}")
    plt.close()
