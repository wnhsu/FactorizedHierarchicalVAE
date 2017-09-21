import os
import time
import scipy
import numpy as np
import soundfile as sf

def mel_scale(freq):
    return 1127.0 * np.log(1.0 + float(freq)/700)

def inv_mel_scale(mel_freq):
    return 700 * (np.exp(float(mel_freq)/1127) - 1)

class MelBank(object):
    def __init__(self, 
                 low_freq=20, 
                 high_freq=8000, 
                 num_bins=80, 
                 sample_freq=16000, 
                 frame_size=32):

        self.low_freq = low_freq
        self.high_freq = high_freq
        self.num_bins = num_bins
        self.sample_freq = sample_freq
        self.frame_size = frame_size

        # frame_size in millisecond
        self.window_size = self.sample_freq * 0.001 * self.frame_size
        self.fft_freqs = np.linspace(
                0, self.sample_freq / 2, self.window_size / 2 + 1)[:-1]

        self.mel_low_freq = mel_scale(self.low_freq)
        self.mel_high_freq = mel_scale(self.high_freq)
        
        mel_freqs = np.linspace(
                self.mel_low_freq, self.mel_high_freq, self.num_bins+2)
        self.mel_windows = [mel_freqs[i:i+3] for i in xrange(self.num_bins)]

        def _weight(mel_window, mel_freq):
            mel_low, mel_center, mel_high = mel_window
            if mel_freq > mel_low and mel_freq < mel_high:
                if mel_freq <= mel_center:
                    return (mel_freq - mel_low) / (mel_center - mel_low)
                else:
                    return (mel_high - mel_freq) / (mel_high - mel_center)
            else:
                return 0
            
        self.mel_banks = [[_weight(window, mel_scale(freq)) \
                for freq in self.fft_freqs] for window in self.mel_windows]
        self.center_freqs = [inv_mel_scale(mel_freq) \
                for mel_freq in mel_freqs[1:-1]]

def hann(n):
    """
    n   : length of the window
    """
    w=np.zeros(n)
    for x in xrange(n):
        w[x] = 0.5*(1 - np.cos(2*np.pi*x/n)) 	
    return w

def stft_index(wave, frame_size_n, frame_starts_n, fft_size=None, win=None):
    """
    wave            : 1-d float array
    frame_size_n    : number of samples in each frame
    frame_starts_n  : a list of int denoting starting sample index of each frame
    fft_size        : number of frequency bins
    win             : windowing function on amplitude; len(win) == frame_size_n
    """
    wave = np.asarray(wave)
    frame_starts_n = np.int32(frame_starts_n)
    if fft_size is None:
        fft_size = frame_size_n
    if win is None:
        win = np.sqrt(hann(frame_size_n))

    # sanity check
    if not wave.ndim == 1:
        raise ValueError('wave is not mono')
    elif not frame_starts_n.ndim == 1:
        raise ValueError('frame_starts_n is not 1-d')
    elif not len(win) == frame_size_n:
        raise ValueError('win does not match frame_starts_n (%s != %s)', len(win), frame_size_n)
    elif fft_size % 2 == 1:
        raise ValueError('odd ffts not yet implemented')
    elif np.min(frame_starts_n) < 0 or np.max(frame_starts_n) > wave.shape[0]-frame_size_n:
        raise ValueError('Your starting indices contain values outside the allowed range')

    spec = np.asarray([scipy.fft(wave[n:n+frame_size_n]*win, n=fft_size)[:fft_size/2+1] \
                       for n in frame_starts_n])
    return spec	
   
def istft_index(spec, frame_size_n, frame_starts_n, fft_size=None, win=None, awin=None):
    """
    spec            : 1-d complex array
    frame_size_n    : number of samples in each frame
    frame_starts_n  : a list of int denoting starting sample index of each frame
    fft_size        : number of frequency bins
    win             : windowing function on spectrogram; len(win) == frame_size_n
    awin            : original windowing function on amplitude; len(win) == frame_size_n
    """
    frame_starts_n = np.int32(frame_starts_n)
    if fft_size is None:
        fft_size = frame_size_n
    if win is None:
       win=np.sqrt(hann(frame_size_n))
    if awin is None:
       awin=np.sqrt(hann(frame_size_n))
    pro_win = win * awin

    # sanity check
    if not frame_starts_n.ndim == 1:
        raise ValueError('frame_starts_n is not 1-d')
    elif not len(win) == frame_size_n:
        raise ValueError('win does not match frame_starts_n (%s != %s)', len(win), frame_size_n)
    elif not len(awin) == frame_size_n:
        raise ValueError('awin does not match frame_starts_n (%s != %s)', len(win), frame_size_n)
    elif spec.shape[0] < frame_starts_n.shape[0]:
        raise ValueError('Number of frames in the spectrogram cannot be \
                          less than the size of frame starts') 

    N = frame_starts_n[-1] + frame_size_n
    
    signal = np.zeros(N)
    normalizer = np.zeros(N, dtype=np.float32)

    n_range = np.arange(frame_size_n)
    for i, n_offset in enumerate(frame_starts_n):
	frames = np.real(scipy.ifft(np.concatenate((spec[i], spec[i][-2:0:-1].conjugate())),
                                    n=fft_size))[:frame_size_n]
        signal[n_offset+n_range] += frames * win
        normalizer[n_offset+n_range] += pro_win

    nonzero = np.where(normalizer>0)
    rest = np.where(normalizer<=0)
    signal[nonzero] = signal[nonzero]/normalizer[nonzero]
    signal[rest] = 0
    return signal

def comp_spec_image(wave, decom, frame_size_n, shift_size_n, fft_size, awin, log_floor):
    """
    RETURN: 
        float matrix of shape (2, T, F)
    """
    frame_starts_n = np.arange(0, wave.shape[0]-frame_size_n, step=shift_size_n)
    spec = stft_index(wave, frame_size_n, frame_starts_n, fft_size, awin)
      
    if decom == "mp":
        phase = np.angle(spec)
        dbmag = np.log10(np.absolute(spec))
        # print("max amplitude %s, max magnitude %s, max phase %s" % (
        #     np.max(wave), np.max(np.absolute(spec)), np.max(phase)))
        dbmag[dbmag < log_floor] = log_floor
        dbmag = 20 * dbmag
        spec_image = np.concatenate([dbmag[None,...], phase[None,...]], axis=0)
    elif decom == "ri":
        real = np.real(spec)
        imag = np.imag(spec)
        # print("max amplitude %s, max real %s, max imag %s" % (
        #     np.max(wave), np.max(np.absolute(real)), np.max(np.absolute(imag))))
        spec_image = np.concatenate([real[None,...], imag[None,...]], axis=0)
    else:
        raise ValueError("decomposition type %s not supported" % decom)

    return spec_image

def est_phase_from_mag_spec(
        mag_spec, frame_size_n, shift_size_n, fft_size, 
        awin=None, k=1000, min_avg_diff=1e-9, debug=False):
    """
    for quality min_avg_diff 1e-9 is recommended

    mag_spec    - magnitude spectrogram (in linear) of shape (n_time, n_frequency)
    """
    start_time = time.time()
    debug_x = []
    frame_starts_n = np.arange(len(mag_spec)) * shift_size_n
    
    # initialize with white noise
    # wave_len = frame_starts_n[-1] + frame_size_n + 1
    # x = np.random.normal(0, 1, size=(wave_len))
    X_phase = None
    X = mag_spec * np.exp(1j * np.random.uniform(-np.pi, np.pi, mag_spec.shape))
    x = istft_index(X, frame_size_n, frame_starts_n, fft_size, awin, awin)
    for i in xrange(k):
        X_phase = np.angle(stft_index(x, frame_size_n, frame_starts_n, fft_size, awin))
        X = mag_spec * np.exp(1j * X_phase)
        new_x = istft_index(X, frame_size_n, frame_starts_n, fft_size, awin, awin)
        avg_diff = np.mean((x - new_x)**2)
        x = new_x
        
        if avg_diff < min_avg_diff:
            break

        if debug and i % 100 == 0:
            print "done %s iterations, avg_diff is %s" % (i, avg_diff)
            debug_x.append(x)
    if debug:
        print "time elapsed = %.2f" % (time.time() - start_time)

    return X_phase, debug_x

def convert_to_complex_spec(
        X, X_phase, decom, phase_type, add_dc=False, est_phase_opts=None):
    """
    X/X_phase       - matrix of shape (..., n_channel, n_time, n_frequency)
    decom           - `mp`: magnitude (in dB) / phase (in rad) decomposition
                      `ri`: real / imaginary decomposition
    phase_type      - `true`: X's n_channel = 2
                      `oracle`: use oracle phase X_phase
                      `zero`: use zero matrix as the phase matrix for X
                      `rand`: use random matrix as the phase matrix for X
                      `est`: estimate the phase from magnitude spectrogram
    est_phase_opts  - arguments for est_phase_from_mag_spec

    complex_X is [..., t, f]
    """
    X, X_phase = np.asarray(X), np.asarray(X_phase)
    if X.shape[-3] != 1 and X.shape[-3] != 2:
        raise ValueError("X's n_channel must be 1 or 2 (%s)" % X.shape[-3])
    if np.any(np.iscomplex(X)):
        raise ValueError("X should not be complex")
    if np.any(np.iscomplex(X_phase)):
        raise ValueError("X_phase should not be complex")

    if add_dc:
        X_dc = np.zeros(X.shape[:-1] + (1,))
        X = np.concatenate([X_dc, X], axis=-1)
        if X_phase:
            X_phase_dc = np.zeros(X_phase.shape[:-1] + (1,))
            X_phase = np.concatenate([X_phase_dc, X_phase], axis=-1)

    if decom == "mp":
        X_lin_mag = 10 ** (X[..., 0, :, :] / 20)
        if phase_type == "true" and X.shape[-3] != 2:
            raise ValueError("X should have 2 channels for phase_type %s" % (
                    phase_type,) + " (X shape is %s)" % (X.shape,))
            X_phase = X[..., 1, :, :]
        else:
            if X.shape[-3] != 1:
                print("WARNING: ignoring X's second channel (phase)")

            if phase_type == "oracle":
                if X_phase is None:
                    raise ValueError("X_phase shape %s invalid for phase_type %s" % (
                            X_phase.shape, phase_type))
            elif phase_type == "zero":
                X_phase = np.zeros_like(X_lin_mag)
            elif phase_type == "rand":
                X_phase = np.random.uniform(-np.pi, np.pi, X_lin_mag.shape)
            elif phase_type == "est":
                X_phase, _ = est_phase_from_mag_spec(X_lin_mag, debug=True, **est_phase_opts)
                print("X_lin_mag shape %s" % (X_lin_mag.shape,))
                print("X_phase shape %s" % (X_phase.shape,))
            else:
                raise ValueError("invalid phase type (%s)" % phase_type)
        complex_X = X_lin_mag * np.exp(1j * X_phase)
    elif decom == "ri":
        if phase_type != "true":
            raise ValueError("invalid phase type %s. only `true` is valid" % phase_type)
        complex_X = X[..., 0, :, :] + 1j * X[..., 1, :, :]
    else:
        raise ValueError("invalid decomposition %s (mp|ri)" % decom)

    return complex_X

def complex_spec_to_audio(
        complex_spec, name=None, trim=0, fs=16000,
        frame_size_n=400, shift_size_n=160, fft_size=400, win=None):
    assert(np.asarray(complex_spec).ndim == 2)
    frame_starts_n = np.arange(len(complex_spec)) * shift_size_n
    signal = istft_index(complex_spec, frame_size_n, frame_starts_n, fft_size, win, win)

    if trim > 0:
        signal = signal[trim:-trim]
    
    if name is not None:
        if os.path.splitext(name)[1] != ".wav":
            name = name + ".wav"
        sf.write(file=name, data=signal, samplerate=fs)

    return signal
