import copy
import inspect
if __name__!='__main__':
    import tensorflow as tf
    import neurokit2 as nk
    import numpy as np
    import scipy

    # from functools import lru_cache
    from scipy.fft import fft, ifft
    from scipy.signal import iirnotch, iirpeak, filtfilt
    from scipy.interpolate import CubicSpline

AUGMENTERS_DICT = {}
AUGMENTERS_PARAMS = {}
AUGMENTERS_HPS = {}

def register_internal_serializable(path, symbol):
    global AUGMENTERS_DICT, AUGMENTERS_PARAMS, AUGMENTERS_HPS
    if isinstance(path, (list, tuple)):
        name = path[0]
    else:
        name = path
    signature = inspect.signature(symbol)
    params = {
        k: v.default if v.default is not inspect.Parameter.empty else None
        for k, v in signature.parameters.items()
    }
    AUGMENTERS_DICT[name] = symbol
    AUGMENTERS_PARAMS[name] = params
    AUGMENTERS_HPS[name] = symbol.hp
    
class aug_export:
    def __init__(self, path):
        self.path = path
        
    def __call__(self, symbol):
        register_internal_serializable(self.path, symbol)
        return symbol

class DataAugmenter():
    def __init__(self, augment_cfg=[], prob=0.5):
        self.augmenters = [AUGMENTERS_DICT[k](**v) for k,v,*_ in augment_cfg]
        self.prob = prob if isinstance(prob, list) else [prob]*len(self.augmenters)
        
    def __call__(self, x, xl, xr):
        shape = tf.shape(x)
        for p, aug in zip(self.prob, self.augmenters):
            if (p > tf.random.uniform(maxval=1, shape=[1])):
                x = tf.numpy_function(aug, [x, xl, xr], [tf.float32])[0]
                x = tf.reshape(x, shape)
        return x

@aug_export('GaussianNoise_Det')
class GaussianNoiseDeterministic:
    hp = {
        'sigma_scale': (0.0, 1.2)
    }
    def __init__(self, sigma_scale=0.1):
        """
        :param sigma_scale: factor to use in computing sigma parameter for noise distribution
            sigma = mean(abs(diff between signal & mean))) * sigma_scale
        """
        self.sigma_scale = sigma_scale

    def __call__(self, x, *args):
        # x = np.squeeze(x, -1)
        mean_power_diff = np.mean(np.abs(x - np.mean(x)))
        noise_sigma = mean_power_diff * self.sigma_scale
        noise = np.random.normal(scale=noise_sigma, size=x.shape)
        # print(noise_sigma)
        return (x + noise).astype(np.float32)

@aug_export('GaussianNoise_Sto')
class GaussianNoiseStochastic:
    hp = {
        'sigma_scale_min': (0.0, 0.5),
        'sigma_scale_max': (0.5, 1.5)
    }
    def __init__(self, sigma_scale_min=0.0, sigma_scale_max=0.5):
        """
        :param sigma_scale_min: min factor to use in computing sigma parameter for noise distribution
        :param sigma_scale_max: max factor to use in computing sigma parameter for noise distribution
            sample sigma_scale uniformly in [sigma_scale_min, sigma_scale_max)
            sigma = mean(abs(diff between signal & mean))) * sigma_scale
        """
        self.sigma_scale_min = sigma_scale_min
        self.sigma_scale_max = sigma_scale_max

    def __call__(self, x, *args):
        # sample sigma scale
        # x = np.squeeze(x, -1)
        sigma_scale = np.random.uniform(self.sigma_scale_min, self.sigma_scale_max)
        # print(sigma_scale.shape)
        mean_power_diff = np.mean(np.abs(x - np.mean(x)))
        noise_sigma = mean_power_diff * sigma_scale
        noise = np.random.normal(scale=noise_sigma, size=x.shape)
        return (x + noise).astype(np.float32)

@aug_export('LowPassFilter_Det')
class LowPassFilterDeterministic():
    hp = {
        'highcut_hz': (0.05, 0.1)
    }
    def __init__(self, data_freq=4, highcut_hz=0.05):
        """
        Apply low pass filter to remove frequency bands >= highcut_hz
        :param data_freq: frequency of data to apply filter to (e.g., 4Hz for EDA)
        :param highcut_hz: lower bound on frequency bands to remove
        """
        self.data_freq = data_freq
        self.highcut_hz = highcut_hz
        self.b, self.a = scipy.signal.butter(4, [highcut_hz], btype="lowpass", output="ba", fs=data_freq)

    def __call__(self, x, *args):
        # print(x.shape)
        segment_filtered = scipy.signal.filtfilt(self.b, self.a, x, axis=0)
        return segment_filtered.astype(np.float32)

@aug_export('BandstopFilter_Det')
class BandstopFilterDeterministic:
    hp = {
        'remove_freq': (0.1, 1.0)
    }
    def __init__(self, data_freq=4, remove_freq=0.25, Q=0.707):
        """
        Selects frequency band to remove.
        See https://stackoverflow.com/questions/54320638/how-to-create-a-bandstop-filter-in-python
        :param data_freq: frequency of data to apply filter to (e.g., 4Hz for EDA)
        :param remove_freq: frequency band to remove
        :param Q: "quality factor" Q = remove_freq / width of filter
        see - https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.iirnotch.html
        and https://en.wikipedia.org/wiki/Q_factor
        """
        self.data_freq = data_freq
        self.remove_freq = remove_freq
        self.Q = Q
        self.b, self.a = iirnotch(self.remove_freq, self.Q, fs=self.data_freq)

    def __call__(self, x, *args):
        return filtfilt(self.b, self.a, x, axis=0).astype(np.float32)

@aug_export('TimeShift_Det')
class TimeShiftDeterministic:
    hp = {
        'shift_len': (1, 200)
    }
    """ Shifts the window left or right by a number of samples """
    def __init__(self, shift_len=120):
        self.shift_len = int(shift_len)

    def __call__(self, x, left_buffer, right_buffer):
        # print(x.shape, left_buffer.shape, right_buffer.shape)
        # drop nans from left and right buffer segment
        l_len = np.bitwise_and.reduce(~np.isnan(left_buffer), axis=1).sum()
        r_len = np.bitwise_and.reduce(~np.isnan(right_buffer), axis=1).sum()
        signal = np.concatenate([left_buffer, x, right_buffer])
        mask = np.bitwise_and.reduce(~np.isnan(signal), axis=1)
        signal = signal[mask]
        # sample shift to apply --- make sure not out-of-bounds!!
        left_shift_len = min(self.shift_len, l_len)
        right_shift_len = min(self.shift_len, r_len)
        shift = np.random.choice([-left_shift_len, right_shift_len])  # choose whether to shift left or right in time
        start_index = l_len + shift
        x_trf = signal[start_index:start_index+len(x)]
        return x_trf.astype(np.float32)

@aug_export('TimeShift_Sto')
class TimeShiftStochastic:
    hp = {
        'shift_min': (0, 100),
        'shift_max': (100, 239)
    }
    """ Shifts the window left or right by a number of samples """
    def __init__(self, shift_len_min=120, shift_len_max=240):
        self.shift_min = int(shift_len_min)
        self.shift_max = int(shift_len_max)
        self.shift_lens = np.arange(self.shift_min, self.shift_max, 1)

    def __call__(self, x, left_buffer, right_buffer):
        # drop nans from left and right buffer segment
        l_len = np.bitwise_and.reduce(~np.isnan(left_buffer), axis=1).sum()
        r_len = np.bitwise_and.reduce(~np.isnan(right_buffer), axis=1).sum()
        signal = np.concatenate([left_buffer, x, right_buffer])
        mask = np.bitwise_and.reduce(~np.isnan(signal), axis=1)
        signal = signal[mask]
        # sample shift len to apply
        shift_len = np.random.choice(self.shift_lens)
        # adjust so shift is in bounds & sample whether to apply it on left or right
        left_shift_len = min(shift_len, l_len)
        right_shift_len = min(shift_len, r_len)
        shift = np.random.choice([-left_shift_len, right_shift_len])  # choose whether to shift left or right in time
        start_index = l_len + shift
        x_trf = signal[start_index:start_index+len(x)]
        return x_trf.astype(np.float32)

@aug_export('HighFreqNoise_Det')
class HighFrequencyNoiseDeterministic:
    hp = {
        'sigma_scale': (0.0, 1.0),
    }

    def __init__(self, sigma_scale=0.1, freq_bin_start_idx=60, freq_bin_stop_idx=120):
        self.sigma_scale = sigma_scale
        self.freq_bin_start_idx = freq_bin_start_idx
        self.freq_bin_stop_idx = freq_bin_stop_idx  # fixed typo: was 'req_bin_stop_idx'
        self.freq_bin_idxs = np.arange(freq_bin_start_idx, freq_bin_stop_idx)

    def __call__(self, x, left_buffer, right_buffer):
        x = np.squeeze(x)
        
        # Compute FFT
        x_fft = fft(x)
        
        x_fft_real = np.real(x_fft).reshape(-1)
        x_fft_imag = np.imag(x_fft).reshape(-1)
        
        # Compute noise scale
        mean_fft_val = np.mean(np.abs(x_fft))
        sigma = self.sigma_scale * mean_fft_val
        
        # Generate 1D complex noise
        noise_real = np.random.normal(scale=sigma, size=len(self.freq_bin_idxs))
        noise_imag = np.random.normal(scale=sigma, size=len(self.freq_bin_idxs))

        # Add noise to positive frequencies
        x_fft_real[self.freq_bin_idxs] += noise_real
        x_fft_imag[self.freq_bin_idxs] += noise_imag

        # Add noise to negative frequencies (conjugate symmetry)
        neg_end_idx = len(x) + 1 - self.freq_bin_start_idx
        neg_start_idx = neg_end_idx - len(self.freq_bin_idxs)
        x_fft_real[neg_start_idx:neg_end_idx] += noise_real
        x_fft_imag[neg_start_idx:neg_end_idx] -= noise_imag

        # Recombine and inverse FFT
        x_fft_noised = x_fft_real + 1j * x_fft_imag
        
        x_ifft = np.abs(ifft(x_fft_noised))
        return x_ifft.astype(np.float32)

@aug_export('LooseSensorArtifact_Det')
class LooseSensorArtifactDeterministic:
    hp = {
        'width': (4, 20)
    }
    def __init__(self, width=4, smooth_width_min=2, smooth_width_max=80):
        self.width = int(width)
        self.smooth_width_min = smooth_width_min
        self.smooth_width_max = smooth_width_max

    def __call__(self, x, left_buffer, right_buffer):
        # sample width of artifact
        artifact_width = self.width
        # sample artifact start
        artifact_start = np.random.choice(np.arange(0, len(x) - artifact_width + 1))
        # compute artifact end (inclusive)
        artifact_end = artifact_start + artifact_width - 1
        
        # don't smooth if artifact goes all the way to boundary
        smooth_left = (artifact_start != 0)
        smooth_right = (artifact_end != len(x) - 1)
        
        # sample smoothing edge widths
        smooth_max = min(self.smooth_width_max, int(artifact_width/2))
        smooth_width1 = np.random.choice(np.arange(self.smooth_width_min, smooth_max + 1)) if smooth_left else 0
        smooth_width2 = np.random.choice(np.arange(self.smooth_width_min, smooth_max + 1)) if smooth_right else 0
        
        # add drop to non-smoothed regions of artifact
        noisy_segment = copy.deepcopy(x)
        drop_start = artifact_start + smooth_width1
        drop_end = artifact_end - smooth_width2  # (inclusive)
        # get mean amplitude of signal in this range
        mean_amp = np.mean(noisy_segment[drop_start:drop_end + 1])  # +1 so inclusive
        # subtract from signal
        noisy_segment[drop_start:drop_end + 1] -= mean_amp
        # zero out negative entries
        noisy_segment[noisy_segment < 0] = 0
        
        # fill in parts to be smoothed
        # fit cubic spline
        # get pre-smooth, unsmoothed artifact, post-smooth
        train_x = np.concatenate([
            np.arange(artifact_start),  # don't include artifact start
            np.arange(drop_start, drop_end + 1),  # include drop end
            np.arange(artifact_end + 1, len(x)) # don't include artifact end
        ])
        train_y = np.concatenate([
            noisy_segment[:artifact_start],
            noisy_segment[drop_start:drop_end + 1],
            noisy_segment[artifact_end + 1:]
        ])
        spline = CubicSpline(train_x, train_y)
        # fill in smoothed parts
        if artifact_start != drop_start:
            noisy_segment[artifact_start:drop_start] = spline(np.arange(artifact_start, drop_start))
        if artifact_end != drop_end:
            noisy_segment[drop_end + 1:artifact_end + 1] = spline(np.arange(drop_end + 1, artifact_end + 1))  # include artifact end
        return noisy_segment.astype(np.float32)

@aug_export('JumpArtifact_Det')
class JumpArtifactDeterministic:
    hp = {
        'max_n_jumps': (2, 5),
        'shift_factor': (0.1, 1.0)
    }
    def __init__(self, max_n_jumps=2, shift_factor=0.1, smooth_width_min=2, smooth_width_max=12):
        self.max_n_jumps = int(max_n_jumps)
        self.shift_factor = shift_factor
        self.smooth_width_min = smooth_width_min
        self.smooth_width_max = smooth_width_max

    def __call__(self, x, left_buffer, right_buffer):
        noisy_segment = x.copy()
        
        # time flip so we can apply the logic below in either direction
        time_flip = np.random.choice([-1, 1]) 
        if time_flip == -1:
            noisy_segment = np.flip(noisy_segment)
        
        # sample n artifacts
        n_jumps = np.random.choice(np.arange(1, self.max_n_jumps + 1)) # make inclusive
        
        # sample artifact starts and shift factors
        min_start = 1 # don't start at 0 because this would shift whole segment instead of creating jump
        # needs to start early enough that there is enough room to smooth jump (with smallest smoothing window)
        max_start = len(x) - self.smooth_width_min - 2 
        artifact_starts = np.sort(np.random.choice(np.arange(min_start, max_start + 1), size=n_jumps, replace=False))
        artifact_shift_factors = self.shift_factor * np.random.choice([-1, 1], size=n_jumps)
        
        # loop through & apply shifts
        for idx, a_start in enumerate(artifact_starts):
            # sample smoothing window (how many samples to smooth)
            # smooth window needs to fit in between a_start and end of x with at least a one sample gap
            _smooth_max = min(self.smooth_width_max, len(x) - a_start - 2)
            a_smooth_win = np.random.choice(np.arange(self.smooth_width_min, _smooth_max + 1)) # make inclusive
            x_post_smooth = noisy_segment[a_start + a_smooth_win:]
            # add jump to x_post_smooth, scale it by width of smooth window (want to control jump/sec)
            x_post_smooth += artifact_shift_factors[idx] * (a_smooth_win / 4)  # get smooth win in secs
            
            # fill in parts to be smoothed
            # fit cubic spline
            # get pre-smooth, unsmoothed artifact, post-smooth
            train_x = np.concatenate([
                np.arange(a_start),  # everywhere but where smoothing occurs
                np.arange(a_start + a_smooth_win, len(x))
            ])
            train_y = np.concatenate([
                noisy_segment[:a_start],
                noisy_segment[a_start + a_smooth_win:],
            ])
            spline = CubicSpline(train_x, train_y)
            # fill in smoothed parts
            noisy_segment[a_start:a_start + a_smooth_win] = spline(np.arange(a_start, a_start + a_smooth_win))
            
            # zero out negative entries
            noisy_segment[noisy_segment < 0] = 0
            
        # Flip the segment back to original time order if it was reversed
        if time_flip == -1:
            noisy_segment = np.flip(noisy_segment)
        
        return noisy_segment.astype(np.float32)
    
@aug_export('Permute_Det')
class PermuteDeterministic:
    hp = {
        'n_splits': (2, 20)
    }
    """ Splits segments into chunks and permutes them """
    def __init__(self, n_splits=10):
        self.n_splits = int(n_splits)

    def __call__(self, x, left_buffer, right_buffer):
        orig_steps = np.arange(x.shape[0])
        splits = np.array_split(orig_steps, self.n_splits)
        np.random.shuffle(splits)
        warp_idx = np.concatenate(splits)
        x_warped = x[warp_idx]
        return x_warped.astype(np.float32)

@aug_export('ConstantAmplitudeScale_Det')
class ConstantAmplitudeScalingDeterministic:
    hp = {
        'scale': (0.1, 2.0)
    }
    """ Scale EDA by constant factor across the window """
    def __init__(self, scale=1):
        self.scale = scale

    def __call__(self, x, left_buffer, right_buffer):
        return (x * self.scale).astype(np.float32)

@aug_export('Flip')
class Flip:
    hp = {}
    """ Flips segments around horizontal axis """
    def __init__(self):
        pass

    def __call__(self, x, left_buffer, right_buffer):
        flip = -1
        x_flip = flip * x + (2 * np.mean(x))
        return x_flip.astype(np.float32)

class ExtractComponent:
    hp = {}
    def __init__(self, component, method="highpass"):
        self.component = component
        self.method = method

    def __call__(self, x, left_buffer, right_buffer):
        # print("method", self.method)
        decomposed = nk.eda_phasic(x, sampling_rate=4, method=self.method)
        return decomposed[f"EDA_{self.component}"].to_numpy().astype(np.float32)

@aug_export('ExtractPhasic')
class ExtractPhasic(ExtractComponent):
    def __init__(self, method="highpass"):
        print("init extract phasic")
        super().__init__("Phasic", method)

@aug_export('ExtractTonic')
class ExtractTonic(ExtractComponent):
    def __init__(self, method="highpass"):
        print("init extract tonic")
        super().__init__("Tonic", method)

if __name__=='__main__':
    import json
    with open('augment_params.json', 'w') as f:
        augment_cfg = {'augment_cfg': [[k, v, AUGMENTERS_HPS[k]]for k,v in AUGMENTERS_PARAMS.items()]}
        json.dump(augment_cfg, f, indent=4)