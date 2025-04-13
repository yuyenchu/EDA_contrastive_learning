import copy
import tensorflow as tf
import neurokit2 as nk
import numpy as np
import scipy

from functools import lru_cache
from sys import exit
from scipy.fft import fft, ifft
from scipy.signal import iirnotch, iirpeak, filtfilt
from scipy.interpolate import CubicSpline

AUGMENTERS_DICT = {}

def register_internal_serializable(path, symbol):
    global AUGMENTERS_DICT
    if isinstance(path, (list, tuple)):
        name = path[0]
    else:
        name = path
    AUGMENTERS_DICT[name] = symbol
    
class aug_export:
    def __init__(self, path):
        self.path = path
        
    def __call__(self, symbol):
        register_internal_serializable(self.path, symbol)
        return symbol

class DataAugmenter():
    def __init__(self, augment_dict={}, prob=0.5):
        self.augmenters = [AUGMENTERS_DICT[k](**v) for k,v in augment_dict.items()]
        self.prob = prob if isinstance(prob, list) else [prob]*len(self.augmenters)
        
    def __call__(self, x, xl, xr):
        shape = tf.shape(x)
        for p, aug in zip(self.prob, self.augmenters):
            if (p > tf.random.uniform(maxval=1, shape=[1])):
                x = tf.numpy_function(aug, [x, xl, xr], [tf.float32])[0]
                x = tf.reshape(x, shape)
        return x

@aug_export('LowPassFilter_Det')
class LowPassFilterDeterministic():
    def __init__(self, data_freq=4, highcut_hz=0.05, **kwargs):
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

@aug_export('GaussianNoise_Det')
class GaussianNoiseDeterministic:
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

@aug_export('BandstopFilter_Det')
class BandstopFilterDeterministic:
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
    """ Shifts the window left or right by a number of samples """
    def __init__(self, shift_len=120):
        self.shift_len = shift_len

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
    """ Shifts the window left or right by a number of samples """
    def __init__(self, shift_len_min=120, shift_len_max=240):
        self.shift_min = shift_len_min
        self.shift_max = shift_len_max
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