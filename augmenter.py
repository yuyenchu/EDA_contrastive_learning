import copy
from sys import exit
from functools import lru_cache

import neurokit2 as nk
import numpy as np
import scipy
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
        self.prob = prob
        
    def __call__(self, x):
        for aug in self.augmenters:
            if (self.prob > tf.random.uniform(maxval=1, shape=[1])):
                x = tf.numpy_function(aug, [x], [tf.float32])[0]
        return x

@aug_export('LowPassFilter_Det')
class LowPassFilterDeterministic():
    def __init__(self, data_freq=4, highcut_hz=0.05):
        """
        Apply low pass filter to remove frequency bands >= highcut_hz
        :param data_freq: frequency of data to apply filter to (e.g., 4Hz for EDA)
        :param highcut_hz: lower bound on frequency bands to remove
        """
        self.data_freq = data_freq
        self.highcut_hz = highcut_hz
        self.b, self.a = scipy.signal.butter(4, [highcut_hz], btype="lowpass", output="ba", fs=data_freq)

    def __call__(self, x):
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

    def __call__(self, x):
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

    def __call__(self, x):
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

    def __call__(self, x):
        return filtfilt(self.b, self.a, x, axis=0).astype(np.float32)