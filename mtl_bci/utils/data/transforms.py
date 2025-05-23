from typing import Any
import numpy as np
import scipy.signal as signal
import torch


class Pipeline:
    """Pipeline of transforms
    Sequentially apply a list of transforms. Intermediate steps of the pipeline must be ‘transforms’, 
    that is, they must implement __call__ method. 

    Args:
        steps (List of transform): implementing __call__ that are chained in sequential order. 

    Example:
            transforms.Pipeline([
                Resample(...),
                Pick_channels(...),
            ])
    """

    def __init__(self, steps):
        self.steps = steps

    def __call__(self, data):
        for s in self.steps:
            data = s(data)
        return data

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        for t in self.steps:
            format_string += "\n"
            format_string += "    {0}".format(t)
        format_string += "\n)"
        return format_string

class Resample:
    """Resample data to new sampling frequency in time or frequency domain using Fourier method along the given axis

    Args:
        orig_sfreq (int): original sampling frequency.
        new_sfreq (int): new sampling frequency.
        axis (int, optional): EEG time axis. Defaults to -1.
        domain (str, optional): A string indicating the domain of the input data. Defaults to "time": 
            `time` Consider the input data as time-domain, 
            `freq` Consider the input data as frequency-domain.
    """
    def __init__(self, orig_sfreq, new_sfreq, axis=-1, domain="time"):
        self.orig_sfreq = orig_sfreq
        self.new_sfreq = new_sfreq
        self.axis = axis
        self.domain = domain

    def __call__(self, data):
        """built-in method to call Resample.
        
        Args:
            data (ndarray): EEG signal to be resampled
        """
        if self.new_sfreq < self.orig_sfreq:
            print("Resample data from {} Hz to {} Hz.".format(self.orig_sfreq, self.new_sfreq))
            new_smp_point = int((data.shape[self.axis] / self.orig_sfreq) * self.new_sfreq)
            data_resampled = signal.resample(data, new_smp_point, axis=self.axis, domain=self.domain)
        else:
            raise Exception("Sorry, new_sfreq must less than orig_sfreq")
        return data_resampled


class Pick_channels:
    """Pick some channels form EEG data.
    
    Args:
            orig_ch_names (int): original channels
            pick_ch_names (int): pick channels
            axis (int, optional): _description_. Defaults to -2.
    """
    def __init__(self, orig_ch_names, pick_ch_names, axis=-2):
        self.orig_ch_names = np.char.upper(orig_ch_names)
        self.pick_ch_names = np.char.upper(pick_ch_names)
        self.axis = axis

    def __call__(self, data):
        """built-in method to call Pick_channels.
        
        Args:
            data (ndarray): EEG data

        Returns:
            ndarray: picked EEG data
        """
        self.indices = [
            np.where(self.orig_ch_names == c)[0][0] for c in self.pick_ch_names
        ]
        print("Pick {} channels.".format(len(self.pick_ch_names)))
        new_data = np.take(data, self.indices, axis=self.axis)
        return new_data


class Crop:
    """Crop a time interval from the trial.

    Args:
        tmin (float): Start time of selection in seconds.
        tmax (float): Stop time of selection in seconds.
        times (ndarray): Time vector in seconds.
        axis (int, optional): axis of the time. Defaults to -1.
    """

    def __init__(self, tmin, tmax, times, axis=-1):
        self.tmin = tmin
        self.tmax = tmax
        self.times = times
        self.axis = axis

    def __call__(self, data):
        """built-in method to call Crop.
        
        Args:
            data (ndarray): EEG data
            
        Returns:
            ndarray: cropped EEG data
        """
        print("Crop tmin={}, tmax={}".format(self.tmin, self.tmax))
        selected = (self.times >= self.tmin) * (self.times < self.tmax)
        self.indices = np.where(selected)[0]
        new_data = np.take(data, self.indices, axis=self.axis)
        return new_data

class ButterFilterBank:
    """The Butterworth filter that can filter EEG data as filter-bank.

        Args:
            filt_bank (array_like): bands to cut
            sfreq (int): sampling frequency.
            order (int, optional): filter order. Defaults to 2.
            axis (int, optional): axis. Defaults to -1.
            btype (str, optional): available options are 'low', 'high' and 'band'. Defaults to "band".
            filtering (str, optional): available options are 'filtfilt' and 'filter'. Defaults to "filtfilt".
        """
    def __init__(
        self, filt_bank, sfreq, order=2, axis=-1, btype="band", filtering="filtfilt"
    ):
        if isinstance(filt_bank[0], int):
            filt_bank = [
                filt_bank
            ]  # change type: filt_bank = [8, 30] --> filt_bank = [[8, 30]]
        self.filt_bank = filt_bank
        self.sfreq = sfreq
        self.order = order
        self.axis = axis
        self.btype = btype
        self.filtering = filtering
        
    def butter_filter(self, data, band, fs, order):
        nyq = 0.5 * fs
        if isinstance(band, int) or isinstance(band, float): # btype = low, high
            Wn = band / nyq
        else:
            Wn = list(np.array(band) / nyq) # btype = band
        b, a = signal.butter(N=order, Wn=Wn, btype=self.btype)
        y = signal.__dict__[self.filtering](b, a, data, axis=self.axis)
        return y
    
    def __call__(self, data):
        """built-in method to call ButterFilterBank.
        
        Args:
            data (ndarray): EEG data
            
        Returns:
            ndarray: filtered EEG or filter-bank EEG
        """
        # print("filtering band {}, order {}".format(self.filt_bank, self.order))
        # initialize output
        out = np.zeros([len(self.filt_bank), *data.shape])

        # repetitively filter the data.
        for i, band in enumerate(self.filt_bank):
            out[i] = self.butter_filter(
                data=data,
                band=band,
                fs=self.sfreq,
                order=self.order
            )

        # remove any redundant 3rd dimension
        if len(self.filt_bank) == 1:
            out = np.squeeze(out, axis=0)
        else:
            out = np.moveaxis(out, 0, -3) # move band axis to -3 (..., band, channel, time)
        # new_data = torch.from_numpy(out).float()
        return out