import zarr
import numpy as np
import os
import wget
import scipy.io as sio
import copy
import json
import mne

from .. import transforms
from .BaseDataset import BaseDataset

class SMR_BCI(BaseDataset):
    def __init__(self, master_path, pick_events=[0, 1], **kwargs):
        self.dataset_name = "SMR_BCI"
        self.event_ids = {"right_hand": 0, "feet": 1}
        self.n_subject = 14
        self.phases_name = ["T", "E"]
        self.n_phase = 2
        self.tmin = 0
        self.tmax = 4
        self.sfreq = 512
        self.ch_names = ['FCC3', 'FCCz', 'FCC4',
                         'C5h', 'C3', 'C3h', 'C1h', 'Cz', 'C2h', 'C4h', 'C4', 'C6h',
                         'CCP3', 'CCPz', 'CCP4'],
        self.n_time = int((self.tmax-self.tmin)*self.sfreq)
        self.n_channel = len(self.ch_names)
        super(SMR_BCI, self).__init__(master_path, pick_events, **kwargs)
        
    def __call__(self, **kwargs):
        return super().__call__(**kwargs)
    
    def _get_attrs(self):
        return super()._get_attrs()
    
    def _write_metadata(self, dest):
        return super()._write_metadata(dest)
    
    def load(self):
        return super().load()
    
    def is_path_empty(self, path):
        return super().is_path_empty(path)

    def download(self):
        pass
    
    def _set_mode(self, n_band=0):
        if n_band > 1:
            self.mode = "multi-view"
            self.epoch_axis = -3 # (...,band, channel, time)
            self.shape_name = ("trial", "band", "channel", "time")
        else:
            self.mode = "raw"
            self.epoch_axis = -2 # (..., channel, time)
            self.shape_name = ("trial", "channel", "time")

    def convert_raw_to_zarr(self):
        source = os.path.join(self.master_path, self.dataset_name, "raw.mat")
        dest = os.path.join(self.master_path, self.dataset_name, "raw.zarr")
            
        s_name = "S{:02d}{}.mat"
        d_name = "S{:02d}{}.zarr"
        
        self._set_mode()
        
        for subject in range(1, self.n_subject + 1):
            print("transforming npy to zarr S{}".format(subject))
            for phase in self.phases_name:
                
                source_path = os.path.join(source, s_name.format(subject, phase))
                dest_path = os.path.join(dest, d_name.format(subject, phase))
                
                if self.is_path_empty(os.path.join(dest_path, "x")):
                        
                    mat = sio.loadmat(source_path.format(subject, phase))["data"][0]
                    x, y = [], []
                    for run in range(mat.size):
                        x_run = np.transpose(mat[run][0,0][0])
                        t_run = np.squeeze(mat[run][0,0][1])
                        y_run = np.squeeze(mat[run][0,0][2])
                        # if t_run.size:
                        for t, y_ in zip(t_run, y_run):
                            start = t+int(self.tmin*self.sfreq)
                            stop = t+int(self.tmax*self.sfreq)
                            x.append(x_run[:, start:stop])
                            y.append(y_ - 1)
                        # y.append(y_run - 1)
                    x = np.array(x)
                    y = np.array(y)
                    # y = np.concatenate(y)
                     
                    z = zarr.open_group(dest_path, mode="w")
                    
                    x_zarr = z.create_dataset("x", shape=x.shape, chunks=(x.shape[0],))
                    x_zarr[:] = x
                    
                    y_zarr = z.create_dataset("y", shape=y.shape, chunks=(y.shape[0],))
                    y_zarr[:] = y
                    
                    times = np.arange(self.tmin, self.tmax, 1/(self.sfreq)) # tmin=0, tmax=4
                    times_zarr = z.create_dataset("times", shape=times.shape, chunks=(len(times)))
                    times_zarr[:] = times
                    print("raw data are saved at: ", dest_path)
                else:
                    print("raw data are saved at: ", dest_path)

        # write metadata
        self._write_metadata(dest=dest)

    def fetch_zarr(self, name="filter.zarr", **kwargs):
        dest = os.path.join(self.master_path, self.dataset_name, name)

        self.data = []
        d_name = "S{:02d}{}.zarr"
        for subject in range(1, self.n_subject + 1):
            data_phase = []
            for phase in self.phases_name:
                dest_path = os.path.join(dest, d_name.format(subject, phase))
                print("feching: ", dest_path)
                z = zarr.open_group(dest_path, mode="r")
                data_phase.append(z)
            self.data.append(data_phase)
        
        with open(os.path.join(dest, ".metadata"), "r") as json_file:
            kwargs = json.load(json_file)
            for name in kwargs:
                if name not in ["master_path", "pick_events"]:
                    setattr(self, name, kwargs[name])
                
    def preprocess(
        self,
        ch_names=['FCC3', 'FCCz', 'FCC4',
                  'C5h', 'C3', 'C3h', 'C1h', 'Cz', 'C2h', 'C4h', 'C4', 'C6h',
                  'CCP3', 'CCPz', 'CCP4'],
        crop=[0, 4],
        sfreq=100,
        filt_bank=[8, 30],
        order=5,
        name="filter.zarr",
        **kwargs
    ):
        
        dest = os.path.join(self.master_path, self.dataset_name, name)
        
        if isinstance(filt_bank[0], int) or isinstance(filt_bank[0], float):
            filt_bank = [
                filt_bank
            ]  # change format: filt_bank = [8, 30] --> filt_bank = [[8, 30]]
        
        self.fetch_zarr(name="raw.zarr")
        
        self.orig_ch_names = copy.deepcopy(self.ch_names)
        self.ch_names = ch_names
        self.tmin, self.tmax  = crop
        self.orig_sfreq = copy.deepcopy(self.sfreq)
        self.sfreq = sfreq
        self.filt_bank = filt_bank
        self.n_channel = len(ch_names)
        self.n_time = int(self.sfreq * (self.tmax - self.tmin))

        self._set_mode(n_band=len(filt_bank))
            
        transform_obj = transforms.Pipeline(
            steps=[
                transforms.Pick_channels(
                    orig_ch_names=self.orig_ch_names, pick_ch_names=self.ch_names, axis=-2
                ),
                transforms.Crop(tmin=self.tmin, tmax=self.tmax, times=self.data[0][0]["times"][:], axis=-1),
                transforms.Resample(orig_sfreq=self.orig_sfreq, new_sfreq=self.sfreq, axis=-1),
                transforms.ButterFilterBank(filt_bank=filt_bank, sfreq=self.sfreq, order=order, axis=-1),
            ]
        )
        
        # loop to reduce memory usage.
        d_name = "S{:02d}{}.zarr"
        for subject in range(len(self.data)):
            print("\nTransforming data of subject {}.".format(subject+1))
            for idx, phase in enumerate(self.phases_name):
                dest_path = os.path.join(dest, d_name.format(subject+1, phase))
                if self.is_path_empty(os.path.join(dest_path, "x")):
                    z = zarr.open_group(dest_path, mode="w")
                    
                    x = transform_obj(self.data[subject][idx]["x"][:])
                    x_chunks = self.data[subject][idx]["x"].chunks[0]
                    x_zarr = z.create_dataset("x", shape=x.shape, chunks=(x_chunks,))
                    x_zarr[:] = x
                    
                    y = self.data[subject][idx]["y"][:]
                    y_chunks = self.data[subject][idx]["y"].chunks[0]
                    y_zarr = z.create_dataset("y", shape=y.shape, chunks=(y_chunks,))
                    y_zarr[:] = y
                    
                    times = np.arange(self.tmin, self.tmax, 1/(self.sfreq))
                    times_zarr = z.create_dataset("times", shape=times.shape, chunks=(len(times)))
                    times_zarr[:] = times
                    print("processed data are saved at: ", dest_path)
                else:
                    print("processed data are saved at: ", dest_path)

        # write metadata
        self._write_metadata(dest=dest)
    
    def split(self, scheme, subject_idx, n_splits=5, random_state=42, **kwargs):
        return super().split(scheme, subject_idx, n_splits, random_state, **kwargs)
    
    def subject_dependent(self, subject_idx, n_splits=5, random_state=42, **kwargs):
        
        x = copy.deepcopy(self.data[subject_idx][0]["x"][:])  # train phase
        y = copy.deepcopy(self.data[subject_idx][0]["y"][:])
        x_test = copy.deepcopy(self.data[subject_idx][1]["x"][:])  # test phase
        y_test = copy.deepcopy(self.data[subject_idx][1]["y"][:])
        train_data = (x, y)
        test_data = (x_test, y_test)
        return super()._split_subject_dependent(train_data, test_data, subject_idx, n_splits, random_state, **kwargs)

    def subject_independent(self, subject_idx, n_splits=5, random_state=42, **kwargs):

        x, y = [], []
        for subject in range(self.n_subject):
            x_, y_ = [], []
            for phase in range(len(self.phases_name)):
                x_.append(self.data[subject][phase]["x"][:]) # all trials
                y_.append(self.data[subject][phase]["y"][:])
            x.append(np.concatenate(x_))
            y.append(np.concatenate(y_))
        x_test = copy.deepcopy(self.data[subject_idx][1]["x"][:])  # phase test
        y_test = copy.deepcopy(self.data[subject_idx][1]["y"][:])
        train_data = (x, y)
        test_data = (x_test, y_test)
        return super()._split_subject_independent(train_data, test_data, subject_idx, n_splits, random_state, **kwargs)

    def get_data(self, fold):
        return super().get_data(fold)