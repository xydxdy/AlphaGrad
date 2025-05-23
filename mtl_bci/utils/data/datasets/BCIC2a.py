import zarr
import numpy as np
import os
import wget
import scipy.io as sio
import copy
import json

from .. import transforms
from .BaseDataset import BaseDataset

class BCIC2a(BaseDataset):
    def __init__(self, master_path, pick_events=[0, 1], **kwargs):
        self.dataset_name = "BCIC2a"
        self.event_ids = {"right": 0, "left": 1, "feet": 2, "tongue": 3}
        self.n_subject = 9
        self.n_session = 2
        self.session_name = ["T", "E"]
        self.n_trials = 288
        self.tmin = 0
        self.tmax = 4
        self.sfreq = 250
        self.ch_names = ["Fz", "FC3", "FC1", "FCz", "FC2", "FC4",
                         "C5", "C3", "C1", "Cz", "C2", "C4", "C6",
                         "CP3", "CP1", "CPz", "CP2", "CP4",
                         "P1", "Pz", "P2", "POz",
                         "EOG-1", "EOG-2", "EOG-3"]
        self.n_time = int((self.tmax-self.tmin)*self.sfreq)
        self.n_channel = len(self.ch_names)
        super(BCIC2a, self).__init__(master_path, pick_events, **kwargs)
        
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
            self.shape_name = ("subject", "session", "trial", "band", "channel", "time")
            self.x_shape = (self.n_subject, self.n_session, self.n_trials, n_band, self.n_channel, self.n_time)
            self.y_shape = (self.n_subject, self.n_session, self.n_trials)
            self.times_shape = (self.n_time)
        else:
            self.mode = "raw"
            self.epoch_axis = -2 # (..., channel, time)
            self.shape_name = ("subject", "session", "trial", "channel", "time")
            self.x_shape = (self.n_subject, self.n_session, self.n_trials , self.n_channel, self.n_time)
            self.y_shape = (self.n_subject, self.n_session, self.n_trials)
            self.times_shape = (self.n_time)
            
    def convert_raw_to_zarr(self):
        source = os.path.join(self.master_path, self.dataset_name, "raw.mat")
        dest = os.path.join(self.master_path, self.dataset_name, "raw.zarr")

        self._set_mode()

        if self.is_path_empty(os.path.join(dest, "x")):
            self.data = zarr.open_group(dest, mode="w")
            x_zarr = self.data.create_dataset(
                "x",
                shape=self.x_shape,
                chunks=(1, 1),
            )
            y_zarr = self.data.create_dataset(
                "y",
                shape=self.y_shape,
                chunks=(1, 1),
            )
            times_zarr = self.data.create_dataset(
                "times",
                shape=self.times_shape,
                chunks=(self.n_time),
            )
            
            file_name = "A{:02d}{}.mat"
            for i, subject in enumerate(range(1, self.n_subject + 1)):
                print("transforming mat to zarr S{}".format(subject))
                for j, session in enumerate(self.session_name):
                    mat = sio.loadmat(
                        os.path.join(source, file_name.format(subject, session))
                    )["data"][0]

                    x, y = [], []
                    for run in range(mat.size):
                        x_run = np.transpose(mat[run][0,0][0])
                        t_run = np.squeeze(mat[run][0,0][1])
                        y_run = np.squeeze(mat[run][0,0][2])
                        if t_run.size:
                            for t in t_run:
                                start = t+int(self.tmin*self.sfreq)
                                stop = t+int(self.tmax*self.sfreq)
                                x.append(x_run[:, start:stop])
                            y.append(y_run - 1)
                    x = np.array(x)
                    y = np.concatenate(y)
                    x_zarr[i, j] = x
                    y_zarr[i, j] = y
                        
            times_zarr[:] = np.arange(self.tmin, self.tmax, 1/(self.sfreq)) # tmin=0, tmax=4

            # write metadata
            self._write_metadata(dest=dest)
            print("raw data are saved at: ", dest)
        else:
            print("raw data are saved at: ", dest)
            pass
        
    def fetch_zarr(self, name="filter.zarr", **kwargs):
        return super().fetch_zarr(name, **kwargs)
    
    def preprocess(
        self,
        ch_names=['FC3', 'FC1', 'FCz', 'FC2', 'FC4', 
                  'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6',
                  'CP3', 'CP1', 'CPz', 'CP2', 'CP4',
                  'P1', 'Pz', 'P2'],
        crop=[0, 4],
        sfreq=100,
        filt_bank=[8, 30],
        order=5,
        name="filter.zarr",
        **kwargs
    ):
        return super().preprocess(ch_names, crop, sfreq, filt_bank, order, name, **kwargs)
    
    def split(self, scheme, subject_idx, n_splits=5, random_state=42, **kwargs):
        return super().split(scheme, subject_idx, n_splits, random_state, **kwargs)
    
    def subject_dependent(self, subject_idx, n_splits=5, random_state=42, **kwargs):
        
        x = self.data["x"][subject_idx, 0]  # T
        y = self.data["y"][subject_idx, 0]
        x_test = self.data["x"][subject_idx, 1]  # E
        y_test = self.data["y"][subject_idx, 1]
        train_data = (x, y)
        test_data = (x_test, y_test)
        return super()._split_subject_dependent(train_data, test_data, subject_idx, n_splits, random_state, **kwargs)

    def subject_independent(self, subject_idx, n_splits=5, random_state=42, **kwargs):
        
        x = self.data["x"][:]
        y = self.data["y"][:]
        x = x.reshape((x.shape[0],) + (np.prod(x.shape[1:self.epoch_axis]),) + (x.shape[self.epoch_axis:]))
        y = y.reshape(y.shape[0], -1)
        x_test = self.data["x"][subject_idx, 1] # E
        y_test = self.data["y"][subject_idx, 1]
        train_data = (x, y)
        test_data = (x_test, y_test)
        return super()._split_subject_independent(train_data, test_data, subject_idx, n_splits, random_state, **kwargs)

    def get_data(self, fold):
        return super().get_data(fold)