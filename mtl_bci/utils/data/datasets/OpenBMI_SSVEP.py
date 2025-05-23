import zarr
import numpy as np
import os
import wget
import scipy.io as sio
import copy
import json

from .. import transforms
from .BaseDataset import BaseDataset

class OpenBMI_SSVEP(BaseDataset):
    def __init__(self, master_path, pick_events=None, **kwargs):
        self.dataset_name = "OpenBMI_SSVEP"
        self.event_ids = {"up": 0, "left": 1, "right": 2, "down": 3} # 12, 8.57, 6.67, 5.45 Hz
        self.n_subject = 54
        self.n_session = 2
        self.phases_name = ["EEG_SSVEP_train", "EEG_SSVEP_test"]
        self.n_phase = 2
        self.n_trials = 100
        self.tmin = 0
        self.tmax = 4
        self.sfreq = 1000
        self.ch_names = ['Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'FC5', 'FC1', 'FC2', 'FC6', 'T7', 
                        'C3','Cz','C4', 'T8', 'TP9', 'CP5', 'CP1', 'CP2', 'CP6', 'TP10', 'P7', 'P3', 
                        'Pz', 'P4', 'P8', 'PO9', 'O1', 'Oz', 'O2', 'PO10', 'FC3', 'FC4', 'C5', 'C1',
                        'C2', 'C6', 'CP3','CPz', 'CP4', 'P1', 'P2', 'POz', 'FT9', 'FTT9h', 'TTP7h', 
                        'TP7', 'TPP9h', 'FT10','FTT10h','TPP8h', 'TP8', 'TPP10h', 'F9', 'F10', 
                        'AF7', 'AF3', 'AF4', 'AF8', 'PO3','PO4']
        self.n_time = int((self.tmax-self.tmin)*self.sfreq)
        self.n_channel = len(self.ch_names)
        super(OpenBMI_SSVEP, self).__init__(master_path, pick_events, **kwargs)
        
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
        try:
            save_path = os.path.join(self.master_path, self.dataset_name, "raw.mat")
            if save_path is not None:
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
            for session in range(1, self.n_session + 1):
                for subject in range(1, self.n_subject + 1):
                    file_name = "sess{:02d}_subj{:02d}_EEG_SSVEP.mat".format(
                        session, subject
                    )
                    file_path = os.path.join(save_path, file_name)
                    if not os.path.exists(file_path):
                        # os.remove(file_path)
                        print(
                            "\nDownload is being processed on session: {} subject: {}".format(
                                session, subject
                            )
                        )
                        url = "ftp://parrot.genomics.cn/gigadb/pub/10.5524/100001_101000/100542/session{}/s{}/{}".format(
                            session, subject, file_name
                        )
                        wget.download(url, file_path)
                    else:
                        pass
        except:
            raise Exception(
                "Path Error: file does not exist, please direccly download at http://gigadb.org/dataset/100542"
            )

    def _set_mode(self, n_band=0):
        if n_band > 1:
            self.mode = "multi-view"
            self.epoch_axis = -3 # (...,band, channel, time)
            self.shape_name = ("subject", "session", "phase", "trial", "band", "channel", "time")
            self.x_shape = (self.n_subject, self.n_session, self.n_phase, self.n_trials, n_band, self.n_channel, self.n_time)
            self.y_shape = (self.n_subject, self.n_session, self.n_phase, self.n_trials)
            self.times_shape = (self.n_time)
        else:
            self.mode = "raw"
            self.epoch_axis = -2 # (..., channel, time)
            self.shape_name = ("subject", "session", "phase", "trial", "channel", "time")
            self.x_shape = (self.n_subject, self.n_session, self.n_phase, self.n_trials , self.n_channel, self.n_time)
            self.y_shape = (self.n_subject, self.n_session, self.n_phase, self.n_trials)
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
                chunks=(1, 1, 1),
            )
            y_zarr = self.data.create_dataset(
                "y",
                shape=self.y_shape,
                chunks=(1, 1, 1),
            )
            times_zarr = self.data.create_dataset(
                "times",
                shape=self.times_shape,
                chunks=(self.n_time),
            )
            
            file_name = "sess{:02d}_subj{:02d}_EEG_SSVEP.mat"
            for i, subject in enumerate(range(1, self.n_subject + 1)):
                print("transforming mat to zarr S{}".format(subject))
                for j, session in enumerate(range(1, self.n_session + 1)):
                    mat = sio.loadmat(
                        os.path.join(source, file_name.format(session, subject))
                    )
                    if self.ch_names is None:
                        self.ch_names = [
                            m.item()
                            for m in mat["EEG_SSVEP_train"][0, 0]["chan"]
                            .squeeze()
                            .tolist()
                        ]

                    for k, phase in enumerate(self.phases_name):
                        x = mat[phase]["smt"][0, 0] # (4000, 100, 62)
                        y = mat[phase]["y_dec"][0, 0].squeeze() - 1
                        x = np.transpose(x, axes=(1, 2, 0)) # (100, 62, 4000)
                        x_zarr[i, j, k] = x
                        y_zarr[i, j, k] = y
                        
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
        ch_names=["P7", "P3", "Pz", "P4", "P8",
                  "PO9", "PO10", 
                  "O1", "Oz", "O2"],
        crop=[0, 4],
        sfreq=100,
        filt_bank=[0.5, 25],
        order=5,
        name="filter.zarr",
        **kwargs
    ):
        return super().preprocess(ch_names, crop, sfreq, filt_bank, order, name, **kwargs)
    
    def split(self, scheme, subject_idx, n_splits=5, random_state=42, **kwargs):
        return super().split(scheme, subject_idx, n_splits, random_state, **kwargs)
    
    def subject_dependent(self, subject_idx, n_splits=5, random_state=42, **kwargs):
        
        x = np.concatenate(self.data["x"][subject_idx, :, 0], axis=0)  # EEG_SSVEP_train from all session
        y = np.concatenate(self.data["y"][subject_idx, :, 0], axis=0)
        x_test = np.concatenate(self.data["x"][subject_idx, :, 1], axis=0)  # EEG_SSVEP_test from all session
        y_test = np.concatenate(self.data["y"][subject_idx, :, 1], axis=0)
        train_data = (x, y)
        test_data = (x_test, y_test)
        return super()._split_subject_dependent(train_data, test_data, subject_idx, n_splits, random_state, **kwargs)

    def subject_independent(self, subject_idx, n_splits=5, random_state=42, **kwargs):
        
        x = self.data["x"][:]
        y = self.data["y"][:]
        x = x.reshape((x.shape[0],) + (np.prod(x.shape[1:self.epoch_axis]),) + (x.shape[self.epoch_axis:]))
        y = y.reshape(y.shape[0], -1)
        x_test = np.concatenate(self.data["x"][subject_idx, :, 1], axis=0)  # EEG_SSVEP_test from all session
        y_test = np.concatenate(self.data["y"][subject_idx, :, 1], axis=0)
        train_data = (x, y)
        test_data = (x_test, y_test)
        return super()._split_subject_independent(train_data, test_data, subject_idx, n_splits, random_state, **kwargs)

    def get_data(self, fold):
        return super().get_data(fold)