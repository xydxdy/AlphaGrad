import zarr
import numpy as np
import os
import wget
import scipy.io as sio
import copy
import json

from .. import transforms
from .BaseDataset import BaseDataset

class OpenBMI_ERP(BaseDataset):
    def __init__(self, master_path, pick_events=None, random_under_sampling=True, **kwargs):
        self.dataset_name = "OpenBMI_ERP"
        self.event_ids = {"target": 0, "nontarget": 1}
        self.n_subject = 54
        self.n_session = 2
        self.phases_name = ["EEG_ERP_train", "EEG_ERP_test"]
        self.n_phase = 2
        self.n_trials = [1980, 2160] # ["EEG_ERP_train", "EEG_ERP_test"]
        self.tmin = 0
        self.tmax = 0.8
        self.sfreq = 1000
        self.ch_names = ['Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'FC5', 'FC1', 'FC2', 'FC6', 'T7', 
                        'C3','Cz','C4', 'T8', 'TP9', 'CP5', 'CP1', 'CP2', 'CP6', 'TP10', 'P7', 'P3', 
                        'Pz', 'P4', 'P8', 'PO9', 'O1', 'Oz', 'O2', 'PO10', 'FC3', 'FC4', 'C5', 'C1',
                        'C2', 'C6', 'CP3','CPz', 'CP4', 'P1', 'P2', 'POz', 'FT9', 'FTT9h', 'TTP7h', 
                        'TP7', 'TPP9h', 'FT10','FTT10h','TPP8h', 'TP8', 'TPP10h', 'F9', 'F10', 
                        'AF7', 'AF3', 'AF4', 'AF8', 'PO3','PO4']
        self.n_time = int((self.tmax-self.tmin)*self.sfreq)
        self.n_channel = len(self.ch_names)
        super(OpenBMI_ERP, self).__init__(master_path, pick_events, random_under_sampling, **kwargs)
        
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
                    file_name = "sess{:02d}_subj{:02d}_EEG_ERP.mat".format(
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

    def _set_mode(self, phase=0, n_band=0):
        if n_band > 1:
            self.mode = "multi-view"
            self.epoch_axis = -3 # (...,band, channel, time)
            self.shape_name = ("subject", "session", "trial", "band", "channel", "time")
            self.x_shape = (self.n_subject, self.n_session, self.n_trials[phase], n_band, self.n_channel, self.n_time)
            self.y_shape = (self.n_subject, self.n_session, self.n_trials[phase])
            self.times_shape = (self.n_time)
        else:
            self.mode = "raw"
            self.epoch_axis = -2 # (..., channel, time)
            self.shape_name = ("subject", "session", "trial", "channel", "time")
            self.x_shape = (self.n_subject, self.n_session, self.n_trials[phase], self.n_channel, self.n_time)
            self.y_shape = (self.n_subject, self.n_session, self.n_trials[phase])
            self.times_shape = (self.n_time)
            
    def convert_raw_to_zarr(self):
        source = os.path.join(self.master_path, self.dataset_name, "raw.mat")
        dest = os.path.join(self.master_path, self.dataset_name, "raw.zarr")

        d_name = "{}.zarr"
        
        # create empty zarr
        self.data = []
        x_zarr, y_zarr, times_zarr = [], [], []
        for k, phase in enumerate(self.phases_name):
            
            self._set_mode(phase=k)
            dest_path = os.path.join(dest, d_name.format(phase))
            if self.is_path_empty(os.path.join(dest_path, "x")):
                self.data.append(zarr.open_group(dest_path, mode="w"))
                x_zarr.append(self.data[k].create_dataset(
                    "x",
                    shape=self.x_shape,
                    chunks=(1, 1),
                ))
                y_zarr.append(self.data[k].create_dataset(
                    "y",
                    shape=self.y_shape,
                    chunks=(1, 1),
                ))
                times_zarr.append(self.data[k].create_dataset(
                    "times",
                    shape=self.times_shape,
                    chunks=(self.n_time),
                ))
                print("raw data are saving at: ", dest_path)
            else:
                # Exit the function if the Zarr file already exists
                print("raw data are saved at: ", dest)
                return
            
        file_name = "sess{:02d}_subj{:02d}_EEG_ERP.mat"
        for i, subject in enumerate(range(1, self.n_subject + 1)):
            print("transforming mat to zarr S{}".format(subject))
            for j, session in enumerate(range(1, self.n_session + 1)):
                mat = sio.loadmat(
                    os.path.join(source, file_name.format(session, subject))
                )
                if self.ch_names is None:
                    self.ch_names = [
                        m.item()
                        for m in mat["EEG_ERP_train"][0, 0]["chan"]
                        .squeeze()
                        .tolist()
                    ]

                for k, phase in enumerate(self.phases_name):
                    x = mat[phase]["smt"][0, 0] # (800, 1980, 62) 
                    y = mat[phase]["y_dec"][0, 0].squeeze() - 1
                    x = np.transpose(x, axes=(1, 2, 0)) # (1980, 62, 800)
                    x_zarr[k][i, j] = x
                    y_zarr[k][i, j] = y
                    times_zarr[k][:] = np.arange(self.tmin, self.tmax, 1/(self.sfreq)) # tmin=0, tmax=0.8

            # write metadata
            self._write_metadata(dest=dest)
        
    def fetch_zarr(self, name="filter.zarr", **kwargs):
        dest = os.path.join(self.master_path, self.dataset_name, name)

        self.data = []
        d_name = "{}.zarr"
        for phase in self.phases_name:
            dest_path = os.path.join(dest, d_name.format(phase))
            print("feching: ", dest_path)
            self.data.append(zarr.open_group(dest_path, mode="r"))
        
        with open(os.path.join(dest, ".metadata"), "r") as json_file:
            kwargs = json.load(json_file)
            for name in kwargs:
                if name != "master_path":
                    setattr(self, name, kwargs[name])
    
    def preprocess(
        self,
        ch_names=["FP1", "FP2",
                  "F7", "F3", "Fz", "F4", "F8",
                  "FC5", "FC1", "FC2", "FC6", 
                  "T7", "T8",
                  "C3", "Cz", "C4",
                  "TP9", "TP10",
                  "CP5", "CP1", "CP2", "CP6",
                  "P7", "P3", "Pz", "P4", "P8",
                  "PO9", "PO10",
                  "O1", "Oz", "O2"],
        crop=[0, 0.8],
        sfreq=100,
        filt_bank=[0.5, 40],
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
            
        transform_obj = transforms.Pipeline(
            steps=[
                transforms.Pick_channels(
                    orig_ch_names=self.orig_ch_names, pick_ch_names=self.ch_names, axis=-2
                ),
                transforms.Crop(tmin=self.tmin, tmax=self.tmax, times=self.data[0]["times"][:], axis=-1),
                transforms.Resample(orig_sfreq=self.orig_sfreq, new_sfreq=self.sfreq, axis=-1),
                transforms.ButterFilterBank(filt_bank=filt_bank, sfreq=self.sfreq, order=order, axis=-1),
            ]
        )
        
        # loop to reduce memory usage.
        d_name = "{}.zarr"
        for idx, phase in enumerate(self.phases_name):
            self._set_mode(phase=idx, n_band=len(filt_bank))
            dest_path = os.path.join(dest, d_name.format(phase))
            if self.is_path_empty(os.path.join(dest_path, "x")):
                z = zarr.open_group(dest_path, mode="w")
                
                # x = transform_obj(self.data[idx]["x"][:])
                x_chunks = self.data[idx]["x"].chunks[0]
                x_zarr = z.create_dataset("x", shape=self.x_shape, chunks=(x_chunks,))
                
                # y = self.data[idx]["y"][:]
                y_chunks = self.data[idx]["y"].chunks[0]
                y_zarr = z.create_dataset("y", shape=self.y_shape, chunks=(y_chunks,))
                
                for subj in range(self.n_subject):
                    x_zarr[subj] = transform_obj(self.data[idx]["x"][subj])
                    y_zarr[subj] = self.data[idx]["y"][subj]
                
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
        
        x = np.concatenate(self.data[0]["x"][subject_idx, :], axis=0)  # EEG_ERP_train from all session
        y = np.concatenate(self.data[0]["y"][subject_idx, :], axis=0)
        x_test = np.concatenate(self.data[1]["x"][subject_idx, :], axis=0)  # EEG_ERP_test from all session
        y_test = np.concatenate(self.data[1]["y"][subject_idx, :], axis=0)
        train_data = (x, y)
        test_data = (x_test, y_test)
        return super()._split_subject_dependent(train_data, test_data, subject_idx, n_splits, random_state, **kwargs)

    def subject_independent(self, subject_idx, n_splits=5, random_state=42, **kwargs):
        
        x = np.concatenate([self.data[0]["x"][:], self.data[1]["x"][:]], axis=2) # axis=2 --> trial | (subj, sess, trial, ...)
        y = np.concatenate([self.data[0]["y"][:], self.data[1]["y"][:]], axis=2)
        x = x.reshape((x.shape[0],) + (np.prod(x.shape[1:self.epoch_axis]),) + (x.shape[self.epoch_axis:]))
        y = y.reshape(y.shape[0], -1)
        x_test = np.concatenate(self.data[1]["x"][subject_idx, :], axis=0)  # EEG_ERP_test from all session
        y_test = np.concatenate(self.data[1]["y"][subject_idx, :], axis=0)
        train_data = (x, y)
        test_data = (x_test, y_test)
        return super()._split_subject_independent(train_data, test_data, subject_idx, n_splits, random_state, **kwargs)

    def get_data(self, fold):
        return super().get_data(fold)