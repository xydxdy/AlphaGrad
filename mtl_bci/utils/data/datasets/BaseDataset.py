import numpy as np
import zarr
import os
import json
import copy
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn import preprocessing
from imblearn.under_sampling import RandomUnderSampler
from .. import transforms

from abc import abstractmethod

class BaseDataset:
    def __init__(self, master_path, pick_events=None, random_under_sampling=None, **kwargs):
        self.master_path = os.path.join(master_path, "datasets")
        self.pick_events = pick_events
        self.data = None # data is zarr when calling fetch_zarr()
        self.mode = "raw"  # raw, multi_view
        self.epoch_axis = -2 # (..., channel, time)
        self.random_under_sampling = random_under_sampling
        for k in kwargs.keys():
            self.__setattr__(k, kwargs[k])
        
    def __call__(self, load=False, name="filter.zarr", **kwargs):
        """ prepared dataset for split to training
        """
        if load:
            self.load()
        self.preprocess(name, **kwargs)
        self.fetch_zarr(name)
        
    def _get_attrs(self):
        """ get class attributes as a dict
        """
        attrs = copy.deepcopy(self.__dict__)
        try:
            del attrs["data"]  # remove preload data from dict
            del attrs["master_path"]
            del attrs["pick_events"]
        except:
            pass
        return attrs
    
    def _write_metadata(self, dest):
        metadata = self._get_attrs()
        json_object = json.dumps(metadata, indent=4)
        with open(os.path.join(dest, ".metadata"), "w") as outfile:
            outfile.write(json_object)
    
    def load(self):
        """fetch raw data

        Args:
            data_path (str, optional): main path to save datasets. Defaults to "datasets".
        """
        print("loading: raw data")
        self.download()
        self.convert_raw_to_zarr()

    def is_path_empty(self, path):
        if os.path.exists(path):
            dir = os.listdir(path)
            try:
                dir.remove(".zarray")  # remove form list | incase of empty .zarr
            except:
                pass
            if len(dir) == 0:  # empty
                return True  # path exist and empty
            else:
                return False  # path exist and not empty
        else:
            return True  # path not exist
    
    @abstractmethod
    def download(self):
        pass
    
    @abstractmethod
    def _set_mode(self, n_band=None):
        """prepare data shape for create zarr
        """
        pass
    
    @abstractmethod
    def convert_raw_to_zarr(self):
        """
        """
        pass

    def fetch_zarr(self, name="filter.zarr", **kwargs):
        """get .zarr data

        Args:
            data_path (str, optional): main path of datasets. Defaults to "datasets".
            name (str, optional): name of zarr file. Defaults to "filter.zarr".

        Returns:
            data: zarr data
        """
        dest = os.path.join(self.master_path, self.dataset_name, name)

        print("feching: ", dest)
        self.data = zarr.open_group(dest, mode="r")
        with open(os.path.join(dest, ".metadata"), "r") as json_file:
            kwargs = json.load(json_file)
            for name in kwargs:
                if name not in ["master_path", "pick_events"]:
                    setattr(self, name, kwargs[name])
    
    def preprocess(
        self,
        ch_names=["FC5", "FC3", "FC1", "FC2", "FC4", "FC6", 
                  "C5", "C3", "C1", "Cz", "C2", "C4", "C6",
                  "CP5", "CP3", "CP1", "CPz", "CP2", "CP4", "CP6"],
        crop=[0, 4],
        sfreq=100,
        filt_bank=[8, 30],
        order=5,
        name="filter.zarr",
        **kwargs
    ):
        source = os.path.join(self.master_path, self.dataset_name, "raw.zarr")
        dest = os.path.join(self.master_path, self.dataset_name, name)
        
        if isinstance(filt_bank[0], int) or isinstance(filt_bank[0], float):
            filt_bank = [
                filt_bank
            ]  # change format: filt_bank = [8, 30] --> filt_bank = [[8, 30]]
            
        self.fetch_zarr(name="raw.zarr") # fetch to self.data
        
        self.orig_ch_names= copy.deepcopy(self.ch_names)
        self.ch_names = ch_names
        self.tmin, self.tmax  = crop
        self.orig_sfreq = copy.deepcopy(self.sfreq)
        self.sfreq = sfreq
        self.filt_bank = filt_bank
        self.n_channel = len(ch_names)
        self.n_time = int(self.sfreq * (self.tmax - self.tmin))

        self._set_mode(n_band=len(filt_bank))

        if self.is_path_empty(os.path.join(dest, "x")):
            
            z = zarr.open_group(dest, mode="w")
            x_zarr = z.create_dataset(
                "x",
                shape=self.x_shape,
                chunks=(
                    1,
                    1,
                ),
            )
            y_zarr = z.create_dataset(
                "y",
                shape=self.y_shape,
                chunks=(
                    1,
                    1,
                ),
            )
            times_zarr = z.create_dataset(
                "times",
                shape=self.times_shape,
                chunks=(
                    self.n_time,
                ),
            )
            
            transform_obj = transforms.Pipeline(
                steps=[
                    transforms.Pick_channels(
                        orig_ch_names=self.orig_ch_names, pick_ch_names=self.ch_names, axis=-2
                    ),
                    transforms.Crop(tmin=self.tmin, tmax=self.tmax, times=self.data["times"][:], axis=-1),
                    transforms.Resample(orig_sfreq=self.orig_sfreq, new_sfreq=self.sfreq, axis=-1),
                    transforms.ButterFilterBank(filt_bank=filt_bank, sfreq=self.sfreq, order=order, axis=-1),
                ]
            )
            # loop to reduce memory usage.
            for subject in range(len(self.data["x"])):
                print("\nTransforming data of subject {}.".format(subject+1))
                x_zarr[subject] = transform_obj(self.data["x"][subject])
                y_zarr[subject] = self.data["y"][subject]
            times_zarr[:] = np.arange(self.tmin, self.tmax, 1/(self.sfreq))

            # write metadata
            self._write_metadata(dest=dest)
        else:
            print("Data has already been processed: ", dest)
            pass

    def split(self, scheme, subject_idx, n_splits=5, random_state=42, **kwargs):
        self.scheme = scheme
        if scheme == "subject_dependent":
            self.subject_dependent(subject_idx, n_splits=n_splits, random_state=random_state, **kwargs)
        elif scheme == "subject_independent":
            self.subject_independent(subject_idx, n_splits=n_splits, random_state=random_state, **kwargs)
    
    @abstractmethod
    def subject_dependent(self, subject_idx, n_splits=5, random_state=42, **kwargs):
        train_data, test_data = (None, None), (None, None)
        self._split_subject_dependent(train_data, test_data, subject_idx, n_splits, random_state, **kwargs)
    
    @abstractmethod
    def subject_independent(self, subject_idx, n_splits=5, random_state=42, **kwargs):
        train_data, test_data = (None, None), (None, None)
        self._split_subject_independent(train_data, test_data, subject_idx, n_splits, random_state, **kwargs)
    
    def _split_subject_dependent(self, train_data, test_data, subject_idx, n_splits=5, random_state=42, **kwargs):

        self.x, self.y = train_data
        self.x_test, self.y_test = test_data
        
        if self.pick_events:
            self.x, self.y = self._pick_events(self.x, self.y, random_state, self.random_under_sampling)
            self.x_test, self.y_test = self._pick_events(self.x_test, self.y_test, random_state)

        self.train_list, self.val_list = [], []
        skf = StratifiedKFold(
            n_splits=n_splits, random_state=random_state, shuffle=True
        )
        for fold, (train_index, val_index) in enumerate(skf.split(self.x, self.y)):
            print(
                "FOLD:",
                fold,
                "\nTRAIN:",
                len(train_index),
                "\nVALIDATION:",
                len(val_index),
            )
            self.train_list.append(train_index)
            self.val_list.append(val_index)

    def _split_subject_independent(self, train_data, test_data, subject_idx, n_splits=5, random_state=42, **kwargs):
  
        self.x, self.y = train_data
        self.x_test, self.y_test = test_data
        
        mark_train = [True] * self.n_subject
        mark_train[subject_idx] = False
        train_subject_idxs = np.arange(self.n_subject)[mark_train]  # remove test subject from index list
        
        if self.pick_events:
            self.x, self.y = self._pick_events(self.x, self.y, random_state, self.random_under_sampling)
            self.x_test, self.y_test = self._pick_events(self.x_test, self.y_test, random_state)

        self.train_list, self.val_list = [], []
        kf = KFold(n_splits=n_splits, random_state=random_state, shuffle=True)
        for fold, (train_index, val_index) in enumerate(kf.split(train_subject_idxs)):
            train, val = list(train_subject_idxs[train_index]), list(
                train_subject_idxs[val_index]
            )
            self.train_list.append(train)
            self.val_list.append(val)
            print("Fold:", fold)
            print("TRAIN:", train, "\nVALIDATION:", val, "\nTEST:", subject_idx)
    
    def get_data(self, fold):
        x_train, y_train = [], []
        x_val, y_val = [], []
        train_list, val_list = self.train_list[fold], self.val_list[fold]
        if self.scheme == "subject_dependent":
            x_train = np.array([self.x[i] for i in train_list])
            y_train = np.array([self.y[i] for i in train_list])
            x_val = np.array([self.x[i] for i in val_list])
            y_val = np.array([self.y[i] for i in val_list])
        elif self.scheme == "subject_independent":
            x_train = np.concatenate([self.x[i] for i in train_list], axis=0)
            y_train = np.concatenate([self.y[i] for i in train_list], axis=0)
            x_val = np.concatenate([self.x[i] for i in val_list], axis=0)
            y_val = np.concatenate([self.y[i] for i in val_list], axis=0)
       
        return (x_train, y_train), (x_val, y_val), (self.x_test, self.y_test)
    
    
    def _pick_events(self, x, y, random_state, random_under_sampling=False):
        
        # Ensure pick_events is a list
        if isinstance(self.pick_events, (int, str)):
            self.pick_events = [self.pick_events]

        # Convert event names to event ids
        if isinstance(self.pick_events[0], str):
            self.pick_events = [self.event_ids[p] for p in self.pick_events]
        
        # mark selected samples based on y dimensions
        if isinstance(y, np.ndarray):
            y_dim = y.ndim
        elif isinstance(y, list):
            y_dim = 2 # list of np.ndarray

        if y_dim == 2:
            sel = [[y_i in self.pick_events for y_i in y_] for y_ in y]
        elif y_dim == 1:
            sel = [y_i in self.pick_events for y_i in y]
        
        # Process x, y based on type
        if isinstance(x, list):   
            x_sel, y_sel = [], []
            for x_, y_, s_ in zip(x, y, sel):
                x_tmp, y_tmp = x_[s_], y_[s_]
                if random_under_sampling:
                    x_tmp, y_tmp = self.random_under_sampler(x_tmp, y_tmp, random_state)
                x_sel.append(x_tmp)
                y_sel.append(y_tmp)

        elif isinstance(x, np.ndarray):
            if y.ndim == 2:
                x_sel, y_sel = [], []
                for x_, y_, s_ in zip(x, y, sel):
                    x_tmp, y_tmp = x_[s_], y_[s_]
                    if random_under_sampling:
                        x_tmp, y_tmp = self.random_under_sampler(x_tmp, y_tmp, random_state)
                    x_sel.append(x_tmp)
                    y_sel.append(y_tmp)
                x_sel, y_sel = np.array(x_sel), np.array(y_sel)
            elif y.ndim ==1:
                x_sel = x[sel]
                y_sel = y[sel]
                if random_under_sampling:
                    x_sel, y_sel = self.random_under_sampler(x_sel, y_sel, random_state)
        
        return x_sel, y_sel
    
    def random_under_sampler(self, x, y, random_state=42):
        rus = RandomUnderSampler(random_state=random_state)
        x_ids = np.array([np.arange(len(y)), np.arange(len(y))]).T # ids, just for fit_resample()
        x_ids_res, _ = rus.fit_resample(x_ids, y)
        ids = x_ids_res[:,0]
        ids.sort()
        
        x_res = x[ids]
        y_res = y[ids]
    
        print("******random_under_sampling*******", x_res.shape, y_res.shape)
        return x_res, y_res