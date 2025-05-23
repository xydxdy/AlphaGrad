import copy

def config(dataset, monitor="val_acc", classes=[0, 1], adaptive=None, data_name="filter.zarr"):

    n_class = len(classes)
    n_channel = 20
    n_time = 400
    crop = [0, 4]
    filt_bank = [8, 30]
    sfreq = 100
    
    if dataset == "SMR_BCI":
        n_channel = 15
    if dataset == "BCIC2b":
        n_channel = 3
            
    if dataset.endswith("SSVEP"):
        classes = [0, 1, 2, 3]
        n_class = len(classes)
        n_channel = 10
        n_time = 400
        filt_bank = [0.5, 25]
    elif dataset.endswith("ERP"):
        classes = [0, 1]
        n_class = len(classes)
        n_channel = 32
        n_time = 80
        crop = [0, 0.8]
        filt_bank = [0.5, 40]
        
    # checkpoint
    if monitor == "val_loss":
        mode = "min"
    elif monitor == "val_acc":
        mode = "max"
        
    adaptive = None

    exp = {
        "network": {"n_channel": n_channel, "n_time": n_time, "n_class": n_class, "dropout": 0.5},
        "classes": classes,
        "batch_size": 10,
        "losses": {
            "CrossEntropyLoss": {
                "kwargs": {"reduction": "mean"},
                "weight": 1.0
            },
        },
        "classifier_index": -1,
        "adaptive_loss": adaptive,
        "optimizer": {
            "Adam": {"kwargs": {"lr": 1e-2, "betas": (0.9, 0.999), "eps": 1e-08}}
        },
        "scheduler": {
            "ReduceLROnPlateau": {
                "kwargs": {
                    "mode": mode,
                    "factor": 0.5,
                    "patience": 5,
                    "min_lr": 1e-2,
                    "verbose": True,
                },
                "monitor": monitor,
            }
        },
        "check_stoploop": {
            "EarlyStopping": {
                "kwargs": {
                    "monitor": monitor,
                    "patience": 20,
                    "mode": mode,
                    "min_delta": 1e-6,
                    "min_epoch": 50,
                    "max_epoch": 100,
                }
            }
        },
        "checkpoint": {
            "Checkpoint": {
                "kwargs": {"monitor": monitor, "mode": mode}
            }
        },
    }

    subject_dependent = copy.deepcopy(exp)
    subject_dependent["batch_size"] = 10
    subject_dependent["network"]["dropout"] = 0.5

    subject_independent = copy.deepcopy(exp)
    subject_independent["batch_size"] = 100
    subject_independent["network"]["dropout"] = 0.25

    # experimental configurations
    config = {
        "seed": 20230803,
        "preprocess": {
            "crop": crop,
            "sfreq": sfreq,
            "filt_bank": filt_bank,
            "name": data_name,
        },
        "dataset": {"transform": None, "expand_dims_axis": 0, "to_tensor": True},
        "subject_dependent": subject_dependent,
        "subject_independent": subject_independent,
    }

    return config