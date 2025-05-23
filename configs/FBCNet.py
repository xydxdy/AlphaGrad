import copy

def config(dataset, monitor="val_acc", classes=[0, 1], adaptive=None, data_name="filterbank.zarr"):
    
    n_class = len(classes)
    n_channel = 20
    n_time = 400
    crop = [0, 4]
    filt_bank = [[4,8],[8,12],[12,16],[16,20],[20,24],[24,28],[28,32],[32,36],[36,40]]
    n_band = len(filt_bank)
    sfreq = 100
    
    if dataset == "SMR_BCI":
        n_channel = 15
    if dataset == "BCIC2b":
        n_channel = 3
        
    # checkpoint
    if monitor == "val_loss":
        mode = "min"
    elif monitor == "val_acc":
        mode = "max"
        
    adaptive = None

    exp = {
        "network": {"n_channel": n_channel, "n_time": n_time, "n_class": n_class, "n_band": n_band},
        "classes": classes,
        "batch_size": 10,
        "losses": {
            "NLLLoss": {
                "kwargs": {"reduction": "mean"},
                "weight": 1.0
            },
        },
        "classifier_index": -1,
        "adaptive_loss": adaptive,
        "optimizer": {
            "Adam": {"kwargs": {"lr": 1e-3, "betas": (0.9, 0.999), "eps": 1e-08}}
        },
        "scheduler": {
            "ReduceLROnPlateau": {
                "kwargs": {
                    "mode": mode,
                    "factor": 0.5,
                    "patience": 5,
                    "min_lr": 1e-3,
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

    subject_independent = copy.deepcopy(exp)
    subject_independent["batch_size"] = 100


    # experimental configurations
    config = {
        "seed": 20230803,
        "preprocess": {
            "crop": crop,
            "sfreq": sfreq,
            "filt_bank": filt_bank,
            "name": data_name,
        },
        "dataset": {"transform": None, "to_tensor": True},
        "subject_dependent": subject_dependent,
        "subject_independent": subject_independent,
    }

    return config