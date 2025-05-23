import copy

def config(dataset, monitor="val_acc", classes=[0, 1], adaptive=None, data_name="filter.zarr"):
    
    filt_bank = [8, 30]
    n_band = 1
    expand_dims_axis = -1
    transpose_axes = None
    sfreq = 100
    subsampling = 100
    n_class = len(classes)
    n_channel = 20
    n_time = 400
    crop = [0, 4]
    
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
        subsampling = 10
        
    # checkpoint
    if monitor == "val_loss":
        mode = "min"
    elif monitor == "val_acc":
        mode = "max"

    if adaptive == 'AlphaGrad-1.0':
        adaptive = {"AlphaGrad": {"kwargs": {"p":2, "lr":1.0}}}
    elif adaptive == 'AlphaGrad-0.1':
        adaptive =  {"AlphaGrad": {"kwargs": {"p":2, "lr":0.1}}}
    elif adaptive == 'AlphaGrad-0.01':
        adaptive =  {"AlphaGrad": {"kwargs": {"p":2, "lr":0.01}}}
    elif adaptive == 'AlphaGrad-0.001':
        adaptive =  {"AlphaGrad": {"kwargs": {"p":2, "lr":0.001}}}
    elif adaptive == 'AlphaGrad-0.0001':
        adaptive =  {"AlphaGrad": {"kwargs": {"p":2, "lr":0.0001}}}
    elif adaptive == 'AdaMT':
        adaptive = "AdaMT"
    elif adaptive == 'GA':
        adaptive = {"GradApprox": {"kwargs": {"policy": "HistoricalTangentSlope",  "warmup_epoch": 2, "overlap": 1}}}
    elif adaptive == 'FX':
        adaptive = None # FX
        
    exp = {
        "network": {"n_channel": n_channel, "n_time": n_time, "n_band": n_band, "subsampling": subsampling, "n_class": n_class},
        "classes": classes,
        "batch_size": 10,
        "losses": {
            "TripletLoss": {"kwargs": {"margin": 1.0}, "weight": 1.0},
            "MSELoss": {
                "kwargs": {"reduction": "mean"},
                "weight": 1.0
            },
            "CrossEntropyLoss": {
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
                    "min_lr": 1e-4,
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
        "dataset": {"transform": None, "expand_dims_axis": expand_dims_axis, "to_tensor": True, "transpose_axes": transpose_axes},
        "subject_dependent": subject_dependent,
        "subject_independent": subject_independent,
    }

    return config