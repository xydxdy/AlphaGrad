import os
import argparse
import mtl_bci.utils as utils
from mtl_bci import networks, Trainer

"""How to run an experiment

python run.py MIN2Net --dataset OpenBMI --classes 0 1 --data_name filter.zarr --device cuda:6
"""


def main(data, subject):
    
    data.split(
        scheme=args.scheme,
        subject_idx=subject - 1,
        n_splits=args.n_splits,
        random_state=PARAMS.seed + subject,
    )

    for fold in range(args.n_splits):
            
        if fold < args.start_fold:
            continue
        else:
            args.start_fold = 0
            
        if fold == 0:
            utils.write_log(
                filepath=os.path.join(output_path, f"subj{subject:02d}"),
                name="results.csv",
                data=["subject", "fold", "accuracy", "f1_score"],
                mode="w",
            )

        utils.set_random_seed(seed=PARAMS.seed)

        (x_train, y_train), (x_val, y_val), (x_test, y_test) = data.get_data(fold=fold)

        train_data = utils.data.Dataset(x_train, y_train, **PARAMS.dataset)
        val_data = utils.data.Dataset(x_val, y_val, **PARAMS.dataset)
        test_data = utils.data.Dataset(x_test, y_test, **PARAMS.dataset)

        """Define network
        ex.
        net = networks.MIN2Net(...)
        """
        net = networks.__dict__[args.network](**PARAMS[args.scheme].network)
        
        model = Trainer(
            net=net,
            output_path=os.path.join(
                output_path, f"subj{subject:02d}", f"fold{fold:01d}"
            ),
            seed=PARAMS.seed,
            device=args.device,
            **PARAMS[args.scheme],
        )

        accuracy, f1_score = model.train(
            train_data=train_data, val_data=val_data, test_data=test_data
        )
        utils.write_log(
            filepath=os.path.join(output_path, f"subj{subject:02d}"),
            name="results.csv",
            data=[subject, fold, accuracy, f1_score],
            mode="a",
        )
        
        del train_data, val_data, test_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run an experiment.")
    parser.add_argument(
        "network", 
        type=str, 
        help="network name")
    parser.add_argument(
        "--dataset", 
        type=str, 
        default="OpenBMI", 
        help="dataset name (default: OpenBMI)"
    )
    parser.add_argument(
        "--load_raw", 
        type=bool,
        default=False,
        help="load raw data (default: False)"
    )
    parser.add_argument(
        "--preprocess",
        type=bool,
        default=False,
        help="preprocess raw data (default: False)",
    )
    parser.add_argument(
        "--data_name", 
        type=str, 
        default="filter.zarr", 
        help="filtered data to load (default: filter.zarr)"
    )
    parser.add_argument(
        "--classes",
        nargs="+",
        type=int,
        default=[0, 1],
        help="list of classes (default: [0, 1])",
    )
    parser.add_argument(
        "--scheme",
        type=str,
        default="subject_independent",
        help="training scheme: subject_dependent and subject_independent (default: subject_independent)",
    )
    parser.add_argument(
        "--n_splits",
        type=int,
        default=5,
        help="n_splits of k_folds cross-validation (default: 5)",
    )
    parser.add_argument(
        "--subjects",
        nargs="+",
        type=int,
        default=None,
        help="run specific subject if lenght is 1, ex.--subjects 10 | run in subject range if lenght is 2, ex. --subjects 1 10 | list of subject if lenght > 2, ex. --subjects 1 2 3 4 | if None -> run all subjects (default: None)",
    )
    parser.add_argument(
        "--start_fold",
        type=int,
        default=0,
        help="start fold (default: 0)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="A torch.device (default: cuda:0)",
    )
    parser.add_argument(
        "--monitor",
        type=str,
        default='val_acc',
        help="monitor mode",
    )
    parser.add_argument(
        "--adaptive",
        type=str,
        default="FX",
        help="load experimeantal config (default: FX)",
    )
    parser.add_argument(
        "--datetime",
        type=str,
        default=None,
        help="ex: 20240314",
    )
    parser.add_argument(
        "--output_name",
        type=str,
        default="output",
        help="ex: out_multiclass",
    )
    args = parser.parse_args()

    ###################################################################
    ###                                                             ###
    ###                         configuring                         ###
    ###                                                             ###
    ###################################################################
    
    """load experimental parameters
    and convert dict to DotDict notation: support both PARAMS["seed"] or PARAMS.seed
    """
    PARAMS = utils.load_config_function(
        filepath=os.path.join("configs", args.network +".py"),
        dataset=args.dataset, monitor=args.monitor, classes=args.classes,
        adaptive=args.adaptive, data_name=args.data_name,
        to_dotdict=True,
    )
    print(PARAMS)
    
    if args.datetime:
        current_datetime = args.datetime
    else:
        current_datetime = utils.get_datetime()
    master_path = os.path.dirname(os.path.abspath(__file__))
        
    output_path = os.path.join(
        master_path,
        "output",
        args.output_name,
        args.dataset,
        args.scheme,
        args.network + "-" + args.adaptive + "-" + args.monitor,
        current_datetime,
    )
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    ###################################################################
    ###################################################################
    
    """fetch and process dataset
    ex.
    data = utils.data.datasets.OpenBMI(master_path=master_path)
    """
    
    data = utils.data.datasets.__dict__[args.dataset](master_path=master_path, pick_events=PARAMS[args.scheme].classes)
    if args.load_raw:
        data.load_raw() # Dowload raw data and save format to "raw.zarr"
    if args.preprocess:
        data.preprocess(**PARAMS.preprocess) # process "raw.zarr" and save to "filter.zarr"
    data.fetch_zarr(name=PARAMS.preprocess.name)  # fetch filter.zarr
    
    """run experiment for each subject
    """
    if args.subjects == None:
        for subject in range(1, data.n_subject + 1):
            main(data, subject)
    elif len(args.subjects) == 2:
        subj_start, subj_stop = args.subjects
        for subject in range(subj_start, subj_stop + 1):
            main(data, subject)
    else:
        for subject in args.subjects:
            main(data, subject)
