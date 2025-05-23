#!/usr/bin/env python
# coding: utf-8

# To do deep learning
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score
import matplotlib.pyplot as plt
import os
import pickle
import copy
import csv
import math
import time
from tqdm import tqdm

from . import callbacks
from . import losses as losses_module
from . import grads

class Trainer:
    def __init__(
        self,
        net,
        output_path=None,
        seed=None,
        device="cuda:0",
        num_threads=3,
        classes=[0, 1],
        batch_size=128,
        losses="CrossEntropyLoss",
        classifier_index=0,
        adaptive_loss=None,
        optimizer="Adam",
        scheduler="ReduceLROnPlateau",
        check_stoploop="EarlyStopping",
        checkpoint="Checkpoint",
        verbose=1,
        **kwargs
    ):
        self.net = net
        self.output_path = output_path
        self.seed = seed
        self.device = device
        self.num_threads = num_threads
        self.classes = classes
        self.batch_size = batch_size
        self.losses = losses
        self.classifier_index = classifier_index
        self.adaptive_loss = adaptive_loss
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.check_stoploop = check_stoploop
        self.checkpoint = checkpoint
        self.verbose = verbose

        # Set seed
        if isinstance(self.seed, int):
            self.set_random_seed(self.seed)

        params_dict = copy.deepcopy(self.__dict__)
        params_dict["net"] = copy.deepcopy(self.net.to("cpu").state_dict())
        self.logs = {}
        self.logs["kwargs"] = params_dict

        # Set device
        self.net.to(self.device)
        self.set_losses(self.losses)
        self.set_adaptive_loss(self.adaptive_loss)
        self.set_optimizer(self.optimizer)
        self.set_scheduler(self.scheduler)
        self.set_check_stoploop(self.check_stoploop)
        self.set_checkpoint(self.checkpoint)

        if self.output_path is not None:
            if not os.path.exists(self.output_path):
                os.makedirs(self.output_path)
            print("Results will be saved in folder : " + self.output_path)

    def train(self, train_data, val_data, test_data=None):
        # define the classes
        if self.classes is None:
            self.classes = list(set(train_data.labels))

        if self.batch_size is None:
            self.batch_size = len(train_data.labels)

        train_logs, best_loss_weights = self.train_loop(
            train_data=train_data, val_data=val_data
        )

        self.logs["train_logs"] = train_logs
        self.logs["best_loss_weights"] = best_loss_weights
        self.logs["best_net"] = self.net.state_dict()
        if self.adaptive_loss:
            self.logs["weights_history"] = self.adaptive_loss.history

        if test_data is not None:
            prob, y_pred, y_true, loss = self.evaluate(
                data=test_data, loss_weights=best_loss_weights, reduction="mean"
            )
            test_results = self.calculate_results(y_pred, y_true, classes=self.classes)
            test_results["prob"] = prob
            test_results["y_true"] = y_true
            test_results["y_pred"] = y_pred
            test_results["loss"] = loss
            self.logs["results"] = test_results
            self.print_report(self.logs)

        if self.output_path is not None:
            self.write_logs(self.logs, log_csv=True)

        return test_results["accuracy"], test_results["f1_score"]

    def train_loop(self, train_data, val_data):
        logs = []

        epoch_logs = {"epoch": 0, "val_loss": float("inf"), "val_acc": 1, "compute_time": 0}
        best_loss_weights = copy.deepcopy(self.loss_weights.detach().tolist())

        stop = False
        while not stop:
            time_start = time.time()

            (tr_yp, tr_yt, tr_loss), (val_yp, val_yt, val_loss) = self.train_val_step(
                train_data=train_data,
                val_data=val_data,
                epoch=epoch_logs["epoch"],
            )
            time_end = time.time()
            epoch_logs["compute_time"] = time_end - time_start

            train_results = self.calculate_results(tr_yp, tr_yt, classes=self.classes)
            epoch_logs.update(tr_loss)
            epoch_logs["train_acc"] = train_results["accuracy"]

            val_results = self.calculate_results(val_yp, val_yt, classes=self.classes)
            epoch_logs.update(val_loss)
            epoch_logs["val_acc"] = val_results["accuracy"]
            epoch_logs["loss_weights"] = self.loss_weights.detach().tolist()

            stop = self.check_stoploop(epoch_logs)

            if self.scheduler:
                self.scheduler.step(epoch_logs[self.moitor])

            self.checkpoint(
                logs=epoch_logs,
                net=self.net.state_dict(),
                optimizer=self.optimizer.state_dict(),
                loss_weights=self.loss_weights,
            )
            if stop:
                best_loss_weights = self.checkpoint.loss_weights
                self.net.load_state_dict(self.checkpoint.net)
                self.optimizer.load_state_dict(self.checkpoint.optimizer)

            self.print_progress(epoch_logs)
            logs.append(copy.deepcopy(epoch_logs))
            epoch_logs["epoch"] += 1

        return logs, best_loss_weights

    def train_val_step(
        self,
        train_data,
        val_data,
        reduction="mean",
        epoch=None,
    ):

        n_batch = int(math.ceil(train_data.__len__() / self.batch_size))
        val_batch_size = int(math.floor(val_data.__len__() / n_batch))

        tr_loader = DataLoader(
            train_data,
            batch_size=self.batch_size,
            shuffle=True,
            sampler=None,
        )
        val_loader = DataLoader(
            val_data, batch_size=val_batch_size, shuffle=True, sampler=None
        )

        tr_y_pred, tr_y_true, tr_loss = [], [], []
        val_y_pred, val_y_true, val_loss = [], [], []

        for tr_batch in tqdm(tr_loader, ascii=True, desc="Epoch " + str(epoch)):
            tr_prob, tr_yp, tr_yt, tr_l = self.train_step(tr_batch)
            tr_y_pred.extend(tr_yp)
            tr_y_true.extend(tr_yt)
            tr_loss.append(tr_l)

        for val_batch in val_loader:
            val_prob, val_yp, val_yt, val_l = self.val_step(val_batch)
            val_y_pred.extend(val_yp)
            val_y_true.extend(val_yt)
            val_loss.append(val_l)

        if isinstance(self.adaptive_loss, grads.GradApprox):
            self.adaptive_loss.add_losses(
                train_losses=np.array(tr_loss).swapaxes(0, 1),
                valid_losses=np.array(val_loss).swapaxes(0, 1),
            )
            self.loss_weights = self.adaptive_loss.compute_adaptive_weights()

        tr_loss = self.losses_list_to_dict(tr_loss, name="train", reduction=reduction)
        val_loss = self.losses_list_to_dict(val_loss, name="val", reduction=reduction)

        return (tr_y_pred, tr_y_true, tr_loss), (val_y_pred, val_y_true, val_loss)

    def train_step(self, data):
        if isinstance(self.adaptive_loss, (grads.AlphaGrad, grads.AdaMT)):
            self.layer = self.net.get_shared_layer()
        self.net.train()

        with torch.enable_grad():
            self.optimizer.zero_grad()

            preds = self.net(data["data"].to(self.device))
            preds = (
                [preds] if len(self.losses) == 1 else preds
            )

            task_loss = []
            for lossFn, out in zip(self.losses, preds):
                if lossFn.__class__.__name__ in ["MSELoss"]:
                    loss_val = lossFn(
                        out,
                        data["data"].to(self.device),
                    )
                else:
                    loss_val = lossFn(
                        out, data["labels"].type(torch.LongTensor).to(self.device)
                    )
                task_loss.append(loss_val)
            task_loss = torch.stack(task_loss).to(self.device)

            if isinstance(self.adaptive_loss, (grads.AlphaGrad, grads.AdaMT)):
                self.adaptive_loss.device = self.device
                self.adaptive_loss.backward(self.layer, task_loss.clone())
                self.adaptive_loss.step()
                self.loss_weights = self.adaptive_loss.get_loss_weights()
            
            weighted_task_loss = torch.mul(
                self.loss_weights.to(self.device), task_loss
            ).to(self.device)
            total_loss = torch.sum(weighted_task_loss).to(self.device)
            total_loss.backward(retain_graph=True)

            self.optimizer.step()
            
            prob = preds[self.classifier_index]
            pred = torch.argmax(prob, 1)
            y_pred = pred.data.tolist()
            y_true = data["labels"].tolist()
        return prob.tolist(), y_pred, y_true, weighted_task_loss.tolist()

    def val_step(self, data):

        self.net.eval()

        with torch.no_grad():
            preds = self.net(data["data"].to(self.device))
            preds = (
                [preds] if len(self.losses) == 1 else preds
            )

            task_loss = []
            for lossFn, out in zip(self.losses, preds):
                if out.dim() == 1:
                    out = torch.unsqueeze(out, dim=0)

                if lossFn.__class__.__name__ in ["MSELoss"]:
                    loss_val = lossFn(
                        out,
                        data["data"].to(self.device),
                    )
                else:
                    loss_val = lossFn(
                        out, data["labels"].type(torch.LongTensor).to(self.device)
                    )
                task_loss.append(loss_val)
            task_loss = torch.stack(task_loss).to(self.device)

            weighted_task_loss = torch.mul(
                self.loss_weights.to(self.device), task_loss
            ).to(self.device)

            prob = preds[self.classifier_index]
            if prob.dim() == 1:
                prob = torch.unsqueeze(prob, dim=0)
            pred = torch.argmax(prob, 1)
            y_pred = pred.data.tolist()
            y_true = data["labels"].tolist()
        return prob.tolist(), y_pred, y_true, weighted_task_loss.tolist()
    
    
    def test_step(self, data):
        self.net.eval()

        with torch.no_grad():
            preds = self.net(data["data"].to(self.device))
            preds = (
                [preds] if len(self.losses) == 1 else preds
            )

            prob = preds[self.classifier_index]
            if prob.dim() == 1:
                prob = torch.unsqueeze(prob, dim=0)
            pred = torch.argmax(prob, 1)
            y_pred = pred.data.tolist()
        return prob.tolist(), y_pred

    def evaluate(self, data, loss_weights=None, reduction=None):
        if loss_weights is None:
            loss_weights = self.loss_weights

        if self.batch_size is None:
            self.batch_size = len(data.labels)

        prob = []
        y_pred = []
        y_true = []
        losses = []

        dataLoader = DataLoader(data, batch_size=self.batch_size, shuffle=False)
        for data in dataLoader:
            prob_, yp, yt, loss = self.val_step(data)
            prob.extend(prob_)
            y_pred.extend(yp)
            y_true.extend(yt)
            losses.append(loss)
        losses = self.losses_list_to_dict(losses, name="test", reduction=reduction)
        return prob, y_pred, y_true, losses

    def losses_list_to_dict(self, losses_list, name="train", reduction="mean"):
        losses = np.array(losses_list).swapaxes(
            0, 1
        )  # (batch, n_losses) --> (m_losses, n_batch)
        losses_dict = {}
        if reduction == "mean":
            losses_dict = {name + "_loss": losses.sum(0).mean()}
            losses_dict.update(
                dict(
                    zip(
                        [name + "_" + fn.__class__.__name__ for fn in self.losses],
                        np.mean(losses, axis=1),
                    )
                )
            )
        elif reduction == None:
            losses_dict = {name + "_loss": losses.sum(0)}
            losses_dict.update(
                dict(
                    zip(
                        [name + "_" + fn.__class__.__name__ for fn in self.losses],
                        losses,
                    )
                )
            )
        return losses_dict

    def print_report(self, logs):
        if self.verbose >= 0:
            print()
            print("******************************************")
            print("************** Test Results **************")
            print("******************************************")
            print("Accuracy:", logs["results"]["accuracy"])
            print("F1_score:", logs["results"]["f1_score"])
            print("loss:", logs["results"]["loss"])
            print(
                "best_loss_weights:",
                ["{:.5f}".format(n) for n in logs["best_loss_weights"]],
            )
            print("confusion_matrix:", "\n", logs["results"]["confusion_matrix"])
            print("******************************************", "\n")

    def print_progress(self, variables_logs):
        if self.verbose >= 1:
            [
                print(k + ": {:.5f}".format(v), end=" - ")
                for k, v in variables_logs.items()
                if "train" in k
            ]
            print()
            [
                print(k + ": {:.5f}".format(v), end=" - ")
                for k, v in variables_logs.items()
                if "val" in k
            ]
            print(
                "\nloss_weights:",
                ["{:.5f}".format(n) for n in variables_logs["loss_weights"]],
                "\n",
            )

    def calculate_results(self, y_pred, y_true, classes=None, average="macro"):
        """
        Calculate the results matrices based on the actual and predicted class.

        Parameters
        ----------
        y_pred : list
            List of predicted labels.
        y_true : list
            List of actual labels.
        classes : list, optional
            List of labels to index the CM.
            This may be used to reorder or select a subset of class labels.
            If None then, the class labels that appear at least once in
            y_pred or y_true are used in sorted order.
            The default is None.

        Returns
        -------
        dict
            a dictionary with fields:
                acc : accuracy.
                cm  : confusion matrix..

        """

        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average=average)
        if classes is not None:
            cm = confusion_matrix(y_true, y_pred, labels=classes)
        else:
            cm = confusion_matrix(y_true, y_pred)

        return {"accuracy": acc, "f1_score": f1, "confusion_matrix": cm}
    
    def set_random_seed(self, seed):
        """
        Set all the random initializations with a given seed

        Parameters
        ----------
        seed : int
            seed.

        Returns
        -------
        None.

        """
        self.seed = seed

        np.random.seed(self.seed)
        random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)
        torch.set_num_threads(self.num_threads)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def set_losses(self, loss_names):
        """setup loss function and loss weights

        Args:
            loss_names (str, list, set, dict):
                (str): loss name
                (list, set): list of loss name
                (dict): dict of loss_names, each consist of "kwargs" and "weight" keys
        Raises:
            AssertionError: loss function not found.

        Returns:
            list, list: list of loss function and list of loss weights
        """

        loss_objs = []
        loss_weights = []

        if isinstance(loss_names, str):
            loss_names = [loss_names]
        
        n_task = len(loss_names)

        for loss_name in loss_names:
            try:
                kwargs = loss_names[loss_name]["kwargs"]
            except:
                kwargs = {}

            try:
                loss_weights.append(loss_names[loss_name]["weight"])
            except:
                loss_weights.append(1.0/n_task)

            if loss_name in nn.__dict__.keys():
                out = nn.__dict__[loss_name](**kwargs)
            elif loss_name in losses_module.__dict__.keys():
                out = losses_module.__dict__[loss_name](**kwargs)
            else:
                out = loss_names
            loss_objs.append(out)

        self.losses = loss_objs
        self.loss_weights = torch.Tensor(loss_weights)

    def set_adaptive_loss(self, adaptive_loss):
        if adaptive_loss is not None:
            if isinstance(adaptive_loss, str):
                name = copy.deepcopy(adaptive_loss)
                kwargs = {}
            elif isinstance(adaptive_loss, dict):
                name = list(adaptive_loss.keys())[0]
                try:
                    kwargs = adaptive_loss[name]["kwargs"]
                except:
                    kwargs = {}
            if name in grads.__dict__.keys():
                self.loss_weights = torch.nn.Parameter(self.loss_weights)
                self.adaptive_loss = grads.__dict__[name](self.loss_weights, **kwargs)
            else:
                self.adaptive_loss = adaptive_loss

    def set_optimizer(self, optimizer):
        if isinstance(optimizer, str):
            name = copy.deepcopy(optimizer)
            kwargs = {}
        elif isinstance(optimizer, dict):
            name = list(optimizer.keys())[0]
            try:
                kwargs = optimizer[name]["kwargs"]
            except:
                kwargs = {}

        if name in optim.__dict__.keys():
            self.optimizer = optim.__dict__[name](self.net.parameters(), **kwargs)
        else:
            self.optimizer = optimizer

    def set_scheduler(self, scheduler):
        if scheduler is not None:
            if isinstance(scheduler, str):
                name = copy.deepcopy(scheduler)
                kwargs = {}
                self.moitor = "val_loss"
            elif isinstance(scheduler, dict):
                name = list(scheduler.keys())[0]
                try:
                    kwargs = scheduler[name]["kwargs"]
                except:
                    kwargs = {}
                try:
                    self.moitor = scheduler[name]["monitor"]
                except:
                    self.moitor = "val_loss"

            if name in optim.lr_scheduler.__dict__.keys():
                self.scheduler = optim.lr_scheduler.__dict__[name](
                    self.optimizer, **kwargs
                )
            else:
                self.scheduler = scheduler

    def set_check_stoploop(self, check_stoploop):
        if isinstance(check_stoploop, str):
            name = copy.deepcopy(check_stoploop)
            kwargs = {}
        elif isinstance(check_stoploop, dict):
            name = list(check_stoploop.keys())[0]
            try:
                kwargs = check_stoploop[name]["kwargs"]
            except:
                kwargs = {}
        if name in callbacks.__dict__.keys():
            self.check_stoploop = callbacks.__dict__[name](**kwargs)
        else:
            self.check_stoploop = check_stoploop

    def set_checkpoint(self, checkpoint):
        if checkpoint is not None:
            if isinstance(checkpoint, str):
                name = copy.deepcopy(checkpoint)
                kwargs = {}
            elif isinstance(checkpoint, dict):
                name = list(checkpoint.keys())[0]
                try:
                    kwargs = checkpoint[name]["kwargs"]
                except:
                    kwargs = {}

            if name in callbacks.__dict__.keys():
                self.checkpoint = callbacks.__dict__[name](**kwargs)
            else:
                self.checkpoint = checkpoint

    def write_logs(self, logs, log_csv=True):
        with open(
            os.path.join(self.output_path, "logs.dat"),
            "wb",
        ) as fp:
            pickle.dump(logs, fp)

        if log_csv:
            list_dict = logs["train_logs"]
            keys = list_dict[-1].keys()
            with open(
                os.path.join(self.output_path, "train_logs.csv"), "w", newline=""
            ) as output_file:
                dict_writer = csv.DictWriter(output_file, keys)
                dict_writer.writeheader()
                dict_writer.writerows(list_dict)
