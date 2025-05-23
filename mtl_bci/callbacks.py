import copy

class Checkpoint:
    """Callback to copy the model, optimizer, or other variables at some frequency.
    
    `Checkpoint` callback is used in conjunction with training loop
    `train_loop()` to copy a model or weights at some
    interval, so the model or weights can be loaded later to continue the
    training from the best state saved.
    
    Args: 
        monitor (str): The metric name to monitor. Default to "val_loss".
                    * Prefix the name with `"val_`" to monitor validation metrics.
                    * Use `val_loss`" to monitor the model's total loss.
        mode (str): one of {'auto', 'min', 'max'} Default to "auto".
                    the decision to overwrite the current best state is made based on either
                    the maximization or the minimization of the monitored quantity.
                    For `val_acc`, this should be `max`, for `val_loss` this should be
                    `min`, etc. In `auto` mode, the mode is set to `min` if the quantities
                    monitored are 'loss' or the othets are set to `max` for
                    the rest of the quantities.

    """
    def __init__(self, monitor="val_loss", mode="auto"):
        self.monitor = monitor
        self.mode = mode
        if self.mode == "auto":
            if "loss" in self.monitor:
                self.mode = "min"
            else:
                self.mode = "max"
        if self.mode == "min":
            self.best_value = float("inf")
        elif self.mode == "max":
            self.best_value = 0
        else:
            self.best_value = None
            
    def __copybest__(self, logs, **kwargs):
        """Copy the best state

        Args:
            logs (dict): dict of the metrics of the current epoch.
            **kwargs: the other keyword arguments that you want to copy at best state.
        """
        print("Checkpoint: Copy the best state, monitoring={}, mode={}".format(self.monitor, self.mode))
        self.best_value = logs[self.monitor]
        for k in kwargs.keys():
            self.__setattr__(k, copy.deepcopy(kwargs[k]))
        
    def __call__(self, logs, **kwargs):
        """a built-in method to copy a model or weights at some interval
        Args:
            logs (dict): dict of the metrics of the current epoch.
            **kwargs: the other keyword arguments that you want to copy at best state.
        """
        if self.mode == "max" and logs[self.monitor] > self.best_value:
            self.__copybest__(logs, **kwargs)
        elif self.mode == "min" and logs[self.monitor] < self.best_value:
            self.__copybest__(logs, **kwargs)
        elif self.mode == None:
            self.__copybest__(logs, **kwargs)

class EarlyStopping:
    """Stop training when a monitored metric has stopped improving.
    
     Args:
        monitor (str): Quantity to be monitored. Default to "val_loss".
        patience: Number of epochs with no improvement 
            after which training will be stopped. Default to 5.
        mode: One of `{"auto", "min", "max"}`. Default to "auto".
            In `min` mode,
            training will stop when the quantity
            monitored has stopped decreasing; in `"max"`
            mode it will stop when the quantity
            monitored has stopped increasing; in `"auto"`
            mode, the direction is automatically inferred
            from the name of the monitored quantity.
    """
    def __init__(
        self, monitor="val_loss", patience=5, mode="auto", min_delta=0.0, min_epoch=None, max_epoch=None
    ):
        self.monitor = monitor
        self.patience = patience
        self.mode = mode
        self.min_delta = min_delta
        self.min_epoch = 0 if min_epoch is None else min_epoch
        self.max_epoch = float("inf") if max_epoch is None else max_epoch
        if self.mode == "auto":
            if "loss" in self.monitor:
                self.mode = "min"
            else:
                self.mode = "max"
        if self.mode == "min":
            EarlyStop = NoDecrease 
        elif self.mode == "max":
            EarlyStop = NoIncrease
        self.eary_stop = EarlyStop(
            patience=self.patience, monitor=self.monitor, min_delta=self.min_delta
        )

    def __call__(self, logs):
        """a built-in method to check to stop training

        Args:
            logs (dict): dict of the metrics of the current epoch.

        Returns:
            bool: Flag to stop.
        """
        
        stop = self.eary_stop(logs)
        # Force continue/stop
        if logs["epoch"] < self.min_epoch:
            stop = False
        elif logs["epoch"] >= self.max_epoch:
            stop = True
        return stop

class NoDecrease:
    """ Stop on no decrease of a particular variable.
    
    Args:
        patience (int): Number of epochs with no improvement
            after which training will be stopped.
        monitor (str): Quantity to be monitored. Such as "val_loss".
        min_delta (float): Minimum change in the monitored quantity
          to qualify as an improvement, i.e. an absolute
          change of less than min_delta, will count as no improvement.
    """

    def __init__(self, patience, monitor, min_delta=0.0):
        self.patience = patience
        self.monitor = monitor
        self.min_delta = min_delta
        self.min_value = float("inf")
        self.count_noprogress = 0

    def __call__(self, logs):
        """
        a built-in method to check to stop on no decrease of a particular variable

        Args:
            logs (dict): dict of the metrics of the current epoch.

        Returns:
            bool: Flag to stop.
        """
        if logs[self.monitor] <= (1 - self.min_delta) * self.min_value:
            self.min_value = logs[self.monitor]
            logs[self.monitor + "_min"] = self.min_value
            self.count_noprogress = 0
        else:
            self.count_noprogress += 1

        return self.count_noprogress >= self.patience


class NoIncrease:
    """ Stop on no increase of a particular variable.
    
    Args:
        patience (int): Number of epochs with no improvement
            after which training will be stopped.
        monitor (str): Quantity to be monitored. Such as "val_loss".
        min_delta (float): Minimum change in the monitored quantity
          to qualify as an improvement, i.e. an absolute
          change of less than min_delta, will count as no improvement.
    """

    def __init__(self, patience, monitor, min_delta=0.0):
        self.patience = patience
        self.monitor = monitor
        self.min_delta = min_delta
        self.max_value = 0
        self.count_noprogress = 0

    def __call__(self, logs):
        """
        a built-in method to check to stop on no increase of a particular variable

        Args:
            logs (dict): dict of the metrics of the current epoch.

        Returns:
            bool: Flag to stop.
        """
        if logs[self.monitor] >= (1 - self.min_delta) * self.max_value:
            self.max_value = logs[self.monitor]
            logs[self.monitor + "_max"] = self.max_value
            self.count_noprogress = 0
        else:
            self.count_noprogress += 1

        return self.count_noprogress >= self.patience
