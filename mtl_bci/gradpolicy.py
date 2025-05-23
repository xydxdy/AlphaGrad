import numpy as np
from abc import abstractmethod


class BaseGradientPolicy(object):
    def __init__(self, overlap=1.25):
        self.overlap = overlap
        self.average_win = None
        self.valid_average_win = None
        self.hist_size = None
        self.valid_hist_size = None
        self.train_losses = np.array([])
        self.valid_losses = np.array([])
        self.smoothed_train_losses = np.array([])
        self.smoothed_valid_losses = np.array([])

    def add_losses(self, train_losses_val, valid_losses_val): 
        
        if self.average_win is None:   
            self.average_win = len(train_losses_val)
            self.valid_average_win = len(valid_losses_val)
            self.hist_size = int(self.average_win * self.overlap)
            self.valid_hist_size = int(self.valid_average_win * self.overlap)
            
        self.train_losses = np.append(self.train_losses, [train_losses_val])
        self.valid_losses = np.append(self.valid_losses, [valid_losses_val])
        self._moving_average_smoothing()

    def _moving_average_smoothing(self):
        smoothed_train = np.mean(
            self.train_losses[-int(self.average_win * self.overlap) :]
        )
        self.smoothed_train_losses = np.append(
            self.smoothed_train_losses, [smoothed_train]
        )

        smoothed_val = np.mean(
            self.valid_losses[-int(self.valid_average_win * self.overlap) :]
        )
        self.smoothed_valid_losses = np.append(
            self.smoothed_valid_losses, [smoothed_val]
        )

    def _line_fit(self, train_losses, valid_losses):
        t = np.arange(len(train_losses))
        p_train = np.polyfit(t, train_losses, 1)
        assert len(p_train) == 2
        p_train = p_train[0]

        t = np.arange(len(valid_losses))
        p_valid = np.polyfit(t, valid_losses, 1)
        assert len(p_valid) == 2
        p_valid = p_valid[0]

        return p_train, p_valid

    @abstractmethod
    def compute_weights(self):
        """compute and return w, Gk, Ok"""
        pass


class HistoricalTangentSlope(BaseGradientPolicy):
    """
    Update from Huy Phan version
    """

    def __init__(self, overlap=1.25):
        super().__init__(overlap)
        self.train_slope_ref = None
        self.valid_slope_ref = None

    def compute_weights(self):
        if self.train_slope_ref is None and self.valid_slope_ref is None:
            train_losses = self.smoothed_train_losses[-self.hist_size - 1 : -1]
            valid_losses = self.smoothed_valid_losses[-self.valid_hist_size - 1 : -1]
            self.train_slope_ref, self.valid_slope_ref = self._line_fit(
                train_losses, valid_losses
            )

        cur_train_losses = self.smoothed_train_losses[-self.hist_size :]
        cur_valid_losses = self.smoothed_valid_losses[-self.valid_hist_size :]
        train_slope, valid_slope = self._line_fit(cur_train_losses, cur_valid_losses)

        Ok = (valid_slope - train_slope) - (self.valid_slope_ref - self.train_slope_ref)
        Gk = valid_slope - self.valid_slope_ref

        w = Gk / (Ok * Ok + 1e-6)
        if w < 0.0:
            w = 0.0

        # update references
        if self.valid_slope_ref > valid_slope:
            self.train_slope_ref = train_slope
            self.valid_slope_ref = valid_slope

        return w, Gk, Ok


class TangentSlope(BaseGradientPolicy):
    """
    Gradient slope
    """

    def __init__(self, overlap=1.25):
        super().__init__(overlap)

    def compute_weights(self):
        cur_train_losses = self.smoothed_train_losses[-self.hist_size :]
        cur_valid_losses = self.smoothed_valid_losses[-self.valid_hist_size :]
        train_slope, valid_slope = self._line_fit(cur_train_losses, cur_valid_losses)

        Ok = valid_slope - train_slope
        Gk = valid_slope

        w = Gk / (Ok * Ok + 1e-6)
        if w < 0.0:
            w = 0.0

        return w, Gk, Ok


class SecantSlope(BaseGradientPolicy):
    """
    sec slope
    """

    def __init__(self, overlap=1.25):
        super().__init__(overlap)
        self.train_prev_point = None
        self.valid_prev_point = None

    def compute_weights(self):
        if self.train_prev_point is None and self.valid_prev_point is None:
            train_losses = self.smoothed_train_losses[-self.hist_size - 1 : -1]
            valid_losses = self.smoothed_valid_losses[-self.valid_hist_size - 1 : -1]
            self.train_prev_point, self.valid_prev_point = (
                train_losses.mean(),
                valid_losses.mean(),
            )

        cur_train_losses = self.smoothed_train_losses[-self.hist_size :]
        cur_valid_losses = self.smoothed_valid_losses[-self.valid_hist_size :]
        train_cur_point, valid_cur_point = (
            cur_train_losses.mean(),
            cur_valid_losses.mean(),
        )

        Ok = (valid_cur_point - train_cur_point) - (
            self.valid_prev_point - self.train_prev_point
        )
        Gk = valid_cur_point - self.valid_prev_point

        w = Gk / (Ok * Ok + 1e-6)
        if w < 0.0:
            w = 0.0

        # update references
        if self.valid_prev_point > valid_cur_point:
            self.train_prev_point = train_cur_point
            self.valid_prev_point = valid_cur_point

        return w, Gk, Ok


class Threshold(BaseGradientPolicy):
    """
    loss threshold
    """

    def __init__(self, overlap=1.25):
        super().__init__(overlap)

    def compute_weights(self):
        """add historical loss"""
        train_losses = self.smoothed_train_losses[-self.hist_size :]
        valid_losses = self.smoothed_valid_losses[-self.valid_hist_size :]

        Ok = valid_losses - train_losses
        Gk = valid_losses

        w = np.mean(Gk / (Ok * Ok + 1e-6))
        if w < 0.0:
            w = 0.0

        return w, Gk.mean(), Ok.mean()


class BlendingRatio(BaseGradientPolicy):
    """
    Scale to norm losses
    """

    def __init__(self, overlap=1.25):
        super().__init__(overlap)

    def compute_weights(self):
        cur_valid_losses = self.smoothed_valid_losses[-self.valid_hist_size :]

        w = np.mean(cur_valid_losses)

        Ok = None
        Gk = None

        return w, Gk, Ok
