import numpy as np
import random
from copy import copy

class Augment:
    def __init__(self,
                 p_jitter=0.8, sigma=0.02, scale_range=[0.9,1.1],
                 p_dropout=0.3,
                 p_moda=0.5,
                 p_f_dropout=0.3,
                 p_left_right=0.1,          
                 drift_std=0.005,     
                 drift_max=0.25):      
        self.p_jitter  = p_jitter
        self.sigma     = sigma
        self.scale_min, self.scale_max = scale_range
        self.p_dropout = p_dropout
        self.p_moda    = p_moda
        self.p_f_dropout = p_f_dropout
        self.drift_std = drift_std
        self.drift_max = drift_max
        self.p_left_right = p_left_right

    # ---------- Jitter & Scaling ----------
    def jitter_scale(self, x: np.ndarray) -> np.ndarray:
        noise  = np.random.randn(*x.shape) * self.sigma
        scale  = np.random.uniform(self.scale_min,
                                   self.scale_max,
                                   size=(1, x.shape[1]))
        return (x + noise) * scale

    # ---------- Sensor Drop-out ----------
    def sensor_dropout(self,
                       x: np.ndarray,
                       imu_dim: int) -> np.ndarray:

        if random.random() < self.p_dropout:
            x[:, imu_dim:] = 0.0
        return x
    
    def feature_dropout(self, x: np.ndarray) -> np.ndarray:
        """
        ランダムな特徴量を1つ選んで、値をすべて0にする。
        """
        feature_dim = len(x[0])
        num_list = [i for i in range(feature_dim)]
        idx = random.choice(num_list)
        x[:, idx] = 0.0
        return x

    def motion_drift(self, x: np.ndarray, imu_dim: int) -> np.ndarray:

        T = x.shape[0]

        drift = np.cumsum(
            np.random.normal(scale=self.drift_std, size=(T, 1)),
            axis=0
        )
        drift = np.clip(drift, -self.drift_max, self.drift_max)   

        x[:, :6] += drift

        if imu_dim > 6:
            x[:, 6:imu_dim] += drift     
        return x

    def left_right_change(self, x: np.ndarray, demo: np.array):
        """
        利き手の左右反転を行う拡張
        ・acc_xとrot_xの符号を反転
        ・demographicの利き手を反転（0と1を入れ替え）
        ・tof、thmで3と5を入れ替え
        """
        #acc_xとrot_xの反転
        x[:, 0] *= -1
        x[:, 4] *= -1

        #demographicの利き手反転
        demo[3] = 1 - demo[3]

        #thm_3とthm_5を入れ替え
        tmp_thm_3 = copy(x[:, 9])
        tmp_thm_5 = copy(x[:, 11])
        x[:, 9] = tmp_thm_5
        x[:, 11] = tmp_thm_3

        #tof_3とtof_5を入れ替え
        tmp_tof_3 = copy(x[:, 20:23 + 1])
        tmp_tof_5 = copy(x[:, 28:])
        x[:, 20:23 + 1] = tmp_tof_5
        x[:, 28:] = tmp_tof_3

        return x, demo

    
    # ---------- master call ----------
    def __call__(self,
                 x: np.ndarray, demo: np.array,
                 imu_dim: int) -> np.ndarray:
        if random.random() < self.p_left_right:
            x, demo = self.left_right_change(x, demo)

        if random.random() < self.p_jitter:
            x = self.jitter_scale(x)

        if random.random() < self.p_moda:
            x = self.motion_drift(x, imu_dim)

        if random.random() < self.p_f_dropout:
            x = self.feature_dropout(x)

        x = self.sensor_dropout(x, imu_dim)
        return x, demo