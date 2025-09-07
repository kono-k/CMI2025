from scipy.spatial.transform import Rotation as R
from copy import copy
import pandas as pd
import numpy as np

class FE:
    def __init__(self, df, is_train=True):
        self.df = df
        self.tof_cols = [c for c in df.columns if c.startswith('tof_')]
        if is_train:
            self.insert_loc = 16
        else:
            self.insert_loc = 11

    def transform(self):
        #self.df = self._remove_gravity_from_acc()
        #self.df = self._compute_angular_velocity_from_quat()
        #self.df = self._compute_angular_distance()
        df = self._compute_tof_features()
        return df
    
    def _remove_gravity_from_acc(self):
        df = copy(self.df)
        df = df.ffill().bfill().fillna(0)
        acc = df[["acc_x", "acc_y", "acc_z"]].to_numpy()
        rot = df[["rot_x", "rot_y", "rot_z", "rot_w"]].to_numpy()

        linear_accel = np.zeros_like(acc)
        gravity_world = np.array([0, 0, 9.81])
        
        for i in range(len(acc)):
            if np.all(np.isnan(rot[i])) or np.all(np.isclose(rot[i], 0)):
                linear_accel[i, :] = acc[i, :]
                continue
            try:
                rotation = R.from_quat(rot[i])
                gravity_sensor_frame = rotation.apply(gravity_world, inverse=True)
                linear_accel[i, :] = acc[i, :] - gravity_sensor_frame
            except ValueError:
                linear_accel[i, :] = acc[i, :]

        df["acc_x"] = linear_accel.T[0]
        df["acc_y"] = linear_accel.T[1]
        df["acc_z"] = linear_accel.T[2]

        return df

    def _compute_angular_velocity_from_quat(self, time_delta=1/200):
        df = copy(self.df)
        df = df.ffill().bfill().fillna(0)
        rot = df[["rot_x", "rot_y", "rot_z", "rot_w"]].to_numpy()
        angular_vel = np.zeros((len(rot), 3))
        
        for i in range(len(rot) - 1):
            q_t, q_t_plus_dt = rot[i], rot[i+1]
            if np.all(np.isnan(q_t)) or np.all(np.isnan(q_t_plus_dt)):
                continue
            try:
                rot_t = R.from_quat(q_t)
                rot_t_plus_dt = R.from_quat(q_t_plus_dt)
                delta_rot = rot_t.inv() * rot_t_plus_dt
                angular_vel[i, :] = delta_rot.as_rotvec() / time_delta
            except ValueError:
                pass

        df.insert(self.insert_loc, "angular_vel_mag", np.sqrt(angular_vel.T[0]**2 + angular_vel.T[1]**2 + angular_vel.T[2]**2))
        df.insert(self.insert_loc, "angular_vel_mag_jerk", df.groupby("sequence_id")["angular_vel_mag"].diff().fillna(0).to_numpy())
        return df

    def _compute_angular_distance(self):
        df = copy(self.df)
        df = df.ffill().bfill().fillna(0)
        rot = df[["rot_x", "rot_y", "rot_z", "rot_w"]].to_numpy()
        angular_dist = np.zeros(len(rot))
        
        for i in range(len(rot) - 1):
            q1, q2 = rot[i], rot[i+1]
            if np.all(np.isnan(q1)) or np.all(np.isnan(q2)):
                continue
            try:
                r1, r2 = R.from_quat(q1), R.from_quat(q2)
                relative_rotation = r1.inv() * r2
                angular_dist[i] = np.linalg.norm(relative_rotation.as_rotvec())
            except ValueError:
                pass
        
        df.insert(self.insert_loc, "angular_dist", angular_dist.T[0])

        return df

    def _compute_acc_world(self):
        # acc: [:, x, y, z]
        # rot: [:, x, y, z, w]

        df = copy(self.df)
        df = df.ffill().bfill().fillna(0)
        acc = df[["acc_x", "acc_y", "acc_z"]].to_numpy()
        rot = df[["rot_x", "rot_y", "rot_z", "rot_w"]].to_numpy()

        acc_world = []
        for i in range(len(self.df.index)):
            try:
                r = R.from_quat(rot[i])  # shape: (M,)
                acc_world.append(r.apply(acc[i]))
            except ValueError:
                print(rot[i])
                raise(ValueError)

        acc_world = np.array(acc_world)

        df.insert(self.insert_loc, "fe_acc_world_z", acc_world.T[2])
        df.insert(self.insert_loc, "fe_acc_world_y", acc_world.T[1])
        df.insert(self.insert_loc, "fe_acc_world_x", acc_world.T[0])

        return df

    def _compute_tof_features(self):
        df = copy(self.df)
        seq_gp = df.groupby('sequence_id')
        seq_df_list = []
        
        for seq_id, seq_df in seq_gp:
            seq_df_copy = seq_df.copy()
            for i in range(1, 6):
                pixel_cols = [f"tof_{i}_v{p}" for p in range(64)]
                tof_sensor_data = seq_df_copy[pixel_cols].replace(-1, np.nan)
                seq_df_copy[f'tof_{i}_mean'] = tof_sensor_data.mean(axis=1)
                seq_df_copy[f'tof_{i}_std'] = tof_sensor_data.std(axis=1)
                seq_df_copy[f'tof_{i}_min'] = tof_sensor_data.min(axis=1)
                seq_df_copy[f'tof_{i}_max'] = tof_sensor_data.max(axis=1)
                seq_df_copy = seq_df_copy.drop(pixel_cols, axis=1)

            seq_df_list.append(seq_df_copy)
        df = pd.concat(seq_df_list, axis=0)

        return df

def test():
    df = pd.read_csv("kono-k_2/data/train.csv", nrows=10)
    fe = FE(df=df, is_train=False)
    df = fe.transform()
    print(df)
    print(df.columns)

if __name__ == "__main__":
    test()
