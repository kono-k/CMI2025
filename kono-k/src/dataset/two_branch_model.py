import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset

def pad_sequences_torch(sequences, maxlen, padding='post', truncating='post', value=0.0):
    """PyTorch equivalent of Keras pad_sequences"""
    result = []
    for seq in sequences:
        if len(seq) >= maxlen:
            if truncating == 'post':
                seq = seq[:maxlen]
            else:  # 'pre'
                seq = seq[-maxlen:]
        else:
            pad_len = maxlen - len(seq)
            if padding == 'post':
                seq = np.concatenate([seq, np.full((pad_len, seq.shape[1]), value)])
            else:  # 'pre'
                seq = np.concatenate([np.full((pad_len, seq.shape[1]), value), seq])
        result.append(seq)
    return np.array(result, dtype=np.float32)

def preprocess_sequence(df_seq: pd.DataFrame, feature_cols: list, scaler: StandardScaler):
    """Normalizes and cleans the time series sequence"""
    mat = df_seq[feature_cols].ffill().bfill().fillna(0).values
    return scaler.transform(mat).astype('float32')

class CMI3Dataset(Dataset):
    def __init__(self,
                 X_list,
                 y_list,
                 maxlen,
                 mode="train",
                 imu_dim=7,
                 augment=None,
                 demo_features=None):
        self.X_list = X_list
        self.mode = mode
        self.y_list = y_list
        self.maxlen = maxlen
        self.imu_dim = imu_dim     
        self.augment = augment
        self.demo_features = demo_features   

    def pad_sequences_torch(self, seq, maxlen, padding='post', truncating='post', value=0.0):

        if seq.shape[0] >= maxlen:
            if truncating == 'post':
                seq = seq[:maxlen]
            else:  # 'pre'
                seq = seq[-maxlen:]
        else:
            pad_len = maxlen - seq.shape[0]
            if padding == 'post':
                seq = np.concatenate([seq, np.full((pad_len, seq.shape[1]), value)])
            else:  # 'pre'
                seq = np.concatenate([np.full((pad_len, seq.shape[1]), value), seq])
        return seq  
        
    def __getitem__(self, index):
        X = self.X_list[index]
        y = self.y_list[index]

        # ---------- (A)  Augmentation ----------
        if self.mode == "train" and self.augment is not None:
            if self.demo_features is not None:
                demo = self.demo_features[index]
                X, demo = self.augment(X, demo, self.imu_dim)
            else:
                X = self.augment(X, self.imu_dim)
        else:
            if self.demo_features is not None:
                demo = self.demo_features[index]

        X = self.pad_sequences_torch(X, self.maxlen, 'pre', 'pre')
        
        if self.demo_features is not None:
            return X, y, demo
        else:
            return X, y
    
    def __len__(self):
        return len(self.X_list)