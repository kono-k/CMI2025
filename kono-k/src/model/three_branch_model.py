import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from scipy.signal import firwin

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

mean = torch.tensor([
    0,  0, 0, 0, 0,
    0,  9.0319e-03,  1.0849e+00, -2.6186e-03,  3.7651e-03,
    -5.3660e-03, -2.8177e-03,  1.3318e-03, -1.5876e-04,  6.3495e-01,
     6.2877e-01,  6.0607e-01,  6.2142e-01,  6.3808e-01,  6.5420e-01,
     7.4102e-03, -3.4159e-03, -7.5237e-03, -2.6034e-02,  2.9704e-02,
    -3.1546e-02, -2.0610e-03, -4.6986e-03, -4.7216e-03, -2.6281e-02,
     1.5799e-02,  1.0016e-02
], dtype=torch.float32).view(1, -1, 1).to(device)         

std = torch.tensor([
    1, 1, 1, 1, 1, 1, 0.2067, 0.8583, 0.3162,
    0.2668, 0.2917, 0.2341, 0.3023, 0.3281, 1.0264, 0.8838, 0.8686, 1.0973,
    1.0267, 0.9018, 0.4658, 0.2009, 0.2057, 1.2240, 0.9535, 0.6655, 0.2941,
    0.3421, 0.8156, 0.6565, 1.1034, 1.5577
], dtype=torch.float32).view(1, -1, 1).to(device) + 1e-8  

class ImuFeatureExtractor(nn.Module):
    def __init__(self, fs=100., add_quaternion=False):
        super().__init__()
        self.fs = fs
        self.add_quaternion = add_quaternion

        k = 15
        self.lpf = nn.Conv1d(6, 6, kernel_size=k, padding=k//2,
                             groups=6, bias=False)
        nn.init.kaiming_uniform_(self.lpf.weight, a=math.sqrt(5))

        self.lpf_acc  = nn.Conv1d(3, 3, k, padding=k//2, groups=3, bias=False)
        self.lpf_gyro = nn.Conv1d(3, 3, k, padding=k//2, groups=3, bias=False)

    def forward(self, imu):
        # imu: 
        B, C, T = imu.shape
        acc  = imu[:, 0:3, :]                 # acc_x, acc_y, acc_z
        gyro = imu[:, 3:6, :]                 # gyro_x, gyro_y, gyro_z
        extra = imu[:, 7:, :]

        # 1) magnitude
        acc_mag  = torch.norm(acc,  dim=1, keepdim=True)          # (B,1,T)
        gyro_mag = torch.norm(gyro, dim=1, keepdim=True)

        # 2) jerk 
        jerk = F.pad(acc[:, :, 1:] - acc[:, :, :-1], (1,0))       # (B,3,T)
        gyro_delta = F.pad(gyro[:, :, 1:] - gyro[:, :, :-1], (1,0))

        # 3) energy
        acc_pow  = acc ** 2
        gyro_pow = gyro ** 2

        # 4) LPF / HPF 
        acc_lpf  = self.lpf_acc(acc)
        acc_hpf  = acc - acc_lpf
        gyro_lpf = self.lpf_gyro(gyro)
        gyro_hpf = gyro - gyro_lpf

        if len(extra[0]) == 0:
            features = [
                acc, gyro,
                acc_mag, gyro_mag,
                jerk, gyro_delta,
                acc_pow, gyro_pow,
                acc_lpf, acc_hpf,
                gyro_lpf, gyro_hpf
            ]
        else:
            features = [
                acc, gyro, extra,
                acc_mag, gyro_mag,
                jerk, gyro_delta,
                acc_pow, gyro_pow,
                acc_lpf, acc_hpf,
                gyro_lpf, gyro_hpf
            ]
        return torch.cat(features, dim=1)  # (B, C_out, T)


class SEBlock(nn.Module):
    def __init__(self, channels, reduction=8):
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool1d(1)
        self.excitation = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        b, c, _ = x.size()
        y = self.squeeze(x).view(b, c)
        y = self.excitation(y).view(b, c, 1)
        return x * y.expand_as(x)

class ResidualSECNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, pool_size=2, dropout=0.3, weight_decay=1e-4):
        super().__init__()
        
        # First conv block
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size//2, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        
        # Second conv block
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, padding=kernel_size//2, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        
        # SE block
        self.se = SEBlock(out_channels)
        
        # Shortcut connection
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, 1, bias=False),
                nn.BatchNorm1d(out_channels)
            )
        
        self.pool = nn.MaxPool1d(pool_size)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        shortcut = self.shortcut(x)
        
        # First conv
        out = F.relu(self.bn1(self.conv1(x)))
        # Second conv
        out = self.bn2(self.conv2(out))
        
        # SE block
        out = self.se(out)
        
        # Add shortcut
        out += shortcut
        out = F.relu(out)
        
        # Pool and dropout
        out = self.pool(out)
        out = self.dropout(out)
        
        return out

class AttentionLayer(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attention = nn.Linear(hidden_dim, 1)
        
    def forward(self, x):
        # x shape: (batch, seq_len, hidden_dim)
        scores = torch.tanh(self.attention(x))  # (batch, seq_len, 1)
        weights = F.softmax(scores.squeeze(-1), dim=1)  # (batch, seq_len)
        context = torch.sum(x * weights.unsqueeze(-1), dim=1)  # (batch, hidden_dim)
        return context
    
class DemoAttentionLayer(nn.Module):
    """セルフアテンションモジュール"""
    
    def __init__(self, hidden_dim: int, num_heads: int = 4):
        """
        初期化
        
        Args:
            hidden_dim: 隠れ層の次元
            num_heads: アテンションヘッド数
        """
        super().__init__()
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads)
        self.norm = nn.LayerNorm(hidden_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        順伝播
        
        Args:
            x: 入力テンソル (batch_size, seq_len, hidden_dim)
            
        Returns:
            torch.Tensor: 出力テンソル (batch_size, seq_len, hidden_dim)
        """
        # 入力の順番を変更 (batch_size, seq_len, hidden_dim) -> (seq_len, batch_size, hidden_dim)
        x_t = x.transpose(0, 1)
        
        # セルフアテンションの適用
        attn_out, _ = self.attention(x_t, x_t, x_t)
        
        # 残差接続と正規化
        attn_out = attn_out + x_t
        #attn_out = self.norm(attn_out)
        
        # 出力の順番を戻す (seq_len, batch_size, hidden_dim) -> (batch_size, seq_len, hidden_dim)
        return attn_out.transpose(0, 1)

class DemographicBranch(nn.Module):
    def __init__(self, demo_dim=6, hidden_dim=64, dropout=0.3):
        super().__init__()
        # demo_dim: adult_child, age, sex, handedness, height_cm, shoulder_to_wrist_cm, elbow_to_wrist_cm
        self.demo_dim = demo_dim
        
        # Normalization layers for continuous features
        self.age_norm = nn.BatchNorm1d(1)#nn.LayerNorm(1)
        self.height_norm = nn.BatchNorm1d(1)#nn.LayerNorm(1)
        self.shoulder_norm = nn.BatchNorm1d(1)#nn.LayerNorm(1)
        self.elbow_norm = nn.BatchNorm1d(1)#nn.LayerNorm(1)
        
        # Dense layers for demographic features
        self.demo_fc1 = nn.Linear(demo_dim, hidden_dim, bias=False)
        self.demo_bn1 = nn.BatchNorm1d(hidden_dim)
        self.demo_drop1 = nn.Dropout(dropout)
        
        self.demo_fc2 = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.demo_bn2 = nn.BatchNorm1d(hidden_dim)
        self.demo_drop2 = nn.Dropout(dropout)
        
    def forward(self, demo_features):
        # demo_features: (batch, demo_dim)
        # Normalize continuous features
        adult_child = demo_features[:, 0:1]  # binary
        age = self.age_norm(demo_features[:, 1:2])  # continuous
        sex = demo_features[:, 2:3]  # binary
        handedness = demo_features[:, 3:4]  # binary
        height = self.height_norm(demo_features[:, 4:5])  # continuous
        shoulder = self.shoulder_norm(demo_features[:, 5:6])  # continuous
        
        # Handle elbow_to_wrist_cm if present
        if demo_features.shape[1] > 6:
            elbow = self.elbow_norm(demo_features[:, 6:7])  # continuous
            normalized_features = torch.cat([adult_child, age, sex, handedness, height, shoulder, elbow], dim=1)
        else:
            normalized_features = torch.cat([adult_child, age, sex, handedness, height, shoulder], dim=1)
        
        # Process through dense layers
        #x = F.relu(self.demo_bn1(self.demo_fc1(normalized_features)))
        #x = self.demo_drop1(x)
        #x = F.relu(self.demo_bn2(self.demo_fc2(x)))
        #x = self.demo_drop2(x)
        
        return demo_features
    
class ThreeBranchModel(nn.Module):
    def __init__(self, pad_len, imu_dim_raw, tof_dim, thm_dim, demo_dim, n_classes, 
                 dropouts=[0.3, 0.3, 0.3, 0.3, 0.4, 0.5, 0.3], 
                 feature_engineering=True, **kwargs):
        super().__init__()
        self.feature_engineering = feature_engineering
        if feature_engineering:
            self.imu_fe = ImuFeatureExtractor(**kwargs)
            imu_dim = imu_dim_raw + 25
            #print(f"imu_dim:{imu_dim}")            
        else:
            self.imu_fe = nn.Identity()
            imu_dim = imu_dim_raw   
            
        self.imu_dim = imu_dim
        self.tof_dim = tof_dim + thm_dim
        self.demo_dim = demo_dim

        self.fir_nchan = imu_dim_raw

        weight_decay = 3e-3

        numtaps = 33  
        fir_coef = firwin(numtaps, cutoff=1.0, fs=10.0, pass_zero=False)
        fir_kernel = torch.tensor(fir_coef, dtype=torch.float32).view(1, 1, -1)
        fir_kernel = fir_kernel.repeat(imu_dim_raw, 1, 1)  # (imu_dim, 1, numtaps)
        self.register_buffer("fir_kernel", fir_kernel)
        
        # IMU deep branch
        self.imu_block1 = ResidualSECNNBlock(imu_dim, 64, 3, dropout=dropouts[0], weight_decay=weight_decay)
        self.imu_block2 = ResidualSECNNBlock(64, 128, 5, dropout=dropouts[1], weight_decay=weight_decay)
        
        # TOF branch
        self.tof_conv1 = nn.Conv1d(self.tof_dim, 64, 3, padding=1, bias=False)
        self.tof_bn1 = nn.BatchNorm1d(64)
        self.tof_pool1 = nn.MaxPool1d(2)
        self.tof_drop1 = nn.Dropout(dropouts[2])
        
        self.tof_conv2 = nn.Conv1d(64, 128, 3, padding=1, bias=False)
        self.tof_bn2 = nn.BatchNorm1d(128)
        self.tof_pool2 = nn.MaxPool1d(2)
        self.tof_drop2 = nn.Dropout(dropouts[3])
        
        # Demographics branch
        self.demo_branch = DemographicBranch(demo_dim, hidden_dim=64, dropout=dropouts[2])
        
        # BiLSTM (increased input size for 3 branches)
        # IMU: 128, TOF: 128, THM: 128 -> 384 total
        self.bilstm = nn.LSTM(256, 128, bidirectional=True, batch_first=True)
        self.lstm_dropout = nn.Dropout(dropouts[4])
        
        # Attention
        self.attention = AttentionLayer(256)  # 128*2 for bidirectional
        self.demo_attention = DemoAttentionLayer(1, num_heads=1)

        # Dense layers (combining LSTM output + demo features)
        # LSTM attention output: 256, Demo branch: 64 -> 320 total
        self.dense1 = nn.Linear(256 + 7, 256, bias=False)
        self.bn_dense1 = nn.BatchNorm1d(256)
        self.drop1 = nn.Dropout(dropouts[5])
        
        self.dense2 = nn.Linear(256, 128, bias=False)
        self.bn_dense2 = nn.BatchNorm1d(128)
        self.drop2 = nn.Dropout(dropouts[6])
        
        self.classifier = nn.Linear(128, n_classes)
        
    def forward(self, x, demo_features=None):
        # Split input - expecting order: IMU -> TOF -> THM
        imu = x[:, :, :self.fir_nchan].transpose(1, 2)  # (batch, imu_dim, seq_len)
        # Separate TOF and THM from the remaining sensors
        tof = x[:, :, self.fir_nchan:].transpose(1, 2)  # (batch, tof_dim + thm_dim, seq_len)

        imu = self.imu_fe(imu)   # (B, imu_dim, T)
        filtered = F.conv1d(
            imu[:, :self.fir_nchan, :],        # (B,7,T)
            self.fir_kernel,
            padding=self.fir_kernel.shape[-1] // 2,
            groups=self.fir_nchan,
        )
        
        imu = torch.cat([filtered, imu[:, self.fir_nchan:, :]], dim=1)  
        imu = (imu - mean) / std 
        
        # IMU branch
        x1 = self.imu_block1(imu)
        x1 = self.imu_block2(x1)
        
        # TOF/THM branch
        x2 = F.relu(self.tof_bn1(self.tof_conv1(tof)))
        x2 = self.tof_drop1(self.tof_pool1(x2))
        x2 = F.relu(self.tof_bn2(self.tof_conv2(x2)))
        x2 = self.tof_drop2(self.tof_pool2(x2))
        
        # Demographics branch
        if demo_features is not None:
            demo_out = self.demo_branch(demo_features)  # (batch, 64)
        else:
            # If no demo features provided, use zeros
            demo_out = torch.zeros(x.shape[0], 64, device=x.device)
        
        # Concatenate IMU, TOF, and THM branches for LSTM
        merged = torch.cat([x1, x2], dim=1).transpose(1, 2)  # (batch, seq_len, 256)
        
        # BiLSTM
        lstm_out, _ = self.bilstm(merged)
        lstm_out = self.lstm_dropout(lstm_out)
        
        # Attention
        attended = self.attention(lstm_out)  # (batch, 256)
        
        # Concatenate LSTM output with demographics
        combined = torch.cat([attended, demo_out], dim=1)  # (batch, 256 + 7)

        #Attention with demographics
        x = self.demo_attention(combined.unsqueeze(-1))
        x = torch.squeeze(x, -1)
        
        # Dense layers
        x = F.relu(self.bn_dense1(self.dense1(x)))
        x = self.drop1(x)
        x = F.relu(self.bn_dense2(self.dense2(x)))
        x = self.drop2(x)
        
        # Classification
        logits = self.classifier(x)
        return logits