import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
import numpy as np
import joblib
import torch
import os
import polars as pl

from src.metric.cmi_2025_metric import CompetitionMetric
from src.dataset.two_branch_model import preprocess_sequence, pad_sequences_torch, CMI3Dataset
from src.dataset.transform import Augment
from src.model.two_branch_model import TwoBranchModel
from src.model.three_branch_model import ThreeBranchModel
from src.utils.train_utils import EarlyStopping, EMA
from src.utils.config import CFG as config
from src.dataset.features import FE

CFG = config("../input/config.yaml")
CFG.RAW_DIR = Path(CFG.RAW_DIR)
CFG.PRETRAINED_DIR = Path(CFG.PRETRAINED_DIR)
CFG.EXPORT_DIR = Path(CFG.EXPORT_DIR)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print("▶ INFERENCE MODE – loading artefacts from", CFG.PRETRAINED_DIR)
feature_cols = np.load(CFG.PRETRAINED_DIR / "feature_cols.npy", allow_pickle=True).tolist()
demo_feature_cols = np.load(CFG.PRETRAINED_DIR / "demo_features.npy", allow_pickle=True).tolist()
pad_len = int(np.load(CFG.PRETRAINED_DIR / "sequence_maxlen.npy"))
scaler = joblib.load(CFG.PRETRAINED_DIR / "scaler.pkl")
demo_scaler = joblib.load(CFG.PRETRAINED_DIR / "demo_scaler.pkl")
gesture_classes = np.load(CFG.PRETRAINED_DIR / "gesture_classes.npy", allow_pickle=True)

imu_cols = [c for c in feature_cols if not (c.startswith('thm_') or c.startswith('tof_'))]
tof_cols = [c for c in feature_cols if c.startswith('tof_')]
thm_cols = [c for c in feature_cols if c.startswith('thm_')]

# Reorder feature columns to match model expectation: IMU -> TOF -> THM
feature_cols_ordered = imu_cols + tof_cols + thm_cols

    
# Load model
MODELS = [f'gesture_threebranchmodel_fold{i}.pth' for i in range(5)]
    
models = []
for path in MODELS:
    checkpoint = torch.load(CFG.PRETRAINED_DIR / path, map_location=device)
        
    model = ThreeBranchModel(
        checkpoint['pad_len'], 
        checkpoint['imu_dim'], 
        checkpoint['tof_dim'], 
        checkpoint['thm_dim'],
        checkpoint['demo_dim'],
        checkpoint['n_classes']
        ).to(device)
        
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    models.append(model)

print("  model, scaler, pads loaded – ready for evaluation")

def predict(sequence: pl.DataFrame, demographics: pl.DataFrame) -> str:
    """Prediction function for Kaggle competition"""
    global gesture_classes
    if gesture_classes is None:
        gesture_classes = np.load(CFG.PRETRAINED_DIR / "gesture_classes.npy", allow_pickle=True)

    df_seq = sequence.to_pandas()
    fe = FE(df=df_seq)
    df_seq = fe.transform()
    mat = preprocess_sequence(df_seq, feature_cols_ordered, scaler)
    pad = pad_sequences_torch([mat], maxlen=pad_len, padding='pre', truncating='pre')
    demo_array = demographics.to_pandas()[demo_feature_cols].to_numpy(dtype=np.float32)
    scaled_demo_array = demo_scaler.transform(demo_array)

    with torch.no_grad():
        x = torch.FloatTensor(pad).to(device)
        demo_data = torch.FloatTensor(scaled_demo_array).to(device)
        outputs = None
        for model in models:
            model.eval()
            p = torch.softmax(model(x, demo_data), dim=1)
            if outputs is None: outputs = p
            else: outputs += p
        outputs /= len(models)
            
        idx = int(outputs.argmax(dim=1)[0].cpu().numpy())
        
    return str(gesture_classes[idx])

# Kaggle competition interface
import kaggle_evaluation.cmi_inference_server
inference_server = kaggle_evaluation.cmi_inference_server.CMIInferenceServer(predict)

if os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
    inference_server.serve()
else:
    inference_server.run_local_gateway(
        data_paths=(
            '/kaggle/input/cmi-detect-behavior-with-sensor-data/test.csv',
            '/kaggle/input/cmi-detect-behavior-with-sensor-data/test_demographics.csv',
        )
    )
