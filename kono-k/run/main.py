import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import StratifiedKFold
from timm.scheduler import CosineLRScheduler
import joblib
import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
import torch.nn.functional as F
import wandb
from copy import copy
import json
import random
import os

from src.metric.cmi_2025_metric import CompetitionMetric
from src.dataset.two_branch_model import preprocess_sequence, pad_sequences_torch, CMI3Dataset
from src.dataset.transform import Augment
from src.model.two_branch_model import TwoBranchModel
from src.model.three_branch_model import ThreeBranchModel
from src.utils.train_utils import EarlyStopping, EMA
from src.utils.config import CFG as config
from src.dataset.features import FE

#class CFG:
    #RAW_DIR = Path("../data")
    #PRETRAINED_DIR = Path("/kaggle/input/cmi3-models-p") 
    #EXPORT_DIR = Path("../result")
    #PAD_PERCENTILE = 100
    #maxlen = PAD_PERCENTILE
    #LR_INIT = 1e-3
    #WD = 3e-3
    # MIXUP_ALPHA = 0.4
    #PATIENCE = 40
    #FOLDS = 5
    #random_state = 42
    #epochs_warmup = 20
    #warmup_lr_init = 1.822126131809773e-05
    #lr_min = 3.810323058740104e-09

CFG = config("../input/config.yaml")
CFG.RAW_DIR = Path(CFG.RAW_DIR)
CFG.EXPORT_DIR = Path(CFG.EXPORT_DIR)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def seed_evrything():
    os.environ['PYTHONHASHSEED'] = str(CFG.random_state)
    random.seed(CFG.random_state)
    np.random.seed(CFG.random_state)
    torch.manual_seed(CFG.random_state)
    torch.cuda.manual_seed(CFG.random_state)
    torch.backends.cudnn.deterministic = True

seed_evrything()

def main():
    df = pd.read_csv(CFG.RAW_DIR / "train.csv")
    df_demo = pd.read_csv(CFG.RAW_DIR / "train_demographics.csv")

    with open(CFG.RAW_DIR / "remove_id.json", "r") as f:
        d = json.load(f)
        remove_id = d["seq_id"]

    #df = df[~df["sequence_id"].isin(remove_id)]

    # Feature Engineering
    fe = FE(df=df)
    df = fe.transform()

    # Label encoding
    le = LabelEncoder()
    df['gesture_int'] = le.fit_transform(df['gesture'])
    np.save(CFG.EXPORT_DIR / "gesture_classes.npy", le.classes_)
    
    # Process demographics data
    demo_features = ['adult_child', 'age', 'sex', 'handedness', 'height_cm', 'shoulder_to_wrist_cm']
    if 'elbow_to_wrist_cm' in df_demo.columns:
        demo_features.append('elbow_to_wrist_cm')
    
    # Create subject to demographics mapping
    subject_to_demo = {}
    for _, row in df_demo.iterrows():
        subject_to_demo[row['subject']] = row[demo_features].values.astype(np.float32)
    
    # Add demographics to main dataframe
    df = pd.merge(df, df_demo, on="subject")
    
    # Save demographics info
    np.save(CFG.EXPORT_DIR / "demo_features.npy", np.array(demo_features))
    joblib.dump(subject_to_demo, CFG.EXPORT_DIR / "subject_to_demo.pkl")

    # Feature list
    meta_cols = {'gesture', 'gesture_int', 'sequence_type', 'behavior', 'orientation',
                 'row_id', 'subject', 'phase', 'sequence_id', 'sequence_counter'}
    feature_cols = [c for c in df.columns if c not in meta_cols]
    feature_cols = [c for c in feature_cols if c not in demo_features]

    imu_cols = [c for c in feature_cols if not (c.startswith('thm_') or c.startswith('tof_'))]
    tof_cols = [c for c in feature_cols if c.startswith('tof_')]
    thm_cols = [c for c in feature_cols if c.startswith('thm_')]
    
    # Reorder feature columns to match model expectation: IMU -> TOF -> THM
    feature_cols_ordered = imu_cols + tof_cols + thm_cols
    print(f"  IMU {len(imu_cols)} | TOF {len(tof_cols)} | THM {len(thm_cols)} | total {len(feature_cols_ordered)} features")

    # Global scaler
    scaler = StandardScaler().fit(df[feature_cols_ordered].ffill().bfill().fillna(0).values)
    joblib.dump(scaler, CFG.EXPORT_DIR / "scaler.pkl")
    demo_scaler = StandardScaler().fit(df_demo[demo_features].values)
    joblib.dump(demo_scaler, CFG.EXPORT_DIR / "demo_scaler.pkl")

    # Build sequences
    seq_gp = df.groupby('sequence_id')
    X_list, y_list, id_list, demo_list = [], [], [], []
    for seq_id, seq in seq_gp:
        if CFG.scaling == "global":
            mat = preprocess_sequence(seq, feature_cols_ordered, scaler)
        elif CFG.scaling == "local":
            local_scaler = StandardScaler()
            local_scaler.fit(seq[feature_cols_ordered].ffill().bfill().fillna(0).values)
            mat = local_scaler.transform(seq[feature_cols_ordered].ffill().bfill().fillna(0).values)

        X_list.append(mat)
        y_list.append(seq['gesture_int'].iloc[0])
        id_list.append(seq_id)
        # Get demographics for this sequence
        demo_data = np.array([seq[demo_features].iloc[0].to_numpy()])
        scaled_demo_data = demo_scaler.transform(demo_data)
        demo_list.append(scaled_demo_data[0])
        # lens.append(len(mat))
    
    pad_len = CFG.PAD_PERCENTILE#int(np.percentile(lens, PAD_PERCENTILE))
    np.save(CFG.EXPORT_DIR / "sequence_maxlen.npy", pad_len)
    np.save(CFG.EXPORT_DIR / "feature_cols.npy", np.array(feature_cols))
    id_list = np.array(id_list)
    X_list_all = pad_sequences_torch(X_list, maxlen=pad_len, padding='pre', truncating='pre')
    y_list_all = np.eye(len(le.classes_))[y_list].astype(np.float32)  # One-hot encoding
    demo_list_all = np.array(demo_list)

    augmenter = Augment(
        p_jitter=CFG.p_jitter, sigma=0.03291295776089293, scale_range=(0.7542342630597011,1.1625052821731077),
        p_dropout=0.41782786013520684, p_f_dropout=CFG.p_f_dropout, p_left_right=CFG.p_left_right,
        p_moda=CFG.p_moda, drift_std=0.0040285239353308015, drift_max=0.3929358950258158    
    )

    # Split
    skf = StratifiedKFold(n_splits=CFG.FOLDS, shuffle=True, random_state=CFG.random_state)
    models = []
    cv_acc_list = []

    run = wandb.init(
        # Set the wandb entity where your project will be logged (generally your team name).
        entity="CMI2025",
        # Set the wandb project where this run will be logged.
        project="CMI2025",
        name=CFG.exp_name,
        config=CFG.to_dict()
    )
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(id_list, np.argmax(y_list_all, axis=1))):

        train_list= X_list_all[train_idx]
        train_y_list= y_list_all[train_idx]
        train_demo_list = demo_list_all[train_idx]
        val_list = X_list_all[val_idx]
        val_y_list= y_list_all[val_idx]
        val_demo_list = demo_list_all[val_idx]

        
        # Data loaders
        if CFG.model_name == "ThreeBranchModel":
            train_dataset = CMI3Dataset(train_list, train_y_list, CFG.maxlen, mode="train", imu_dim=len(imu_cols),
                                    augment=augmenter, demo_features=train_demo_list)
            val_dataset = CMI3Dataset(val_list, val_y_list, CFG.maxlen, mode="val", demo_features=val_demo_list, augment=augmenter)
        else:
            train_dataset = CMI3Dataset(train_list, train_y_list, CFG.maxlen, mode="train", imu_dim=len(imu_cols),
                                    augment=augmenter)
            val_dataset = CMI3Dataset(val_list, val_y_list, CFG.maxlen, mode="val")
        
        train_loader = DataLoader(train_dataset, batch_size=CFG.BATCH_SIZE, shuffle=True, num_workers=4,drop_last=True)
        val_loader = DataLoader(val_dataset, batch_size=CFG.BATCH_SIZE, shuffle=False, num_workers=4,drop_last=True)

    
        # Model
        if CFG.model_name == "ThreeBranchModel":
            model = ThreeBranchModel(CFG.maxlen, len(imu_cols), len(tof_cols), len(thm_cols), len(demo_features),
                          len(le.classes_)).to(device)
        else:
            model = TwoBranchModel(CFG.maxlen, len(imu_cols), len(tof_cols), 
                          len(le.classes_)).to(device)
        ema = EMA(model, decay=0.999)
        # Optimizer and scheduler
        optimizer = Adam(model.parameters(), lr=float(CFG.LR_INIT), weight_decay=float(CFG.WD))
        
        steps_per_epoch = len(train_loader)
        nbatch = len(train_loader)
        warmup = CFG.epochs_warmup * nbatch
        nsteps = CFG.EPOCHS * nbatch
        scheduler = CosineLRScheduler(optimizer,
                          warmup_t=warmup, warmup_lr_init=CFG.warmup_lr_init, warmup_prefix=True,
                          t_initial=(nsteps - warmup), lr_min=float(CFG.lr_min), k_decay=CFG.k_decay) 
    
        early_stopping = EarlyStopping(patience=CFG.PATIENCE, restore_best_weights=True)
    
        train_loss = 0.0
        train_acc = 0.0
        val_loss = 0.0
        val_acc = 0.0
        val_best_acc = 0.0
        i_scheduler = 0
        best_model = None
        
        # Training loop
        print("▶ Starting training...")
        for epoch in range(CFG.EPOCHS):
            model.train()
            train_preds = []
            train_targets = []
            for batch in (train_loader):  
                if CFG.model_name == "ThreeBranchModel":
                    X, y, demo = batch
                    X, y, demo = X.float().to(device), y.to(device), demo.float().to(device)
                    logits = model(X, demo)
                else:
                    X, y = batch
                    X, y = X.float().to(device), y.to(device)
                    logits = model(X)
                optimizer.zero_grad()
    
                loss = -torch.sum(F.log_softmax(logits, dim=1) * y, dim=1).mean()
                loss.backward()
                optimizer.step()
                ema.update(model)
                train_preds.extend(logits.argmax(dim=1).cpu().numpy())
                train_targets.extend(y.argmax(dim=1).cpu().numpy())
                scheduler.step(i_scheduler)
                i_scheduler +=1
    
                train_loss += loss.item()
                
            model.eval()
            with torch.inference_mode():
                val_preds = []
                val_targets = []
                for batch in (val_loader):  
                    if CFG.model_name == "ThreeBranchModel":
                        X, y, demo = batch
                        demo = demo.float().to(device)
                    else:
                        X, y = batch
                        demo = None
                    
                    half = CFG.BATCH_SIZE // 2         

                    x_front = X[:half]               
                    x_back  = X[half:].clone()      
                    
                    x_back[:, :, 7:] = 0.0    
                    X = torch.cat([x_front, x_back], dim=0)  # (B, C, T)
                    X, y = X.float().to(device), y.to(device)
                    
                    if demo is not None:
                        demo_combined = torch.cat([demo[:half], demo[half:]], dim=0)
                        logits = model(X, demo_combined)
                    else:
                        logits = model(X)
                    val_preds.extend(logits.argmax(dim=1).cpu().numpy())
                    val_targets.extend(y.argmax(dim=1).cpu().numpy())
                    
                    loss = F.cross_entropy(logits, y)
                    val_loss += loss.item()
    
            train_acc = CompetitionMetric().calculate_hierarchical_f1(
                pd.DataFrame({'gesture': le.classes_[train_targets]}),
                pd.DataFrame({'gesture': le.classes_[train_preds]}))
            val_acc = CompetitionMetric().calculate_hierarchical_f1(
                pd.DataFrame({'gesture': le.classes_[val_targets]}),
                pd.DataFrame({'gesture': le.classes_[val_preds]}))
            train_loss /= len(train_loader)
            val_loss /= len(val_loader)
            if val_acc > val_best_acc:
                val_best_acc = copy(val_acc)
                best_model = copy(model)

            wandb.log(
                {
                    "epoch": epoch, 
                    f"train_loss_{fold}": train_loss, 
                    f"val_loss_{fold}": val_loss,
                    f"train_acc_{fold}": train_acc, 
                    f"val_acc_{fold}": val_acc,
                    f"val_best_acc_{fold}": val_best_acc,
                },
                #step=epoch
                )
        models.append(model)
        cv_acc_list.append(val_best_acc)
        
        # Save model
        save_dict = {
            'model_state_dict': best_model.state_dict(),
            'imu_dim': len(imu_cols),
            'tof_dim': len(tof_cols),
            'n_classes': len(le.classes_),
            'pad_len': pad_len,
            'model_name': CFG.model_name
        }
        if CFG.model_name == "ThreeBranchModel":
            save_dict['demo_dim'] = len(demo_features)
            save_dict['thm_dim'] = len(thm_cols)
        
        model_filename = f"gesture_{CFG.model_name.lower()}_fold{fold}.pth"
        torch.save(save_dict, CFG.EXPORT_DIR / model_filename)
        print(f"fold: {fold} val_all_acc: {val_acc:.4f}")
        print("✔ Training done – artefacts saved in", CFG.EXPORT_DIR)

    wandb.log(
                    {
                        "cv_mean_score": np.mean(cv_acc_list),
                        "cv_std_score": np.std(cv_acc_list),
                        "cv_scores": cv_acc_list,
                        "n_folds": CFG.FOLDS,
                    }
                )

if __name__ == "__main__":
    main()
