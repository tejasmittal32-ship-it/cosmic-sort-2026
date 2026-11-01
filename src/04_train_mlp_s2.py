"""
Stage-2 Dedicated Point-Source MLP Training Script
Seeds: [42, 123, 999] x 5 Folds Clean Nested Validation (10% inner validation split)
Trained strictly on unresolved point sources (y != 0).
Recombined with Stage 1 P(POINT) from clean XGBoost hierarchical model.
"""
import os, time, warnings
os.environ["OMP_NUM_THREADS"] = "4"
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import log_loss, accuracy_score
warnings.filterwarnings('ignore')

torch.set_num_threads(4)
device = torch.device('cpu')

cache_dir = '/Users/tejasmittal32/College/Cosmic Sort/cached_predictions'
X_tr = np.load(os.path.join(cache_dir, 'X_train_77.npy'))
y = np.load(os.path.join(cache_dir, 'y.npy'))
X_te = np.load(os.path.join(cache_dir, 'X_test_77.npy'))

# Clean Stage 1 P(Point) from XGBoost
clean_xgb_h = np.load(os.path.join(cache_dir, 'clean_oof_xgb_h.npy'))
p_pt_va_all = 1.0 - clean_xgb_h[:, 0]
clean_test_xgb_h = np.load(os.path.join(cache_dir, 'clean_test_xgb_h.npy'))
p_pt_te_all = 1.0 - clean_test_xgb_h[:, 0]

n_train, n_test = len(y), len(X_te)
n_features = X_tr.shape[1]
SEEDS = [42, 123, 999]
N_SEEDS = len(SEEDS)

class TabularResBlock(nn.Module):
    def __init__(self, hidden_dim, dropout_rate=0.15):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(dropout_rate)
        )
    def forward(self, x):
        return x + self.block(x)

class PointSourceResMLP(nn.Module):
    def __init__(self, in_features, hidden_dim=256, dropout=0.15):
        super().__init__()
        self.input_bn = nn.BatchNorm1d(in_features)
        self.stem = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        self.res1 = TabularResBlock(hidden_dim, dropout)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 2)
        )
    def forward(self, x):
        h = self.stem(self.input_bn(x))
        h = self.res1(h)
        return self.head(h)

acc_s2_mlp = np.zeros((n_train, 3))
tacc_s2_mlp = np.zeros((n_test, 3))
losses_s2 = np.zeros((N_SEEDS, 5))

t0 = time.time()
print("=" * 70, flush=True)
print("TRAINING STAGE-2 DEDICATED POINT-SOURCE MLP (3 SEEDS x 5 FOLDS)", flush=True)
print("=" * 70, flush=True)

for s_idx, SEED in enumerate(SEEDS):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    oof_s2 = np.zeros((n_train, 3))
    
    for fold, (tr_idx, va_idx) in enumerate(skf.split(X_tr, y)):
        f_t0 = time.time()
        inner_tr, inner_val = train_test_split(
            tr_idx, test_size=0.10, stratify=y[tr_idx], random_state=SEED + fold
        )
        
        pt_tr = (y[inner_tr] != 0)
        pt_val = (y[inner_val] != 0)
        
        scaler = RobustScaler()
        X_tr_sc = scaler.fit_transform(X_tr[inner_tr][pt_tr]).astype(np.float32)
        X_val_sc = scaler.transform(X_tr[inner_val][pt_val]).astype(np.float32)
        X_va_sc = scaler.transform(X_tr[va_idx]).astype(np.float32)
        X_te_sc = scaler.transform(X_te).astype(np.float32)
        
        y_tr_bin = (y[inner_tr][pt_tr] == 1).astype(int)
        y_val_bin = (y[inner_val][pt_val] == 1).astype(int)
        
        mlp = PointSourceResMLP(n_features, hidden_dim=256, dropout=0.15).to(device)
        opt = torch.optim.AdamW(mlp.parameters(), lr=2e-3, weight_decay=1e-4)
        crit = nn.CrossEntropyLoss()
        
        epochs = 25
        bs = 256
        n_batches = int(np.ceil(len(X_tr_sc) / bs))
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=2e-3, total_steps=epochs * n_batches, pct_start=0.2)
        
        best_vl = 999.0
        best_st = None
        for ep in range(epochs):
            mlp.train()
            perm = np.random.permutation(len(X_tr_sc))
            for b in range(n_batches):
                bi = perm[b*bs:(b+1)*bs]
                bx = torch.tensor(X_tr_sc[bi], dtype=torch.float32, device=device)
                by = torch.tensor(y_tr_bin[bi], dtype=torch.long, device=device)
                opt.zero_grad()
                crit(mlp(bx), by).backward()
                opt.step()
                sched.step()
            
            mlp.eval()
            with torch.no_grad():
                vl = []
                for b in range(int(np.ceil(len(X_val_sc) / bs))):
                    bx = torch.tensor(X_val_sc[b*bs:(b+1)*bs], dtype=torch.float32, device=device)
                    vl.append(mlp(bx))
                vp = F.softmax(torch.cat(vl, dim=0), dim=1).numpy()
                vloss = log_loss(y_val_bin, vp)
                if vloss < best_vl:
                    best_vl = vloss
                    best_st = {k: v.clone() for k, v in mlp.state_dict().items()}
        
        mlp.load_state_dict(best_st)
        mlp.eval()
        with torch.no_grad():
            vl_va = []
            for b in range(int(np.ceil(len(X_va_sc) / bs))):
                bx = torch.tensor(X_va_sc[b*bs:(b+1)*bs], dtype=torch.float32, device=device)
                vl_va.append(mlp(bx))
            p_qso_cond = F.softmax(torch.cat(vl_va, dim=0), dim=1)[:, 1].numpy()
            
            vl_te = []
            for b in range(int(np.ceil(len(X_te_sc) / bs))):
                bx = torch.tensor(X_te_sc[b*bs:(b+1)*bs], dtype=torch.float32, device=device)
                vl_te.append(mlp(bx))
            p_qso_te_cond = F.softmax(torch.cat(vl_te, dim=0), dim=1)[:, 1].numpy()
            
        p_pt_va = p_pt_va_all[va_idx]
        recomb_va = np.column_stack([
            1.0 - p_pt_va,
            p_pt_va * p_qso_cond,
            p_pt_va * (1.0 - p_qso_cond)
        ])
        recomb_va /= recomb_va.sum(axis=1, keepdims=True)
        
        recomb_te = np.column_stack([
            1.0 - p_pt_te_all,
            p_pt_te_all * p_qso_te_cond,
            p_pt_te_all * (1.0 - p_qso_te_cond)
        ])
        recomb_te /= recomb_te.sum(axis=1, keepdims=True)
        
        oof_s2[va_idx] = recomb_va
        tacc_s2_mlp += (recomb_te / 5.0) / N_SEEDS
        ll = log_loss(y[va_idx], recomb_va)
        losses_s2[s_idx, fold] = ll
        print(f"  Seed {SEED} Fold {fold+1}: Stage-2 MLP Recombined OOF LL = {ll:.5f} ({time.time()-f_t0:.1f}s)", flush=True)
        
    acc_s2_mlp += oof_s2 / N_SEEDS
    print(f"Seed {SEED} Complete: Stage-2 MLP OOF LL = {log_loss(y, oof_s2):.5f}", flush=True)

np.save(os.path.join(cache_dir, 'clean_oof_mlp_s2.npy'), acc_s2_mlp)
np.save(os.path.join(cache_dir, 'clean_test_mlp_s2.npy'), tacc_s2_mlp)
np.save(os.path.join(cache_dir, 'fold_losses_mlp_s2.npy'), losses_s2)

print(f"\nStage-2 MLP Done in {time.time()-t0:.1f}s.")
print(f"Stage-2 MLP 3-Seed OOF: {log_loss(y, acc_s2_mlp):.5f} ± {np.std(losses_s2)/np.sqrt(15):.5f}")
