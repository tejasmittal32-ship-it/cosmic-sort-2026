"""
ResMLP-2 Training Script
Seeds: [42, 123, 999] x 5 Folds Clean Nested Validation (10% inner validation split)
2 Residual Blocks, 256 hidden dim, GELU, Dropout 0.15, QuantileTransformer normal, OneCycleLR 25 epochs.
"""
import os, time, warnings
os.environ["OMP_NUM_THREADS"] = "4"
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import QuantileTransformer
from sklearn.metrics import log_loss, accuracy_score
warnings.filterwarnings('ignore')

torch.set_num_threads(4)
device = torch.device('cpu')

cache_dir = '/Users/tejasmittal32/College/Cosmic Sort/cached_predictions'
X_tr = np.load(os.path.join(cache_dir, 'X_train_77.npy'))
y = np.load(os.path.join(cache_dir, 'y.npy'))
X_te = np.load(os.path.join(cache_dir, 'X_test_77.npy'))

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

class ResMLP_2(nn.Module):
    def __init__(self, in_features, hidden_dim=256, num_classes=3, dropout_rate=0.15):
        super().__init__()
        self.input_bn = nn.BatchNorm1d(in_features)
        self.stem = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate)
        )
        self.res1 = TabularResBlock(hidden_dim, dropout_rate)
        self.res2 = TabularResBlock(hidden_dim, dropout_rate)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim // 2, num_classes)
        )
    def forward(self, x):
        h = self.stem(self.input_bn(x))
        h = self.res1(h)
        h = self.res2(h)
        return self.head(h)

acc_mlp2 = np.zeros((n_train, 3))
tacc_mlp2 = np.zeros((n_test, 3))
losses_mlp2 = np.zeros((N_SEEDS, 5))

t0 = time.time()
print("=" * 70, flush=True)
print("TRAINING RESMLP-2 (3 SEEDS x 5 FOLDS)", flush=True)
print("=" * 70, flush=True)

for s_idx, SEED in enumerate(SEEDS):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    oof_mlp2 = np.zeros((n_train, 3))
    
    for fold, (tr_idx, va_idx) in enumerate(skf.split(X_tr, y)):
        f_t0 = time.time()
        inner_tr, inner_val = train_test_split(
            tr_idx, test_size=0.10, stratify=y[tr_idx], random_state=SEED + fold
        )
        
        scaler = QuantileTransformer(output_distribution='normal', random_state=SEED, n_quantiles=2000)
        X_tr_sc = scaler.fit_transform(X_tr[inner_tr]).astype(np.float32)
        X_val_sc = scaler.transform(X_tr[inner_val]).astype(np.float32)
        X_va_sc = scaler.transform(X_tr[va_idx]).astype(np.float32)
        X_te_sc = scaler.transform(X_te).astype(np.float32)
        
        mlp = ResMLP_2(n_features, hidden_dim=256, num_classes=3, dropout_rate=0.15).to(device)
        crit = nn.CrossEntropyLoss()
        opt = torch.optim.AdamW(mlp.parameters(), lr=2e-3, weight_decay=1e-4)
        
        epochs = 25
        bs = 512
        n_batches = int(np.ceil(len(X_tr_sc) / bs))
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=2e-3, total_steps=epochs * n_batches, pct_start=0.3)
        
        best_vl = 999.0
        best_st = None
        for ep in range(epochs):
            mlp.train()
            perm = np.random.permutation(len(X_tr_sc))
            for b in range(n_batches):
                bi = perm[b*bs:(b+1)*bs]
                bx = torch.tensor(X_tr_sc[bi], dtype=torch.float32, device=device)
                by = torch.tensor(y[inner_tr][bi], dtype=torch.long, device=device)
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
                vloss = log_loss(y[inner_val], vp)
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
            p_va = F.softmax(torch.cat(vl_va, dim=0), dim=1).numpy()
            
            vl_te = []
            for b in range(int(np.ceil(len(X_te_sc) / bs))):
                bx = torch.tensor(X_te_sc[b*bs:(b+1)*bs], dtype=torch.float32, device=device)
                vl_te.append(mlp(bx))
            p_te = F.softmax(torch.cat(vl_te, dim=0), dim=1).numpy()
            
        oof_mlp2[va_idx] = p_va
        tacc_mlp2 += (p_te / 5.0) / N_SEEDS
        ll = log_loss(y[va_idx], p_va)
        losses_mlp2[s_idx, fold] = ll
        print(f"  Seed {SEED} Fold {fold+1}: ResMLP-2 OOF LL = {ll:.5f} ({time.time()-f_t0:.1f}s)", flush=True)
        
    acc_mlp2 += oof_mlp2 / N_SEEDS
    print(f"Seed {SEED} Complete: ResMLP-2 OOF LL = {log_loss(y, oof_mlp2):.5f}", flush=True)

np.save(os.path.join(cache_dir, 'clean_oof_resmlp2.npy'), acc_mlp2)
np.save(os.path.join(cache_dir, 'clean_test_resmlp2.npy'), tacc_mlp2)
np.save(os.path.join(cache_dir, 'fold_losses_resmlp2.npy'), losses_mlp2)

print(f"\nResMLP-2 Done in {time.time()-t0:.1f}s.")
print(f"ResMLP-2 3-Seed OOF: {log_loss(y, acc_mlp2):.5f} ± {np.std(losses_mlp2)/np.sqrt(15):.5f}")
