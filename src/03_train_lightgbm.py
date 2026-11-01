"""
LightGBM Training Script: 3-Class and Hierarchical
Seeds: [42, 123, 999] x 5 Folds Clean Nested Validation (10% inner validation split)
"""
import os, time, warnings
os.environ["OMP_NUM_THREADS"] = "4"
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import log_loss, accuracy_score
warnings.filterwarnings('ignore')

cache_dir = '/Users/tejasmittal32/College/Cosmic Sort/cached_predictions'
X_tr = np.load(os.path.join(cache_dir, 'X_train_77.npy'))
y = np.load(os.path.join(cache_dir, 'y.npy'))
X_te = np.load(os.path.join(cache_dir, 'X_test_77.npy'))
size_r_tr = np.load(os.path.join(cache_dir, 'size_r_tr.npy'))
size_r_te = np.load(os.path.join(cache_dir, 'size_r_te.npy'))

n_train, n_test = len(y), len(X_te)
SEEDS = [42, 123, 999]
N_SEEDS = len(SEEDS)

# Stage 1 binary: 0=GALAXY, 1=POINT (STAR/QSO)
y_s1 = np.where(y == 0, 0, 1)

acc_lgb_3c = np.zeros((n_train, 3))
acc_lgb_h  = np.zeros((n_train, 3))
tacc_lgb_3c = np.zeros((n_test, 3))
tacc_lgb_h  = np.zeros((n_test, 3))

losses_3c = np.zeros((N_SEEDS, 5))
losses_h  = np.zeros((N_SEEDS, 5))

t0 = time.time()
print("=" * 70, flush=True)
print("TRAINING LIGHTGBM 3-CLASS & HIERARCHICAL (3 SEEDS x 5 FOLDS)", flush=True)
print("=" * 70, flush=True)

for s_idx, SEED in enumerate(SEEDS):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    oof_3c = np.zeros((n_train, 3))
    oof_h  = np.zeros((n_train, 3))
    
    for fold, (tr_idx, va_idx) in enumerate(skf.split(X_tr, y)):
        f_t0 = time.time()
        inner_tr, inner_val = train_test_split(
            tr_idx, test_size=0.10, stratify=y[tr_idx], random_state=SEED + fold
        )
        
        # 1. LightGBM 3-Class
        ds_tr = lgb.Dataset(X_tr[inner_tr], label=y[inner_tr], free_raw_data=False)
        ds_va = lgb.Dataset(X_tr[inner_val], label=y[inner_val], free_raw_data=False)
        params_3c = {
            'objective': 'multiclass',
            'num_class': 3,
            'metric': 'multi_logloss',
            'learning_rate': 0.03,
            'num_leaves': 80,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': SEED + fold,
            'n_jobs': 4,
            'verbose': -1
        }
        bst_3c = lgb.train(
            params_3c, ds_tr, num_boost_round=1500,
            valid_sets=[ds_va], callbacks=[lgb.early_stopping(50, verbose=False)]
        )
        p_va_3c = bst_3c.predict(X_tr[va_idx])
        p_te_3c = bst_3c.predict(X_te)
        oof_3c[va_idx] = p_va_3c
        tacc_lgb_3c += (p_te_3c / 5.0) / N_SEEDS
        ll_3c = log_loss(y[va_idx], p_va_3c)
        losses_3c[s_idx, fold] = ll_3c
        
        # 2. LightGBM Hierarchical
        # Stage 1: Galaxy vs Point
        ds_s1_tr = lgb.Dataset(X_tr[inner_tr], label=y_s1[inner_tr], free_raw_data=False)
        ds_s1_va = lgb.Dataset(X_tr[inner_val], label=y_s1[inner_val], free_raw_data=False)
        params_bin = {
            'objective': 'binary',
            'metric': 'binary_logloss',
            'learning_rate': 0.03,
            'num_leaves': 80,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': SEED + fold,
            'n_jobs': 4,
            'verbose': -1
        }
        bst_s1 = lgb.train(
            params_bin, ds_s1_tr, num_boost_round=1500,
            valid_sets=[ds_s1_va], callbacks=[lgb.early_stopping(50, verbose=False)]
        )
        p_pt_va = bst_s1.predict(X_tr[va_idx])
        p_pt_te = bst_s1.predict(X_te)
        
        # Stage 2: Star (y=2 -> 0) vs QSO (y=1 -> 1) on point sources
        pt_tr = (y[inner_tr] != 0)
        pt_val = (y[inner_val] != 0)
        ds_s2_tr = lgb.Dataset(X_tr[inner_tr][pt_tr], label=(y[inner_tr][pt_tr] == 1).astype(int), free_raw_data=False)
        ds_s2_va = lgb.Dataset(X_tr[inner_val][pt_val], label=(y[inner_val][pt_val] == 1).astype(int), free_raw_data=False)
        bst_s2 = lgb.train(
            params_bin, ds_s2_tr, num_boost_round=1500,
            valid_sets=[ds_s2_va], callbacks=[lgb.early_stopping(50, verbose=False)]
        )
        p_qso_va = bst_s2.predict(X_tr[va_idx])
        p_qso_te = bst_s2.predict(X_te)
        
        recomb_va = np.column_stack([
            1.0 - p_pt_va,
            p_pt_va * p_qso_va,
            p_pt_va * (1.0 - p_qso_va)
        ])
        recomb_va /= recomb_va.sum(axis=1, keepdims=True)
        recomb_te = np.column_stack([
            1.0 - p_pt_te,
            p_pt_te * p_qso_te,
            p_pt_te * (1.0 - p_qso_te)
        ])
        recomb_te /= recomb_te.sum(axis=1, keepdims=True)
        
        oof_h[va_idx] = recomb_va
        tacc_lgb_h += (recomb_te / 5.0) / N_SEEDS
        ll_h = log_loss(y[va_idx], recomb_va)
        losses_h[s_idx, fold] = ll_h
        
        print(f"  Seed {SEED} Fold {fold+1}: LGB 3c LL = {ll_3c:.5f} | LGB Hier LL = {ll_h:.5f} ({time.time()-f_t0:.1f}s)", flush=True)
        
    acc_lgb_3c += oof_3c / N_SEEDS
    acc_lgb_h  += oof_h / N_SEEDS
    print(f"Seed {SEED} Complete: 3c OOF LL = {log_loss(y, oof_3c):.5f} | Hier OOF LL = {log_loss(y, oof_h):.5f}", flush=True)

np.save(os.path.join(cache_dir, 'clean_oof_lgb_3c.npy'), acc_lgb_3c)
np.save(os.path.join(cache_dir, 'clean_test_lgb_3c.npy'), tacc_lgb_3c)
np.save(os.path.join(cache_dir, 'fold_losses_lgb_3c.npy'), losses_3c)

np.save(os.path.join(cache_dir, 'clean_oof_lgb_h.npy'), acc_lgb_h)
np.save(os.path.join(cache_dir, 'clean_test_lgb_h.npy'), tacc_lgb_h)
np.save(os.path.join(cache_dir, 'fold_losses_lgb_h.npy'), losses_h)

print(f"\nLightGBM Done in {time.time()-t0:.1f}s.")
print(f"LGB 3-Class 3-Seed OOF:  {log_loss(y, acc_lgb_3c):.5f} ± {np.std(losses_3c)/np.sqrt(15):.5f}")
print(f"LGB Hier    3-Seed OOF:  {log_loss(y, acc_lgb_h):.5f} ± {np.std(losses_h)/np.sqrt(15):.5f}")
