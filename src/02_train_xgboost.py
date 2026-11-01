"""
XGBoost Training Script: 3-Class and Hierarchical
Seeds: [42, 123, 999] x 5 Folds Clean Nested Validation (10% inner validation split)
"""
import os, time, warnings
os.environ["OMP_NUM_THREADS"] = "4"
import numpy as np
import xgboost as xgb
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

acc_xgb_3c = np.zeros((n_train, 3))
acc_xgb_h  = np.zeros((n_train, 3))
tacc_xgb_3c = np.zeros((n_test, 3))
tacc_xgb_h  = np.zeros((n_test, 3))

losses_3c = np.zeros((N_SEEDS, 5))
losses_h  = np.zeros((N_SEEDS, 5))

t0 = time.time()
print("=" * 70, flush=True)
print("TRAINING XGBOOST 3-CLASS & HIERARCHICAL (3 SEEDS x 5 FOLDS)", flush=True)
print("=" * 70, flush=True)

# Pre-create test DMatrix once to save time
dtest = xgb.DMatrix(X_te)

for s_idx, SEED in enumerate(SEEDS):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    oof_3c = np.zeros((n_train, 3))
    oof_h  = np.zeros((n_train, 3))
    
    for fold, (tr_idx, va_idx) in enumerate(skf.split(X_tr, y)):
        f_t0 = time.time()
        inner_tr, inner_val = train_test_split(
            tr_idx, test_size=0.10, stratify=y[tr_idx], random_state=SEED + fold
        )
        
        douter = xgb.DMatrix(X_tr[va_idx])
        
        # 1. XGBoost 3-Class
        dtr = xgb.DMatrix(X_tr[inner_tr], label=y[inner_tr])
        dva = xgb.DMatrix(X_tr[inner_val], label=y[inner_val])
        params_3c = {
            'objective': 'multi:softprob',
            'num_class': 3,
            'eval_metric': 'mlogloss',
            'learning_rate': 0.035,
            'max_depth': 7,
            'min_child_weight': 3,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'seed': SEED + fold,
            'nthread': 4
        }
        bst_3c = xgb.train(
            params_3c, dtr, num_boost_round=1500,
            evals=[(dva, 'val')], early_stopping_rounds=50, verbose_eval=False
        )
        p_va_3c = bst_3c.predict(douter, iteration_range=(0, bst_3c.best_iteration))
        p_te_3c = bst_3c.predict(dtest, iteration_range=(0, bst_3c.best_iteration))
        oof_3c[va_idx] = p_va_3c
        tacc_xgb_3c += (p_te_3c / 5.0) / N_SEEDS
        ll_3c = log_loss(y[va_idx], p_va_3c)
        losses_3c[s_idx, fold] = ll_3c
        
        # 2. XGBoost Hierarchical
        # Stage 1: Galaxy vs Point
        dtr_s1 = xgb.DMatrix(X_tr[inner_tr], label=y_s1[inner_tr])
        dva_s1 = xgb.DMatrix(X_tr[inner_val], label=y_s1[inner_val])
        params_bin = {
            'objective': 'binary:logistic',
            'eval_metric': 'logloss',
            'learning_rate': 0.035,
            'max_depth': 7,
            'min_child_weight': 3,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'seed': SEED + fold,
            'nthread': 4
        }
        bst_s1 = xgb.train(
            params_bin, dtr_s1, num_boost_round=1500,
            evals=[(dva_s1, 'val')], early_stopping_rounds=50, verbose_eval=False
        )
        p_pt_va = bst_s1.predict(douter, iteration_range=(0, bst_s1.best_iteration))
        p_pt_te = bst_s1.predict(dtest, iteration_range=(0, bst_s1.best_iteration))
        
        # Stage 2: Point sources only (y != 0)
        pt_tr = (y[inner_tr] != 0)
        pt_val = (y[inner_val] != 0)
        dtr_s2 = xgb.DMatrix(X_tr[inner_tr][pt_tr], label=(y[inner_tr][pt_tr] == 1).astype(int))
        dva_s2 = xgb.DMatrix(X_tr[inner_val][pt_val], label=(y[inner_val][pt_val] == 1).astype(int))
        bst_s2 = xgb.train(
            params_bin, dtr_s2, num_boost_round=1500,
            evals=[(dva_s2, 'val')], early_stopping_rounds=50, verbose_eval=False
        )
        p_qso_va = bst_s2.predict(douter, iteration_range=(0, bst_s2.best_iteration))
        p_qso_te = bst_s2.predict(dtest, iteration_range=(0, bst_s2.best_iteration))
        
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
        tacc_xgb_h += (recomb_te / 5.0) / N_SEEDS
        ll_h = log_loss(y[va_idx], recomb_va)
        losses_h[s_idx, fold] = ll_h
        
        print(f"  Seed {SEED} Fold {fold+1}: XGB 3c LL = {ll_3c:.5f} | XGB Hier LL = {ll_h:.5f} ({time.time()-f_t0:.1f}s)", flush=True)
        
    acc_xgb_3c += oof_3c / N_SEEDS
    acc_xgb_h  += oof_h / N_SEEDS
    print(f"Seed {SEED} Complete: 3c OOF LL = {log_loss(y, oof_3c):.5f} | Hier OOF LL = {log_loss(y, oof_h):.5f}", flush=True)

np.save(os.path.join(cache_dir, 'clean_oof_xgb_3c.npy'), acc_xgb_3c)
np.save(os.path.join(cache_dir, 'clean_test_xgb_3c.npy'), tacc_xgb_3c)
np.save(os.path.join(cache_dir, 'fold_losses_xgb_3c.npy'), losses_3c)

np.save(os.path.join(cache_dir, 'clean_oof_xgb_h.npy'), acc_xgb_h)
np.save(os.path.join(cache_dir, 'clean_test_xgb_h.npy'), tacc_xgb_h)
np.save(os.path.join(cache_dir, 'fold_losses_xgb_h.npy'), losses_h)

print(f"\nXGBoost Done in {time.time()-t0:.1f}s.")
print(f"XGB 3-Class 3-Seed OOF:  {log_loss(y, acc_xgb_3c):.5f} ± {np.std(losses_3c)/np.sqrt(15):.5f}")
print(f"XGB Hier    3-Seed OOF:  {log_loss(y, acc_xgb_h):.5f} ± {np.std(losses_h)/np.sqrt(15):.5f}")
