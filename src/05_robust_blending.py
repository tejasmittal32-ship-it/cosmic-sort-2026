"""
Step 1, Step 2, and Step 3:
1. Overfitting Audit of Post-Processing (Nested CV vs In-Sample OOF)
2. Robust Blending with Equalised Seeds, Weight Capping (0.40, 0.50, 0.60), and Member Pruning Checks
3. Tail-Risk Analysis (p < 0.001 confident errors) and Calibration (Floors, Scalar vs Per-Class T, ECE)
"""
import os, time, warnings
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.metrics import log_loss, accuracy_score
from sklearn.model_selection import StratifiedKFold
warnings.filterwarnings('ignore')

cache_dir = '/Users/tejasmittal32/College/Cosmic Sort/cached_predictions'
y = np.load(os.path.join(cache_dir, 'y.npy'))
n_train = len(y)
classes = ['GALAXY', 'QSO', 'STAR'] # 0, 1, 2

# True macro log loss
def macro_logloss(y_true, proba, eps=1e-15):
    p = np.clip(proba, eps, 1.0 - eps)
    p = p / p.sum(axis=1, keepdims=True)
    c_losses = []
    for c in range(3):
        mask = (y_true == c)
        c_losses.append(-np.mean(np.log(p[mask, c])))
    return np.mean(c_losses), c_losses

# Load model OOF and Test predictions
models = ['resmlp2', 'xgb_h', 'xgb_3c', 'lgb_h', 'lgb_3c', 'mlp_s2']
model_names = {
    'resmlp2': 'ResMLP-2',
    'xgb_h': 'XGB_Hier',
    'xgb_3c': 'XGB_3c',
    'lgb_h': 'LGB_Hier',
    'lgb_3c': 'LGB_3c',
    'mlp_s2': 'MLP_S2'
}

oof_dict = {m: np.load(os.path.join(cache_dir, f'clean_oof_{m}.npy')) for m in models}
test_dict = {m: np.load(os.path.join(cache_dir, f'clean_test_{m}.npy')) for m in models}

print("=" * 80)
print("BASE MODELS 3-SEED CLEAN OOF PERFORMANCE (STEP 0 RECOMPUTED):")
print("=" * 80)
for m in models:
    pred = oof_dict[m]
    micro = log_loss(y, pred)
    mac, per_c = macro_logloss(y, pred)
    acc = accuracy_score(y, np.argmax(pred, axis=1))
    print(f"{model_names[m]:15s} | Micro LL: {micro:.5f} | Correct Macro LL: {mac:.5f} | Acc: {acc*100:.2f}% | Per-class: G={per_c[0]:.4f}, Q={per_c[1]:.4f}, S={per_c[2]:.4f}")

# ==============================================================================
# STEP 1: OVERFITTING AUDIT OF POST-PROCESSING (NESTED CV vs IN-SAMPLE)
# ==============================================================================
print("\n" + "=" * 80)
print("STEP 1: OVERFITTING AUDIT OF POST-PROCESSING VIA 5-FOLD NESTED CV")
print("=" * 80)

# Unconstrained convex blend optimizer
def fit_blend_weights(oof_subset, y_subset, cap=None):
    K = len(oof_subset)
    def loss_func(w_raw):
        if cap is not None:
            # Parametrized sigmoid projection to strictly enforce 0 <= w_i <= cap
            w = 1.0 / (1.0 + np.exp(-w_raw))
            w = w * cap
            if w.sum() == 0: return 999.0
            w = w / w.sum()
            # Double check cap violation
            if (w > cap + 1e-5).any():
                return 999.0 + np.sum(np.maximum(0, w - cap))*100
        else:
            w = np.exp(w_raw) / np.sum(np.exp(w_raw))
        blend = sum(w[i] * oof_subset[i] for i in range(K))
        p = np.clip(blend, 1e-15, 1.0 - 1e-15)
        p = p / p.sum(axis=1, keepdims=True)
        return log_loss(y_subset, p)
    
    init_w = np.zeros(K)
    res = minimize(loss_func, init_w, method='Nelder-Mead', options={'maxiter': 2500, 'xatol': 1e-5, 'fatol': 1e-6})
    if cap is not None:
        w = 1.0 / (1.0 + np.exp(-res.x)) * cap
        w = w / w.sum()
    else:
        w = np.exp(res.x) / np.sum(np.exp(res.x))
    return w

# Calibration optimizer (Temperature T and floor eps)
def fit_calibration(oof_p, y_subset, per_class=False):
    if not per_class:
        def obj(params):
            T, floor_log10 = params
            eps = 10.0**floor_log10
            p_adj = np.power(np.clip(oof_p, 1e-15, 1.0), 1.0 / T)
            p_adj /= p_adj.sum(axis=1, keepdims=True)
            p_adj = np.clip(p_adj, eps, 1.0 - eps)
            p_adj /= p_adj.sum(axis=1, keepdims=True)
            return log_loss(y_subset, p_adj)
        res = minimize(obj, [1.01, -6.0], bounds=[(0.8, 1.5), (-10.0, -2.0)], method='L-BFGS-B')
        return res.x[0], 10.0**res.x[1]
    else:
        def obj(params):
            T = params[:3]
            floor_log10 = params[3]
            eps = 10.0**floor_log10
            p_adj = np.power(np.clip(oof_p, 1e-15, 1.0), 1.0 / T)
            p_adj /= p_adj.sum(axis=1, keepdims=True)
            p_adj = np.clip(p_adj, eps, 1.0 - eps)
            p_adj /= p_adj.sum(axis=1, keepdims=True)
            return log_loss(y_subset, p_adj)
        res = minimize(obj, [1.01, 1.01, 1.01, -6.0], bounds=[(0.8, 1.5)]*3 + [(-10.0, -2.0)], method='L-BFGS-B')
        return res.x[:3], 10.0**res.x[3]

def apply_calib(p, T, eps):
    p_adj = np.power(np.clip(p, 1e-15, 1.0), 1.0 / T)
    p_adj /= p_adj.sum(axis=1, keepdims=True)
    p_adj = np.clip(p_adj, eps, 1.0 - eps)
    p_adj /= p_adj.sum(axis=1, keepdims=True)
    return p_adj

# Run 5-fold nested CV on post-processing
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oof_matrix_list = [oof_dict[m] for m in models]

nested_oof_unconstrained = np.zeros_like(oof_dict['resmlp2'])
fold_weights_unconstrained = []
fold_T_unconstrained = []
fold_eps_unconstrained = []

for fi, (tr_idx, va_idx) in enumerate(skf.split(X_tr_sc := oof_dict['resmlp2'], y)):
    y_tr_sub, y_va_sub = y[tr_idx], y[va_idx]
    tr_oofs = [m_oof[tr_idx] for m_oof in oof_matrix_list]
    va_oofs = [m_oof[va_idx] for m_oof in oof_matrix_list]
    
    # Fit weights on 4 folds
    w_f = fit_blend_weights(tr_oofs, y_tr_sub, cap=None)
    fold_weights_unconstrained.append(w_f)
    
    # Blend train and val
    tr_blend = sum(w_f[i] * tr_oofs[i] for i in range(len(models)))
    tr_blend /= tr_blend.sum(axis=1, keepdims=True)
    
    va_blend = sum(w_f[i] * va_oofs[i] for i in range(len(models)))
    va_blend /= va_blend.sum(axis=1, keepdims=True)
    
    # Fit T and floor on 4 folds
    T_f, eps_f = fit_calibration(tr_blend, y_tr_sub, per_class=False)
    fold_T_unconstrained.append(T_f)
    fold_eps_unconstrained.append(eps_f)
    
    # Evaluate on held-out 5th fold
    nested_oof_unconstrained[va_idx] = apply_calib(va_blend, T_f, eps_f)

nested_ll_unconstrained = log_loss(y, nested_oof_unconstrained)
nested_mac_unconstrained, _ = macro_logloss(y, nested_oof_unconstrained)

# In-sample full-OOF fit
w_insample = fit_blend_weights(oof_matrix_list, y, cap=None)
insample_blend = sum(w_insample[i] * oof_matrix_list[i] for i in range(len(models)))
insample_blend /= insample_blend.sum(axis=1, keepdims=True)
T_insample, eps_insample = fit_calibration(insample_blend, y, per_class=False)
insample_calib = apply_calib(insample_blend, T_insample, eps_insample)
insample_ll = log_loss(y, insample_calib)

optimism_gap = nested_ll_unconstrained - insample_ll
print(f"In-Sample OOF Fit Micro LL:       {insample_ll:.5f}")
print(f"Nested-CV Out-of-Sample Micro LL:  {nested_ll_unconstrained:.5f}")
print(f"Quantified Post-Processing Bias:   +{optimism_gap:.5f} (optimistic leakage in in-sample fitting)")
print(f"Mean Nested T: {np.mean(fold_T_unconstrained):.4f} | In-Sample T: {T_insample:.4f}")
print(f"Mean Nested Floor eps: {np.mean(fold_eps_unconstrained):.2e} | In-Sample Floor eps: {eps_insample:.2e}")

# ==============================================================================
# STEP 2: ROBUST BLENDING WITH WEIGHT CAPPING & STABILITY AUDIT
# ==============================================================================
print("\n" + "=" * 80)
print("STEP 2: ROBUST BLENDING & WEIGHT CAPPING SWEEP (0.40, 0.50, 0.60)")
print("=" * 80)

caps = [0.40, 0.50, 0.60, None]
cap_results = {}

for cap in caps:
    cap_str = f"Cap {int(cap*100)}%" if cap else "Unconstrained"
    nested_oof_cap = np.zeros_like(oof_dict['resmlp2'])
    weights_per_fold = []
    
    for fi, (tr_idx, va_idx) in enumerate(skf.split(oof_matrix_list[0], y)):
        y_tr_sub, y_va_sub = y[tr_idx], y[va_idx]
        tr_oofs = [m_oof[tr_idx] for m_oof in oof_matrix_list]
        va_oofs = [m_oof[va_idx] for m_oof in oof_matrix_list]
        
        w_f = fit_blend_weights(tr_oofs, y_tr_sub, cap=cap)
        weights_per_fold.append(w_f)
        
        tr_blend = sum(w_f[i] * tr_oofs[i] for i in range(len(models)))
        tr_blend /= tr_blend.sum(axis=1, keepdims=True)
        va_blend = sum(w_f[i] * va_oofs[i] for i in range(len(models)))
        va_blend /= va_blend.sum(axis=1, keepdims=True)
        
        T_f, eps_f = fit_calibration(tr_blend, y_tr_sub, per_class=False)
        nested_oof_cap[va_idx] = apply_calib(va_blend, T_f, eps_f)
    
    n_ll = log_loss(y, nested_oof_cap)
    n_mac, _ = macro_logloss(y, nested_oof_cap)
    n_acc = accuracy_score(y, np.argmax(nested_oof_cap, axis=1))
    mean_w = np.mean(weights_per_fold, axis=0)
    std_w = np.std(weights_per_fold, axis=0)
    
    cap_results[cap_str] = {
        'micro': n_ll, 'macro': n_mac, 'acc': n_acc,
        'mean_w': mean_w, 'std_w': std_w, 'oof': nested_oof_cap
    }
    print(f"\n--- {cap_str} ---")
    print(f"  Nested Micro LL: {n_ll:.5f} | Macro LL: {n_mac:.5f} | Acc: {n_acc*100:.2f}%")
    print("  Mean Weights across folds (mean ± std):")
    for i, m in enumerate(models):
        print(f"    {model_names[m]:15s}: {mean_w[i]*100:5.2f}% ± {std_w[i]*100:4.2f}%")

# Ablation: Test pruning LGB Hierarchical or Stage-2 MLP
print("\n" + "=" * 80)
print("DIVERSITY ABLATION: COST OF PRUNING LGB HIERARCHICAL & STAGE-2 MLP")
print("=" * 80)

# Full 6 models vs Drop LGB_Hier vs Drop MLP_S2 vs Drop Both
subsets = {
    'Full 6 Models': models,
    'Drop LGB_Hier': [m for m in models if m != 'lgb_h'],
    'Drop MLP_S2':   [m for m in models if m != 'mlp_s2'],
    'Drop Both (4 Models)': ['resmlp2', 'xgb_h', 'xgb_3c', 'lgb_3c']
}

for sname, smodels in subsets.items():
    s_indices = [models.index(m) for m in smodels]
    nested_oof_sub = np.zeros_like(oof_dict['resmlp2'])
    for fi, (tr_idx, va_idx) in enumerate(skf.split(oof_matrix_list[0], y)):
        y_tr_sub, y_va_sub = y[tr_idx], y[va_idx]
        tr_oofs = [oof_matrix_list[idx][tr_idx] for idx in s_indices]
        va_oofs = [oof_matrix_list[idx][va_idx] for idx in s_indices]
        
        w_f = fit_blend_weights(tr_oofs, y_tr_sub, cap=0.50)
        tr_blend = sum(w_f[i] * tr_oofs[i] for i in range(len(smodels)))
        tr_blend /= tr_blend.sum(axis=1, keepdims=True)
        va_blend = sum(w_f[i] * va_oofs[i] for i in range(len(smodels)))
        va_blend /= va_blend.sum(axis=1, keepdims=True)
        
        T_f, eps_f = fit_calibration(tr_blend, y_tr_sub, per_class=False)
        nested_oof_sub[va_idx] = apply_calib(va_blend, T_f, eps_f)
    
    s_ll = log_loss(y, nested_oof_sub)
    delta_vs_full = s_ll - cap_results['Cap 50%']['micro']
    print(f"  {sname:22s}: Nested Micro LL = {s_ll:.5f} (Delta vs Full: {delta_vs_full:+.5f})")

# ==============================================================================
# STEP 3: TAIL-RISK & CALIBRATION AUDIT
# ==============================================================================
print("\n" + "=" * 80)
print("STEP 3: TAIL-RISK AUDIT (TRUE-CLASS p < 0.001 CONFIDENT ERRORS)")
print("=" * 80)

def compute_tail_risk(probs, y_true):
    n = len(y_true)
    true_p = np.array([probs[i, y_true[i]] for i in range(n)])
    sample_losses = -np.log(np.clip(true_p, 1e-15, 1.0))
    total_loss = np.sum(sample_losses)
    
    tail_mask = (true_p < 0.001)
    tail_count = np.sum(tail_mask)
    tail_loss = np.sum(sample_losses[tail_mask])
    tail_loss_share = (tail_loss / total_loss) * 100
    
    return tail_count, tail_loss_share, np.max(sample_losses)

for m in models:
    tc, ts, max_l = compute_tail_risk(oof_dict[m], y)
    print(f"  {model_names[m]:15s}: Confident Errors (p<0.001): {tc:4d} ({tc/n_train*100:.3f}%) | Share of Total Loss: {ts:5.2f}% | Max Loss: {max_l:.2f}")

tc_unc, ts_unc, max_l_unc = compute_tail_risk(nested_oof_unconstrained, y)
print(f"  {'Unconstrained Blend':15s}: Confident Errors (p<0.001): {tc_unc:4d} ({tc_unc/n_train*100:.3f}%) | Share of Total Loss: {ts_unc:5.2f}% | Max Loss: {max_l_unc:.2f}")

tc_c50, ts_c50, max_l_c50 = compute_tail_risk(cap_results['Cap 50%']['oof'], y)
print(f"  {'Cap 50% Blend':15s}: Confident Errors (p<0.001): {tc_c50:4d} ({tc_c50/n_train*100:.3f}%) | Share of Total Loss: {ts_c50:5.2f}% | Max Loss: {max_l_c50:.2f}")

# Floor Grid Search in {1e-6, 1e-5, 1e-4, 1e-3}
print("\n" + "=" * 80)
print("PROBABILITY FLOOR GRID SEARCH (NESTED CV):")
print("=" * 80)
floors = [1e-6, 1e-5, 1e-4, 1e-3]
for fl in floors:
    nested_oof_fl = np.zeros_like(oof_dict['resmlp2'])
    for fi, (tr_idx, va_idx) in enumerate(skf.split(oof_matrix_list[0], y)):
        y_tr_sub, y_va_sub = y[tr_idx], y[va_idx]
        tr_oofs = [m_oof[tr_idx] for m_oof in oof_matrix_list]
        va_oofs = [m_oof[va_idx] for m_oof in oof_matrix_list]
        w_f = fit_blend_weights(tr_oofs, y_tr_sub, cap=0.50)
        tr_b = sum(w_f[i] * tr_oofs[i] for i in range(len(models)))
        tr_b /= tr_b.sum(axis=1, keepdims=True)
        va_b = sum(w_f[i] * va_oofs[i] for i in range(len(models)))
        va_b /= va_b.sum(axis=1, keepdims=True)
        # Fit T with fixed floor
        def obj_T(T):
            p = apply_calib(tr_b, T[0], fl)
            return log_loss(y_tr_sub, p)
        res = minimize(obj_T, [1.01], bounds=[(0.8, 1.5)], method='L-BFGS-B')
        nested_oof_fl[va_idx] = apply_calib(va_b, res.x[0], fl)
    
    fl_ll = log_loss(y, nested_oof_fl)
    fl_tc, fl_ts, fl_max_l = compute_tail_risk(nested_oof_fl, y)
    print(f"  Floor = {fl:1.0e}: Nested Micro LL = {fl_ll:.5f} | Confident Errors: {fl_tc:4d} | Tail Loss Share: {fl_ts:5.2f}% | Max Loss: {fl_max_l:.2f}")

# Compare Single T vs Per-Class T
print("\n" + "=" * 80)
print("SINGLE TEMPERATURE vs PER-CLASS VECTOR TEMPERATURE (NESTED CV):")
print("=" * 80)
nested_oof_vecT = np.zeros_like(oof_dict['resmlp2'])
for fi, (tr_idx, va_idx) in enumerate(skf.split(oof_matrix_list[0], y)):
    y_tr_sub, y_va_sub = y[tr_idx], y[va_idx]
    tr_oofs = [m_oof[tr_idx] for m_oof in oof_matrix_list]
    va_oofs = [m_oof[va_idx] for m_oof in oof_matrix_list]
    w_f = fit_blend_weights(tr_oofs, y_tr_sub, cap=0.50)
    tr_b = sum(w_f[i] * tr_oofs[i] for i in range(len(models)))
    tr_b /= tr_b.sum(axis=1, keepdims=True)
    va_b = sum(w_f[i] * va_oofs[i] for i in range(len(models)))
    va_b /= va_b.sum(axis=1, keepdims=True)
    
    T_vec_f, eps_vec_f = fit_calibration(tr_b, y_tr_sub, per_class=True)
    nested_oof_vecT[va_idx] = apply_calib(va_b, T_vec_f, eps_vec_f)

vecT_ll = log_loss(y, nested_oof_vecT)
print(f"  Single Scalar T:   Nested Micro LL = {cap_results['Cap 50%']['micro']:.5f}")
print(f"  Per-Class Vector T: Nested Micro LL = {vecT_ll:.5f} (Delta: {vecT_ll - cap_results['Cap 50%']['micro']:+.5f})")

# ECE and Calibration curve
def compute_multiclass_ece(probs, labels, n_bins=10):
    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    accuracies = (predictions == labels)
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        in_bin = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i+1])
        prop_in_bin = in_bin.mean()
        if prop_in_bin > 0:
            ece += np.abs(confidences[in_bin].mean() - accuracies[in_bin].mean()) * prop_in_bin
    # Per-class ECE
    per_class_ece = []
    for c in range(3):
        y_c = (labels == c).astype(int)
        p_c = probs[:, c]
        ece_c = 0.0
        for b in range(n_bins):
            mask = (p_c >= bin_boundaries[b]) & (p_c < bin_boundaries[b+1])
            if mask.sum() > 0:
                ece_c += (mask.sum()/len(labels)) * abs(p_c[mask].mean() - y_c[mask].mean())
        per_class_ece.append(ece_c)
    return ece, per_class_ece

ece_total, ece_per_c = compute_multiclass_ece(cap_results['Cap 50%']['oof'], y)
print(f"\nExpected Calibration Error (ECE): {ece_total*100:.3f}% (GALAXY: {ece_per_c[0]*100:.3f}%, QSO: {ece_per_c[1]*100:.3f}%, STAR: {ece_per_c[2]*100:.3f}%)")

np.save(os.path.join(cache_dir, 'robust_blend_oof.npy'), cap_results['Cap 50%']['oof'])
print("Audit successfully completed and cached.")
