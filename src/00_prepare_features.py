"""
Feature Preparation Script for Cosmic Sort 2026
Extracts 77 domain-validated astrophysical features and saves arrays persistently.
Strictly enforces Rule 7.2: redshift is completely excluded.
"""
import time, os, warnings
import numpy as np
import pandas as pd
warnings.filterwarnings('ignore')

out_dir = '/Users/tejasmittal32/College/Cosmic Sort/cached_predictions'
os.makedirs(out_dir, exist_ok=True)

train_path = '/Users/tejasmittal32/College/Cosmic Sort/train.csv'
test_path = '/Users/tejasmittal32/College/Cosmic Sort/test.csv'

print(f"Loading data from {train_path} and {test_path}...")
train = pd.read_csv(train_path)
test = pd.read_csv(test_path)

# Verify Rule 7.2: redshift is strictly excluded
assert 'redshift' in train.columns
feature_cols = [c for c in train.columns if c not in ['object_id', 'class', 'redshift']]
assert 'redshift' not in feature_cols, "CRITICAL ERROR: 'redshift' in feature_cols!"
print(f"Base feature columns ({len(feature_cols)}): {feature_cols}")

# Target mapping: 0=GALAXY, 1=QSO, 2=STAR
target_map = {'GALAXY': 0, 'QSO': 1, 'STAR': 2}
y = train['class'].map(target_map).values
np.save(os.path.join(out_dir, 'y.npy'), y)

def engineer_features(df):
    d = df[feature_cols].copy()
    
    # 1. Adjacent colors
    for m1, m2 in [('u','g'),('g','r'),('r','i'),('i','z')]:
        d[f'{m1}_minus_{m2}'] = df[m1] - df[m2]
        
    # 2. Long-range colors
    for m1, m2 in [('u','r'),('u','z'),('g','z'),('g','i'),('r','z'),('u','i')]:
        d[f'{m1}_minus_{m2}'] = df[m1] - df[m2]
        
    # 3. Johnson-Morgan Q-color index
    d['Q_ug_gr'] = (df['u'] - df['g']) - 0.72 * (df['g'] - df['r'])
    d['Q_gr_ri'] = (df['g'] - df['r']) - 0.85 * (df['r'] - df['i'])
    
    # 4. Spectral curvature / finite differences
    d['curv_ugr'] = df['u'] - 2 * df['g'] + df['r']
    d['curv_gri'] = df['g'] - 2 * df['r'] + df['i']
    d['curv_riz'] = df['r'] - 2 * df['i'] + df['z']
    
    # 5. Non-linear color interactions
    d['ug_x_gr'] = (df['u']-df['g']) * (df['g']-df['r'])
    d['gr_x_ri'] = (df['g']-df['r']) * (df['r']-df['i'])
    d['ri_x_iz'] = (df['r']-df['i']) * (df['i']-df['z'])
    
    # 6. Error aggregations
    errs = [df[f'err_{b}'] for b in 'ugriz']
    d['total_err'] = np.sqrt(sum(e**2 for e in errs))
    d['mean_err'] = np.mean(errs, axis=0)
    d['max_err'] = np.max(errs, axis=0)
    d['min_err'] = np.min(errs, axis=0)
    d['err_range'] = d['max_err'] - d['min_err']
    
    # 7. Propagated color uncertainties
    for m1, m2 in [('u','g'),('g','r'),('r','i'),('i','z')]:
        d[f'err_{m1}{m2}'] = np.sqrt(df[f'err_{m1}']**2 + df[f'err_{m2}']**2)
        
    # 8. Photometric signal-to-noise ratio proxies
    for b in 'ugriz':
        d[f'snr_{b}'] = df[b] / (df[f'err_{b}'] + 1e-10)
        
    # 9. Log transformations of skewed error variables
    for col in ['err_u','err_g','err_r','err_i','err_z','size_r']:
        d[f'log_{col}'] = np.log1p(df[col])
        
    # 10. Magnitude summary statistics
    mags = df[list('ugriz')]
    d['mean_mag'] = mags.mean(axis=1)
    d['std_mag'] = mags.std(axis=1)
    d['range_mag'] = mags.max(axis=1) - mags.min(axis=1)
    d['median_mag'] = mags.median(axis=1)
    
    # 11. Spatial & morphology features
    d['abs_gal_b'] = np.abs(df['gal_b'])
    d['size_r_over_r'] = df['size_r'] / (df['r'] + 1e-10)
    d['is_point_source'] = (df['size_r'] < 1.8).astype(float)
    
    # 12. Flux ratios in linear energy scale
    d['flux_ratio_ug'] = np.power(10.0, -0.4 * (df['u'] - df['g']))
    d['flux_ratio_gr'] = np.power(10.0, -0.4 * (df['g'] - df['r']))
    d['flux_ratio_ri'] = np.power(10.0, -0.4 * (df['r'] - df['i']))
    d['flux_ratio_iz'] = np.power(10.0, -0.4 * (df['i'] - df['z']))
    
    # 13. Linear Physical Fluxes and Fractional SED Shapes
    flux_cols = []
    for b in 'ugriz':
        f_name = f'flux_{b}'
        d[f_name] = np.power(10.0, -0.4 * df[b])
        d[f'err_flux_{b}'] = 0.4 * np.log(10) * d[f_name] * df[f'err_{b}']
        flux_cols.append(f_name)
        
    tot_flux = d[flux_cols].sum(axis=1)
    for b in 'ugriz':
        d[f'frac_flux_{b}'] = d[f'flux_{b}'] / (tot_flux + 1e-15)
        
    return d

print("Engineering 77 features for train and test...")
t0 = time.time()
X_tr = engineer_features(train)
X_te = engineer_features(test)

assert 'redshift' not in X_tr.columns, "Leakage check failed in X_tr!"
assert 'redshift' not in X_te.columns, "Leakage check failed in X_te!"
assert X_tr.shape[1] == 77, f"Expected 77 features, got {X_tr.shape[1]}"
assert X_te.shape[1] == 77, f"Expected 77 features, got {X_te.shape[1]}"
print(f"Features created successfully in {time.time()-t0:.2f}s: {X_tr.shape[1]} features.")

X_tr_np = X_tr.values.astype(np.float32)
X_te_np = X_te.values.astype(np.float32)

np.save(os.path.join(out_dir, 'X_train_77.npy'), X_tr_np)
np.save(os.path.join(out_dir, 'X_test_77.npy'), X_te_np)
np.save(os.path.join(out_dir, 'size_r_tr.npy'), train['size_r'].values.astype(np.float32))
np.save(os.path.join(out_dir, 'size_r_te.npy'), test['size_r'].values.astype(np.float32))
np.save(os.path.join(out_dir, 'test_object_id.npy'), test['object_id'].values)

print(f"Saved persistent arrays to {out_dir}. All format assertions PASSED.")
