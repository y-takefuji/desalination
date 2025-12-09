import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.cluster import FeatureAgglomeration
import xgboost as xgb
import shap

# Load the dataset
data = pd.read_csv('Normalized_dataset.csv')
X = data.drop('CO2 Solubility (mol/kg)', axis=1)
y = data['CO2 Solubility (mol/kg)']

# Define mapping for abbreviated feature names
feature_abbreviations = {
    'Pressure (Mpa)': 'P',
    'Temperature(Kelvin)': 'T',
    'NaCl Concentration (ppm)': 'NaCl',
    'KCl Concentration (ppm)': 'KCl',
    'CaCl2 Concentration (ppm)': 'CaCl2',
    'MgCl2 Concentration (ppm)': 'MgCl2'
}

# Function to abbreviate feature names
def abbreviate_features(feature_list):
    return [feature_abbreviations.get(f, f) for f in feature_list]

# Function to get cross-validation score with specified number of folds
def get_cv_score(X_subset, y, model_type='RF', n_folds=5):
    if model_type == 'RF':
        model = RandomForestRegressor(random_state=42)
    elif model_type == 'XGB':
        model = xgb.XGBRegressor(random_state=42)
    
    cv_scores = cross_val_score(model, X_subset, y, cv=n_folds, scoring='r2')
    cv_mean = np.mean(cv_scores)
    
    return round(cv_mean, 4)

# Results dictionary
results = {}

# 1. Random Forest Feature Importance
rf_model = RandomForestRegressor(random_state=42)
rf_model.fit(X, y)
rf_importances = pd.Series(rf_model.feature_importances_, index=X.columns).sort_values(ascending=False)

# RF top 5 features (set1)
rf_top5 = rf_importances.index[:5].tolist()
rf_set1 = X[rf_top5]
rf_set1_cv5 = get_cv_score(rf_set1, y, 'RF', 5)

# RF reduced dataset (removing top feature) and top 4 (set2)
top_feature = rf_importances.index[0]
rf_reduced = X.drop(top_feature, axis=1)
rf_model_reduced = RandomForestRegressor(random_state=42)
rf_model_reduced.fit(rf_reduced, y)
rf_importances_reduced = pd.Series(rf_model_reduced.feature_importances_, index=rf_reduced.columns).sort_values(ascending=False)
rf_top4_reduced = rf_importances_reduced.index[:4].tolist()
rf_set2 = rf_reduced[rf_top4_reduced]
rf_set2_cv4 = get_cv_score(rf_set2, y, 'RF', 4)

results['RF'] = {
    'CV5': rf_set1_cv5,
    'CV4': rf_set2_cv4,
    'Set1': rf_top5,
    'Set2': rf_top4_reduced
}

# 2. XGBoost Feature Importance
xgb_model = xgb.XGBRegressor(random_state=42)
xgb_model.fit(X, y)
xgb_importances = pd.Series(xgb_model.feature_importances_, index=X.columns).sort_values(ascending=False)

# XGB top 5 features (set1)
xgb_top5 = xgb_importances.index[:5].tolist()
xgb_set1 = X[xgb_top5]
xgb_set1_cv5 = get_cv_score(xgb_set1, y, 'XGB', 5)

# XGB reduced dataset (removing top feature) and top 4 (set2)
xgb_top_feature = xgb_importances.index[0]
xgb_reduced = X.drop(xgb_top_feature, axis=1)
xgb_model_reduced = xgb.XGBRegressor(random_state=42)
xgb_model_reduced.fit(xgb_reduced, y)
xgb_importances_reduced = pd.Series(xgb_model_reduced.feature_importances_, index=xgb_reduced.columns).sort_values(ascending=False)
xgb_top4_reduced = xgb_importances_reduced.index[:4].tolist()
xgb_set2 = xgb_reduced[xgb_top4_reduced]
xgb_set2_cv4 = get_cv_score(xgb_set2, y, 'XGB', 4)

results['XGB'] = {
    'CV5': xgb_set1_cv5,
    'CV4': xgb_set2_cv4,
    'Set1': xgb_top5,
    'Set2': xgb_top4_reduced
}

# 3. Feature Agglomeration
# Using variance to cluster-central distance ratio of 0.9 to 0.1
n_clusters = min(5, X.shape[1])  # At most 5 clusters
fa = FeatureAgglomeration(n_clusters=n_clusters)
fa.fit(X)

# Calculate variance for each feature
feature_variances = np.var(X.values, axis=0)

# Calculate feature scores
feature_scores = []
for i, feature_name in enumerate(X.columns):
    # Use variance with 0.9 weight
    score = 0.9 * feature_variances[i] / np.max(feature_variances)  # Normalized variance
    feature_scores.append((feature_name, score))

# Sort features by score
fa_features_ranked = sorted(feature_scores, key=lambda x: x[1], reverse=True)
fa_top5 = [f[0] for f in fa_features_ranked[:5]]

# FA top 5 features (set1)
fa_set1 = X[fa_top5]
fa_set1_cv5 = get_cv_score(fa_set1, y, 'RF', 5)

# FA reduced dataset (removing top feature) and top 4 (set2)
fa_top_feature = fa_features_ranked[0][0]
fa_reduced = X.drop(fa_top_feature, axis=1)

# Recalculate FA on reduced dataset
fa_reduced_var = np.var(fa_reduced.values, axis=0)
fa_reduced_scores = []

for i, feature_name in enumerate(fa_reduced.columns):
    score = 0.9 * fa_reduced_var[i] / np.max(fa_reduced_var)  # Normalized variance
    fa_reduced_scores.append((feature_name, score))

fa_reduced_ranked = sorted(fa_reduced_scores, key=lambda x: x[1], reverse=True)
fa_top4_reduced = [f[0] for f in fa_reduced_ranked[:4]]
fa_set2 = fa_reduced[fa_top4_reduced]
fa_set2_cv4 = get_cv_score(fa_set2, y, 'RF', 4)

results['FA'] = {
    'CV5': fa_set1_cv5,
    'CV4': fa_set2_cv4,
    'Set1': fa_top5,
    'Set2': fa_top4_reduced
}

# 4. Highly Variable Gene Selection (HVGS)
# For regression task, we'll use variance as a proxy for feature importance
variances = X.var().sort_values(ascending=False)
hvgs_top5 = variances.index[:5].tolist()

# HVGS top 5 features (set1)
hvgs_set1 = X[hvgs_top5]
hvgs_set1_cv5 = get_cv_score(hvgs_set1, y, 'RF', 5)

# HVGS reduced dataset (removing top feature) and top 4 (set2)
hvgs_top_feature = variances.index[0]
hvgs_reduced = X.drop(hvgs_top_feature, axis=1)
hvgs_reduced_var = hvgs_reduced.var().sort_values(ascending=False)
hvgs_top4_reduced = hvgs_reduced_var.index[:4].tolist()
hvgs_set2 = hvgs_reduced[hvgs_top4_reduced]
hvgs_set2_cv4 = get_cv_score(hvgs_set2, y, 'RF', 4)

results['HVGS'] = {
    'CV5': hvgs_set1_cv5,
    'CV4': hvgs_set2_cv4,
    'Set1': hvgs_top5,
    'Set2': hvgs_top4_reduced
}

# 5. Spearman Correlation
spearman_corr = pd.DataFrame().assign(target=y).join(X).corr(method='spearman')['target'].drop('target')
spearman_abs_corr = spearman_corr.abs().sort_values(ascending=False)
spearman_top5 = spearman_abs_corr.index[:5].tolist()

# Spearman top 5 features (set1)
spearman_set1 = X[spearman_top5]
spearman_set1_cv5 = get_cv_score(spearman_set1, y, 'RF', 5)

# Spearman reduced dataset (removing top feature) and top 4 (set2)
spearman_top_feature = spearman_abs_corr.index[0]
spearman_reduced = X.drop(spearman_top_feature, axis=1)
spearman_reduced_corr = pd.DataFrame().assign(target=y).join(spearman_reduced).corr(method='spearman')['target'].drop('target')
spearman_reduced_abs_corr = spearman_reduced_corr.abs().sort_values(ascending=False)
spearman_top4_reduced = spearman_reduced_abs_corr.index[:4].tolist()
spearman_set2 = spearman_reduced[spearman_top4_reduced]
spearman_set2_cv4 = get_cv_score(spearman_set2, y, 'RF', 4)

results['Spearman'] = {
    'CV5': spearman_set1_cv5,
    'CV4': spearman_set2_cv4,
    'Set1': spearman_top5,
    'Set2': spearman_top4_reduced
}

# 6. RF SHAP
rf_explainer = shap.TreeExplainer(rf_model)
rf_shap_values = rf_explainer.shap_values(X)
rf_shap_importance = pd.Series(np.abs(rf_shap_values).mean(0), index=X.columns).sort_values(ascending=False)
rf_shap_top5 = rf_shap_importance.index[:5].tolist()

# RF SHAP top 5 features (set1)
rf_shap_set1 = X[rf_shap_top5]
rf_shap_set1_cv5 = get_cv_score(rf_shap_set1, y, 'RF', 5)

# RF SHAP reduced dataset (removing top feature) and top 4 (set2)
rf_shap_top_feature = rf_shap_importance.index[0]
rf_shap_reduced = X.drop(rf_shap_top_feature, axis=1)
rf_reduced_model = RandomForestRegressor(random_state=42)
rf_reduced_model.fit(rf_shap_reduced, y)
rf_reduced_explainer = shap.TreeExplainer(rf_reduced_model)
rf_reduced_shap_values = rf_reduced_explainer.shap_values(rf_shap_reduced)
rf_reduced_shap_importance = pd.Series(np.abs(rf_reduced_shap_values).mean(0), index=rf_shap_reduced.columns).sort_values(ascending=False)
rf_shap_top4_reduced = rf_reduced_shap_importance.index[:4].tolist()
rf_shap_set2 = rf_shap_reduced[rf_shap_top4_reduced]
rf_shap_set2_cv4 = get_cv_score(rf_shap_set2, y, 'RF', 4)

results['RF-SHAP'] = {
    'CV5': rf_shap_set1_cv5,
    'CV4': rf_shap_set2_cv4,
    'Set1': rf_shap_top5,
    'Set2': rf_shap_top4_reduced
}

# 7. XGB SHAP
xgb_explainer = shap.TreeExplainer(xgb_model)
xgb_shap_values = xgb_explainer.shap_values(X)
xgb_shap_importance = pd.Series(np.abs(xgb_shap_values).mean(0), index=X.columns).sort_values(ascending=False)
xgb_shap_top5 = xgb_shap_importance.index[:5].tolist()

# XGB SHAP top 5 features (set1)
xgb_shap_set1 = X[xgb_shap_top5]
xgb_shap_set1_cv5 = get_cv_score(xgb_shap_set1, y, 'XGB', 5)

# XGB SHAP reduced dataset (removing top feature) and top 4 (set2)
xgb_shap_top_feature = xgb_shap_importance.index[0]
xgb_shap_reduced = X.drop(xgb_shap_top_feature, axis=1)
xgb_reduced_model = xgb.XGBRegressor(random_state=42)
xgb_reduced_model.fit(xgb_shap_reduced, y)
xgb_reduced_explainer = shap.TreeExplainer(xgb_reduced_model)
xgb_reduced_shap_values = xgb_reduced_explainer.shap_values(xgb_shap_reduced)
xgb_reduced_shap_importance = pd.Series(np.abs(xgb_reduced_shap_values).mean(0), index=xgb_shap_reduced.columns).sort_values(ascending=False)
xgb_shap_top4_reduced = xgb_reduced_shap_importance.index[:4].tolist()
xgb_shap_set2 = xgb_shap_reduced[xgb_shap_top4_reduced]
xgb_shap_set2_cv4 = get_cv_score(xgb_shap_set2, y, 'XGB', 4)

results['XGB-SHAP'] = {
    'CV5': xgb_shap_set1_cv5,
    'CV4': xgb_shap_set2_cv4,
    'Set1': xgb_shap_top5,
    'Set2': xgb_shap_top4_reduced
}

# Create summary dataframe with abbreviated feature names
summary = []
for method, values in results.items():
    # Store original feature names for CSV
    original_set1 = ', '.join(values['Set1'])
    original_set2 = ', '.join(values['Set2'])
    
    # Create abbreviated feature names for display
    abbreviated_set1 = ', '.join(abbreviate_features(values['Set1']))
    abbreviated_set2 = ', '.join(abbreviate_features(values['Set2']))
    
    summary.append({
        'Method': method,
        'CV5': values['CV5'],
        'CV4': values['CV4'],
        'Top5_Features': abbreviated_set1,
        'Top4_Features_Reduced': abbreviated_set2,
        'Original_Top5_Features': original_set1,
        'Original_Top4_Features_Reduced': original_set2
    })

summary_df = pd.DataFrame(summary)

# Create a version with abbreviated feature names for display and CSV export
display_df = summary_df[['Method', 'CV5', 'CV4', 'Top5_Features', 'Top4_Features_Reduced']]
display_df.to_csv('result.csv', index=False)

# Display summary
print(display_df)

# Also save a version with original feature names for reference
summary_df[['Method', 'CV5', 'CV4', 'Original_Top5_Features', 'Original_Top4_Features_Reduced']].to_csv('result_full_names.csv', index=False)
