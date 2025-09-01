import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.cluster import FeatureAgglomeration
from xgboost import XGBRegressor
import scipy.stats as stats
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Load dataset
data = pd.read_csv("Normalized_dataset.csv")

# Separate features and target
X = data.drop('CO2 Solubility (mol/kg)', axis=1)
y = data['CO2 Solubility (mol/kg)']

# Define function for calculating cross-validation scores
def get_cv_score(model, X, y, cv=5):
    scores = cross_val_score(model, X, y, cv=cv, scoring='r2')
    return scores.mean(), scores.std()

# Define function to get feature importances from Random Forest
def get_rf_importances(X, y):
    rf = RandomForestRegressor(random_state=42)
    rf.fit(X, y)
    importances = pd.Series(rf.feature_importances_, index=X.columns)
    return importances.sort_values(ascending=False)

# Define function to get feature importances from XGBoost
def get_xgb_importances(X, y):
    xgb = XGBRegressor(random_state=42)
    xgb.fit(X, y)
    importances = pd.Series(xgb.feature_importances_, index=X.columns)
    return importances.sort_values(ascending=False)

# Define function for Feature Agglomeration
def feature_agglomeration_importances(X, y, ratio=0.9):
    fa = FeatureAgglomeration(n_clusters=3)
    fa.fit(X)
    
    # Get the cluster assignments for each feature
    clusters = fa.labels_
    
    # For each cluster, calculate the importance
    cluster_importances = {}
    for cluster in np.unique(clusters):
        cluster_features = X.columns[clusters == cluster]
        
        # Get correlation with target
        corr_scores = {}
        for feature in cluster_features:
            corr_scores[feature] = abs(stats.spearmanr(X[feature], y)[0])
        
        # Get variance
        var_scores = {}
        for feature in cluster_features:
            var_scores[feature] = np.var(X[feature])
        
        # Normalize scores
        if len(corr_scores) > 0:
            max_corr = max(corr_scores.values()) if max(corr_scores.values()) > 0 else 1
            max_var = max(var_scores.values()) if max(var_scores.values()) > 0 else 1
            
            for feature in cluster_features:
                corr_scores[feature] /= max_corr
                var_scores[feature] /= max_var
        
        # Combine scores with the given ratio
        combined_scores = {}
        for feature in cluster_features:
            combined_scores[feature] = ratio * var_scores[feature] + (1-ratio) * corr_scores[feature]
        
        # Store the highest scoring feature for this cluster
        if combined_scores:
            cluster_importances.update(combined_scores)
    
    return pd.Series(cluster_importances).sort_values(ascending=False)

# Define function for Highly Variable Gene Selection (adapted for features)
def hvgs_importances(X, y):
    # Calculate variance for each feature
    variances = X.var().sort_values(ascending=False)
    return variances

# Define function for Spearman's correlation
def spearman_importances(X, y):
    corr_values = {}
    p_values = {}
    
    for col in X.columns:
        corr, p = stats.spearmanr(X[col], y)
        corr_values[col] = abs(corr)
        p_values[col] = p
    
    corr_series = pd.Series(corr_values)
    p_series = pd.Series(p_values)
    
    return corr_series.sort_values(ascending=False), p_series

# 1. Original Feature Set Analysis
print("===== Original Feature Set Analysis =====")

# Random Forest
rf_importances = get_rf_importances(X, y)
print("\nRandom Forest Feature Importances:")
print(rf_importances)
rf_cv_score, rf_cv_std = get_cv_score(RandomForestRegressor(random_state=42), X, y)
print(f"Random Forest CV R² Score: {rf_cv_score:.4f} ± {rf_cv_std:.4f}")

# XGBoost
xgb_importances = get_xgb_importances(X, y)
print("\nXGBoost Feature Importances:")
print(xgb_importances)
xgb_cv_score, xgb_cv_std = get_cv_score(XGBRegressor(random_state=42), X, y)
print(f"XGBoost CV R² Score: {xgb_cv_score:.4f} ± {xgb_cv_std:.4f}")

# Feature Agglomeration
fa_importances = feature_agglomeration_importances(X, y, ratio=0.9)
print("\nFeature Agglomeration Importances:")
print(fa_importances)
fa_cv_score, fa_cv_std = get_cv_score(RandomForestRegressor(random_state=42), X, y)
print(f"Random Forest with Feature Agglomeration CV R² Score: {fa_cv_score:.4f} ± {fa_cv_std:.4f}")

# Highly Variable Gene Selection
hvgs_importances_result = hvgs_importances(X, y)
print("\nHighly Variable Gene Selection Importances:")
print(hvgs_importances_result)
hvgs_cv_score, hvgs_cv_std = get_cv_score(RandomForestRegressor(random_state=42), X, y)
print(f"Random Forest with HVGS CV R² Score: {hvgs_cv_score:.4f} ± {hvgs_cv_std:.4f}")

# Spearman's Correlation
spearman_corr, spearman_p = spearman_importances(X, y)
print("\nSpearman's Correlation Importances:")
print(spearman_corr)
print("\nP-values:")
print(spearman_p)
spearman_cv_score, spearman_cv_std = get_cv_score(RandomForestRegressor(random_state=42), X, y)
print(f"Random Forest with Spearman CV R² Score: {spearman_cv_score:.4f} ± {spearman_cv_std:.4f}")

# 2. Reduced Feature Set Analysis
print("\n===== Reduced Feature Set Analysis =====")

# Create reduced datasets by removing second most important feature from each method
rf_reduced = X.drop(rf_importances.index[1], axis=1)
xgb_reduced = X.drop(xgb_importances.index[1], axis=1)
fa_reduced = X.drop(fa_importances.index[1], axis=1)
hvgs_reduced = X.drop(hvgs_importances_result.index[1], axis=1)
spearman_reduced = X.drop(spearman_corr.index[1], axis=1)

# Random Forest with reduced features
rf_reduced_importances = get_rf_importances(rf_reduced, y)
print("\nRandom Forest Reduced Feature Importances:")
print(rf_reduced_importances)
rf_reduced_cv_score, rf_reduced_cv_std = get_cv_score(RandomForestRegressor(random_state=42), rf_reduced, y)
print(f"Random Forest Reduced CV R² Score: {rf_reduced_cv_score:.4f} ± {rf_reduced_cv_std:.4f}")

# XGBoost with reduced features
xgb_reduced_importances = get_xgb_importances(xgb_reduced, y)
print("\nXGBoost Reduced Feature Importances:")
print(xgb_reduced_importances)
xgb_reduced_cv_score, xgb_reduced_cv_std = get_cv_score(XGBRegressor(random_state=42), xgb_reduced, y)
print(f"XGBoost Reduced CV R² Score: {xgb_reduced_cv_score:.4f} ± {xgb_reduced_cv_std:.4f}")

# Feature Agglomeration with reduced features
fa_reduced_importances = feature_agglomeration_importances(fa_reduced, y, ratio=0.9)
print("\nFeature Agglomeration Reduced Importances:")
print(fa_reduced_importances)
fa_reduced_cv_score, fa_reduced_cv_std = get_cv_score(RandomForestRegressor(random_state=42), fa_reduced, y)
print(f"Random Forest with Feature Agglomeration Reduced CV R² Score: {fa_reduced_cv_score:.4f} ± {fa_reduced_cv_std:.4f}")

# Highly Variable Gene Selection with reduced features
hvgs_reduced_importances = hvgs_importances(hvgs_reduced, y)
print("\nHighly Variable Gene Selection Reduced Importances:")
print(hvgs_reduced_importances)
hvgs_reduced_cv_score, hvgs_reduced_cv_std = get_cv_score(RandomForestRegressor(random_state=42), hvgs_reduced, y)
print(f"Random Forest with HVGS Reduced CV R² Score: {hvgs_reduced_cv_score:.4f} ± {hvgs_reduced_cv_std:.4f}")

# Spearman's Correlation with reduced features
spearman_reduced_corr, spearman_reduced_p = spearman_importances(spearman_reduced, y)
print("\nSpearman's Correlation Reduced Importances:")
print(spearman_reduced_corr)
print("\nReduced P-values:")
print(spearman_reduced_p)
spearman_reduced_cv_score, spearman_reduced_cv_std = get_cv_score(RandomForestRegressor(random_state=42), spearman_reduced, y)
print(f"Random Forest with Spearman Reduced CV R² Score: {spearman_reduced_cv_score:.4f} ± {spearman_reduced_cv_std:.4f}")

# Summary of CV scores
print("\n===== Summary of Cross-Validation Scores =====")
print(f"Original Random Forest: {rf_cv_score:.4f} ± {rf_cv_std:.4f}")
print(f"Reduced Random Forest: {rf_reduced_cv_score:.4f} ± {rf_reduced_cv_std:.4f}")

print(f"Original XGBoost: {xgb_cv_score:.4f} ± {xgb_cv_std:.4f}")
print(f"Reduced XGBoost: {xgb_reduced_cv_score:.4f} ± {xgb_reduced_cv_std:.4f}")

print(f"Original Feature Agglomeration: {fa_cv_score:.4f} ± {fa_cv_std:.4f}")
print(f"Reduced Feature Agglomeration: {fa_reduced_cv_score:.4f} ± {fa_reduced_cv_std:.4f}")

print(f"Original HVGS: {hvgs_cv_score:.4f} ± {hvgs_cv_std:.4f}")
print(f"Reduced HVGS: {hvgs_reduced_cv_score:.4f} ± {hvgs_reduced_cv_std:.4f}")

print(f"Original Spearman: {spearman_cv_score:.4f} ± {spearman_cv_std:.4f}")
print(f"Reduced Spearman: {spearman_reduced_cv_score:.4f} ± {spearman_reduced_cv_std:.4f}")
