"""
Script 5: Détection de Data Drift
==================================
Ce script démontre comment:
- Détecter les changements dans la distribution des données (data drift)
- Comparer les statistiques entre train et production
- Logger les métriques de drift dans MLflow
- Alerter en cas de drift significatif

Note: Ce script utilise des méthodes statistiques simples. En production,
utilisez des librairies comme Evidently, Alibi Detect, ou Great Expectations.
"""

import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

# Configuration MLflow
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("05-Data-Drift-Detection")

def calculate_psi(expected, actual, buckets=10):
    """
    Population Stability Index (PSI)
    Mesure le drift entre deux distributions
    PSI < 0.1: Pas de changement significatif
    0.1 < PSI < 0.2: Changement modéré
    PSI > 0.2: Changement significatif
    """
    def scale_range(input_array, min_val, max_val):
        input_array_scaled = (input_array - min_val) / (max_val - min_val)
        return input_array_scaled
    
    # Normaliser les données entre 0 et 1
    min_val = min(expected.min(), actual.min())
    max_val = max(expected.max(), actual.max())
    
    expected_scaled = scale_range(expected, min_val, max_val)
    actual_scaled = scale_range(actual, min_val, max_val)
    
    # Créer les bins
    breakpoints = np.linspace(0, 1, buckets + 1)
    
    # Calculer les fréquences
    expected_freq = np.histogram(expected_scaled, breakpoints)[0] / len(expected_scaled)
    actual_freq = np.histogram(actual_scaled, breakpoints)[0] / len(actual_scaled)
    
    # Éviter les divisions par zéro
    expected_freq = np.where(expected_freq == 0, 0.0001, expected_freq)
    actual_freq = np.where(actual_freq == 0, 0.0001, actual_freq)
    
    # Calculer le PSI
    psi = np.sum((actual_freq - expected_freq) * np.log(actual_freq / expected_freq))
    
    return psi

def kolmogorov_smirnov_test(reference, current):
    """
    Test de Kolmogorov-Smirnov
    Test statistique pour comparer deux distributions
    p-value < 0.05 indique un drift significatif
    """
    statistic, p_value = stats.ks_2samp(reference, current)
    return statistic, p_value

def clean_feature_name(feature_name):
    """
    Nettoie le nom de la feature pour être compatible avec MLflow
    MLflow accepte: alphanumerics, underscores, dashes, periods, spaces, colons, slashes
    """
    # Remplacer les parenthèses et autres caractères spéciaux par des underscores
    cleaned = feature_name.replace('(', '_').replace(')', '_').replace(',', '')
    # Supprimer les underscores multiples
    while '__' in cleaned:
        cleaned = cleaned.replace('__', '_')
    # Supprimer les underscores en début et fin
    cleaned = cleaned.strip('_')
    return cleaned

# ====================
# SCÉNARIO 1: Données normales (pas de drift)
# ====================
print("=" * 70)
print("📊 SCÉNARIO 1: Données de production normales (PAS DE DRIFT)")
print("=" * 70)

data = load_iris()
X_train, X_test, y_train, y_test = train_test_split(
    data.data, data.target, test_size=0.2, random_state=42
)

# Simuler des données de production similaires au training
X_production_normal = X_test  # Utiliser le test set comme "production"

with mlflow.start_run(run_name="drift_check_normal"):
    mlflow.set_tag("drift_check_type", "normal")
    
    feature_names = data.feature_names
    drift_detected = False
    
    for i, feature_name in enumerate(feature_names):
        reference_feature = X_train[:, i]
        current_feature = X_production_normal[:, i]
        
        # Nettoyer le nom de la feature pour MLflow
        clean_name = clean_feature_name(feature_name)
        
        # Calculer les métriques de drift
        psi = calculate_psi(reference_feature, current_feature)
        ks_stat, ks_p_value = kolmogorov_smirnov_test(reference_feature, current_feature)
        
        # Statistiques descriptives
        mean_diff = abs(np.mean(reference_feature) - np.mean(current_feature))
        std_diff = abs(np.std(reference_feature) - np.std(current_feature))
        
        # Logger dans MLflow (avec nom nettoyé)
        mlflow.log_metric(f"psi_{clean_name}", psi)
        mlflow.log_metric(f"ks_stat_{clean_name}", ks_stat)
        mlflow.log_metric(f"ks_pvalue_{clean_name}", ks_p_value)
        mlflow.log_metric(f"mean_diff_{clean_name}", mean_diff)
        mlflow.log_metric(f"std_diff_{clean_name}", std_diff)
        
        # Détection du drift
        drift_status = "✅ OK"
        if psi > 0.2 or ks_p_value < 0.05:
            drift_status = "⚠️  DRIFT DÉTECTÉ"
            drift_detected = True
        elif psi > 0.1:
            drift_status = "🟡 DRIFT MODÉRÉ"
        
        print(f"\n{feature_name}:")
        print(f"  PSI: {psi:.4f} | KS p-value: {ks_p_value:.4f} | {drift_status}")
    
    mlflow.log_metric("drift_detected", 1 if drift_detected else 0)
    print(f"\n{'⚠️  ALERTE: Drift détecté!' if drift_detected else '✅ Pas de drift détecté'}")

# ====================
# SCÉNARIO 2: Drift artificiel
# ====================
print("\n" + "=" * 70)
print("📊 SCÉNARIO 2: Données de production avec DRIFT")
print("=" * 70)

# Simuler un drift en ajoutant du bruit et un shift
X_production_drift = X_test.copy()
X_production_drift[:, 0] += 1.5  # Shift sur la première feature
X_production_drift[:, 1] *= 1.3  # Scaling sur la deuxième feature
X_production_drift += np.random.normal(0, 0.3, X_production_drift.shape)  # Bruit

with mlflow.start_run(run_name="drift_check_with_drift"):
    mlflow.set_tag("drift_check_type", "with_drift")
    
    drift_detected = False
    drift_features = []
    
    for i, feature_name in enumerate(feature_names):
        reference_feature = X_train[:, i]
        current_feature = X_production_drift[:, i]
        
        # Nettoyer le nom de la feature pour MLflow
        clean_name = clean_feature_name(feature_name)
        
        # Calculer les métriques de drift
        psi = calculate_psi(reference_feature, current_feature)
        ks_stat, ks_p_value = kolmogorov_smirnov_test(reference_feature, current_feature)
        
        # Statistiques descriptives
        mean_diff = abs(np.mean(reference_feature) - np.mean(current_feature))
        std_diff = abs(np.std(reference_feature) - np.std(current_feature))
        
        # Logger dans MLflow (avec nom nettoyé)
        mlflow.log_metric(f"psi_{clean_name}", psi)
        mlflow.log_metric(f"ks_stat_{clean_name}", ks_stat)
        mlflow.log_metric(f"ks_pvalue_{clean_name}", ks_p_value)
        mlflow.log_metric(f"mean_diff_{clean_name}", mean_diff)
        mlflow.log_metric(f"std_diff_{clean_name}", std_diff)
        
        # Détection du drift
        drift_status = "✅ OK"
        if psi > 0.2 or ks_p_value < 0.05:
            drift_status = "⚠️  DRIFT DÉTECTÉ"
            drift_detected = True
            drift_features.append(feature_name)
        elif psi > 0.1:
            drift_status = "🟡 DRIFT MODÉRÉ"
        
        print(f"\n{feature_name}:")
        print(f"  PSI: {psi:.4f} | KS p-value: {ks_p_value:.4f} | {drift_status}")
        print(f"  Mean diff: {mean_diff:.4f} | Std diff: {std_diff:.4f}")
        
        # Créer un graphique de comparaison
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Histogramme
        ax1.hist(reference_feature, bins=20, alpha=0.5, label='Training', color='blue')
        ax1.hist(current_feature, bins=20, alpha=0.5, label='Production', color='red')
        ax1.set_xlabel(feature_name)
        ax1.set_ylabel('Frequency')
        ax1.set_title(f'{feature_name} - Distribution')
        ax1.legend()
        
        # Box plot
        ax2.boxplot([reference_feature, current_feature], labels=['Training', 'Production'])
        ax2.set_ylabel(feature_name)
        ax2.set_title(f'{feature_name} - Box Plot')
        
        plt.tight_layout()
        plot_filename = f"drift_{feature_name.replace(' ', '_')}.png"
        plt.savefig(plot_filename)
        mlflow.log_artifact(plot_filename)
        plt.close()
    
    mlflow.log_metric("drift_detected", 1 if drift_detected else 0)
    mlflow.log_metric("num_features_with_drift", len(drift_features))
    
    if drift_detected:
        mlflow.set_tag("drift_features", ", ".join(drift_features))
    
    print(f"\n{'⚠️  ALERTE: Drift détecté sur ' + str(len(drift_features)) + ' feature(s)!' if drift_detected else '✅ Pas de drift détecté'}")
    if drift_features:
        print(f"Features concernées: {', '.join(drift_features)}")

# ====================
# RECOMMANDATIONS
# ====================
print("\n" + "=" * 70)
print("💡 RECOMMANDATIONS EN CAS DE DRIFT")
print("=" * 70)
print("""
1. 🔄 Ré-entraîner le modèle avec les nouvelles données
2. 📊 Investiguer la cause du drift (changement business, erreur data pipeline)
3. 🎯 Adapter les features ou le preprocessing
4. 📈 Monitorer la performance du modèle en production
5. ⚠️  Mettre en place des alertes automatiques (email, Slack, etc.)
6. 🔍 Utiliser des outils dédiés: Evidently, Alibi Detect, WhyLabs

Interprétation des métriques:
- PSI < 0.1: Pas de drift
- 0.1 < PSI < 0.2: Drift modéré → surveiller
- PSI > 0.2: Drift significatif → action requise
- KS p-value < 0.05: Distributions statistiquement différentes
""")

print("\n🎉 Analyse de drift terminée!")
print("💡 Consultez MLflow UI pour voir les graphiques de distribution.")
