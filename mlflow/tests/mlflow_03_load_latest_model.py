"""
Script 3: Chargement automatique de la dernière version du modèle
==================================================================
Ce script démontre comment:
- Charger un modèle depuis le Registry par stage (Production, Staging)
- Charger une version spécifique
- Charger la dernière version entraînée
- Utiliser le modèle pour faire des prédictions
"""

import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
from sklearn.datasets import load_iris
import numpy as np

# Configuration MLflow
mlflow.set_tracking_uri("http://localhost:5000")

model_name = "iris_classifier_registry"
client = MlflowClient()

print("🔍 Recherche des modèles disponibles...\n")

# ====================
# MÉTHODE 1: Charger par STAGE (Production, Staging)
# ====================
print("📦 MÉTHODE 1: Chargement du modèle en PRODUCTION")
try:
    model_production_uri = f"models:/{model_name}/Production"
    model_production = mlflow.sklearn.load_model(model_production_uri)
    print(f"✅ Modèle en Production chargé: {model_production_uri}")
    
    # Obtenir les détails de la version
    production_versions = client.get_latest_versions(model_name, stages=["Production"])
    if production_versions:
        version_info = production_versions[0]
        print(f"   Version: {version_info.version}")
        print(f"   Run ID: {version_info.run_id}")
        print(f"   Created: {version_info.creation_timestamp}")
except Exception as e:
    print(f"⚠️  Aucun modèle en Production: {e}")
    model_production = None

# ====================
# MÉTHODE 2: Charger par VERSION SPÉCIFIQUE
# ====================
print("\n📦 MÉTHODE 2: Chargement d'une version spécifique (version 1)")
try:
    model_v1_uri = f"models:/{model_name}/1"
    model_v1 = mlflow.sklearn.load_model(model_v1_uri)
    print(f"✅ Version 1 chargée: {model_v1_uri}")
except Exception as e:
    print(f"⚠️  Impossible de charger la version 1: {e}")
    model_v1 = None

# ====================
# MÉTHODE 3: Charger la DERNIÈRE VERSION (peu importe le stage)
# ====================
print("\n📦 MÉTHODE 3: Chargement de la dernière version entraînée")
try:
    # Récupérer toutes les versions
    all_versions = client.search_model_versions(f"name='{model_name}'")
    
    if all_versions:
        # Trier par numéro de version (décroissant)
        latest_version = sorted(all_versions, key=lambda x: int(x.version), reverse=True)[0]
        
        model_latest_uri = f"models:/{model_name}/{latest_version.version}"
        model_latest = mlflow.sklearn.load_model(model_latest_uri)
        
        print(f"✅ Dernière version chargée: {latest_version.version}")
        print(f"   Stage: {latest_version.current_stage}")
        print(f"   Run ID: {latest_version.run_id}")
    else:
        print("⚠️  Aucune version disponible")
        model_latest = None
except Exception as e:
    print(f"⚠️  Erreur: {e}")
    model_latest = None


# ====================
# UTILISATION DU MODÈLE POUR DES PRÉDICTIONS
# ====================
print("\n" + "="*60)
print("🎯 TEST DU MODÈLE EN PRODUCTION")
print("="*60)

if model_production:
    # Charger quelques données de test
    data = load_iris()
    X_test = data.data[:5]  # Prendre 5 échantillons
    
    print("\n📊 Données d'entrée (5 premières fleurs):")
    for i, features in enumerate(X_test):
        print(f"  Fleur {i+1}: {features}")
    
    # Faire des prédictions
    predictions = model_production.predict(X_test)
    
    print("\n🔮 Prédictions:")
    target_names = data.target_names
    for i, pred in enumerate(predictions):
        print(f"  Fleur {i+1}: {target_names[pred]} (classe {pred})")
    
    # Prédictions avec probabilités (si disponible)
    if hasattr(model_production, 'predict_proba'):
        probas = model_production.predict_proba(X_test)
        print("\n📊 Probabilités:")
        for i, proba in enumerate(probas):
            print(f"  Fleur {i+1}:")
            for j, class_proba in enumerate(proba):
                print(f"    - {target_names[j]}: {class_proba:.2%}")
else:
    print("\n⚠️  Aucun modèle en production disponible pour les tests.")
    print("💡 Exécutez d'abord mlflow_02_model_registry.py pour créer un modèle.")

# ====================
# AFFICHER TOUS LES MODÈLES DISPONIBLES
# ====================
print("\n" + "="*60)
print("📋 RÉSUMÉ DES MODÈLES DISPONIBLES")
print("="*60)

try:
    registered_models = client.search_registered_models()
    
    if registered_models:
        for rm in registered_models:
            print(f"\n🏷️  Modèle: {rm.name}")
            versions = client.search_model_versions(f"name='{rm.name}'")
            
            for version in sorted(versions, key=lambda x: int(x.version)):
                print(f"   📌 Version {version.version}")
                print(f"      Stage: {version.current_stage}")
                print(f"      Status: {version.status}")
    else:
        print("Aucun modèle enregistré.")
except Exception as e:
    print(f"⚠️  Erreur lors de la récupération des modèles: {e}")

print("\n🎉 Script terminé!")
print("💡 TIP: En production, utilisez toujours 'models:/{model_name}/Production'")
