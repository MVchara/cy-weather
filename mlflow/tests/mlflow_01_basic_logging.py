"""
Script 1: Enregistrement basique avec MLflow
============================================
Ce script démontre comment logger:
- Des paramètres (hyperparamètres)
- Des métriques (accuracy, f1-score, etc.)
- Un modèle entraîné
- Des artefacts (graphiques, fichiers)
"""

import mlflow
import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import matplotlib.pyplot as plt
import numpy as np

# Configuration MLflow
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("01-Basic-Logging")

# Chargement des données
print("📊 Chargement du dataset Iris...")
data = load_iris()
X_train, X_test, y_train, y_test = train_test_split(
    data.data, data.target, test_size=0.2, random_state=42
)

# Entraînement avec logging MLflow
print("\n🚀 Entraînement du modèle avec MLflow...")
with mlflow.start_run(run_name="random_forest_iris"):
    
    # 1. Logger les paramètres
    params = {
        "n_estimators": 100,
        "max_depth": 5,
        "min_samples_split": 2,
        "random_state": 42
    }
    mlflow.log_params(params)
    
    # 2. Entraîner le modèle
    model = RandomForestClassifier(**params)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    
    # 3. Logger les métriques
    metrics = {
        "accuracy": accuracy_score(y_test, predictions),
        "f1_score_macro": f1_score(y_test, predictions, average='macro'),
        "precision_macro": precision_score(y_test, predictions, average='macro'),
        "recall_macro": recall_score(y_test, predictions, average='macro'),
        "train_size": len(X_train),
        "test_size": len(X_test)
    }
    mlflow.log_metrics(metrics)
    
    print(f"✅ Accuracy: {metrics['accuracy']:.4f}")
    print(f"✅ F1-Score: {metrics['f1_score_macro']:.4f}")
    
    # 4. Logger le modèle
    mlflow.sklearn.log_model(
        model, 
        "model",
        registered_model_name="iris_classifier"
    )
    
    # 5. Logger un artefact (graphique)
    fig, ax = plt.subplots(figsize=(10, 6))
    feature_importance = model.feature_importances_
    features = data.feature_names
    ax.barh(features, feature_importance)
    ax.set_xlabel('Importance')
    ax.set_title('Feature Importance')
    plt.tight_layout()
    
    # Sauvegarder et logger le graphique
    plt.savefig("feature_importance.png")
    mlflow.log_artifact("feature_importance.png")
    plt.close()
    
    # 6. Logger des tags pour organiser les runs
    mlflow.set_tags({
        "model_type": "RandomForest",
        "dataset": "iris",
        "team": "data-science",
        "environment": "training"
    })
    
    print(f"\n🎉 Run terminé! Consultez MLflow UI pour voir les résultats.")
    print(f"Run ID: {mlflow.active_run().info.run_id}")
