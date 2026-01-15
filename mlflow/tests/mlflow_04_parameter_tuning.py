"""
Script 4: Monitoring et comparaison de modèles (Hyperparameter Tuning)
========================================================================
Ce script démontre comment:
- Entraîner plusieurs modèles avec différents hyperparamètres
- Logger et comparer les performances
- Utiliser des runs parents/enfants pour organiser les expérimentations
- Trouver le meilleur modèle automatiquement
"""

import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, f1_score
import numpy as np
from itertools import product

# Configuration MLflow
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("04-Hyperparameter-Tuning")

# Chargement des données
print("📊 Chargement du dataset Iris...")
data = load_iris()
X_train, X_test, y_train, y_test = train_test_split(
    data.data, data.target, test_size=0.2, random_state=42
)

# Définir la grille d'hyperparamètres à tester
param_grid = {
    'n_estimators': [10, 50, 100, 200],
    'max_depth': [3, 5, 10, None],
    'min_samples_split': [2, 5, 10]
}

print(f"\n🔬 Test de {len(param_grid['n_estimators']) * len(param_grid['max_depth']) * len(param_grid['min_samples_split'])} combinaisons d'hyperparamètres...\n")

# Créer un run parent pour grouper tous les tests
with mlflow.start_run(run_name="hyperparameter_tuning_experiment") as parent_run:
    
    mlflow.set_tag("experiment_type", "grid_search")
    mlflow.set_tag("model_family", "RandomForest")
    
    best_accuracy = 0
    best_params = None
    best_run_id = None
    all_results = []
    
    # Tester toutes les combinaisons
    for n_est, max_d, min_split in product(
        param_grid['n_estimators'],
        param_grid['max_depth'],
        param_grid['min_samples_split']
    ):
        
        # Créer un run enfant pour chaque combinaison
        with mlflow.start_run(
            run_name=f"RF_n{n_est}_d{max_d}_s{min_split}",
            nested=True
        ) as child_run:
            
            # Paramètres
            params = {
                'n_estimators': n_est,
                'max_depth': max_d,
                'min_samples_split': min_split,
                'random_state': 42
            }
            
            mlflow.log_params(params)
            
            # Entraînement
            model = RandomForestClassifier(**params)
            model.fit(X_train, y_train)
            
            # Prédictions
            predictions = model.predict(X_test)
            
            # Métriques
            accuracy = accuracy_score(y_test, predictions)
            f1 = f1_score(y_test, predictions, average='macro')
            
            # Cross-validation score
            cv_scores = cross_val_score(model, X_train, y_train, cv=5)
            cv_mean = cv_scores.mean()
            cv_std = cv_scores.std()
            
            metrics = {
                'accuracy': accuracy,
                'f1_score': f1,
                'cv_mean': cv_mean,
                'cv_std': cv_std
            }
            
            mlflow.log_metrics(metrics)
            
            # Logger le modèle
            mlflow.sklearn.log_model(model, "model")
            
            # Garder trace du meilleur modèle
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_params = params
                best_run_id = child_run.info.run_id
            
            all_results.append({
                'params': params,
                'accuracy': accuracy,
                'f1_score': f1,
                'cv_mean': cv_mean,
                'run_id': child_run.info.run_id
            })
            
            print(f"✓ n_est={n_est:3d}, max_depth={str(max_d):4s}, min_split={min_split:2d} → Accuracy: {accuracy:.4f}, F1: {f1:.4f}")
    
    # Logger les résultats du meilleur modèle dans le run parent
    mlflow.log_params({f"best_{k}": v for k, v in best_params.items()})
    mlflow.log_metric("best_accuracy", best_accuracy)
    mlflow.set_tag("best_run_id", best_run_id)
    
    print("\n" + "="*70)
    print("🏆 MEILLEUR MODÈLE TROUVÉ")
    print("="*70)
    print(f"Accuracy: {best_accuracy:.4f}")
    print(f"Paramètres:")
    for param, value in best_params.items():
        print(f"  - {param}: {value}")
    print(f"Run ID: {best_run_id}")

# ====================
# ANALYSE DES RÉSULTATS
# ====================
print("\n" + "="*70)
print("📊 ANALYSE DES RÉSULTATS")
print("="*70)

# Trier par accuracy
all_results_sorted = sorted(all_results, key=lambda x: x['accuracy'], reverse=True)

print("\n🥇 TOP 5 DES MODÈLES:")
for i, result in enumerate(all_results_sorted[:5], 1):
    print(f"\n{i}. Accuracy: {result['accuracy']:.4f} | F1: {result['f1_score']:.4f} | CV: {result['cv_mean']:.4f}")
    print(f"   Paramètres: {result['params']}")

# Statistiques globales
accuracies = [r['accuracy'] for r in all_results]
print(f"\n📈 STATISTIQUES GLOBALES:")
print(f"Accuracy moyenne: {np.mean(accuracies):.4f}")
print(f"Accuracy médiane: {np.median(accuracies):.4f}")
print(f"Accuracy min: {np.min(accuracies):.4f}")
print(f"Accuracy max: {np.max(accuracies):.4f}")
print(f"Écart-type: {np.std(accuracies):.4f}")

# ====================
# REQUÊTE MLFLOW POUR RETROUVER LES MEILLEURS RUNS
# ====================
print("\n" + "="*70)
print("🔍 RECHERCHE AVEC L'API MLFLOW")
print("="*70)

client = MlflowClient()

# Rechercher les runs avec une accuracy > 0.95
high_accuracy_runs = client.search_runs(
    experiment_ids=[mlflow.get_experiment_by_name("04-Hyperparameter-Tuning").experiment_id],
    filter_string="metrics.accuracy > 0.95",
    order_by=["metrics.accuracy DESC"],
    max_results=5
)

print(f"\n🎯 Runs avec Accuracy > 0.95 ({len(high_accuracy_runs)} trouvés):")
for run in high_accuracy_runs:
    if run.data.metrics:  # Vérifier que le run a des métriques
        accuracy = run.data.metrics.get('accuracy', 'N/A')
        n_est = run.data.params.get('n_estimators', 'N/A')
        max_d = run.data.params.get('max_depth', 'N/A')
        print(f"  - Accuracy: {accuracy:.4f} | n_estimators={n_est}, max_depth={max_d}")

print("\n🎉 Expérience terminée!")
print("💡 Consultez MLflow UI pour visualiser les comparaisons graphiques.")
print("💡 Dans l'UI, utilisez 'Compare' pour voir les runs côte à côte.")
