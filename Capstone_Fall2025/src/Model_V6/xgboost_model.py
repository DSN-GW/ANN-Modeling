import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score
import pickle
import os
import json
from pathlib import Path

import optuna
from optuna.exceptions import TrialPruned
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner
optuna.logging.set_verbosity(optuna.logging.WARNING)


data_version = "Data_v6"
model_version = "V6"

def convert_numpy_types(obj):
    if isinstance(obj, (np.integer, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    else:
        return obj

class XGBoostClassifier:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = None
        self.feature_importance = None
        
    def load_data(self, data_path=None):
        if data_path is None:
            current_dir = Path(__file__).parent
            project_root = current_dir.parent.parent
            data_path = project_root / "data" / data_version / f"LLM_data_train_preprocessed_{model_version}.csv"
        
        df = pd.read_csv(data_path)
        return df
    
    def prepare_features(self, df):
        exclude_columns = []

        feature_columns = []
        for col in df.columns:
            if col not in exclude_columns:
                if df[col].dtype in ['int64', 'float64', 'bool']:
                    feature_columns.append(col)

        feature_columns = [
            # 'sub',
            # 'td_or_asd',
            # 'FSR',
            # 'BIS',
            # 'SRS.Raw',
            # 'free_response_TDprof_norm',
            # 'TDNorm_avg_PE_scaled',
            # 'overall_avg_PE_scaled',
            'TDNorm_avg_PE',
            'overall_avg_PE',
            'TDnorm_concept_learning',
            'overall_concept_learning',
            'FSR_scaled',

            'personality_inference_mentioned',
            'personality_inference_positive',
            'personality_inference_negative',
            'personality_inference_neutral',
            'sweets_mentioned',
            'sweets_positive',
            'sweets_negative',
            'sweets_neutral',
            'Fruits_and_vegetables_mentioned',
            'Fruits_and_vegetables_positive',
            'Fruits_and_vegetables_negative',
            'Fruits_and_vegetables_neutral',
            'healthy_savory_food_mentioned',
            'healthy_savory_food_positive',
            'healthy_savory_food_negative',
            'healthy_savory_food_neutral',
            'food_mentioned',
            'food_positive',
            'food_negative',
            'food_neutral',
            'cosmetics_mentioned',
            'cosmetics_positive',
            'cosmetics_negative',
            'cosmetics_neutral',
            'fashion_mentioned',
            'fashion_positive',
            'fashion_negative',
            'fashion_neutral',
            'toys_gadgets_and_games_mentioned',
            'toys_gadgets_and_games_positive',
            'toys_gadgets_and_games_negative',
            'toys_gadgets_and_games_neutral',
            'sports_mentioned',
            'sports_positive',
            'sports_negative',
            'sports_neutral',
            'music_mentioned',
            'music_positive',
            'music_negative',
            'music_neutral',
            'arts_and_crafts_mentioned',
            'arts_and_crafts_positive',
            'arts_and_crafts_negative',
            'arts_and_crafts_neutral',
            'word_count',
            'sentence_count',
            'char_count',
            'avg_word_length',
            'avg_sentence_length',
            'shortness_score',
            'lexical_diversity',
            'sentiment_polarity',
            'sentiment_subjectivity',
            'positive_word_count',
            'negative_word_count',
            'positive_word_ratio',
            'negative_word_ratio',
            'flesch_reading_ease',
            'flesch_kincaid_grade',

        ]

        X = df[feature_columns].copy()
        y = df['td_or_asd'].copy()
        
        X = X.fillna(0)
        
        self.feature_names = feature_columns
        
        print(f"Selected {len(feature_columns)} features.")
        # print(f"Feature columns: {feature_columns}")
        return X, y

   ##################### OPTUNA ########################
    def tune_with_optuna(
            self,
            X,
            y,
            n_trials: int = 200,
            cv_splits: int = 5,
            random_state: int = 42,
            direction: str = "maximize",
            metric: str = "accuracy"
    ):
        """
        Hyperparameter tuning for XGBoost using Optuna + xgb.cv.
        Works consistently across XGBoost versions and ensures reproducibility.
        """
        # Stratified splits
        skf = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=random_state)

        # Handle potential class imbalance
        pos = (y == 1).sum() if set(pd.Series(y).unique()) <= {0, 1} else None
        neg = len(y) - pos if pos is not None else None
        spw_guess = (neg / pos) if (pos and pos > 0) else 1.0

        # DMatrix is more efficient; XGBoost doesn’t need feature scaling
        dtrain = xgb.DMatrix(X.values, label=y.values)

        # Set metric
        if metric == "accuracy":
            xgb_metric = "error"  # 1 - accuracy

            def extract_score(cv_df):  # maximize accuracy
                return float(1.0 - cv_df["test-error-mean"].min())
        else:
            xgb_metric = "auc"

            def extract_score(cv_df):  # maximize AUC
                return float(cv_df["test-auc-mean"].max())

        # Silence Optuna logs except progress bar
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        CV_SEEDS = lambda base_seed: [base_seed + i for i in range(3)]

        def objective(trial: optuna.Trial) -> float:
            booster = trial.suggest_categorical("booster", ["gbtree", "dart"])

            params = {
                "objective": "binary:logistic",
                "eval_metric": xgb_metric,
                "tree_method": "hist",
                "booster": booster,

                # depth / structure
                "max_depth": trial.suggest_int("max_depth", 3, 30),  # was up to 12
                "min_child_weight": trial.suggest_int("min_child_weight", 1, 128),
                "max_leaves": trial.suggest_int("max_leaves", 0, 8192),  # for lossguide
                "grow_policy": trial.suggest_categorical("grow_policy", ["depthwise", "lossguide"]),

                # learning rate
                "eta": trial.suggest_float("learning_rate", 1e-5, 0.5, log=True),

                # sampling (much wider)
                "subsample": trial.suggest_float("subsample", 0.1, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.1, 1.0),
                "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.1, 1.0),
                "colsample_bynode": trial.suggest_float("colsample_bynode", 0.1, 1.0),

                # regularization
                "gamma": trial.suggest_float("gamma", 0.0, 50.0),
                "alpha": trial.suggest_float("reg_alpha", 1e-8, 300.0, log=True),
                "lambda": trial.suggest_float("reg_lambda", 1e-8, 300.0, log=True),

                # imbalance
                "scale_pos_weight": trial.suggest_float(
                    "scale_pos_weight",
                    max(0.05, (spw_guess or 1.0) * 0.1),
                    max(50.0, (spw_guess or 1.0) * 6.0),
                    log=True
                )
                if spw_guess else 1.0,

                # bins & predictor
                "max_bin": trial.suggest_int("max_bin", 64, 2048),

                # reproducibility
                "seed": random_state,
                "nthread": 1,
            }

            if booster == "dart":
                params.update({
                    "sample_type": trial.suggest_categorical("sample_type", ["uniform", "weighted"]),
                    "normalize_type": trial.suggest_categorical("normalize_type", ["tree", "forest"]),
                    "rate_drop": trial.suggest_float("rate_drop", 0.0, 0.8),
                    "skip_drop": trial.suggest_float("skip_drop", 0.0, 0.8),
                })

            num_boost_round = trial.suggest_int("n_estimators", 800, 20000, step=200)  # more rounds; early stopping will cap it

            early_stopping_rounds = trial.suggest_int("early_stopping_rounds", 100, 400, step=50)

            # Try xgb.cv with early stopping; fallback if not supported
            # === Repeat CV across several seeds and average the metric ===
            scores = []
            best_rounds = []
            for seed_i in CV_SEEDS(random_state):
                try:
                    cv_res = xgb.cv(
                        params=params,
                        dtrain=dtrain,
                        num_boost_round=num_boost_round,
                        nfold=cv_splits,
                        stratified=True,
                        seed=seed_i,
                        early_stopping_rounds=early_stopping_rounds,
                        verbose_eval=False,
                    )
                except TypeError:
                    # fallback w/o early stopping if version mismatch
                    cv_res = xgb.cv(
                        params=params,
                        dtrain=dtrain,
                        num_boost_round=num_boost_round,
                        nfold=cv_splits,
                        stratified=True,
                        seed=seed_i,
                        verbose_eval=False,
                    )

                scores.append(extract_score(cv_res))
                best_rounds.append(int(cv_res.shape[0]))

            # Track avg score; also store stability info per trial
            avg_score = float(np.mean(scores))
            trial.set_user_attr("cv_scores", [float(s) for s in scores])
            trial.set_user_attr("best_boost_round_mean", int(np.round(np.mean(best_rounds))))
            trial.set_user_attr("best_boost_round_min", int(np.min(best_rounds)))
            trial.set_user_attr("best_boost_round_max", int(np.max(best_rounds)))
            return avg_score

        # Create seeded study for reproducibility
        sampler = TPESampler(
            seed=random_state,
            multivariate=True,  # lets TPE model joint interactions
            group=True,  # groups related params (helps with conditionals)
            consider_prior=True,
            n_startup_trials=40
        )

        pruner = MedianPruner(n_warmup_steps=15)

        study = optuna.create_study(
            direction=direction,
            sampler=sampler,
            pruner=pruner,
        )

        print(f"[Optuna] Starting hyperparameter search ({n_trials} trials)...")
        study.optimize(objective,
                       n_trials=n_trials,
                       show_progress_bar=True,
                       timeout=22000, n_jobs=1, gc_after_trial=True)

        print("Best value:", study.best_value)
        print("Best params:", study.best_params)
        print("Attrs:", study.best_trial.user_attrs)

        # Extract best parameters
        best_params = study.best_trial.params.copy()
        best_boost_round = study.best_trial.user_attrs.get(
            "best_boost_round",
            best_params.get("n_estimators", 300)
        )

        # Map low-level to sklearn API param names
        best_params["n_estimators"] = int(best_boost_round)
        if "learning_rate" not in best_params and "eta" in best_params:
            best_params["learning_rate"] = best_params.pop("eta")

        self.best_params_ = best_params
        self.best_score_ = float(study.best_value)

        print(f"[Optuna] Best {metric.upper()}: {self.best_score_:.5f}")
        print(f"[Optuna] Best Params:\n{best_params}")

        return best_params, self.best_score_
# #############################################################################
#     def train_model(self, X, y, random_state=42):
#         X_scaled = self.scaler.fit_transform(X)
#
#         self.model = xgb.XGBClassifier(**{
#             "booster": "dart",
#             "max_depth": 6,
#             "min_child_weight": 45,
#             "max_leaves": 1030,
#             "grow_policy": "depthwise",
#             "learning_rate": 0.10810109913557502,
#             "subsample": 0.9504094778274954,
#             "colsample_bytree": 0.9835838303754004,
#             "colsample_bylevel": 0.9745161890663262,
#             "colsample_bynode": 0.47576016392813497,
#             "gamma": 1.8162036355522655,
#             "reg_alpha": 0.0813636308766135,
#             "reg_lambda": 10.310586595956945,
#             "scale_pos_weight": 1.3321875637042824,
#             "max_bin": 161,
#             "sample_type": "weighted",
#             "normalize_type": "tree",
#             "rate_drop": 0.43991549330770086,
#             "skip_drop": 0.6122020811068443,
#             "n_estimators": 61
#         })
#
#         self.model.fit(X_scaled, y)
#
#         cv_scores = cross_val_score(self.model, X_scaled, y, cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state), scoring='accuracy')
#
#         self.feature_importance = dict(zip(self.feature_names, self.model.feature_importances_))
#
#         training_results = {
#             'cv_accuracy_mean': float(cv_scores.mean()),
#             'cv_accuracy_std': float(cv_scores.std()),
#             'cv_scores': cv_scores.tolist(),
#             'n_features': len(self.feature_names),
#             'feature_names': self.feature_names
#         }
#
#         return training_results

    def train_model(self, X, y, random_state=42, params: dict | None = None):
        X_scaled = self.scaler.fit_transform(X)

        base_params = dict(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.7,
            colsample_bytree=0.7,
            min_child_weight=3,
            gamma=0.1,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=random_state,
            eval_metric='logloss'
        )
        if params:
            base_params.update(params)

        self.model = xgb.XGBClassifier(**base_params)
        self.model.fit(X_scaled, y)

        # report both accuracy and roc_auc via CV
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
        cv_acc = cross_val_score(self.model, X_scaled, y, cv=skf, scoring='accuracy')
        cv_auc = cross_val_score(self.model, X_scaled, y, cv=skf, scoring='roc_auc')

        self.feature_importance = dict(zip(self.feature_names, self.model.feature_importances_))

        training_results = {
            'cv_accuracy_mean': float(cv_acc.mean()),
            'cv_accuracy_std': float(cv_acc.std()),
            'cv_auc_mean': float(cv_auc.mean()),
            'cv_auc_std': float(cv_auc.std()),
            'cv_scores_accuracy': cv_acc.tolist(),
            'cv_scores_auc': cv_auc.tolist(),
            'n_features': len(self.feature_names),
            'feature_names': self.feature_names,
            'used_params': base_params
        }
        return training_results


    def get_feature_importance_sorted(self, top_n=20):
        if self.feature_importance is None:
            return None
        return sorted(self.feature_importance.items(), key=lambda x: x[1], reverse=True)[:top_n]
    
    def save_model(self, model_dir=None):
        if model_dir is None:
            current_dir = Path(__file__).parent
            project_root = current_dir.parent.parent
            model_dir = project_root / "Results" / model_version
        
        model_dir = Path(model_dir)
        model_dir.mkdir(parents=True, exist_ok=True)
        
        with open(model_dir / f'xgboost_model_{model_version}.pkl', 'wb') as f:
            pickle.dump(self.model, f)
        
        with open(model_dir / f'scaler_{model_version}.pkl', 'wb') as f:
            pickle.dump(self.scaler, f)
        
        with open(model_dir / f'feature_names_{model_version}.pkl', 'wb') as f:
            pickle.dump(self.feature_names, f)
        
        if self.feature_importance:
            feature_importance_serializable = convert_numpy_types(self.feature_importance)
            with open(model_dir / f'feature_importance_{model_version}.json', 'w') as f:
                json.dump(feature_importance_serializable, f, indent=2)

        # Save optuna best params if available
        if hasattr(self, "best_params_"):
            with open(model_dir / f'best_params_optuna_{model_version}.json', 'w') as f:
                json.dump(convert_numpy_types(self.best_params_), f, indent=2)
        if hasattr(self, "best_score_"):
            with open(model_dir / f'best_score_optuna_{model_version}.txt', 'w') as f:
                f.write(str(self.best_score_))


        print(f"Model artifacts saved to: {model_dir}")
    
    def load_model(self, model_dir=None):
        if model_dir is None:
            current_dir = Path(__file__).parent
            project_root = current_dir.parent.parent
            model_dir = project_root / "Results" / model_version
        
        model_dir = Path(model_dir)
        
        with open(model_dir / f'xgboost_model_{model_version}.pkl', 'rb') as f:
            self.model = pickle.load(f)
        
        with open(model_dir / f'scaler_{model_version}.pkl', 'rb') as f:
            self.scaler = pickle.load(f)
        
        with open(model_dir / f'feature_names_{model_version}.pkl', 'rb') as f:
            self.feature_names = pickle.load(f)
        
        feature_importance_path = model_dir / f'feature_importance_{model_version}.json'
        if feature_importance_path.exists():
            with open(feature_importance_path, 'r') as f:
                self.feature_importance = json.load(f)
        
        print(f"Model loaded from: {model_dir}")
    
    def predict(self, X):
        if self.model is None:
            raise ValueError("Model not trained or loaded. Please train or load a model first.")
        
        X_scaled = self.scaler.transform(X)
        predictions = self.model.predict(X_scaled)
        probabilities = self.model.predict_proba(X_scaled)
        
        return predictions, probabilities

# def train_xgboost_model():
#     classifier = XGBoostClassifier()
#
#     df = classifier.load_data()
#     print(f"Loaded training data with shape: {df.shape}")
#
#     X, y = classifier.prepare_features(df)
#     print(f"Feature matrix shape: {X.shape}")
#     print(f"Target distribution: {y.value_counts().to_dict()}")
#
#     training_results = classifier.train_model(X, y)
#
#     classifier.save_model()
#
#     current_dir = Path(__file__).parent
#     project_root = current_dir.parent.parent
#     results_dir = project_root / "Results" / f"{model_version}"
#
#     training_results_serializable = convert_numpy_types(training_results)
#     with open(results_dir / f'training_results_{model_version}.json', 'w') as f:
#         json.dump(training_results_serializable, f, indent=2)
#
#     print(f"Training completed successfully!")
#     print(f"Cross-validation accuracy: {training_results['cv_accuracy_mean']:.4f} (+/- {training_results['cv_accuracy_std']:.4f})")
#
#     return classifier, training_results

def train_xgboost_model(run_optuna: bool = True, n_trials: int = 200):
    classifier = XGBoostClassifier()

    df = classifier.load_data()
    print(f"Loaded training data with shape: {df.shape}")

    X, y = classifier.prepare_features(df)
    print(f"Feature matrix shape: {X.shape}")
    print(f"Target distribution: {y.value_counts().to_dict()}")

    best_params = None
    if run_optuna:
        print("[Optuna] Starting hyperparameter search...")
        best_params, best_score = classifier.tune_with_optuna(
            X, y,
            n_trials=n_trials,
            cv_splits=5,
            random_state=42,
            direction="maximize",
            metric="accuracy"
        )
        print(f"[Optuna] Done. Best Accuracy = {best_score:.5f}")

    training_results = classifier.train_model(X, y, params=best_params)

    classifier.save_model()

    current_dir = Path(__file__).parent
    project_root = current_dir.parent.parent
    results_dir = project_root / "Results" / f"{model_version}"

    training_results_serializable = convert_numpy_types(training_results)
    with open(results_dir / f'training_results_{model_version}.json', 'w') as f:
        json.dump(training_results_serializable, f, indent=2)

    print("Training completed successfully!")
    print(
        f"CV ACC: {training_results['cv_accuracy_mean']:.4f} (+/- {training_results['cv_accuracy_std']:.4f}) | "
        f"CV AUC: {training_results['cv_auc_mean']:.4f} (+/- {training_results['cv_auc_std']:.4f})"
    )

    return classifier, training_results

if __name__ == "__main__":
    classifier, results = train_xgboost_model(run_optuna=True, n_trials=200)
    # classifier, results = train_xgboost_model()


