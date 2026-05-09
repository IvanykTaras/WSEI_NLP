from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, f1_score


# Stałe z siatkami parametrów do GridSearch
PARAM_GRIDS = {
    "nb":     {"classifier__alpha": [0.1, 0.5, 1.0]},
    "rf":     {"classifier__n_estimators": [100, 300], "classifier__max_depth": [None, 10, 20]},
    "logreg": {"classifier__C": [0.1, 1, 10]},
    "mlp":    {"classifier__hidden_layer_sizes": [(128,), (256, 128)]},
}

ALL_MODELS = ["nb", "rf", "logreg", "mlp"]


def _build_classifier(model_name: str, seed: int):
    """Zwraca instancję klasyfikatora dla podanej nazwy."""
    if model_name == "nb":
        return MultinomialNB()
    elif model_name == "rf":
        return RandomForestClassifier(random_state=seed, n_jobs=-1)
    elif model_name == "logreg":
        return LogisticRegression(random_state=seed, max_iter=1000)
    elif model_name == "mlp":
        return MLPClassifier(random_state=seed, max_iter=500)
    else:
        raise NotImplementedError(
            f"Model '{model_name}' nie jest zaimplementowany. "
            f"Dostępne: {', '.join(ALL_MODELS)}"
        )


class ClassificationProvider:
    """
    Warstwa Logiki Biznesowej (Core ML).
    Odpowiada za trenowanie modeli, testowanie (w tym GridSearch) i zwracanie metryk.
    """

    def run_experiment(
        self, X: list, y: list, vectorizer, model_name: str,
        seed: int, use_gridsearch: bool = False
    ):
        """
        Trenuje pojedynczy model i zwraca metryki.
        Jeśli model_name == 'all', deleguje do run_all_models().
        Zwraca: (accuracy, macro_f1, y_test, y_pred, pipeline)
        """
        model_name = model_name.lower()
        if model_name == "all":
            raise ValueError(
                "Użyj run_all_models() dla method=all."
            )

        classifier = _build_classifier(model_name, seed)
        pipeline = Pipeline([
            ('vectorizer', vectorizer),
            ('classifier', classifier)
        ])

        if use_gridsearch:
            param_grid = PARAM_GRIDS.get(model_name, {})
            print(f"GridSearch dla modelu {model_name}...")
            pipeline = GridSearchCV(pipeline, param_grid, cv=3, n_jobs=-1, verbose=0)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=seed
        )

        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)

        if use_gridsearch:
            print(f"Najlepsze parametry: {pipeline.best_params_}")

        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)

        return acc, f1, y_test, y_pred, pipeline

    def run_all_models(
        self, X: list, y: list, vectorizer_factory,
        seed: int, use_gridsearch: bool = False
    ) -> list:
        """
        Uruchamia wszystkie modele (nb, rf, logreg, mlp) na tym samym zbiorze.
        vectorizer_factory: callable() -> nowy vectorizer (bo każdy pipeline potrzebuje własnego)
        Zwraca listę słowników z wynikami.
        """
        results = []
        for name in ALL_MODELS:
            print(f"\n=== Uruchamiam model: {name.upper()} ===")
            try:
                vectorizer = vectorizer_factory()
                acc, f1, y_test, y_pred, pipeline = self.run_experiment(
                    X=X, y=y, vectorizer=vectorizer,
                    model_name=name, seed=seed, use_gridsearch=use_gridsearch
                )
                results.append({
                    "model_name": name,
                    "acc": acc,
                    "f1": f1,
                    "y_test": y_test,
                    "y_pred": y_pred,
                    "pipeline": pipeline,
                    "vectorizer": vectorizer,
                })
                print(f"  {name.upper()} — Accuracy: {acc:.4f} | Macro F1: {f1:.4f}")
            except Exception as e:
                print(f"  {name.upper()} — błąd: {e}")
        return results