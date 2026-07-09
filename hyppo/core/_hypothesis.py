"""Hypothesis/Model OWL classes and model-selection scoring (Definition 10, Chapter 2).

Extends the ``virtual_experiment_onto`` classes from
:mod:`hyppo.core._base` with the behaviour needed at runtime: AIC/BIC scoring
(Definition 10), model ranking, and pairwise prediction comparison used by
:func:`hyppo.core._epistemic.evaluate_status`.
"""

import math

import numpy as np
from scipy import stats
from sklearn.metrics import mean_squared_error, r2_score

from hyppo.core._base import virtual_experiment_onto

# ---------------------------------------------------------------------------
# Module-level pure functions (Definitions from Chapter 2)
# ---------------------------------------------------------------------------


def get_aic(n_params: int, log_likelihood: float) -> float:
    """Compute the Akaike Information Criterion (Definition 10, Chapter 2).

    Args:
        n_params: Number of free parameters k in the model.
        log_likelihood: Log-likelihood ln(L) of the fitted model.

    Returns:
        AIC = 2k - 2ln(L); lower is better.
    """
    return 2 * n_params - 2 * log_likelihood


def get_bic(n_params: int, n_observations: int, log_likelihood: float) -> float:
    """Compute the Bayesian Information Criterion (Definition 10, Chapter 2).

    Args:
        n_params: Number of free parameters k in the model.
        n_observations: Number of observations n used to fit the model.
        log_likelihood: Log-likelihood ln(L) of the fitted model.

    Returns:
        BIC = k*ln(n) - 2ln(L); lower is better.
    """
    return n_params * math.log(n_observations) - 2 * log_likelihood


def range_models(
    scores: dict[str, float], threshold: float = 0.7
) -> list[tuple[str, float]]:
    """Rank models by a metric, dropping those below a threshold.

    Args:
        scores: Mapping of model name to its score.
        threshold: Minimum score (inclusive) for a model to be kept.

    Returns:
        (model_name, score) pairs above ``threshold``, sorted descending by score.
    """
    filtered = [(name, score) for name, score in scores.items() if score >= threshold]
    return sorted(filtered, key=lambda x: x[1], reverse=True)


with virtual_experiment_onto:

    class Hypothesis(virtual_experiment_onto.Artefact):
        """Runtime ``Hypothesis`` individual: scoring/comparison behaviour
        layered on top of the OWL class declared in
        :mod:`hyppo.core._base` (Definition 1)."""

        namespace = virtual_experiment_onto.get_namespace(
            "http://synthesis.ipi.ac.ru/virtual_experiment.owl"
        )
        # def __init__(self):
        #     self._parameters = dict()

        @property
        def parameters(self):
            """dict: Model/fit parameters attached to this hypothesis."""
            return self._parameters

        @parameters.setter
        def parameters(self, value):
            self._parameters = value

        @parameters.getter
        def parameters(self):
            return self._parameters

        def range_models(self, X, y, metrics, threshold):
            """Score and filter the models implementing this hypothesis.

            Args:
                X: Input features passed to each model's ``predict``.
                y: Target/observed values to compare predictions against.
                metrics: One of ``"mae"``, ``"r2"``, ``"mse"``.
                threshold: Cutoff applied to the metric (kept if the model
                    passes the metric-specific comparison against it).

            Returns:
                dict: Mapping of model key to its score, restricted to models
                that pass the threshold for the chosen metric.
            """
            models = self.is_implemented_by_model()
            result = {}

            if metrics == "mae":
                for key in models.keys():
                    error = models.predict(X) - y
                    if error <= threshold:
                        result[key] = np.mean(np.abs(error))

            elif metrics == "r2":
                for key in models.keys():
                    error = r2_score(y, models[key].predict(X))
                    if error >= threshold:
                        result[key] = error

            elif metrics == "mse":
                for key in models.keys():
                    error = mean_squared_error(y, models[key].predict(X))
                    if error <= threshold:
                        result[key] = error

            return result

        # TODO add several > 2 models
        def compare_preds_on_single_dataset(self, dataset, stat_test):
            """Compare two competing models' predictions on one dataset.

            Args:
                dataset: Input passed to both ``model_1.predict`` and
                    ``model_2.predict``.
                stat_test: Statistical test to run; only ``"wilcoxon"`` is
                    supported.

            Returns:
                The result object returned by ``scipy.stats.wilcoxon``.

            Raises:
                NotImplementedError: If ``stat_test`` is not ``"wilcoxon"``.
            """
            models = self.is_implemented_by_model()
            linear_prediction = models["model_1"].predict(dataset)
            gp_prediction = models["model_2"].predict(dataset)
            if stat_test == "wilcoxon":
                result = stats.wilcoxon(linear_prediction, gp_prediction)
            else:
                raise NotImplementedError(f"stat_test={stat_test!r} not supported")
            return result

        def compare_preds_on_different_datasets(self, dataset_1, dataset_2, stat_test):
            """Compare two competing models' predictions on two datasets.

            Args:
                dataset_1: Input passed to ``model_1.predict``.
                dataset_2: Input passed to ``model_2.predict``.
                stat_test: Statistical test to run; only ``"wilcoxon"`` is
                    supported.

            Returns:
                The result object returned by ``scipy.stats.wilcoxon``.

            Raises:
                NotImplementedError: If ``stat_test`` is not ``"wilcoxon"``.
            """
            models = self.is_implemented_by_model()
            linear_prediction = models["model_1"].predict(dataset_1)
            gp_prediction = models["model_2"].predict(dataset_2)
            if stat_test == "wilcoxon":
                result = stats.wilcoxon(linear_prediction, gp_prediction)
            else:
                raise NotImplementedError(f"stat_test={stat_test!r} not supported")
            return result


class Model(virtual_experiment_onto.Artefact):
    """Base ``Model`` individual: the implementation paired 1:1 with a
    ``Hypothesis`` via ``is_implemented_by_model`` (Theorem 1). Subclasses
    provide the actual ``fit``/``predict`` behaviour."""

    namespace = virtual_experiment_onto.get_namespace(
        "http://synthesis.ipi.ac.ru/virtual_experiment.owl"
    )

    def __init__(self):
        """Initialize an empty model. No-op on the base class."""
        pass

    def fit(self, X, y):
        """Fit the model to data. No-op on the base class; override in subclasses.

        Args:
            X: Training features.
            y: Training targets.
        """
        pass

    def predict(self, X):
        """Predict outputs for X. No-op on the base class; override in subclasses.

        Args:
            X: Input features.
        """
        pass

    def update_bayesian_probability(self, X, y, prior):
        """Compute the unnormalised Bayesian posterior weight for this hypothesis.

        ``prior * exp(log-likelihood)`` with a Gaussian likelihood at the
        MLE residual variance. Normalise across competing hypotheses with
        :func:`hyppo.comparison.bayesian_posterior`.

        Args:
            X: Input features passed to ``predict``.
            y: Observed targets.
            prior: Prior probability of this hypothesis.

        Returns:
            float: Unnormalised posterior weight (``inf`` if the
            log-likelihood is ``inf``).
        """
        from hyppo.comparison.compare import gaussian_log_likelihood

        ll = gaussian_log_likelihood(y, self.predict(X))
        return float("inf") if ll == float("inf") else prior * math.exp(ll)

    def bayesian_score(self, X, y):
        """Compute the Gaussian log-likelihood of predictions on (X, y).

        Args:
            X: Input features passed to ``predict``.
            y: Observed targets.

        Returns:
            float: Gaussian log-likelihood of the model's predictions.
        """
        from hyppo.comparison.compare import gaussian_log_likelihood

        return gaussian_log_likelihood(y, self.predict(X))


class NonLinearModel(Model):
    """Symbolic-regression ``Model`` (gplearn genetic programming)."""

    namespace = virtual_experiment_onto.get_namespace(
        "http://synthesis.ipi.ac.ru/virtual_experiment.owl"
    )

    def fit(self, X, y):
        """Fit a symbolic regressor via genetic programming (gplearn).

        Args:
            X: Training features.
            y: Training targets.

        Returns:
            The fitted ``gplearn.genetic.SymbolicRegressor`` instance.
        """
        from gplearn.genetic import SymbolicRegressor

        est_gp = SymbolicRegressor(
            population_size=1000,
            tournament_size=20,
            generations=150,
            stopping_criteria=0.001,
            const_range=(-1, 1),
            p_crossover=0.7,
            p_subtree_mutation=0.12,
            p_hoist_mutation=0.06,
            p_point_mutation=0.12,
            p_point_replace=1,
            init_depth=(6, 10),
            function_set=("mul", "sub", "div", "add", "cos"),
            max_samples=0.9,
            verbose=1,
            metric="mse",
            parsimony_coefficient=0.0005,
            random_state=0,
            n_jobs=1,
        )

        est_gp.fit(X, y)
        return est_gp

    def compute_aic(self, X, y):
        """Compute an AIC-like score for this symbolic model (Definition 10).

        Args:
            X: Input features.
            y: Observed targets.

        Returns:
            float: ``log_likelihood - 2 * complexity`` where complexity is
            the model's ``size`` (program length).
        """
        prediction = self.predict(X)
        error = prediction - y
        complexity = self.size
        log_likelihood = np.log(np.mean(error**2 / error.shape[0]))
        return log_likelihood - 2 * complexity

    def compute_bic(self, X, y):
        """Compute a BIC-like score for this symbolic model (Definition 10).

        Args:
            X: Input features.
            y: Observed targets.

        Returns:
            float: ``log_likelihood - log(n) * complexity`` where n is the
            number of observations and complexity is the model's ``size``.
        """
        prediction = self.predict(X)
        error = prediction - y
        log_likelihood = -np.mean(error**2 / error.shape[0])
        complexity = self.size
        return log_likelihood - np.log(y.shape[0]) * complexity


if __name__ == "__main__":
    h = Hypothesis()
    h.has_for_name = "first"
    h1 = Hypothesis()
    h.has_for_probability = 0.0
    h1.has_for_name = "second"
    h2 = Hypothesis()
    h2.has_for_name = "third"

    h.competes = [h1, h2]

    m = Model()
    print([i.has_for_name for i in h.competes])
    print([i.has_for_name for i in h1.INDIRECT_competes])
