import numpy as np
from scipy import stats
from sklearn.metrics import r2_score, mean_squared_error

from hyppo.core._base import virtual_experiment_onto


with virtual_experiment_onto:
    class Hypothesis(virtual_experiment_onto.Artefact):
        namespace = virtual_experiment_onto.get_namespace(
            "http://synthesis.ipi.ac.ru/virtual_experiment.owl")
    # def __init__(self):
    #     self._parameters = dict()

        @property
        def parameters(self):
            return self._parameters

        @parameters.setter
        def parameters(self, value):
            self._parameters = value

        @parameters.getter
        def parameters(self):
            return self._parameters

        def range_models(self, X, y, metrics, threshold):
            models = self.is_implemented_by_model()
            result = {}

            if metrics == 'mae':
                for key in models.keys():
                    error = models.predict(X) - y
                    if error <= threshold:
                        result[key] = np.mean(np.abs(error))

            elif metrics == 'r2':
                for key in models.keys():
                    error = r2_score(y, models[key].predict(X))
                    if error >= threshold:
                        result[key] = error

            elif metrics == 'mse':
                for key in models.keys():
                    error = mean_squared_error(y, models[key].predict(X))
                    if error <= threshold:
                        result[key] = error

        # TODO add several > 2 models
        def compare_preds_on_single_dataset(self, dataset, stat_test):
            models = self.is_implemented_by_model()
            linear_prediction = models['model_1'].predict(dataset)
            gp_prediction = models['model_2'].predict(dataset)
            if stat_test == 'wilcoxon':
                result = stats.wilcoxon(linear_prediction, gp_prediction)
            else:
                raise NotImplemented()
            return result

        def compare_preds_on_different_datasets(self, dataset_1, dataset_2, stat_test):
            models = self.is_implemented_by_model()
            linear_prediction = models['model_1'].predict(dataset_1)
            gp_prediction = models['model_2'].predict(dataset_2)
            if stat_test == 'wilcoxon':
                result = stats.wilcoxon(linear_prediction, gp_prediction)
            else:
                raise NotImplemented()
            return result


class Model(virtual_experiment_onto.Artefact):
    namespace = virtual_experiment_onto.get_namespace("http://synthesis.ipi.ac.ru/virtual_experiment.owl")
    def __init__(self):
        pass

    def fit(self, X, y):
        pass

    def predict(self, X):
        pass

    def update_bayesian_probability(self, X, prior):

        likelihood = stats.norm.cdf(self.predict(X),
                                    self.predict(X).mean,
                                    self.predict(X).std)

        unnormalized_posterior = prior * likelihood
        posterior = unnormalized_posterior / np.nan_to_num(unnormalized_posterior).sum()
        return posterior

    def bayesian_score(self, X, y):
        error = y - self.predict(X)
        std = np.std(error)
        probability = np.sum(1/np.sqrt(2*std**2*np.pi)*np.exp(1/(2*std**2)*(error)))
        return probability


class NonLinearModel(Model):
    namespace = virtual_experiment_onto.get_namespace("http://synthesis.ipi.ac.ru/virtual_experiment.owl")

    def fit(self, X, y):
        est_gp = SymbolicRegressor(population_size=1000,
                                   tournament_size=20,
                                   generations=150, stopping_criteria=0.001,
                                   const_range=(-1, 1),
                                   p_crossover=0.7, p_subtree_mutation=0.12,
                                   p_hoist_mutation=0.06, p_point_mutation=0.12,
                                   p_point_replace=1,
                                   init_depth=(6, 10),
                                   function_set=('mul', 'sub', 'div', 'add', 'cos'),
                                   max_samples=0.9,
                                   verbose=1,
                                   metric='mse',
                                   parsimony_coefficient=0.0005,
                                   random_state=0,
                                   n_jobs=1)

        est_gp.fit(X, y)
        return est_gp

    @property
    def nonlinear_aic(self, X, y):
        prediction = self.predict(X)
        error = prediction - y
        complexity = self.size
        likehood = np.log(np.mean(error ** 2 / error.shape[0]))
        value = likehood - 2 * complexity
        return self._nonlinear_aic

    @property
    def nonlinear_bic(self, X, y):
        prediction = self.predict(X)
        error = prediction - y
        likehood = -np.mean(error ** 2 / error.shape[0])
        complexity = self.size

        value = likehood - np.log(y.shape[0]) * complexity
        return self._nonlinear_bic


if __name__ == '__main__':
    h = Hypothesis()
    h.has_for_name = 'first'
    h1 = Hypothesis()
    h.has_for_probability = 0.0
    h1.has_for_name = 'second'
    h2 = Hypothesis()
    h2.has_for_name = 'third'

    h.competes = [h1, h2]

    m = Model()
    print([i.has_for_name for i in h.competes])
    print([i.has_for_name for i in h1.INDIRECT_competes])
