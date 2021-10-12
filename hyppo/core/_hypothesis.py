import numpy as np

from hyppo.core._base import virtual_experiment_onto


class Hypothesis(virtual_experiment_onto.Artefact):
    namespace = virtual_experiment_onto.get_namespace("http://synthesis.ipi.ac.ru/virtual_experiment.owl")
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

    def _parse_specification(self):
        pass


class Model(virtual_experiment_onto.Artefact):
    namespace = virtual_experiment_onto.get_namespace("http://synthesis.ipi.ac.ru/virtual_experiment.owl")
    def __init__(self):
        pass

    def fit(self, X, y):
        pass

    def predict(self, X):
        pass


class NonLinearModel(Model):
    namespace = virtual_experiment_onto.get_namespace("http://synthesis.ipi.ac.ru/virtual_experiment.owl")

    def nonlinear_aic(self, X, y):
        prediction = self.predict(X)
        error = prediction - y
        complexity = self.size
        likehood = np.log(np.mean(error ** 2 / error.shape[0]))
        value = likehood - 2 * complexity
        return value

    def nonlinear_bic(self, X, y):
        prediction = self.predict(X)
        error = prediction - y
        std = np.std(error)
        likehood = -np.mean(error ** 2 / error.shape[0])
        complexity = self.size

        value = likehood - np.log(y.shape[0]) * complexity
        return value


def range_models(models, dataset, metrics, threshold):
    result = {}
    if metrics == 'mae':
        for key in models.keys():
            error = models.predict(dataset.X) - dataset.y
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


def update_bayesian_probability(model, dataset, prior_probability):

    likelihood = norm.cdf(model.predict(dataset),
                          model.predict(dataset).mean,
                          model.predict(dataset).std))
    unnormalized_posterior = prior_probability * likelihood
    posterior = unnormalized_posterior / np.nan_to_num(unnormalized_posterior).sum()

    return posterior

def compute_bayesian_hypothesis_score(models, dataset):
    X = dataset.X
    y = dataset.y
    sum_probability = 0
    result = {}
    for key in models.keys():
        error = y-models[key].predict(X)
        std = np.std(error)
        probability = 1/np.sqrt(2*std**2*np.pi)*np.exp(1/(2*std**2)*(error))
        probability = np.sum(probability)
        sum_probability += probability
        result[key] = probability
    return result

def compare_preds_on_single_dataset(models, dataset, stat_test):
    linear_prediction = models['model_1'].predict(dataset)
    gp_prediction = models['model_2'].predict(dataset)
    if stat_test == 'wilcoxon':
        result = scipy.stats.wilcoxon(linear_prediction, gp_prediction)
    else:
        raise NotImplemented()
    return result


def compare_preds_on_different_datasets(models, dataset_1, dataset_2, stat_test):
    linear_prediction = models['model_1'].predict(dataset_1)
    gp_prediction = models['model_2'].predict(dataset_2)
    if stat_test == 'wilcoxon':
        result = scipy.stats.wilcoxon(linear_prediction, gp_prediction)
    else:
        raise NotImplemented()
    return result

def compute_diff_dag(models=None, dataset):
    atlas = datasets.fetch_atlas_harvard_oxford('cort-maxprob-thr25-2mm')
    labels = atlas.labels[1:]

    res = []
    for i in range(47):
        for j in range(i + 1, 48):
            res.append(r2_score(X1[i], X2[j]))

    X1 = dataset[dataset.gender='man'].values # pd.read_csv('men.csv').values  # [:500]
    X2 = dataset[dataset.gender='woman'].values # pd.read_csv('women.csv').values  # [:500]
    changed_nodes = set(range(X1.shape[1]))

    # run DCI algorithm 1 (to limit the number of hypotheses tested limit max_set_size to 3)
    print('Estimating skeleton of the difference DAG (running Algorithm 2 of DCI)...')
    print('This might take a bit of time...')
    if models is None:
        retained_edges, _, _, _ = dci.estimate_ddag_skeleton(X1,
                                                             X2,
                                                             temp,
                                                             changed_nodes,
                                                             0.98,
                                                             max_set_size=20,
                                                             verbose=False)
    else:
        retained_edges = models
    # save results
    RES_FOLDER = './results/'
    sys_utils.ensure_dirs([RES_FOLDER])
    yaml.dump(retained_edges, open(RES_FOLDER + 'estimated_ddag_skeleton.yaml', 'w'), indent=2)

    # run DCI algorithm 2
    print('Assigning edge directions (running Algorithm 3 of DCI)...')
    print('This might take a bit of time...')
    est_ddag = dci.estimate_ddag(X1, X2, retained_edges, changed_nodes, 0.005, max_set_size=3, verbose=False)
    # save results
    yaml.dump(est_ddag, open(RES_FOLDER + 'estimated_ddag.yaml', 'w'), indent=2)

    print('Plot the graph if desired (requires graphviz, pydot and networkx packages in python)')
    g = _make_graph(retained_edges, est_ddag, labels)
    fn = RES_FOLDER + 'graph_thres.gv'
    nx.nx_pydot.write_dot(g, fn)
    graphviz.render('dot', 'png', fn)
    return est_ddag

def _paral_train(men, women):
    numbers = range(48)#[:6]
    a = []
    for i in numbers:
        region_to_predict = i
        men_X = men.drop(['x'+str(region_to_predict)], axis=1).values
        men_y = men['x'+str(region_to_predict)].values
        women_X = women.drop(['x'+str(region_to_predict)], axis=1).values
        women_y = women['x'+str(region_to_predict)].values
        a.append([men_X, men_y, i, 'men'])
        a.append([women_X, women_y, i, 'women'])
    with Pool() as pool:
        result = pool.starmap(_train_models, a)

def _compare_AIC_BIC_linear_GP(models, dataset):
    linear_aic = models['linear'].aic
    gp_aic = get_AIC_nonlinear(models['GP'], dataset)
    linear_bic = models['linear'].bic
    gp_bic = get_BIC_nonlinear(models['GP'], dataset)
    result = {'AIC': {'linear':linear_aic, 'GP':gp_aic},
              'BIC': {'linear':linear_bic, 'GP':gp_bic}}
    return result

def _determine_changed_nodes(mn_diff):
    # determine which nodes have any change from state 1 to state 2
    # a node i is included in the changed node set if there exists a node j such that
    # precision(i,j) in state 1 is not equal to precision(i,j) in state 2
    # markov (undirected) difference graph gives how precision matrix changed across the two states
    # i.e a zero for entry (i,j) means that there is no change in precision(i,j) across the two states
    return set(np.where(np.sum(mn_diff, axis=0) != 0)[0])

def _read_data_graph(center=True):
    # read in the data and center the data around 0 if center is True (default)
    X1 = pd.read_csv(filename1, delimiter=',', index_col=0).T
    X2 = pd.read_csv(filename2, delimiter=',', index_col=0).T
    gene_names = X1.columns.values

    if center:
        X1 = X1 - X1.mean(axis=0)
        X2 = X2 - X2.mean(axis=0)

    # read in markov difference network
    mn_diff = np.loadtxt(difference_undirected_filename, delimiter=',')
    # determine which nodes changed
    changed_nodes = _determine_changed_nodes(mn_diff)
    # get all edges with nonzero precision
    est_dug = set(math_utils.upper_tri_ixs_nonzero(mn_diff))
    return X1.values, X2.values, est_dug, changed_nodes, gene_names

def _make_graph(skeleton, oriented_edges, gene_names, known_edges=set()):
    # create a graph for plotting
    unoriented_edges = skeleton - oriented_edges - {(j, i) for i, j in oriented_edges}
    # make a directed graph
    g = nx.DiGraph()
    for i, j in oriented_edges:
        color = 'black'
        g.add_edge(gene_names[i], gene_names[j], color=color, penwidth=3)

    for i, j in unoriented_edges:
        color = 'black'
        g.add_edge(gene_names[i], gene_names[j], arrowhead='none', color=color, penwidth=3)

    for i, j in known_edges - oriented_edges - unoriented_edges:
        if (i, j) not in skeleton and (j, i) not in skeleton:
            g.add_edge(gene_names[i], gene_names[j], arrowhead='none', color='gray')
    return g

def _train_models(X_train, y_train, roi_number, gender):
    '''

    :param X_train:
    :param y_train:
    :param roi_number:
    :param gender:
    :return:
    '''
    ols = sm.regression.linear_model.OLS(y_train, X_train)
    ols = ols.fit()
    print('linear fitted')
    gp = SymbolicRegressor(population_size=1000,
                           tournament_size=100,
                           generations=300, stopping_criteria=0.001,
                           const_range=(-1, 1),
                           p_crossover=0.7, p_subtree_mutation=0.12,
                           p_hoist_mutation=0.06, p_point_mutation=0.12,
                           p_point_replace=1,
                           init_depth=(6, 10),
                           function_set=('mul', 'sub', 'div', 'add'),
                           max_samples=0.8,
                           verbose=0,
                           metric='mse',
                           parsimony_coefficient=0.00005,
                           random_state=0,
                           n_jobs=1)
    gp.fit(X_train, y_train)
    with open('./functions/{0}_region_{1}.pickle'.format(gender, roi_number), 'wb') as f:
        pickle.dump({'linear':ols, 'GP':gp}, f)
    print('GP fitted')
    return {'linear':ols, 'GP':gp}




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
