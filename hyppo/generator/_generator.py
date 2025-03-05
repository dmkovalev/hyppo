import random
import string
from sympy import Symbol
from hyppo.coa._base import Structure, Equation

def generate_hypothesis():
    pass


def _generate_single_hypothesis():
    pass


def create_random_complete_structure(num_vars=5, min_vars_per_eq=1, max_vars_per_eq=3, exogenous_ratio=0.2):
    """
    Create a random complete structure with the specified number of variables.
    
    Args:
        num_vars (int): Number of variables in the structure
        min_vars_per_eq (int): Minimum number of variables per equation
        max_vars_per_eq (int): Maximum number of variables per equation
        exogenous_ratio (float): Ratio of exogenous variables (equations with only one variable)
        
    Returns:
        Structure: A complete structure with random equations
    """
    # Create variable symbols
    variables = [Symbol(f"x_{i+1}") for i in range(num_vars)]
    
    # Determine how many exogenous variables to create
    num_exogenous = max(1, int(num_vars * exogenous_ratio))
    
    # Create equations
    equations = []
    
    # Create exogenous variable equations (equations with only one variable)
    for i in range(num_exogenous):
        var = variables[i]
        formula = f"f_{i+1}({var})=0"
        equations.append(Equation(formula=formula))
    
    # Create remaining equations to ensure completeness
    for i in range(num_exogenous, num_vars):
        # Determine how many variables to include in this equation
        num_vars_in_eq = random.randint(min_vars_per_eq, min(max_vars_per_eq, num_vars))
        
        # Ensure the equation includes the current variable
        eq_vars = [variables[i]]
        
        # Add additional random variables
        available_vars = [v for v in variables if v != variables[i]]
        additional_vars = random.sample(available_vars, min(num_vars_in_eq - 1, len(available_vars)))
        eq_vars.extend(additional_vars)
        
        # Create a formula string
        var_terms = [f"{var}" for var in eq_vars]
        formula = f"{'+'.join(var_terms)}=0"
        
        equations.append(Equation(formula=formula))
    
    # Create and return the structure
    return Structure(equations=equations)

def create_random_complete_structures(num_structures=5, min_vars=3, max_vars=10, **kwargs):
    """
    Create multiple random complete structures.
    
    Args:
        num_structures (int): Number of structures to create
        min_vars (int): Minimum number of variables per structure
        max_vars (int): Maximum number of variables per structure
        **kwargs: Additional arguments to pass to create_random_complete_structure
        
    Returns:
        list: A list of complete structures with random equations
    """
    structures = []
    for _ in range(num_structures):
        num_vars = random.randint(min_vars, max_vars)
        structures.append(create_random_complete_structure(num_vars=num_vars, **kwargs))
    return structures




def generate_hypothesis_from_data(dataset):
    # generate with perceptron
    X_train, y_train, X_test, y_test = train_test_split(dataset.data, dataset.target)

    loses_by_epoch = {}
    data = mens
    print(data.shape)

    loses_by_epoch[region] = list()

    X_train_torch = torch.FloatTensor(X_train).to(device)
    y_train_torch = torch.FloatTensor(y_train).to(device)

    X_test_torch = torch.FloatTensor(X_test).to(device)
    y_test_torch = torch.FloatTensor(y_test).to(device)

    model = torch.nn.Sequential(
        torch.nn.Linear(47, 100),
        torch.nn.Softplus(),
        torch.nn.Linear(100, 50),
        torch.nn.Softplus(),
        torch.nn.Linear(50, 25),
        torch.nn.Tanh(),
        torch.nn.Linear(25, 1),
    ).to(device)
    loss_fn = torch.nn.MSELoss(reduction='mean')

    learning_rate = 1e-2
    batch_size = 8096
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    for t in range(40000):
        for batch in range(0, X_train.shape[0], batch_size):
            y_pred_train = model(X_train_torch[batch:batch + batch_size])

            loss_train = loss_fn(y_pred_train,
                                 y_train_torch.reshape(-1, 1)[batch:batch + batch_size])

            optimizer.zero_grad()
            loss_train.backward()
            optimizer.step()

        train_predict = model(X_train_torch).detach().cpu().numpy()
        test_predict = model(X_test_torch).detach().cpu().numpy()
        r2_train = r2_score(np.ravel(train_predict), np.ravel(y_train))
        r2_test = r2_score(np.ravel(test_predict), np.ravel(y_test))
        mse_train = mean_squared_error(np.ravel(train_predict), np.ravel(y_train))
        mse_test = mean_squared_error(np.ravel(test_predict), np.ravel(y_test))
        loses_by_epoch[region].append([r2_train, r2_test, mse_train, mse_test])

        # generate with GP
        model_gp = SymbolicRegressor(population_size=1000,
                                   tournament_size=20,
                                   generations=200, stopping_criteria=0.001,
                                   const_range=(-3, 3),
                                   p_crossover=0.7, p_subtree_mutation=0.12,
                                   p_hoist_mutation=0.06, p_point_mutation=0.12,
                                   p_point_replace=1,
                                   init_depth=(10, 18),
                                   function_set=('mul', 'sub', 'div', 'add', 'sin'),
                                   max_samples=0.9,
                                   verbose=1,
                                   metric='mse',
                                   parsimony_coefficient=0.00001,
                                   random_state=0,
                                   n_jobs=20)

        model_gp.fit(X_train, y_train)

        best_model = _compare_scores(model, model_gp)
        return best_model