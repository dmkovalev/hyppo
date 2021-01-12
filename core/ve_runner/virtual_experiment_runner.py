def run_virtual_experiment(ve):
    raised = False
    exception_msg = None
    try:

        workflow = ve.get_workflow()

        config = json.load(open(model_path + '_config.json', 'r'))
        dataset = db._load_dataset()
        for task in workflow:

            hypothesis = ve.get_hypothesis(task)
            if hypothesis is None:
                hypothesis = ve_manager.generate_hypothesis_from_data(dataset)
            for model in get_corresponding_models(hypothesis, ve):
                model.execute(dataset)
                scores = model.evaluate(G)
                print("task launched with model:", model)

                print(range_models(models, zip(X, y)), 'r2', 0.7)
                print(str(key) + 'AIC: ' + get_AIC(models, dataset))
                print(str(key) + 'BIC: ' + get_BIC(models, dataset))
                print(str(key) + 'AIC_nonlinear: ' + get_AIC_nonlinear(models, dataset))
                print(str(key) + 'BIC_nonlinear: ' + get_BIC_nonlinear(models, dataset))

                print(compute_bayesian_hypothesis_score(models, dataset))


    except Exception as e:
        raised = True
        exception_msg = e
    self.assertFalse(raised, f'Exception raised, exception - {exception_msg}')
