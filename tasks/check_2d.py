from core.cv_experiment import CrossValidationExperiment 
from sklearn.datasets import load_iris

dataset = load_iris() 
X = dataset.data 
y = dataset.target

def make_pipeline(fold_no, X_train, X_test):
    pass

experiment = CrossValidationExperiment(
    experiment_id = "Unnamed Experiment", 
    X = X, 
    y = y, 
    pipeline_fn = make_pipeline
)

experiment.run()