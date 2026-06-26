from sklearn.model_selection import RepeatedStratifiedKFold
from imblearn.pipeline import Pipeline
import pandas as pd

class CrossValidationExperiment: 
    def __init__(
        self, 

        # Experiment Id 
        experiment_id = None,

        # Data
        X = None, 
        y = None, 

        # Sampling Configuration 
        holdout_size = 0.2, 
        n_splits = 10, 
        n_repeats = 1,

        # Random Seed 
        random_state = 42,

        # Pipeline Generator Function 
        pipeline_fn = None 

    ): 
        # Experiment ID 
        self.experiment_id = experiment_id

        # Data 
        self.X = pd.DataFrame(X) 
        self.y = pd.DataFrame(y) 

        # Sampling Configuration 
        self.holdout_size = holdout_size
        self.n_splits = n_splits
        self.n_repeats = n_repeats 

        # Cross Validator 
        self.cv = RepeatedStratifiedKFold(
            n_splits=self.n_splits, 
            n_repeats=n_repeats,
            random_state=random_state
        )

        # Random Seed 
        self.random_state = random_state

        # Pipeline Generator Function 
        self.pipeline_fn = pipeline_fn

    def run(self): 
        print(f"#" * 80)
        print(f"Experiment: {self.experiment_id}")
        print(f"#" * 80)

        # Loop over splits.
        fold_no = 0
        for train_index, test_index in self.cv.split(self.X, self.y): 

            # Split X_train and X_test
            X_train, X_test = self.X.loc[train_index], self.X.loc[test_index] 
            y_train, y_test = self.y.loc[train_index], self.y.loc[test_index] 

            # Create Pipeline
            pipeline = self.pipeline_fn(fold_no, X_train, X_test) 

            # Run Split 
            # ...

            # Increment fold number.
            fold_no +=1 



