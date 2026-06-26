
class SplitRun: 
    def __init__(
        self, 

        # Verbose Logging 
        verbose = False,
        indent = 0,

        # Run ID
        id_ = None,

        # Pipeline
        pipeline = None,

        # Splits 
        X_train = None, 
        X_test = None,
        y_train = None, 
        y_test = None,

        # Options 
        plot_decision_boundary = False
    ): 
        # Verbose Logging 
        self.verbose = verbose 
        self.indent = indent

        # Run ID 
        self.id = id_ 
        
        # Splits
        self.X_train = X_train 
        self.X_test = X_test
        self.y_train = y_train 
        self.y_test = y_test 

        # Options 
        self.plot_decision_boundary = plot_decision_boundary

    def train(self): 
        pass 

    def test(self):
        pass 
    
     