from utils.printing import Printing

class Performance(Printing): 
    def __init__(
        self,
        context = None,
        X = None, 
        y = None 
    ):
        # Context 
        self.context = context

        # Train and Test Split 
        self.X = X 
        self.y = y 

    def performance(self): 
        pass 