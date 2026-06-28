class EarlyStopping:
    def __init__(self, patience=5, min_delta=0.0):
        """
        Initializes the EarlyStopping object.
        
        Args:
            patience (int): Number of epochs to wait after last improvement before stopping.
            min_delta (float): Minimum change in the monitored metric to qualify as an improvement.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, metric):
        """
        Checks if training should be stopped based on the monitored metric.
        
        Args:
            metric (float): The current value of the monitored metric (e.g., validation loss).
        """
        if self.best_score is None or metric < self.best_score - self.min_delta:
            self.best_score = metric
            self.counter = 0
        else:
            self.counter += 1
            print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True