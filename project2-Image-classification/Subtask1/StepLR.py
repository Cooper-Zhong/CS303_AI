class StepLR:
    def __init__(self, step_size, gamma=0.1):
        self.step_size = step_size
        self.gamma = gamma

    def __call__(self, iteration, initial_lr):
        return initial_lr * (self.gamma ** (iteration // self.step_size))
