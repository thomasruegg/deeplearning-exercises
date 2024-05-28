class EarlyStoppingCounter:
    def __init__(self, patience=3, improvement_margin=0.0002):
        self._patience = patience
        self._improvement_margin = improvement_margin

        self.reset()

    def reset(self):
        self._best = 0.0
        self._count = 0

    def update(self, current):
        if current < self._best + self._improvement_margin:
            self._count += 1
        else:
            self._count = 0
            self._best = current

    def is_stopping_criteria_reached(self):
        return self._count > self._patience
