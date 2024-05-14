class EarlyStoppingCounter:
    def __init__(self, patience=0, improvement_margin=0.0):
        self._patience = patience
        self._improvement_margin = improvement_margin

        self.reset()

    def reset(self):
        self._best = 0.0
        self._count = 0

    def update(self, current):
        # TODO
        if current > self._best + self._improvement_margin:
            self._best = current
            self._count = 0
        else:
            self._count += 1

    def is_stopping_criteria_reached(self):
        # TODO
        return self._count >= self._patience