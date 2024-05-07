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
        pass

    def is_stopping_criteria_reached(self):
        # TODO
        return False
