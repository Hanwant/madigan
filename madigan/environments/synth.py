from madigan.synth import sine_generator

class Synth:
    def __init__(self, generator=None, init_cash=10000,  levarage=1):
        self.data_generator = generator or iter(sine_generator(0., 1., 0.01))
        self.cash = init_cash
        self.levarage = levarage

    def render(self):
        res = next(self.data_generator)
        return res
