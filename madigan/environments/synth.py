from madigan.synth import sine_generator

class Synth:
    def __init__(self, generator=None):
        self.data_generator = generator or iter(sine_generator(0., 1., 0.01))

    def render(self):
        res = next(self.data_generator)
        return res
