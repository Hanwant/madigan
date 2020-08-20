from madigan.synth import Synth, sine_generator




def main():
    gen = sine_generator
    synth = Synth(gen)
    for i in range(10):
        print(next(synth))


if __name__ == "__main__":
    main()
