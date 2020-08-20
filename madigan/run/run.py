from madigan import synth




def main():
    gen = synth.sine_generator
    synth = synth.Synth(gen)
    for i in range(10):
        print(next(synth))


if __name__ == "__main__":
    main()
