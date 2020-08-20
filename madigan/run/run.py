from madigan.environments import Synth

def main():
    env = Synth()
    for i in range(10):
        print(env.render())

if __name__ == "__main__":
    main()
