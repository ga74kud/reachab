import os
import argparse
def main():
        parser = argparse.ArgumentParser()
        parser.add_argument('envname', type=str)
        parser.add_argument('--render', action='store_true')
        parser.add_argument("--max_timesteps", type=int)
        args = parser.parse_args()


if __name__ == '__main__':
        main()