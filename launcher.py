"""
launcher.py
Main script of the project
"""

import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Launcher for Venus Volcanoes solver.')
    parser.add_argument('--generate-dataset', action='store_true')
    parser.add_argument('--clean-dataset', action='store_true')
    parser.add_argument('-t', action='store_true')
    parser.add_argument('-m', type=str)
    parser.add_argument('-epochs', type=int)
    args = parser.parse_args()

    if args.generate_dataset:
        print('Generating dataset...')
    if args.clean_dataset:
        print('Removing dataset...')
    if args.t:
        print('Training network')
        epochs = parser.epochs