import argparse
import json
import matplotlib.pyplot as plt
import numpy as np

def plot_record(record):
    loss = record['loss']
    correct = record['correct']

    loss_steps = len(loss['train'])
    correct_steps = len(correct['train'])

    x_train = range(loss_steps)
    x_valid = np.linspace(0, loss_steps, len(loss['valid']))

    # figure 1
    plt.plot(x_train, loss['train'], label='Training')
    plt.plot(x_valid, loss['valid'], label='Validation')
    plt.xlabel('steps')
    plt.ylabel('loss')
    plt.legend()
    plt.show()

    # figure 2
    plt.plot(x_train, correct['train'], label='Training')
    plt.plot(x_valid, correct['valid'], label='Validation')
    plt.xlabel('steps')
    plt.ylabel('correct')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Dataset preprocessing program.')
    parser.add_argument('--record', type=str, required=True, help='Record file to save path.')
    args = parser.parse_args()

    with open(args.record, 'r') as f:
        record = json.load(f)
        plot_record(record)