import argparse
import datetime
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import sys
import time
from pathlib import Path
from surprise import Dataset, KNNBasic, accuracy
from surprise.model_selection import train_test_split

import tensorflow as tf
from sklearn.model_selection import train_test_split as sk_train_test_split
from sklearn.metrics import mean_squared_error as MSE


def log(logfile, s):
    """ Log a string into a file and print it. """
    with open(logfile, 'a', encoding='utf8') as f:
        f.write(str(s))
        f.write("\n")
    print(s)


def get_args_parser():
    parser = argparse.ArgumentParser(fromfile_prefix_chars='@')
    parser.add_argument(
        "-ds",
        "--dataset",
        default="ml-100k",
        help="MovieLens dataset: 'ml-100k' or 'ml-1m'"
    )
    parser.add_argument(
        "-s",
        "--seed",
        default=1910299034,
        help="Random seed."
    )
    parser.add_argument(
        "-od",
        "--outdir",
        default='results/'
    )

    return parser


###
# Autoencoder idea by Soumya Ghosh.
# "Recommender system on the Movielens dataset using an Autoencoder and Tensorflow in Python"
# https://medium.com/@connectwithghosh/recommender-system-on-the-movielens-using-an-autoencoder-using-tensorflow-in-python-f13d3e8d600d
##
def autoencoder(dataset, logfile, random_state=1910299034):
    # Save home path
    home = str(Path.home())

    # Hyperparameters
    hidden_layer_nodes = 32
    learn_rate = 0.001

    # Load the MovieLens (download it if needed)
    if dataset == 'ml-100k':
        datafile = 'u.data'
        input_layer_nodes = 1682
        output_layer_nodes = input_layer_nodes
        ratings = pd.read_csv('{}/.surprise_data/{}/{}/{}'.format(home, dataset, dataset, datafile), sep="\t",
                              header=None,
                              engine='python')
        batch_size = 20
        epochs = 200
    else:
        datafile = 'ratings.dat'
        input_layer_nodes = 3706
        output_layer_nodes = input_layer_nodes
        ratings = pd.read_csv('{}/.surprise_data/{}/{}/{}'.format(home, dataset, dataset, datafile), sep="::",
                              header=None,
                              engine='python')
        batch_size = 80
        epochs = 100

    # Create DataFrame without timestamps
    ratings_pivot = pd.pivot_table(ratings[[0, 1, 2]], values=2, index=0, columns=1).fillna(0)

    # 80-20 split
    X_train, X_test = sk_train_test_split(ratings_pivot, test_size=0.2, random_state=random_state)

    # Initialize weights
    hidden_layer_weights = {
        'weights': tf.Variable(tf.random_normal([input_layer_nodes + 1, hidden_layer_nodes], seed=random_state))}
    output_layer_weights = {
        'weights': tf.Variable(tf.random_normal([hidden_layer_nodes + 1, output_layer_nodes], seed=random_state))}

    # Set input placeholder
    input_layer = tf.placeholder('float', [None, input_layer_nodes])

    # Add bias to input
    bias = tf.fill([tf.shape(input_layer)[0], 1], 1.0)
    input_layer_concat = tf.concat([input_layer, bias], 1)

    # Forward and activate with Sigmoid
    hidden_activations = tf.nn.sigmoid(tf.matmul(input_layer_concat, hidden_layer_weights['weights']))

    # Add bias
    bias = tf.fill([tf.shape(hidden_activations)[0], 1], 1.0)
    hidden_activations = tf.concat([hidden_activations, bias], 1)

    # Forward for final output
    output_layer = tf.matmul(hidden_activations, output_layer_weights['weights'])

    # Set output placeholder
    output_true = tf.placeholder('float', [None, output_layer_nodes])

    # Loss
    mse_loss = tf.reduce_mean(tf.square(output_layer - output_true))

    # Optimizer
    optimizer = tf.train.AdamOptimizer(learn_rate).minimize(mse_loss)

    # Tensorflow session initialization
    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)

    # Running model
    for epoch in range(epochs):
        epoch_loss = 0

        for i in range(int(X_train.shape[0] / batch_size)):
            batch_X = X_train[i * batch_size: (i + 1) * batch_size]
            _, c = sess.run([optimizer, mse_loss], feed_dict={input_layer: batch_X, output_true: batch_X})
            epoch_loss += c

        output_train = sess.run(output_layer, feed_dict={input_layer: X_train})
        output_test = sess.run(output_layer, feed_dict={input_layer: X_test})

        log(logfile, 'MSE train ' + str(round(MSE(output_train, X_train), 2)) + ' MSE test ' + str(
            round(MSE(output_test, X_test), 2)))
        log(logfile, 'Epoch ' + str(epoch) + '/' + str(epochs) + ' loss: ' + str(round(epoch_loss, 2)))

    # Final test
    time_start = time.time()
    output_test = sess.run(output_layer, feed_dict={input_layer: X_test})
    time_stop = time.time()
    runtime = round(time_stop - time_start, 4)
    log(logfile, 'Test time: {0:f}'.format(runtime).strip('0'))
    mse = round(MSE(output_test, X_test), 3)
    log(logfile, 'MSE test: ' + str(mse) + '\n')
    return [mse, runtime]


def experiments(config_file):
    args = get_args_parser().parse_args(['@' + config_file])

    # Set seed
    np.random.seed(int(args.seed))

    # Construct output directory
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    outdir = args.outdir + str(args.dataset) + "/" + timestamp + '/'

    # Create results directory
    outdir_path = Path(outdir)
    if not outdir_path.is_dir():
        os.makedirs(outdir)

    # Logging
    logfile = outdir + 'log.txt'
    log(logfile, "Directory " + outdir + " created.")

    # Set dataset
    if str(args.dataset) == 'ml-100k':
        dataset_name = 'MovieLens 100K'
    else:
        dataset_name = 'MovieLens 1M'

    # Load the MovieLens dataset (download it if needed),
    data = Dataset.load_builtin(str(args.dataset))

    # 80-20 split
    train_dataset, test_dataset = train_test_split(data, test_size=.20, random_state=int(args.seed))

    # Run Autoencoder
    [a_mse, a_runtime] = autoencoder(str(args.dataset), logfile, int(args.seed))

    # Set algorithms
    user_based_msd_sim_options = {'name': 'msd', 'user_based': True}
    user_based_pearson_baseline_sim_options = {'name': 'pearson_baseline', 'user_based': True}
    user_based_msd_algo = KNNBasic(sim_options=user_based_msd_sim_options)
    user_based_pearson_baseline_algo = KNNBasic(sim_options=user_based_pearson_baseline_sim_options)

    item_based_sim_options = {'name': 'msd', 'user_based': False}
    item_based_pearson_baseline_sim_options = {'name': 'pearson_baseline', 'user_based': False}
    item_based_msd_algo = KNNBasic(sim_options=item_based_sim_options)
    item_based_pearson_baseline_algo = KNNBasic(sim_options=item_based_pearson_baseline_sim_options)

    algorithms = (
        ("User MSD", user_based_msd_algo),
        ("User Pearson Baseline", user_based_pearson_baseline_algo),
        ("Item MSD", item_based_msd_algo),
        ("Item Pearson Baseline", item_based_pearson_baseline_algo),
    )

    # Plotting
    plt.style.use('dark_background')
    fig, ax = plt.subplots()

    # Autoencoder results
    runtimes = [a_runtime]
    mses = [a_mse]
    # ax.annotate("Autoencoder", (runtimes[0] + .001, mses[0] + .001))

    # Running
    for name, algorithm in algorithms:
        log(logfile, dataset_name + ", " + name)

        # Train
        time_start = time.time()
        algorithm.fit(train_dataset)
        time_stop = time.time()
        log(logfile, 'Train time: {0:f}'.format(round(time_stop - time_start, 2)).strip('0'))

        # Test
        time_start = time.time()
        predictions = algorithm.test(test_dataset)
        time_stop = time.time()
        runtime = round(time_stop - time_start, 2)
        runtimes += [runtime]
        log(logfile, 'Test time: {0:f}'.format(runtime).strip('0'))

        # MSE metric
        mse = accuracy.mse(predictions, verbose=False)
        mses += [mse]
        log(logfile, 'MSE: {0:1.4f}\n'.format(mse))

    # Draw scatter plot
    ax.scatter(runtimes[1:], mses[1:], marker='x', color='red')
    # ax.scatter(runtimes, mses, marker='x', color='red')

    # Annotate scatter plot, i=0 is for Autoencoder
    for i, (name, _) in enumerate(algorithms):
        ax.annotate(name, (runtimes[i + 1] + .001, mses[i + 1] + .001))

    # Set plot settings
    plt.title("{}".format(dataset_name), size=15)
    plt.xlabel('Runtime (s)')
    plt.ylabel('MSE')

    # Save plot
    plt.savefig(outdir + 'plot.png', bbox_inches='tight')


if __name__ == "__main__":
    experiments(config_file=sys.argv[1])
