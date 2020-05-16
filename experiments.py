import argparse
import datetime
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import sys
import time
from pathlib import Path
from sklearn.datasets import load_iris, load_breast_cancer
from sklearn.preprocessing import StandardScaler


def log(logfile, s):
    """ Log a string into a file and print it. """
    with open(logfile, 'a', encoding='utf8') as f:
        f.write(str(s))
        f.write("\n")
    print(s)


def get_args_parser():
    parser = argparse.ArgumentParser(fromfile_prefix_chars='@')
    parser.add_argument(
        "-dp",
        "--datapath",
        default="data/ml-100k/u.data",
        help="Path of data folder. For example: data/ml-100k/u.data"
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


def experiments(config_file):
    args = get_args_parser().parse_args(['@' + config_file])

    # Set seed
    np.random.seed(int(args.seed))

    # Construct output directory
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    outdir = args.outdir + timestamp + '/'

    # Create results directory
    outdir_path = Path(outdir)
    if not outdir_path.is_dir():
        os.makedirs(outdir)

    # Logging
    logfile = outdir + 'log.txt'
    log(logfile, "Directory " + outdir + " created.")

    # Read dataset
    feature_names = ["user_id", "movie_id", "rating", "timestamp"]
    df = pd.read_csv(args.datapath, sep='\t', names=feature_names)

    # Set plot settings
    plt.figure(figsize=(7 * 2 + 6, 12.5))
    plt.subplots_adjust(left=.02, right=.98, bottom=.001, top=.96, wspace=.05, hspace=.01)
    plt.style.use('dark_background')
    plot_num = 1

    datasets = (
        (3, load_iris(return_X_y=True), "Iris"),
        (2, load_breast_cancer(return_X_y=True), "Breast Cancer"),
        (2, noisy_circles, "Noisy Circles")
    )

    # Traverse datasets
    # High-level abstraction is from https://scikit-learn.org/stable/modules/clustering.html
    for i, (n_clusters, dataset, dataset_name) in enumerate(datasets):
        X, y = dataset

        # Normalization of features for easier parameter selection
        X = StandardScaler().fit_transform(X)

        connectivity = kneighbors_graph(X, n_neighbors=10, include_self=False)
        # connectivity = 0.5 * (connectivity + connectivity.T)  # Make connectivity symmetric

        average_linkage = cluster.AgglomerativeClustering(
            linkage="average",
            affinity="cityblock",
            n_clusters=n_clusters,
            connectivity=connectivity)

        ward_linkage = cluster.AgglomerativeClustering(
            linkage="ward",
            n_clusters=n_clusters)

        complete_linkage = cluster.AgglomerativeClustering(
            linkage="complete",
            n_clusters=n_clusters)

        single_linkage = cluster.AgglomerativeClustering(
            linkage="single",
            n_clusters=n_clusters)

        k_means = cluster.KMeans(n_clusters=n_clusters)

        gaussian_mixture = mixture.GaussianMixture(
            n_components=n_clusters,
            covariance_type='full')

        # Set techniques
        techniques = (
            ('Agglomerative Avg', average_linkage),
            ('Agglomerative Single', single_linkage),
            ('Agglomerative Complete', complete_linkage),
            ('Agglomerative Ward', ward_linkage),
            ('kMeans', k_means),
            ('GaussianMixture', gaussian_mixture),
        )

        for name, technique in techniques:
            log(logfile, dataset_name + ", " + name)

            time_start = time.time()

            # Catch warnings related to kneighbors_graph
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message="the number of connected components of the " +
                            "connectivity matrix is [0-9]{1,2}" +
                            " > 1. Completing it to avoid stopping the tree early.",
                    category=UserWarning)
                warnings.filterwarnings(
                    "ignore",
                    message="Graph is not fully connected, spectral embedding" +
                            " may not work as expected.",
                    category=UserWarning)
                technique.fit(X)

            time_stop = time.time()

            # Predictions
            if hasattr(technique, 'labels_'):
                y_pred = technique.labels_.astype(np.int)
            else:
                y_pred = technique.predict(X)

            # Entropy metric
            true_cluster_labels = [y[get_cluster_indices(c, y_pred)] for c in range(n_clusters)]
            overall_entropy = get_overall_entropy(true_cluster_labels, y.shape[0])

            # F-Score metric
            f1_score = metrics.f1_score(y, y_pred, average='weighted')

            log(logfile, "\tOverall entropy: " + str(round(overall_entropy, 3)))
            log(logfile, "\tF1 Score: " + str(round(f1_score, 3)))

            # Plotting
            plt.subplot(len(datasets), len(techniques), plot_num)
            if i == 0:
                plt.title("{}".format(name), size=15)

            colors = np.array(list(islice(cycle(['#377eb8', '#ff7f00', '#4daf4a']), int(max(y_pred) + 1))))
            colors = np.append(colors, ["#000000"])  # Add black color for outliers (if any)
            plt.scatter(X[:, 0], X[:, 1], s=10, color=colors[y_pred], alpha=0.60)

            plt.xlim(-2.5, 2.5)
            plt.ylim(-2.5, 2.5)
            plt.xticks(())
            plt.yticks(())

            plt.text(.15, .01, ('%.2fs' % (time_stop - time_start)).lstrip('0'),
                     transform=plt.gca().transAxes, size=15,
                     horizontalalignment='right')

            plt.text(.99, .07, ('%.2f' % (overall_entropy)).lstrip('0'),
                     transform=plt.gca().transAxes, size=15,
                     horizontalalignment='right')
            plt.text(.99, .01, ('%.2f' % (f1_score)).lstrip('0'),
                     transform=plt.gca().transAxes, size=15,
                     horizontalalignment='right')

            plot_num += 1

    # Plotting
    plt.savefig(outdir + 'plot.png', bbox_inches='tight')


if __name__ == "__main__":
    experiments(config_file=sys.argv[1])
