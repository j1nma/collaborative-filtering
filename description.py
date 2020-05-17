import datetime
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import subprocess
import sys
from experiments import get_args_parser, log
from pathlib import Path


def descriptions(config_file):
    args = get_args_parser().parse_args(['@' + config_file])

    # Set seed
    np.random.seed(int(args.seed))

    # Construct output directory
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    outdir = args.outdir + "description/" + timestamp + '/'

    # Create results directory
    outdir_path = Path(outdir)
    if not outdir_path.is_dir():
        os.makedirs(outdir)

    # Logging
    logfile = outdir + 'log.txt'
    log(logfile, "Directory " + outdir + " created.")

    # Set dataset
    dataset_name = "MovieLens100k"
    feature_names = ["user_id", "movie_id", "rating", "timestamp"]
    df = pd.read_csv(args.datapath, sep='\t', names=feature_names, engine='python')

    # Dataset analysis
    log(logfile, 'Size of dataframe: {}'.format(df.shape))
    log(logfile, df.head())
    log(logfile, "\n")
    log(logfile, df.tail())
    log(logfile, "\n")

    # Plotting
    plt.style.use('dark_background')
    _ = plt.hist(df.rating, bins=5, ec='black')
    plt.xlabel('Rating')
    plt.ylabel('Count')
    plt.title('Histogram of Ratings', fontsize=12)
    plt.tight_layout()
    plt.savefig(str(outdir_path) + '/hist_{}.png'.format(dataset_name))

    f = open(logfile, 'a', encoding='utf8')
    cwd, _ = os.path.split(args.datapath)
    subprocess.run(["head", "u.info"], cwd=cwd, stdout=f)
    log(logfile, "\n")
    subprocess.run(["head", "u.genre"], cwd=cwd, stdout=f)
    log(logfile, "\n")
    subprocess.run(["head", "u.user"], cwd=cwd, stdout=f)
    log(logfile, "\n")
    subprocess.run(["head", "u.item"], cwd=cwd, stdout=f)
    log(logfile, "\n")


if __name__ == "__main__":
    descriptions(config_file=sys.argv[1])
