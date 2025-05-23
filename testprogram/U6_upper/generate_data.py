import os
import numpy as np
from utils.data_utils import check_extension, save_dataset
import torch
import pickle
import argparse

def generate_taop_data(dataset_size, taop_size):
    data = []
    rnd = np.random.RandomState(24610)

    loc = rnd.uniform(0, 1, size=(dataset_size, taop_size + 1, 2))
    depot = loc[:, -1]
    cust = loc[:, :-1]
    prize = np.ones((dataset_size, taop_size))
    maxlength = {
        80: 2.,
        100: 2,
        150: 3,
        200: 3.,
        300: 4.,
        500: 4.
    }

    thedata = list(zip(depot.tolist(),  # Depot location
                       cust.tolist(),
                       prize.tolist(),
                       np.full(dataset_size, maxlength[taop_size]).tolist()
                        ))

    return thedata


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--filename", help="Filename of the dataset to create (ignores datadir)")
    parser.add_argument("--dataset_size", type=int, default=500)
    parser.add_argument("--veh_num", type=int, default=3, help="number of the UAVs; 4 or 6")
    parser.add_argument('--graph_size', type=int, default=80,
                        help="Sizes of problem instances: {80, 100, 150, 200, 300, 500}")

    opts = parser.parse_args()
    data_dir = 'data'
    problem = 'taop'
    datadir = os.path.join(data_dir, problem)
    os.makedirs(datadir, exist_ok=True)
    seed = 24610  # the last seed used for generating TAOP data
    np.random.seed(seed)
    print(opts.dataset_size, opts.graph_size)
    filename = os.path.join(datadir, '{}_{}_seed{}.pkl'.format(problem, opts.graph_size, seed))

    dataset = generate_taop_data(opts.dataset_size, opts.graph_size)
    print(dataset[0])
    save_dataset(dataset, filename)



