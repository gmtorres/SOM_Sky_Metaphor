import argparse
from os.path import basename

import numpy as np

from minisom import MiniSom
from SOMToolBox_Parse import SOMToolBox_Parse


def train_som(input, xsize, ysize, sigma, lr, iter):
    idata = SOMToolBox_Parse(input).read_weight_file()
    som = MiniSom(xsize, ysize, idata['vec_dim'], sigma=sigma, learning_rate=lr)
    som.train(idata['arr'], iter, True, False)
    weights = som._weights.reshape(-1, idata['vec_dim'])
    return weights


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="input")
    parser.add_argument("xsize", type=int, help="xsize")
    parser.add_argument("ysize", type=int, help="ysize")
    parser.add_argument("--sigma", type=float, help="sigma", default=1.0)
    parser.add_argument("--lr", type=float, help="learning rate", default=0.5)
    parser.add_argument("--iter", type=int, help="iterations", default=100)
    args = parser.parse_args()

    name = basename(args.input).split('.')[0]
    weights = train_som(args.input, args.xsize, args.ysize,
                        args.sigma, args.lr, args.iter)
    np.save(
        "{}_{}x{}_s{}_lr{}_{}.npy".format(name, args.xsize, args.ysize,args.sigma, args.lr, args.iter),
        weights
    )
