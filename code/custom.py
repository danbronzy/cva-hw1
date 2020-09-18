from os.path import join

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

import util
import visual_words
import visual_recog
from opts import get_opts
import random
import scipy
import math
import time

from bayes_opt import BayesianOptimization
from bayes_opt.logger import JSONLogger
from bayes_opt.event import Events
from bayes_opt.util import load_logs

def build_scales(numScale, multiplier):
    num = int(numScale)
    xVals = np.asarray(range(num))
    yVals = [multiplier * math.pow(1.5, x) for x in xVals]
    return yVals

def do_the_thing(scalesNum, scaleMult, alpha, K, L):
    #parse args from command line, only matters for reading/writing directories
    opts = get_opts()

    #fix some parameters
    scales = build_scales(scalesNum, scaleMult)
    L = int(L)
    K = int(K)
    alpha = int(alpha)

    #rewrite with optimizable parameters
    opts.filter_scales = scales
    opts.K = K
    opts.L = L
    opts.alpha = alpha

    n_cpu = util.get_num_CPU()
    visual_words.compute_dictionary(opts, n_worker=n_cpu)

    visual_recog.build_recognition_system(opts, n_worker=n_cpu)

    conf, accuracy = visual_recog.evaluate_recognition_system(opts, n_worker=n_cpu)

    return accuracy


def main():
    #instantiate logger
    logger = JSONLogger(path="./logs2.json")

    #bounds on the input parameters
    pbounds = {'scalesNum': (2,7), 'scaleMult': (0.5, 7), 'alpha': (25, 250), 'K': (10, 150), 'L': (1,5)}

    #optimizer object
    optimizer = BayesianOptimization(
        f=do_the_thing,
        pbounds=pbounds,
        random_state=2,
        verbose=2
    )

    #log results
    optimizer.subscribe(Events.OPTIMIZATION_STEP, logger)

    #load logs from last optimization OPTIMIZATION_STEP
    load_logs(optimizer, logs = ["./logs.json"])

    #optimize
    optimizer.maximize(init_points = 5, n_iter = 20)

    print(optimizer.max)



if __name__ == '__main__':
    main()
