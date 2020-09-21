import os, math, multiprocessing
from multiprocessing import Pool
from os.path import join
from copy import copy

import numpy as np
from PIL import Image
import visual_words

from functools import partial

def get_feature_from_wordmap(opts, wordmap):
    '''
    Compute histogram of visual words.

    [input]
    * opts      : options
    * wordmap   : numpy.ndarray of shape (H,W)

    [output]
    * hist: numpy.ndarray of shape (K)
    '''

    K = opts.K

    (hist, bins) = np.histogram(wordmap, bins=K, range=(0,K))
    hist = hist/np.sum(hist)
    # plt.bar(bins[:-1], hist, align='edge')
    # plt.show()
    return hist

def recursive_SPM(opts, wordmap, layersLeft):
    if layersLeft == 1:
        #if we are at the bottom of the recursion stack,
        hist = get_feature_from_wordmap(opts, wordmap)

        #return features and histogram
        if opts.L == 1:
            #if only a single layer, no weighting
            return hist, hist
        else:
            #else, there is weighting
            return hist/2, hist

    #split into quadrants
    wordmapTl = wordmap[:wordmap.shape[0]//2,
                        :wordmap.shape[1]//2]
    wordmapTr = wordmap[:wordmap.shape[0]//2,
                        wordmap.shape[1]//2:]
    wordmapBl = wordmap[wordmap.shape[0]//2:,
                        :wordmap.shape[1]//2]
    wordmapBr = wordmap[wordmap.shape[0]//2:,
                        wordmap.shape[1]//2:]

    #generate features and histograms for the layer below
    featsTl, histTl = recursive_SPM(opts, wordmapTl, layersLeft - 1)
    featsTr, histTr = recursive_SPM(opts, wordmapTr, layersLeft - 1)
    featsBl, histBl = recursive_SPM(opts, wordmapBl, layersLeft - 1)
    featsBr, histBr = recursive_SPM(opts, wordmapBr, layersLeft - 1)

    #combine and normalize features
    combinedLowerFeats = np.hstack((featsTl, featsTr, featsBl, featsBr))
    combinedLowerFeats /= 4 #normalized

    #combine and normalize histograms
    combinedHist = (histTl + histTr + histBl + histBr)/4

    #weight the features of this layer
    divisor = 0.
    if layersLeft == opts.L:#if top layer
        divisor = pow(2, layersLeft - 1)
    else:
        divisor = pow(2, layersLeft)

    weightedFeats = combinedHist / divisor

    #append self-features to feature list and then return features and histogram
    feats = np.hstack((combinedLowerFeats, weightedFeats))

    return feats, combinedHist

def get_feature_from_wordmap_SPM(opts, wordmap):
    '''
    Compute histogram of visual words using spatial pyramid matching.

    [input]
    * opts      : options
    * wordmap   : numpy.ndarray of shape (H,W)

    [output]
    * hist_all: numpy.ndarray of shape (K*(4^L-1)/3)
    '''

    K = opts.K
    L = opts.L
    (feats, hist) = recursive_SPM(opts, wordmap, L)
    return feats


def get_image_feature(opts, img_path, dictionary):
    '''
    Extracts the spatial pyramid matching feature.

    [input]
    * opts      : options
    * img_path  : path of image file to read
    * dictionary: numpy.ndarray of shape (K, 3F)


    [output]
    * feature: numpy.ndarray of shape (K)
    '''
    #get image
    img = Image.open(img_path)
    img = np.array(img).astype(np.float32)/255

    #calculate wordmap
    wordmap = visual_words.get_visual_words(opts, img, dictionary)

    #return spm
    return get_feature_from_wordmap_SPM(opts, wordmap)

def write_features_single_image(opts, dictionary, img_and_label):
    (filename, label) = img_and_label
    img_path = join(opts.data_dir, filename)

    #filename is prepended with the label value, for reading back later
    cleanFileName = str(label) + '_' + filename.replace("/","_")[:-4]

    #get feature vector for this images
    feats = get_image_feature(opts, img_path, dictionary)
    np.save(join(opts.out_dir, 'tempFeatureFiles', cleanFileName), feats)

def build_recognition_system(opts, n_worker=1):
    '''
    Creates a trained recognition system by generating training features from all training images.

    [input]
    * opts        : options
    * n_worker  : number of workers to process in parallel

    [saved]
    * features: numpy.ndarray of shape (N,M)
    * labels: numpy.ndarray of shape (N)
    * dictionary: numpy.ndarray of shape (K,3F)
    * SPM_layer_num: number of spatial pyramid layers
    '''

    data_dir = opts.data_dir
    out_dir = opts.out_dir
    SPM_layer_num = opts.L

    train_files = open(join(data_dir, 'train_files.txt')).read().splitlines()
    train_labels = np.loadtxt(join(data_dir, 'train_labels.txt'), np.int32)
    dictionary = np.load(join(out_dir, 'dictionary.npy'))

    #zip filename and label together for iteration with pool
    z = zip(train_files, train_labels)
    partialFunc = partial(write_features_single_image, opts, dictionary)
    with Pool(processes = n_worker) as pool:
        pool.map(partialFunc, z)

    #read in labels and features written to {out_dir}/tempFeatureFiles/
    featureLen = opts.K * (pow(4, opts.L) - 1)//3
    labels = np.ndarray(0).astype(int)
    features = np.ndarray((0,featureLen))
    for filename in os.listdir(join(out_dir, 'tempFeatureFiles')):
        label = int(filename[0])#by my convention in write_features_single_image
        labels = np.append(labels, label)
        fileLoc = join(out_dir, 'tempFeatureFiles', filename)
        arr = np.load(fileLoc)
        features = np.vstack((features, arr))

    np.savez_compressed(join(out_dir, 'trained_system.npz'),
        features=features,
        labels=labels,
        dictionary=dictionary,
        SPM_layer_num=SPM_layer_num,
    )

def distance_to_set(word_hist, histograms):
    '''
    Compute similarity between a histogram of visual words with all training image histograms.

    [input]
    * word_hist: numpy.ndarray of shape (K)
    * histograms: numpy.ndarray of shape (N,K)

    [output]
    * sim: numpy.ndarray of shape (N)
    '''

    return 1 - np.sum(np.minimum(histograms, word_hist), axis = 1)

def evaluate_single_test(opts, dictionary, trained_features, trained_labels, img_and_label):
    (filename, real_label) = img_and_label
    img_path = join(opts.data_dir, filename)
    feature = get_image_feature(opts, img_path, dictionary)

    dists = distance_to_set(feature, trained_features)
    assigned_label = trained_labels[np.argmin(dists)]

    cleanFileName = filename.replace("/","_")[:-4]
    np.save(join(opts.out_dir, 'tempResultFiles', cleanFileName), [real_label, assigned_label])

def evaluate_recognition_system(opts, n_worker=1):
    '''
    Evaluates the recognition system for all test images and returns the confusion matrix.

    [input]
    * opts        : options
    * n_worker  : number of workers to process in parallel

    [output]
    * conf: numpy.ndarray of shape (8,8)
    * accuracy: accuracy of the evaluated system
    '''

    data_dir = opts.data_dir
    out_dir = opts.out_dir

    trained_system = np.load(join(out_dir, 'trained_system.npz'))
    dictionary = trained_system['dictionary']
    trained_features = trained_system['features']
    trained_labels = trained_system['labels']

    # using the stored options in the trained system instead of opts.py
    test_opts = copy(opts)
    test_opts.K = dictionary.shape[0]
    test_opts.L = trained_system['SPM_layer_num']

    test_files = open(join(data_dir, 'test_files.txt')).read().splitlines()
    test_labels = np.loadtxt(join(data_dir, 'test_labels.txt'), np.int32)

    #zip filename and label together for iteration with pool
    z = zip(test_files, test_labels)
    partialFunc = partial(evaluate_single_test, test_opts, dictionary, trained_features, trained_labels)
    with Pool(processes = n_worker) as pool:
        pool.map(partialFunc, z)

    confusion = np.zeros((8,8))
    for filename in os.listdir(join(out_dir, 'tempResultFiles')):
        fileLoc = join(out_dir, 'tempResultFiles', filename)
        arr = np.load(fileLoc)
        confusion[arr[0], arr[1]] += 1

    trace = np.trace(confusion)
    sumC = np.sum(confusion)
    acc = trace/sumC
    return confusion, acc
