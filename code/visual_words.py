import os, multiprocessing
from multiprocessing import Pool
from os.path import join, isfile
import random
import numpy as np
from PIL import Image
import scipy.ndimage
import skimage.color
from functools import partial
import csv
from sklearn import cluster

def extract_filter_responses(opts, img):
    '''
    Extracts the filter responses for the given image.

    [input]
    * opts    : options
    * img    : numpy.ndarray of shape (H,W) or (H,W,3)
    [output]
    * filter_responses: numpy.ndarray of shape (H,W,3F)
    '''
    imgH = img.shape[0]
    imgW = img.shape[1]

    #scale appropriately
    maxVal = np.max(img)
    minVal = np.min(img)
    if maxVal > 1 or minVal < 0:
        print("Images requires scaling...")
        img = (img - minVal) / (maxVal - minVal)


    #duplicate grayscale images into 3 dimensions
    if img.ndim == 2:
        img = img[..., np.newaxis]
        img = np.concatenate((img, img, img), 2)

    #confirm all images are HxWx3
    assert(img.ndim == 3)
    assert(img.shape[2] == 3)

    img = skimage.color.rgb2lab(img)

    filter_scales = opts.filter_scales

    filter_responses = np.ndarray((imgH,imgW,0))

    for i in filter_scales:
        res1 = np.ndarray((imgH,imgW,0))#container to hold single image with filter applied separately to each channel
        for dim in range(3):
            res = scipy.ndimage.gaussian_filter(img[:,:,dim], i)#apply filter to one channel
            res = res[..., np.newaxis]#make 3D
            res1 = np.concatenate((res1,res), 2)#concatenate

        res2 = np.ndarray((imgH,imgW,0))#container to hold single image with filter applied separately to each channel
        for dim in range(3):
            res = scipy.ndimage.gaussian_laplace(img[:,:,dim], i)#apply filter to one channel
            res = res[..., np.newaxis]#make 3D
            res2 = np.concatenate((res2,res), 2)#concatenate

        res3 = np.ndarray((imgH,imgW,0))#container to hold single image with filter applied separately to each channel
        for dim in range(3):
            res = scipy.ndimage.gaussian_filter(img[:,:,dim], i, (0,1))#apply filter to one channel
            res = res[..., np.newaxis]#make 3D
            res3 = np.concatenate((res3,res), 2)#concatenate

        res4 = np.ndarray((imgH,imgW,0))#container to hold single image with filter applied separately to each channel
        for dim in range(3):
            res = scipy.ndimage.gaussian_filter(img[:,:,dim], i, (1,0))#apply filter to one channel
            res = res[..., np.newaxis]#make 3D
            res4 = np.concatenate((res4,res), 2)#concatenate

        filter_responses = np.concatenate((filter_responses, res1, res2, res3, res4), 2)

    return filter_responses

def compute_dictionary_one_image(opts, filename):
    '''
    Extracts a random subset of filter responses of an image and save it to disk
    This is a worker function called by compute_dictionary

    Your are free to make your own interface based on how you implement compute_dictionary
    '''

    img_path = join(opts.data_dir, filename)
    img = Image.open(img_path)
    img = np.array(img).astype(np.float32)/255

    alpha = opts.alpha
    filtered = extract_filter_responses(opts, img)
    imgH = filtered.shape[0]
    imgW = filtered.shape[1]
    wordDim = filtered.shape[2]

    cleanFileName =  filename.replace("/","_")[:-4]
    words = np.ndarray((wordDim,0))
    for i in range(alpha):
        word = filtered[random.randint(0, imgH - 1), random.randint(0, imgW - 1), :]
        word = word[..., np.newaxis]
        words = np.concatenate((words, word), 1)

    #Each line is a single word vector in the feature-space
    np.savetxt(join(opts.out_dir, 'tempWordFiles', cleanFileName), words.T, delimiter=", ", newline="\n")

def compute_dictionary(opts, n_worker=1):
    '''
    Creates the dictionary of visual words by clustering using k-means.

    [input]
    * opts         : options
    * n_worker     : number of workers to process in parallel

    [saved]
    * dictionary : numpy.ndarray of shape (K,3F)
    '''

    data_dir = opts.data_dir
    feat_dir = opts.feat_dir
    out_dir = opts.out_dir
    K = opts.K

    train_files = open(join(data_dir, 'train_files.txt')).read().splitlines()
    partialFunc = partial(compute_dictionary_one_image, opts)
    with Pool(processes = n_worker) as pool:
        pool.map(partialFunc, train_files)

    #now need to read in the temporary files
    words = []
    for filename in os.listdir(join(out_dir, 'tempWordFiles')):
        with open(join(out_dir, 'tempWordFiles', filename)) as csvfile:
            reader = csv.reader(csvfile, quoting=csv.QUOTE_NONNUMERIC)
            for row in reader:
                words.append(row)
    kmeans = cluster.KMeans(n_clusters=K, n_jobs=n_worker).fit(words)
    dictionary = kmeans.cluster_centers_
    np.save(join(out_dir, 'dictionary.npy'), dictionary)

def get_visual_words(opts, img, dictionary):
    '''
    Compute visual words mapping for the given img using the dictionary of visual words.

    [input]
    * opts    : options
    * img    : numpy.ndarray of shape (H,W) or (H,W,3)

    [output]
    * wordmap: numpy.ndarray of shape (H,W)
    '''
    filteredImage = extract_filter_responses(opts, img)

    imgH = filteredImage.shape[0]
    imgW = filteredImage.shape[1]
    dim = filteredImage.shape[2]
    flattened = filteredImage.reshape(imgH * imgW, dim)

    dists = scipy.spatial.distance.cdist(flattened, dictionary)
    wordDims = dists.shape[1]
    dists = dists.reshape(imgH, imgW, wordDims)#back to image size

    wordmap = np.ndarray((imgH, imgW))
    for hInd in range(imgH):
        for wInd in range(imgW):
            word = np.argmin(dists[hInd, wInd])
            wordmap[hInd, wInd] = word
    return wordmap
