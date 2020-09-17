from os.path import join

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

import util
import visual_words
import visual_recog
from opts import get_opts

import scipy

import time

def main():
    opts = get_opts()

    ## Q1.1
    # img_path = join(opts.data_dir, 'kitchen/sun_aasmevtpkslccptd.jpg')
    # img = Image.open(img_path)
    # img = np.array(img).astype(np.float32)/255
    # filter_responses = visual_words.extract_filter_responses(opts, img)
    # util.display_filter_responses(opts, filter_responses)

    # Q1.2
    time_start = time.perf_counter()
    print('Making dictionary')
    n_cpu = util.get_num_CPU()
    visual_words.compute_dictionary(opts, n_worker=n_cpu)
    dict_time = time.perf_counter()
    print("Dictionary done -- Elapsed time: {}".format(dict_time - time_start))
    # ## Q1.3
    # # img 1: highway/sun_aacqsbumiuidokeh.jpg
    # # img 2: desert/sun_aaqyzvrweabdxjzo.jpg
    # # img 3: kitchen/sun_aasmevtpkslccptd.jpg
    #
    # filename = 'highway/sun_aacqsbumiuidokeh.jpg'
    # img_path = join(opts.data_dir, filename)
    # img = Image.open(img_path)
    # img = np.array(img).astype(np.float32)/255
    # dictionary = np.load(join(opts.out_dir, 'dictionary.npy'))
    # wordmap = visual_words.get_visual_words(opts, img, dictionary)
    # cleanFileName = filename.replace("/","_")[:-4]
    # util.visualize_wordmap(wordmap, join(opts.out_dir, cleanFileName))
    ## Q2.1-2.4
    # visual_recog.get_feature_from_wordmap_SPM(opts, wordmap)
    print("Building recognition system")
    n_cpu = util.get_num_CPU()
    visual_recog.build_recognition_system(opts, n_worker=n_cpu)
    recog_time = time.perf_counter()
    print("Recognition system done - Elapsed time: {}".format(recog_time - dict_time))
    #
    # ## Q2.5
    print("Evaluating system")
    n_cpu = util.get_num_CPU()
    conf, accuracy = visual_recog.evaluate_recognition_system(opts, n_worker=n_cpu)
    eval_time = time.perf_counter()
    print("Done evaluating - Elapsed time: {}".format(eval_time - recog_time))

    print(conf)
    print(accuracy)
    np.savetxt(join(opts.out_dir, 'confmat.csv'), conf, fmt='%d', delimiter=',')
    np.savetxt(join(opts.out_dir, 'accuracy.txt'), [accuracy], fmt='%g')


if __name__ == '__main__':
    main()
