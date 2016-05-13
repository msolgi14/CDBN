#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" main module

This module runs the main loops of the network training and testing

"""

import sys
import os
import time
import Image
import numpy as np
import random
import pylab as pl

import utils
import convDBN
import image_data
import config

import scipy
from scipy.misc import toimage
from PIL import Image
import matplotlib.pyplot as plt


# the main entity of the program - an instance of the convolutional DBN
network = convDBN.Network(None)

# will contain the data set to be used by the network.
data = None

# -- check for the system version
if sys.version_info < (2,6,0):
    sys.stderr.write("You need python 2.6 or later to run this program\n")
    exit(1)
    
# -----------------------------------------------------------------------------
def main(network_loaded=False):
    """

    Creates the network with the loaded parameters, then runs the main loop of
    the simulation

    """
    
    # -- global variables
    global network
    global data
    
    # initialize the random seed
    np.random.seed(run_params['random_seed'])

    # -- create a directory to contains the results (oputputs) of the
    # simualtion, if it doesn't exist already
    if not os.path.exists(misc_params['results_dir']):
        os.mkdir(misc_params['results_dir'])
    

    # -------------------------- Read the Input Images ------------------------
    data = image_data.ImageData(image_params['EXTENSIONS'], \
                                  image_params['max_image_edge'], \
                                  image_params['num_channels'])
                                  
    data.load_images(image_params['image_path'])
    
    # --------------------------- Create the Network --------------------------
    # -- model specifies the network with a list of dicts, each dict declaring
    #       one hidden layer in the network
    if network_loaded == False:
        print "Simulation starts with an unlearned network with random weights..\n"
        network = convDBN.Network(network_params['model'])
        
    else:
        print "Simulation starts with a loaded (learned) network..\n"
        # TEST
        win_shape = network.layers[1].btmup_window_shape
        c = 1e-2
        for base in network.layers[1].bases:
            base.Wb = c * np.random.randn(win_shape[0], win_shape[1], network.layers[1].num_channels)
            #base.Wb = utils.normalize_image(base.Wb, MAX_PIXEL=0.1) ** 5
            #print base.Wb
            #for i in xrange(win_shape[0]):
                #for j in xrange(win_shape[1]):
                    #for k in xrange(network.layers[1].num_channels):
                        ##t = 0
                        ##if np.random.randint(100) < 20:
                            ##t = 1
                        #t = base.Wb[i, j, k]
                        #if t < c:
                            #base.Wb[i, j, k] = 0
            #b = c**3 * (np.random.rand()-.5)
            #print b
            #base.bias = 0
        network.layers[1].std_gaussian = 0.04
        network.layers[1].epsilon = 0.1
            
    if network_params['model'][0]['init_data']:
        # -- initialize first layer weights using random input patches
        for base in network.layers[0].bases:
            img_idx = np.random.randint(0, len(data.images))
            rand_patch = data.get_image_patch(img_idx, image_params['image_patch_shape'])

            x_start = np.random.randint(0, rand_patch.shape[0] - base.Wb.shape[0])
            y_start = np.random.randint(0, rand_patch.shape[1] - base.Wb.shape[1])

            x_end = x_start + base.Wb.shape[0]
            y_end = y_start + base.Wb.shape[1]

            base.Wb = 0.001 * rand_patch[x_start:x_end, y_start:y_end, :]
        
    # -- create the data structure to keep the current input to the network
    h = network.layers[0].input_shape[0]
    w = network.layers[0].input_shape[1]
    d = image_params['num_channels']
    data.curr_input = np.zeros((h, w, d))
    
    # ---------------------------- Run the Network ----------------------------
    crbm_run(network_loaded)
    
# -----------------------------------------------------------------------------
def crbm_run(network_loaded=False):
    """

    Runs the training and testing loop of the simulation

    Input:
        network_loaded -- whether a trained network was loaded from a file

    """

    # -- gloabl vars
    global network
    global data

    # -- to keep track of error made in each epoch
    err_file = open(misc_params['results_dir']+misc_params['err_fname'], 'w')
    num_epochs = sum(run_params['epoch_per_layer'])

    layer_err = []

    # -- to know when the network finishes one and switches to the next layer
    # to learn
    prv_layer_to_learn, layer_to_learn = 0, 0
    
    for epoch_idx in range(num_epochs):
        print "Training trial #%s.." % epoch_idx

        # -- make a random permutation of the list of images
        num_images = len(data.images)
        image_order = random.sample(range(num_images), num_images)  

        for img_idx in range(num_images):
            for patch_idx in range(image_params['samples_per_image']):
                print "\n------ Epoch", epoch_idx, ", batch", img_idx, ", patch", patch_idx, "------"
                
                # -- get an image patch and trim it so the size fits for convolution
                img_patch = data.get_image_patch(img_idx, image_params['image_patch_shape'])
                bu_shape = network.layers[0].btmup_window_shape
                
                pshape = network.layers[0].block_shape
                for cnl in range(image_params['num_channels']):
                    data.curr_input[:, :, cnl] = utils.trim_array_maxpool(arr=img_patch[:, :, cnl], conv_window_shape=bu_shape, pooling_shape=pshape)
                    
                # -- reshape and feed the input image (visible layer) to the first hidden layer
                input_shape = (data.curr_input.shape[0], data.curr_input.shape[1], network.layers[0].num_channels)
                
                # -- feed the input image (visible layer) to the first hidden layer
                network.layers[0].pos_data = np.reshape(data.curr_input, input_shape)


                # -- compute the layer to be learned (using the number of epochs
                # needed for each layer, set in the parameters file)
                sum_epochs = 0
                for layer_idx in range(len(network.layers)):
                    sum_epochs += run_params['epoch_per_layer'][layer_idx]
                    if epoch_idx < sum_epochs:
                        layer_to_learn = layer_idx
                        break

                #if img_idx < 1: layer_to_learn = 0
                #else: layer_to_learn = 1

                # TEST 
                if network_loaded:
                    layer_to_learn = 0 

                # -- If specified in the parameters file, set the weights of
                # the 2nd layer to the output of the first layer before
                # starting to learn the 2nd layer
                if layer_to_learn != prv_layer_to_learn and not network_loaded:
                    network.pickle("results/pickled.pcl")
                    if network_params['model'][1]['init_data']:
                        network.layers[layer_to_learn].init_weights_using_prvlayer()

                    # -- give time to observe the network when in debug mode
                    if config.DEBUG_MODE:
                        time.sleep(10)

                # update the network
                network.update(layer_to_learn)

                #print np.sum(network.layers[layer_to_learn].pos_data)
                
                prv_layer_to_learn = layer_to_learn

            # TEST
            #network.layers[layer_to_learn].weights_for_visualization(0, (4,6), dir_path=misc_params['results_dir'], save=True)

            #negdata = network.layers[0].neg_data
            #scipy.misc.imsave('./results/reconstruction.jpg', negdata[:,:,0])
            #posdata = data.curr_input
            #scipy.misc.imsave('./results/input_image.jpg', posdata[:,:,0])
            #tt = Image.frombuffer('L', data.curr_input.shape[0:2], posdata)
            #tt.save("./results/input_image.png")

        # -- compute mean of error made in the epoch and save it to the file
        mean_err = np.mean(network.layers[0].epoch_err)
        layer_err.append(mean_err)
        err_file.write(str(mean_err)+' ')
        err_file.flush()

        # flush the errors made in the previous epoch
        network.layers[0].epoch_err = []
        
        # -- stop decaying after some point
        # TEST
        curr_layer = network.layers[layer_to_learn]
        if curr_layer.std_gaussian > curr_layer.model['sigma_stop']:
             curr_layer.std_gaussian *= 0.99

        # -- visualize layers and save the network at the end of each epoch

        for lyr in range(layer_to_learn+1):
            if lyr == 0:
                tile_shape = (5, 5)
            elif lyr == 1:
                tile_shape = (5, 5)

            network.layers[layer_to_learn].visualize_to_files(tile_shape, dir_path=misc_params['results_dir'])
        
        network.pickle(misc_params['results_dir'] + misc_params['pickle_fname'])
    
    err_file.close()
    

def print_usage():
    """

    Print the usage of the main module

    """
    
    print "\nUsage: \n    %s <params_filename>" % sys.argv[0]
    print "If <params_filename> is not given, the default file specified in config.py will be used.\n"
    print "Example:"
    print "    %s params_natural_imgs.py" % sys.argv[0]


if __name__ == "__main__":
    
    # -- If not given as a command line argument, use the default
    # file name config.params_filename as the parameters file
    if len(sys.argv) == 1:
        print "Default parameters file 'params_naturaImages.py' was used.."
        pars_fname = config.params_filename

    elif len(sys.argv) == 2:
        if sys.argv[1] == "help":
            print_usage()
            exit(0)
        else:
            pars_fname = sys.argv[1]
    else:
        print_usage()
    print
    
    #-- Read the parameters file. execfile() was not used for the sake of
    # forward compatibility with Python 3.0
    exec(compile(open(pars_fname).read(), pars_fname, 'exec'))
    # TEST
    # network = convDBN.Network.unpickle('./results/pickeled-afterFirstLayerFinished.pcl')
    #main(True)
    main(False)

# -- If the program was run in GUI mode, use default file name specified in
# config.params_filename as the parameters file
else:
    exec(compile(open(config.params_filename).read(), config.params_filename, 'exec'))
