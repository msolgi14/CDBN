#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" parameters module

Contains the all the parameters and structure of the model. One needs to modify
or duplicate this file to create different simulations

"""

# -----------------------------------------------------------------------------
# parameters related to input images
image_params = {
    # path of the directory containing the input images
    'image_path': "../data/natural_imgs",    
    # list of image extentions accepted
    'EXTENSIONS':  ['.png', '.jpg', '.tif'],
    # -- maximum edge size in the images - used to scale all the images to have 
    # the same maximum edge size
    'max_image_edge': 512,
    # shape of the patch of the image to be fed to the network
    'image_patch_shape': (70, 70),
    # number of image patches taken from each image before moving to the next
    'samples_per_image': 10,
    # number of channels in the image (e.g. 1 for balck&white and 3 for RGB)
    'num_channels': 1
    }

# -----------------------------------------------------------------------------
# parameters that control the running protocol of the program
run_params = {
    # if None, initialize the random seed using system time
    'random_seed': None,  
    # -- number of epochs to learn each layer after all previous layers are
    # learned
    'epoch_per_layer': [35, 100]
    }

# -----------------------------------------------------------------------------
# structure and parameters of the network to be created
network_params = {
    # -- Each element of the 'model' list contains the parameters for one layer
    # in the network, in a bottom-up order
    'model': [
        {
            # --------------------------- Layer 1 -----------------------------
            # -- number of bases or "groups" in the layer - equivalent to 
            #       parameter K in the Lee et. al. ICML09 paper
            'num_bases': 24,
            # shape of the bottom-up filter
            'btmup_window_shape': (10, 10),
            # shape of the window used for max pooling
            'block_shape': (2, 2),
            
             # -- set the input data shape and number of channels of the
             # first layer using image parameters
            'input_data_shape': image_params['image_patch_shape'],
            'num_channels': image_params['num_channels'],
            
            
            # sparsity parameter - expected activations of the hidden units
            'pbias': 0.005, 
            # step size towards sparsity equilibrium
            'pbias_lambda': 5,  
            
            # initial value for the bias
            'init_bias': 0.01,
            # visible layer (previous layer) bias
            'vbias': 0.01,
            # regularization factor to keep the length of weight vector small
            'regL2': 0.01,
             # learning rate - the ratio of change taken from gradient descent
            'epsilon': 0.02,
            # start and stop value of the parameter used to control the effect 
            # of input vector (versus bias)
            'sigma_start': 0.2,
            'sigma_stop': 0.1,
            # -- number of steps (loops) performed in the contrastive
            # divergence algorithm
            'CD_steps': 1,
            # -- whether to initialize the weights using input from the previous
            # layer (or the input images for the first layer)
            'init_data': False
            
        }
        ,
        {
            # --------------------------- Layer 2 -----------------------------
            # Please see the first layer for comments about the parameters
            'num_bases': 100, 

            'btmup_window_shape': (10, 10),
            'block_shape': (2, 2),

            'pbias': 0.01,
            'pbias_lambda': 5,
            'init_bias': 0.0,
            'regL2': 0.01,
            'vbias': 0.0,
            'epsilon': 1e-17,
            'sigma_start': 0.2,
            'sigma_stop': 0.1,
            'CD_steps': 1,
            'init_data': False
        }
    ]
    }

# -----------------------------------------------------------------------------
# miscellaneous parameters
misc_params = {
    # directory path to save output images, performance plots, etc.
    'results_dir': './results/',
    # name of the file to pickle (save) the network to
    'pickle_fname': 'cdbn_net-natural_imgs.dump',
    # error output file name
    'err_fname': 'error.txt',
    # rate to refresh the GUI
    'GUI_refresh_timeout': 5000 # mili-seconds
    }
