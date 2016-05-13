#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" convDBN module

This module represents the convolutional deep belief network. It has a list 
of layers that sequentially learn using the convolutional restricted boltzman 
machine algorithm.

"""

import config
from layer import *
import image_data

# -----------------------------------------------------------------------------
class Network:
    def __init__(self, model=None):
        """ Constructor
            
        Input:
            model -- a dict containing the structure and parameters of the 
                     network
            
        """
        
        if model == None:  # It's the case before the network is created
            return
        
        # list of layers in the network, in a bottom-up order
        self.layers = []
        
        # index of the layer to be learned at the time
        self.layer_to_learn = 0

        prv_layer = None
        num_layers = len(model)
        # create the layers using the given model
        for i in range(num_layers):
            print "Creating layer #%s with parameters: " % (i+1) , model[i]
            layer = Layer(model[i], "Layer_%s"%(i+1), prv_layer)
            self.layers.append(layer)
            prv_layer = self.layers[i]

# -----------------------------------------------------------------------------     
    def update(self, layer_to_learn):
        """ Update Network State
        
        Performs one bottom-up pass in the network, samples distributions of 
        every layer and changes the learning parameters for the layer_to_learn
        
        Input:
            layer_to_learn -- index of the layer to be learned
        
        """
#        if config.DEBUG_MODE:
            #assert layer_to_learn <= len(self.layers)
        
        self.layer_to_learn = layer_to_learn
        
        # for measuring the time spent on updating pass in the network
        timer = utils.Timer('network update')
        
        with timer:  # measures the time
            # -- for each layer before the layer to learn
            for i in range(layer_to_learn):
                self.layers[i].update(learn = False)
		

                # -- feed the layer's output to the next layer as input
                if i+1 < len(self.layers):  # if there is a next layer
                    for cnl in range(self.layers[i+1].num_channels):
                        data = self.layers[i].bases[cnl].pooling_units
                        # -- trim the data array to ease the max pooling and
                        # convolution operations
                        wshape = self.layers[i+1].btmup_window_shape
                        pshape = self.layers[i+1].block_shape
                        data = utils.trim_array_maxpool(arr=data, conv_window_shape=wshape, pooling_shape=pshape)
                        
                        # -- Normalize the input vector to the next layer
                        data -= np.mean(data)
                        m = np.mean(data**2)
                        if m != 0:
                            data /= np.sqrt(m)
                        
                        # TEST
                        #data = utils.olshausen_whitening(data) # not good - makes it blurry
                        #c = 0.01
                        data = utils.normalize_image(data, -0.01, 0.01)
                        #
                        #print np.sum(data)
                        #print "Normalized data:", np.sum(data)
                        #print data
                        #print

                        self.layers[i+1].pos_data[:, :, cnl] = data
                      
            # if condition is always true, except when the network is being 
            # loaded from a saved file
            if layer_to_learn < len(self.layers):  
                self.layers[layer_to_learn].update(learn = True) 
        
# -----------------------------------------------------------------------------
    def visualize(self):
        """ Visualize Network

        Visualize the current state of every layer in the network

        """
        for i in xrange(len(self.layers)):
            self.layers[i].visualize()
    
# -----------------------------------------------------------------------------
    def pickle(self, fname):
        """ Pickle Network

        Saves the entire network in a file, using Python's pickling tools

        Input:
            fname -- name of the file to save the network to

        """

        f = file(fname, 'wb')
        cPickle.dump(self, f, protocol=cPickle.HIGHEST_PROTOCOL)
        f.close()
    
# -----------------------------------------------------------------------------
    @classmethod
    def unpickle(self, fname):
        """ Unpickle Network

        Loads the network from a file, using Python's pickling tools

        Input:
            fname -- name of the file to load the network from (must be
            previously saved by the pickling tool)

        """
        f = file(fname, 'rb')
        ret_network = cPickle.load(f)
        f.close()
        return ret_network
