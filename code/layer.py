#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" layer module

A layer is a collection of bases (see base module) that have the same bottom-up
and top-down input from adjacent layers.

"""

import scipy as sp
from scipy.misc import toimage

import utils
import config
from base import *

import pycuda_convolution
#conv = scipy.signal.convolve
conv = pycuda_convolution.convolve_gpu

# -----------------------------------------------------------------------------
class Layer:
    def __init__(self, model, label, prv_layer=None):
        """ Constructor
        
        Input:
            model -- a dict containing the structure and parameters of the layer
            label -- label assigned to the layer (e.g. "Layer 1")
            prv_layer -- pointer to the previous layer in the network
        
        """

        # -- copy parameter values to class data members
        self.model = model
        self.label = label
        self.prv_layer = prv_layer

        # -- copy values from model to class data members, to improve code
        # readability
        self.block_shape = model['block_shape']
        self.btmup_window_shape = model['btmup_window_shape']
        print "LAYER:BUWS = ", self.btmup_window_shape
        self.epsilon = model['epsilon']
        self.regL2 = model['regL2']
        self.pbias = model['pbias']
        self.pbias_lambda = model['pbias_lambda']
        self.init_bias = model['init_bias']
        self.vbias = model['vbias']
        self.std_gaussian = model['sigma_start']
        self.CD_steps = model['CD_steps']

        # amount of change made in the bias of the visible layer while updating
        self.vbias_inc = 0
        
        # a list to hold the errors at each training cycle of the epoch
        self.epoch_err = []  

        # -- determine the shape of the input vector, depending on whether this 
        # is the first layer in the network.
        # -- If this is not the first layer
        if (prv_layer):
            self.num_channels = len(prv_layer.bases)
            inshape = prv_layer.output_shape
        # -- for the first layer
        else:
            self.num_channels = model['num_channels']
            inshape = model['input_data_shape']

        wshape = self.btmup_window_shape  # convolution window shape
        pshape = self.block_shape  # pooling shape

        # trim the output of the previous layer to be used for maxpooling
        self.input_shape = utils.trim_array_maxpool(arr_shape=inshape, \
                                                     conv_window_shape=wshape, \
                                                     pooling_shape=pshape)

        # -- shape of hidden units in each base of the layer
        h = self.input_shape[0] - self.btmup_window_shape[0] + 1
        w = self.input_shape[1] - self.btmup_window_shape[1] + 1
        self.hidden_shape = (h, w)

        # -- shape of output (pooling) units in each base of the layer
        h = self.hidden_shape[0] / self.block_shape[0]
        w = self.hidden_shape[0] / self.block_shape[1]
        self.output_shape = (h,w)

        # negative data, i.e. network's belief
        self.neg_data = np.zeros((self.input_shape[0], \
                                    self.input_shape[1], self.num_channels))
        
        # positive data - input from previus layer (raw input if first layer)
        self.pos_data = np.zeros((self.input_shape[0],
                                    self.input_shape[1], self.num_channels))

        # list of bases in the layer
        self.bases = []
        # -- create the bases
        for i in range(model['num_bases']):
            self.bases.append(Base(self))
            
#-----------------------------------------------------------------------------
    def init_weights_using_prvlayer(self):
        """ Weight Initialization

        Initilizes the weights of the layer using input from the prv_layer.
        Using this sort of initialization is expected to speed up the
        convergence. However, it is not used in this current version since
        using random initial weights is a better verification test for the
        network.
        
        """

        # make sure there is a previous layer
        if config.DEBUG_MODE:
            assert self.prv_layer != None

        # for each base in the layer, grab a random patch of the previous
        # layer's output and assign the base's weights to the output values
        for base in self.bases:
            x_start = np.random.randint(0, self.prv_layer.output_shape[0] - base.Wb.shape[0])
            y_start = np.random.randint(0, self.prv_layer.output_shape[1] - base.Wb.shape[1])
            
            x_end = x_start + base.Wb.shape[0]
            y_end = y_start + base.Wb.shape[1]
            
            for i in range(self.num_channels):
                c = 10e-5
                base.Wb[:, :, i] = c * self.prv_layer.bases[i].pooling_units[x_start:x_end, y_start:y_end]

#-----------------------------------------------------------------------------
    def update(self, learn = False):
        """ Update Layer
        
        Performs Gibbs sampling of the layer's state variables (given the
        previous layer), and then updates the parameters and weights
        accordingly. Here is the steps performed:
            1) Sample each base given 
        
        Input:
            layer_to_learn -- index of the layer to be learned
        
        """
        
        # ------------------------- Prositive Phase --------------------------
        print "\nPositive phase for ", self.label, "..."
        
        #timer = utils.Timer('positive phase')
        #with timer:  # measures the time
        #print "self.pos_data:", self.pos_data
        for base in self.bases:
            base.pos_sample()
            
        # ------------------------- Negative Phase ---------------------------
        # -- computes P(v|h) : Equation at the end of Section 2.3 in the paper
        print "Negative phase for", self.label, "..."
        #timer = utils.Timer('negative phase')
        #with timer:  # measures the time
        # perform the following Gibbs sampling steps, CD_steps times
        for step_idx in xrange(self.CD_steps):
            # -- compute the negative data given the hidden layer
            self.neg_data[:, :, :] = 0
            for base in self.bases:
                for channel in range(base.num_channels):
                    w =  base.Wb[:, :, channel]
                    self.neg_data[:, :, channel] += conv(base.pos_states, w,
                                                                        'full')
            self.neg_data += self.vbias
    
            # -- debugging assertion
            if config.DEBUG_MODE:
                assert ~np.isnan(self.neg_data).any()



            for base in self.bases:
                base.neg_sample()
                

        # -- compute the error as Euclidean distance between positive and
        # negative data
        err = np.mean( (utils.normalize_image(self.pos_data, 0, 1) -
                                    utils.normalize_image(self.neg_data, 0, 1))**2)
        self.epoch_err.append(err)
        print "Mean error so far: %.3f" % np.mean(self.epoch_err)
        

        # -- update the bases only if this layer is being currently        
        # learned
        for base in self.bases:
            # -- reset some book keeping values
            base.bias_inc = 0
            base.Wb_inc = 0
            if (learn == True):
                base.update()

        self.vbias_inc = 0
        if (learn == True):
            # -- update the visible layer (prevous layer) bias
            print "Update phase for", self.label, "..."
            # Gradient Descent change
            dV_GD = np.mean(self.pos_data - self.neg_data)
            self.vbias_inc = self.epsilon * dV_GD
            self.vbias += self.vbias_inc
        
            # print the current state variables of the layer
            self.print_statistics()

#-----------------------------------------------------------------------------
    def print_statistics(self):
        """ Print Statistics

        Prints the current state variables of the network, including sparsity
        of units' activation, length and change of the weight vector, hidden
        and visible baises, as well as length of the positive and negative data
        vectors.

        """
        
        W_sum = 0
        Winc_sum = 0
        Hb_sum = 0
        Hbinc_sum = 0
        S_sum = 0
        # -- update the bases only if this layer is being currently
        # learned
        for base in self.bases:
            W_sum += np.sum(base.Wb) ** 2
            Winc_sum += np.sum(base.Wb_inc) ** 2
            S_sum += np.sum(base.pos_states)
            Hb_sum += base.bias ** 2
            Hbinc_sum += base.bias_inc ** 2
            
        num_units = len(self.bases) * self.hidden_shape[0] *self.hidden_shape[1]
        print self.label, ": Sparsity measure: %.2f percent" % (100 * float(S_sum)/num_units)
        print self.label, ": ||W|| = %.2f ||dW|| = %.5f" % (np.sqrt(W_sum), np.sqrt(Winc_sum))
        print self.label, ": ||Hb|| = %.2f  ||dHb|| = %.5f" % (np.sqrt(Hb_sum), np.sqrt(Hbinc_sum))
        print self.label, ": ||Vb|| = %.5f  ||dVb|| = %.6f" % (abs(self.vbias), abs(self.vbias_inc))


#-----------------------------------------------------------------------------
    def biases_for_visualization(self, tile_shape):
        """ Visualize Biases
        
        Prepares a visualization array for the hidden biases of the bases in
        the layer
        
        Input:
            tile_shape -- shape used to arrange the values for different bases

        Output:
             ret_array -- 2D array containing visualization of biases for 
                          each base in the shape tile_shape
        
        """
        
        all_biases = []
        for base in self.bases:
            all_biases.append(base.bias)
        ret_array = np.array(all_biases).reshape(tile_shape)
        return ret_array
        

#-----------------------------------------------------------------------------
    def simple_activations_for_visualization(self, tile_shape):
        """ Visualize Activations
        
        Prepares a visualization array for the hidden activations of the bases
        in the layer
        
        Input:
            tile_shape -- shape used to arrange the values for different bases

        Output:
             ret_array -- 2D array containing visualization of activations 
                          for each base in the shape tile_shape
        
        """
        
        all_acts = [] 
        for base in self.bases:
            all_acts.append(base.pos_activation)

        ret_array = np.array(all_acts).reshape(tile_shape)
        return ret_array

    def avg_filters_for_visualization(self, tile_shape, dir_path="./", save=False):
        """ Visualize Filters
        
        Prepares a visualization array for the filters of the bases in the
        layer. Filter for each base is computed as a weighted linear
        combination of the filters in the previous layer, where the weight to
        each filter in the previous layer is proportional to the sum of
        weights (weight vector of the network) originated from that base.
        
        
        Inputs:
            tile_shape -- shape used to arrange the values for different bases
            dir_path -- path to save the result image
            save -- whether to save the image to a file as well
        
        Output:
            all_weights -- 2D array containing visualization of filters
                            for each base in the shape tile_shape
        
        """

        # not implemeted for layer 1 yet
        if self.prv_layer == None:
            return

        filt_height = self.prv_layer.btmup_window_shape[0] + self.btmup_window_shape[0] - 1
        filt_width = self.prv_layer.btmup_window_shape[1] + self.btmup_window_shape[1] - 1

        filt_size = filt_height * filt_width
        all_filters = np.zeros((len(self.bases), filt_size))
        
        for base_idx in range(len(self.bases)):
            base_filter = np.zeros((filt_height, filt_width))
            for cnl_idx in range(len(self.prv_layer.bases)):
                # TODO: should be prv_layers's filter
                reg = 1
                if np.isnan((self.bases[base_idx].Wb.any())):
                    print "NaN weights while visualizing filters!"
                    exit(1)
                base_filter += conv(self.bases[base_idx].Wb[:, :, cnl_idx]**reg, self.prv_layer.bases[cnl_idx].Wb[:, :, 0], 'full')

            all_filters[base_idx, :] = np.reshape(base_filter, filt_size)

        img_shape = (filt_height, filt_width)
        all_filters = utils.tile_raster_images(all_filters, img_shape, tile_shape, tile_spacing = (1,1))
        all_filters = utils.normalize_image(all_filters)

        if save:
            # -- save the visualization array to a PNG file
            filename = dir_path + "/" + self.label + "-filters.jpg"
            #img = toimage(all_filters)
            #img.save(filename)
            scipy.misc.imsave(filename, all_filters)


            #if config.DEBUG_MODE:
                #img.show()

            print "Filters of", self.label, "were saved to", filename
        
        return all_filters

#-----------------------------------------------------------------------------
    def weights_for_visualization(self, channel, tile_shape, dir_path="./", save=False):
        """ Visualize Weights
        
        Prepares a visualization array for the bottom-up weights of the bases
        in the layer
        
        Input:
            channel -- index of the channel to whose corresponding weights will
                        be shown
            tile_shape -- shape used to arrange the values for different bases

        Output:
             all_weights -- 2D array containing visualization of weights for 
                            the specified channel of each base in the shape 
                            tile_shape
        
        """
        
        w_size = self.bases[0].Wb.shape[0]*self.bases[0].Wb.shape[1]
        all_weights = np.zeros((len(self.bases), w_size))
        
        for i in range(all_weights.shape[0]):
            if channel == None:
                for cnl in range(self.num_channels):
                    all_weights[i] += np.reshape(self.bases[i].Wb[:, :, cnl], w_size)
            else:
                all_weights[i] = np.reshape(self.bases[i].Wb[:, :, channel], w_size)
            
        img_shape = (self.bases[0].Wb.shape[0], self.bases[0].Wb.shape[1])
        all_weights = utils.tile_raster_images(all_weights, img_shape, tile_shape, tile_spacing = (1,1))
        all_weights = utils.normalize_image(all_weights)

        if save:
            # -- save the visualization array to a PNG file
            filename = dir_path + "/" + self.label + "-cnl_" + str(channel) + "-weights.jpg"
            #img = toimage(all_weights)
            #img.save(filename)
            scipy.misc.imsave(filename, all_weights)

            #if config.DEBUG_MODE:
                #img.show()

            print "Weights of", self.label, "were saved to", filename
            
        return all_weights

#-----------------------------------------------------------------------------
    def posstates_for_visualization(self, tile_shape):
        """ Visualize Positive States
        
        Prepares a visualization array for the states of the hidden units 
        of the bases in the layer inferred by positive data  
        
        Input:
            tile_shape -- shape used to arrange the values for different bases

        Output:
             all_states -- 2D array containing visualization of activations 
                           for each base in the shape tile_shape
        
        """
        
        s_size = self.bases[0].pos_states.shape[0]*self.bases[0].pos_states.shape[1]
        all_states = np.zeros((len(self.bases), s_size))

        for i in xrange(all_states.shape[0]):
            all_states[i] = np.reshape(self.bases[i].pos_states, s_size)
            
        img_shape = self.bases[0].pos_states.shape
        all_states = utils.tile_raster_images(all_states, img_shape, tile_shape, tile_spacing = (1,1))
        all_states = utils.normalize_image(all_states)
        return all_states

#-----------------------------------------------------------------------------
    def output_for_visualization(self, tile_shape, tile_spacing):
        """ Visualize Outputs
        
        Prepares a visualization array for the output of the bases in the 
        layer (taken from the pooling units)
        
        Input:
            tile_shape -- shape used to arrange the values for different bases
            tile_spacing -- number of space to put in between tiles of the
                            output array to make neighboring tiles 
                            distinguishable

        Output:
             ret_array -- 2D array containing visualization of outputs
                          for each base in the shape tile_shape
        
        """
        
        size = self.bases[0].pooling_units.size
        all_outputs = np.zeros((len(self.bases), size))

        for i in xrange(all_outputs.shape[0]):
            all_outputs[i] = np.reshape(self.bases[i].pooling_units, size)
            
        img_shape = self.bases[0].pooling_units.shape
        all_outputs = utils.tile_raster_images(all_outputs, img_shape, tile_shape, tile_spacing)
        all_outputs = utils.normalize_image(all_outputs)
        return all_outputs

#-----------------------------------------------------------------------------
    def visualize_to_files(self, tile_shape, dir_path):
        """

        Saves the weight vector, and filters to files as images. More images
        can easily be added if needed.

        """
        
        for cnl in range(self.num_channels):
            print "Saving to file", cnl
            self.weights_for_visualization(cnl, tile_shape, dir_path, save=True)

        self.avg_filters_for_visualization(tile_shape, dir_path, save=True)

        # -- visualize the positive data
        # -- visualize the negagive data
        for cnl in range(self.num_channels):
            img = toimage(self.pos_data[:, :, cnl])
            filename = dir_path + "/" + self.label + "-cnl_" + str(cnl) + "-pos_data.png"
            img.save(filename)
            
            img = toimage(self.neg_data[:, :, cnl])
            filename = dir_path + "/" + self.label + "-cnl_" + str(cnl) + "-neg_data.png"
            img.save(filename)
        
        filename = dir_path + "/" + self.label + "-poooling.png"
        all_outputs = self.output_for_visualization(tile_shape, tile_spacing = (1,1))
        scipy.misc.imsave(filename, all_outputs)
        
        
#-----------------------------------------------------------------------------
    def visualize_bu_weights(self, label):
        
        all_weights = self.weights_for_visualization()
        img = Image.frombuffer('L', (all_weights.shape[1], all_weights.shape[0]), all_weights, "raw", 'L', 0, 1)
        img = img.resize((img.size[0]*5, img.size[1]*5), Image.NEAREST)
        #img.show()
        #img.save(label + ".tif")
        

