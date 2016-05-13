#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" base module

This module contains definition of a basis which is a filter (unit) that 
convolves over the output from the previous layer (or the input image for the 
very first layer). Units in the base (represented in 2D arrays of states and
probabilities here) share the same weight matrix and bias, but they receive
different (local) input.

Note: Terms "basis", "base" and "group" are used interchangeably here 
and in the paper

"""

import sys
import time
import Image
import numpy as np
import scipy as sp
import scipy.signal
import cPickle
import utils
import config

# "cimport" is used to import special compile-time information
# about the numpy module (this is stored in a file numpy.pxd which is
# currently part of the Cython distribution).
cimport numpy as np
# We now need to fix a datatype for our arrays. I've used the variable
# DTYPE for this, which is assigned to the usual NumPy runtime
# type info object.
DTYPE = np.float32
# "ctypedef" assigns a corresponding compile-time type to DTYPE_t. For
# every type in the numpy module there's a corresponding compile-time
# type with a _t-suffix.
ctypedef np.float32_t DTYPE_t

cimport cython

  
import pycuda_convolution
#conv = scipy.signal.convolve
conv = pycuda_convolution.convolve_gpu

cdef inline float float_sum(np.ndarray[DTYPE_t, ndim=2] arr, int w, int h): 
    cdef float s = 0
    cdef int i, j
    for i in range(h):
        for j in range(w):
            s += arr[i, j]
    return s


# -----------------------------------------------------------------------------
class Base:
    def __init__(self, my_layer):
        """ Constructor
        
        Input:
            my_layer -- the layer to which the base belongs to
        
        """
        
        self.my_layer = my_layer

        # -- copy parameters from my_layer for the sake of readability
        self.hidden_shape = my_layer.hidden_shape
        self.num_channels = my_layer.num_channels
        self.bias = my_layer.init_bias
        # -- shape of the black of hidden units that compete in max pooling,
        # equivalent to C in the paper
        self.block_shape = self.my_layer.block_shape

        if self.my_layer.label == "Layer_1": #TODO:
            c = 0.001
        elif self.my_layer.label == "Layer_2":
            c = 0.01 # 0.0005

        win_shape = my_layer.btmup_window_shape
        # Bottom-up weights vector of the group
        self.Wb = c * np.random.randn(win_shape[0], win_shape[1], self.num_channels)

        # increment in bottom-up weight when updating
        self.Wb_inc = np.zeros((win_shape[0], win_shape[1], self.num_channels))

        # Top-dwon weights vector of the group
        self.Wt = 0  # TODO

        # states of the hidden units during the positive and negative phases
        self.pos_states = np.zeros(self.hidden_shape)
        self.pos_probs = np.zeros(self.hidden_shape)
        
        # increment in bias while updating
        self.bias_inc = 0
        
        # output of the pooling units after max pooling operation
        self.pooling_units = np.zeros(self.my_layer.output_shape)
        
        # -- activation array the hidden units after positive and negative phases 
        self.pos_activation = 0
        self.neg_activation = 0
      
# -----------------------------------------------------------------------------
  
    
    @cython.boundscheck(False) # turn of bounds-checking for entire function
    def sample_prob_max_pooling(self, np.ndarray[DTYPE_t, ndim=2] exps):
        """ Probabilistic Max Pooling
        
        Sample the group, and compute the activation of the hidden and pooling 
        units using the given exponentials of the incoming signals to the units
            
        Input:
            exps -- exponential of the incoming signals (including biases) to 
            the hidden units in the group
        
        Outputs:
            probs -- probability of each unit to become active from its inputs
            states -- binary activation matrix  of the hidden units
            self.pooling_units -- (not a "return" output) - binary activation 
            matrix  of the pooling units
        """
        
        # -- create the numpy output arrays
        cdef np.ndarray[DTYPE_t, ndim=2] probs = np.zeros(self.hidden_shape, dtype=DTYPE)
        cdef np.ndarray[DTYPE_t, ndim=2] states = np.zeros(self.hidden_shape, dtype=DTYPE)
        
        cdef int x_start, y_start, x_end, y_end
        cdef int hidden_shape_0 = self.hidden_shape[0]
        cdef int hidden_shape_1 = self.hidden_shape[1]

        cdef int block_shape_0 = self.block_shape[0]
        cdef int block_shape_1 = self.block_shape[1]

        cdef np.ndarray[DTYPE_t, ndim=2] block_exps = np.zeros((block_shape_0, block_shape_1), dtype=DTYPE)

        cdef float rnd_val
        cdef float sum_exp
        
        # -- Loop for sampling and max pooling operations
        # Each iteration takes care of one block of units.
        for x_start in xrange(0, hidden_shape_0, block_shape_0):
            for y_start in xrange(0, hidden_shape_1, block_shape_1):
                # -- set the start and end indices for the block
                x_end = x_start + block_shape_0
                y_end = y_start + block_shape_1
                
                # block of the units to perform max pooling on
                block_exps = exps[x_start: x_end, y_start: y_end]
                #sum_exp = np.sum(block_exps) # This is so damn slow!!
                sum_exp = float_sum(block_exps, block_shape_0, block_shape_1)
                if config.DEBUG_MODE:
                    assert sum_exp >= 0
                
                # whether another unit in the block is active already
                already_chosen = False

                rnd_val = np.random.rand()

                # TODO: try letting the max index to fire
                # -- compute the state of each unit in the block
                for i in xrange(x_start, x_end):
                    for j in xrange(y_start, y_end):
                        #  Equation at the end of Section 3.6 in the paper
                        probs[i, j] = exps[i, j] / (1 + sum_exp)
                                                
                        if (probs[i, j] > rnd_val) and ~already_chosen:
                            states[i, j] = 1
                            already_chosen = True
                        # -- The two lines below are kept only for code readability.
                        #else:  
                        #    states[i, j] = 0
                
                # -- set the activation of the pooling units using their 
                # corresponding hidden units
                i = x_start / block_shape_0  # row of the pooling unit
                j = y_start / block_shape_1  # column of the pooling unit
                if already_chosen: 
                    self.pooling_units[i, j] = 1
                else:
                    self.pooling_units[i, j] = 0

        #print "self.pooling_units:", np.sum(self.pooling_units)
        return (probs, states)

# -----------------------------------------------------------------------------
    def sample(self, bu_data, td_data=0):
        """ Sampling Given Input Data

        Inputs:
        bu_data -- bottom-up input array
        td_data -- top-down input array

        Outputs:
            probs -- same as probs in function sample_prob_max_pooling()
            states -- same as states in function sample_prob_max_pooling()
            activations -- sum of probabilities to fire (probs) in the group

        """
        cdef float activation

        # -- roughly Equation (3) in the paper
        bu_energy = 0
        
        #timer = utils.Timer('convolution')
        #with timer:  # measures the time
        for channel in range(self.num_channels):
            bu_energy += conv(bu_data[:, :, channel], self.Wb[:, :, channel], 'valid')
            #print "self.Wb[:, :, channel]", channel, self.Wb[:, :, channel]
        
        if config.DEBUG_MODE:        
            assert self.my_layer.std_gaussian != 0
            
        sigma = 1.0/(self.my_layer.std_gaussian**2)
        bu_energy = sigma * bu_energy + self.bias
        
        # -- roughly Equation (4) in the paper
        td_energy = 0 # TODO: top-down input to a layer 
        energy = bu_energy + td_energy
        #energy -= np.mean(energy)
        #energy  /= sqrt(np.mean(energy**2))
        #M = np.max(energy)
        #energy /= M
        exps = np.exp(energy)
        #-- debugging assertions
        if config.DEBUG_MODE:
            if np.isinf(exps).any():
                print "bu_data:", bu_data
                print "bu_energy:", bu_energy
                print "exps:", exps
                
            assert ~np.isnan(bu_energy).any()
            assert ~np.isnan(exps).any()
            assert ~np.isinf(exps).any()
            assert (exps >= 0).all()
            
        
        #timer = utils.Timer('sample_prob_max_pooling')
        #with timer:  # measures the time
        probs, states = self.sample_prob_max_pooling(exps) # P(h|v)
       
        activation = np.sum(probs)



        return (probs, states, activation)

# -----------------------------------------------------------------------------
    def pos_sample(self):
        """ Positive Sampling

        Samples the hidden units during the positive phase of the Gibbs sampling

        """
        
        if config.DEBUG_MODE:
            mystr = "pos_data for "+ self.my_layer.label+" is nan!"
            assert ~np.isnan(self.my_layer.pos_data).any(), mystr #'pos_data for %s is nan!' 
            
        #timer = utils.Timer('pos_sample')
        #with timer:  # measures the time
        self.pos_probs, \
        self.pos_states, \
        self.pos_activation = \
            self.sample(self.my_layer.pos_data)

        # -- debugging assertions
        if config.DEBUG_MODE:
            assert ~np.isnan(self.pos_probs).any()
            assert ~np.isnan(self.pos_states).any()
            assert ~np.isnan(self.pos_activation).any()
            
# -----------------------------------------------------------------------------
    def neg_sample(self):
        """ Negative Sampling

        Samples the hidden units during the negative phase of the Gibbs sampling

        """
        
        self.neg_probs, \
        self.neg_states, \
        self.neg_activations = \
            self.sample(self.my_layer.neg_data)
            
        # -- debugging assertions
        if config.DEBUG_MODE:
            assert ~np.isnan(self.neg_probs).any()
            assert ~np.isnan(self.neg_states).any()
            assert ~np.isnan(self.neg_activations).any()
        
   
# -----------------------------------------------------------------------------
    def update(self):
        """ Update Parameters -- the learning component

        Updates the values of the weight vector and biases using the Gradient
        Descent learning rule

        """
        
        # ---------------- (1) Update the weight vector/matrix ----------------
        # number of hidden units
        cnt = self.hidden_shape[0] * self.hidden_shape[1]
        
        # -- compute the convolution of probabilities matrix over data matrix
        # (both positive and negative) added up across channels
        pos_conv = np.zeros((self.Wb.shape[0], self.Wb.shape[1], self.num_channels))
        neg_conv = np.zeros((self.Wb.shape[0], self.Wb.shape[1], self.num_channels))
        for cnl in range(self.num_channels):
            p = self.my_layer.pos_data[:, :, cnl]
            pos_conv[:, :, cnl] += conv(p, np.flipud(np.fliplr(self.pos_probs)), 'valid')
        
            n = self.my_layer.neg_data[:, :, cnl]
            neg_conv[:, :, cnl] += conv(n, np.flipud(np.fliplr(self.neg_probs)), 'valid')
        
        # Gradient descent change (error) in the weight vector
        dW_GD = (pos_conv - neg_conv)/cnt  
        
        # Regularization change to limit the length of the weight vector
        dW_reg = -1 * self.my_layer.regL2 * self.Wb 
        
        # overall change in the weight vector
        dW = dW_GD  + dW_reg

        # amount of increment in weight vector
        self.Wb_inc =  self.my_layer.epsilon * dW        
        self.Wb += self.Wb_inc

        # ---------------- (2) Update the bias of the group -------------------
        # Regularization parameter to enforce sparsity
        sparsity_reg = np.mean(self.pos_probs) - self.my_layer.pbias
        # Gradient descent change (error) in the bias
        dH_GD = (self.pos_activation-self.neg_activations)/cnt 
        self.bias_inc =  self.my_layer.epsilon * dH_GD - self.my_layer.pbias_lambda * sparsity_reg
        self.bias += self.bias_inc
        # debugging assertions
        if config.DEBUG_MODE:
            assert ~np.isnan(pos_conv.any())
            assert ~np.isnan(neg_conv.any())
