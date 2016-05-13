#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" utils module

This module contains different utility functions that are not specific to the
network, but rather are general purpose utility functions that perform useful
computations needed for visualization, timing, logging, etc.

For example ``tile_raster_images`` helps in generating a easy to grasp 
image from a set of samples or weights.

"""

from __future__ import with_statement
import time
import numpy as np
import Image
import math
import copy
from numpy.fft import fft2, ifft2, fftshift

import config


class Timer(object):
    """
    
        Records the time elapsed to finish running a block of code.
        Usage:
            timer = utils.Timer('label_for_timer')
            with timer:
                CODE_BLOCK
    """
    
    def __init__(self, label):
        self.label = label
        
    def __enter__(self):
        self.__start = time.time()

    def __exit__(self, type, value, traceback):
        # Error handling here
        self.__finish = time.time()
        print 'Time spend on %s: %.10f secs' %(self.label, self.__finish - self.__start)

    def duration_in_seconds(self):
        return self.__finish - self.__start



def trim_array_maxpool(arr=None, arr_shape=None, conv_window_shape=0, pooling_shape=0):
    """
    
        Trim the data array to ease the max pooling and convolution operations
        It trims the sides of the arr so that the width and height of the array 
        resulted from convolution is divisible by pooling_shape.

        Inputs:
            arr -- the array on which the convolution is performed
            arr_shape -- shape of arr
            conv_window_shape -- shape of the convolution array
            pooling_shape -- shape of the pooling units
        
    """
    
    if (arr != None):
        h = arr.shape[0]
        w = arr.shape[1]
    else:
        h = arr_shape[0]
        w = arr_shape[1]
        
    hc = conv_window_shape[0]
    wc = conv_window_shape[1]
    
    h_trim = (h-hc+1) % pooling_shape[0]
    t_trim = math.floor(h_trim/2.0)
    b_trim = math.ceil(h_trim/2.0)
    
    assert t_trim + b_trim == h_trim
    
    w_trim = (w-wc+1) % pooling_shape[1]
    l_trim = math.floor(w_trim/2.0)
    r_trim = math.ceil(w_trim/2.0)
    
    assert l_trim + r_trim == w_trim
    
    new_shape = (h-h_trim, w-w_trim)
    
    if (arr_shape):
        return new_shape    
    
    new_arr = np.zeros(new_shape)
    new_arr[0:new_shape[0], 0:new_shape[1]] = arr[t_trim:h-b_trim, l_trim:w-r_trim]

    assert arr[4,3] == new_arr[4-t_trim, 3-l_trim]
    #assert new_arr[:, :].any() == arr[t_trim:h-b_trim, l_trim:w-r_trim].any()

    return new_arr


def normalize_image(img, MIN_PIXEL=0, MAX_PIXEL=255):
    """

    Normalize an image so every pixel falls between 0 and MAX_PIXEL.

    Inputs:
        img -- the image array to normalize 
        MAX_PIXEL -- maximum pixel
        
    """
    
    m = np.min(img)
    #print "m=",m
    M = np.max(img)
    #print "M=",M
    #return np.int_(MAX_PIXEL * (img - m) / float(M - m))
    if M == m:
        return m * np.ones(img.shape)
        
    return MIN_PIXEL + (MAX_PIXEL-MIN_PIXEL) * (img - m) / float(M - m)

def visualize_array(arr):
    """

    Shows an array using Pythin Image Library.

    Input:
        arr -- the array to show
        
    """
    
    arr = normalize_image(arr)
    img = Image.frombuffer('L', (arr.shape[0], arr.shape[1]), arr)
    img.show()

def tile_raster_images(X, img_shape, tile_shape,tile_spacing=(0,0),
              scale_rows_to_unit_interval=True, output_pixel_vals=True):
    """
    Transform an array with one flattened image per row, into an array in 
    which images are reshaped and layed out like tiles on a floor.

    This function is useful for visualizing datasets whose rows are images, 
    and also columns of matrices for transforming those rows 
    (such as the first layer of a neural net).

    Inputs:
        X -- a 2-D array in which every row is a flattened image.
        img_shape -- the original shape of each image
        tile_shape -- the number of images to tile (rows, cols)
        scale_rows_to_unit_interval -- if the values need to be scaled before
                                       being plotted to [0,1] or not
        output_pixel_vals -- if output should be pixel values (i.e. int8
                             values) or floats


    Outputs:
        out_array -- array suitable for viewing as an image.

    """
 
    assert len(img_shape) == 2
    assert len(tile_shape) == 2
    assert len(tile_spacing) == 2

    # The expression below can be re-written in a more C style as 
    # follows : 
    #
    # out_shape    = [0,0]
    # out_shape[0] = (img_shape[0]+tile_spacing[0])*tile_shape[0] -
    #                tile_spacing[0]
    # out_shape[1] = (img_shape[1]+tile_spacing[1])*tile_shape[1] -
    #                tile_spacing[1]
    out_shape = [(ishp + tsp) * tshp - tsp for ishp, tshp, tsp 
                        in zip(img_shape, tile_shape, tile_spacing)]

    if isinstance(X, tuple):
        assert len(X) == 4
        # Create an output numpy ndarray to store the image 
        if output_pixel_vals:
            out_array = np.zeros((out_shape[0], out_shape[1], 4), dtype='uint8')
        else:
            out_array = np.zeros((out_shape[0], out_shape[1], 4), dtype=X.dtype)

        #colors default to 0, alpha defaults to 1 (opaque)
        if output_pixel_vals:
            channel_defaults = [0,0,0,255]
        else:
            channel_defaults = [0.,0.,0.,1.]

        for i in xrange(4):
            if X[i] is None:
                # if channel is None, fill it with zeros of the correct 
                # dtype
                dt = out_array.dtype
                if output_pixel_vals:
                    dt = 'uint8'
                out_array[:,:,i] = np.zeros(out_shape,
                        dtype=dt)+channel_defaults[i]
            else:
                # use a recurrent call to compute the channel and store it 
                # in the output
                out_array[:,:,i] = tile_raster_images(X[i], img_shape, tile_shape, tile_spacing, scale_rows_to_unit_interval, output_pixel_vals)
        return out_array

    else:
        # if we are dealing with only one channel 
        H, W = img_shape
        Hs, Ws = tile_spacing

        # generate a matrix to store the output
        out_array = 255*np.ones(out_shape)


        for tile_row in xrange(tile_shape[0]):
            for tile_col in xrange(tile_shape[1]):
                if tile_row * tile_shape[1] + tile_col < len(X):
                    if scale_rows_to_unit_interval:
                        # if we should scale values to be between 0 and 1 
                        # do this by calling the `scale_to_unit_interval`
                        # function
                        this_img = scale_to_unit_interval(X[tile_row * tile_shape[1] + tile_col].reshape(img_shape))
                    else:
                        this_img = X[tile_row * tile_shape[1] + tile_col].reshape(img_shape)
                    # add the slice to the corresponding position in the 
                    # output array
                    c = 1
                    if output_pixel_vals:
                        c = 255
                    out_array[
                        tile_row * (H+Hs):tile_row*(H+Hs)+H,
                        tile_col * (W+Ws):tile_col*(W+Ws)+W
                        ] \
                        = this_img * c
        return out_array


def scale_to_unit_interval(ndar,eps=1e-8):
    """

    Scales all values in the ndarray ndar to be between 0 and 1

    """
    
    ndar = ndar.copy()
    ndar -= ndar.min()
    ndar *= 1.0 / (ndar.max()+eps)
    return ndar
 
def olshausen_whitening( img):
    """ Return a whitened image as a numpy array

    Performs image whitening as described in Olshausen & Field's Vision 
    Research article (1997)

    f_0 controls the radius of the decay function. 
    n controls the steepness of the radial decay.

    Input:
        img -- image in numpy array

    Outputs:
        img -- result image in numpy array format

    """
    
    [iw, ih] = img.shape

    # Let all images be 3D (MxNxC, where C is number of channels) for consistency
    img = img.reshape(img.shape[0], img.shape[1])

    stdd = np.std(img)
    
    if stdd != 0:
        img = (img - np.mean(img)) / stdd

    X, Y = np.meshgrid(np.arange(-iw/2, iw/2), np.arange(-ih/2, ih/2))

    f_0 = 0.4 * np.mean([iw,ih])  # another source used min
    n = 4
    rho = np.sqrt(X**2 + Y**2)

    filt = rho * np.exp(-(rho/f_0)**n)  # low_pass filter

    img[:, :] = ifft2(fft2(img[:, :]) * fftshift(filt)).real

    stdd = np.std(img)
    if stdd != 0:
        img /= stdd

    #print img[0:9, 0:9]
    #tt = Image.frombuffer('L', (iw, ih), img)
    #tt.show()

    return img
