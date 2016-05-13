#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" image_data module

All the operations performed on image data

"""

import os
import Image
import scipy as sp
import numpy as np 

from numpy.fft import fft2, ifft2, fftshift
from math import *

#import pylab as pl   # matplotlib

import utils

class ImageData:
      
    def __init__(self, EXTENSIONS, max_image_edge, num_channels):
        self.extentions = EXTENSIONS
        self.num_channels = num_channels
        self.max_image_edge = max_image_edge
        self.images = []  # list of images to be loaded
        self.curr_input = None
        
    # -----------------------------------------------------------------------------
    def load_images(self, img_path):
        """ Load all the image files recursively

        Input:
          img_path -- the absolute path of a directory containing images
        
        Output:
          appends all the images to globals.images
        
        """
        
        # -- convert to absolute path and verify the path
        img_path = os.path.abspath(img_path)
        print "Image source:", img_path
        if not os.path.isdir(img_path):
            raise ValueError, "%s is not a directory" % (img_path)
        
        # -- extract the file names
        tree = os.walk(img_path)
        filelist = []
        #categories = tree.next()[1]    
        for path, dirs, files in tree:
            if dirs != []:
                msgs = ["invalid image tree structure:"]
                for d in dirs:
                    msgs += ["  "+"/".join([root, d])]
                msg = "\n".join(msgs)
                raise Exception, msg
            filelist += [ path+'/'+f for f in files if os.path.splitext(f)[-1] in self.extentions ]
        filelist.sort()    
        
        # -- load and preprocess images
        for img_fname in filelist: #[0:1]:
            img = self.load_process_image(img_fname)
            self.images.append(img)
            #utils.visualize_array(img)
        
        #print len(categories), "categories found:"
        #print categories
        


    # -----------------------------------------------------------------------------
    def load_process_image(self, img_fname):
        """ Return a preprocessed image as a numpy array

        Inputs:
          img_fname -- image filename

        Outputs:
          imga -- result

        """

        print "loading "+img_fname.split('/')[-1]
        # -- open image
        img = Image.open(img_fname)
        
        print "preprocessing "+img_fname.split('/')[-1]+" ...\n"
        # -- resize and whiten image
        img = self.preprocess(img)
        
        return img
        
    # -----------------------------------------------------------------------------
    def preprocess(self, img):
        """ Return a resized and whitened image as a numpy array
        
        The following steps are performed:
        
        1) The image is resized so the longest edge is of size max_edge, while 
        keeping the width-height ratio intact.
        2) The resized image is whitened using the Olshausen & Field (1997) 
        alogorithm.
        
        Inputs:
          img -- image in python Image format
          max_edge -- maximum edge length

        Outputs:
          imga -- result

        """
        
        iw, ih = img.size
        
        # -- resize so that the biggest edge is max_edge (keep aspect ratio)
        if iw > ih:
            new_iw = self.max_image_edge
            new_ih = int(round(1.* self.max_image_edge * ih/iw))
        else:
            new_iw = int(round(1.* self.max_image_edge * iw/ih))
            new_ih = self.max_image_edge
        
        img = img.resize((new_iw, new_ih), Image.BICUBIC)

        # -- convert the image to greyscale (mode "L" for luminance)
        if img.mode != 'L':
            img = img.convert('L')
        
        # -- perform olshausen whitening on the image
        imga = self.olshausen_whitening(img)

        
        # -- perform some extra normalization
        imga -= np.mean(imga)
        imga /= sqrt(np.mean(imga**2))
        imga *= sqrt(0.1) # for some tricks?! :TODO
        #imga = utils.normalize_image(imga, -0.01, 0.01)
        return imga

    # -----------------------------------------------------------------------------
    def olshausen_whitening(self, img):
        """ Return a whitened image as a numpy array
        
        Performs image whitening as described in Olshausen & Field's Vision 
        Research article (1997)

        f_0 controls the radius of the decay function. 
        n controls the steepness of the radial decay.

        Input:
          img -- image in python Image format

        Outputs:
          img -- result image in numpy array format

        """
        iw, ih = img.size
        # -- The different color channels are stored in the third dimension, 
        # such that a grey-image is MxN, an RGB-image MxNx3 and an RGBA-image MxNx4.
        img = sp.misc.fromimage(img, flatten = False).real  # convert image to numpy array
        
        # Let all images be 3D (MxNxC, where C is number of channels) for consistency
        img = img.reshape(img.shape[0], img.shape[1], self.num_channels)
        
        img = (img - np.mean(img)) / np.std(img)

        X, Y = np.meshgrid(np.arange(-iw/2, iw/2), np.arange(-ih/2, ih/2))
        
        f_0 = 0.4 * np.mean([iw,ih])  # another source used min
        n = 4
        rho = np.sqrt(X**2 + Y**2)
        
        filt = rho * np.exp(-(rho/f_0)**n)  # low_pass filter
        
        for cnl in range(self.num_channels):
            img[:, :, cnl] = ifft2(fft2(img[:, :, cnl]) * fftshift(filt)).real

        img /= np.std(img)
        
        #print img[0:9, 0:9]
        #tt = Image.frombuffer('L', (iw, ih), img)
        #tt.show()
        
        
        return img

    # -----------------------------------------------------------------------------
    def get_image_patch(self, img_index, patch_shape):
        """ Return a random patch of the image (preprocessed) with the given index
        
        Inputs:
          img_index -- index of the image to be used
          patch_shape -- shape of the image patch to be returned

        Outputs:
          img_patch -- path from the image with the given size

        """
        rows, cols = self.images[img_index].shape[0], self.images[img_index].shape[1]
        
        cx = np.random.randint(rows - patch_shape[0]) 
        cy = np.random.randint(cols - patch_shape[1])
        
        image_patch = self.images[img_index][cx:cx+patch_shape[0], cy:cy+patch_shape[1], :]
        

        # -- preprocess image_patch
        image_patch -= np.mean(image_patch)

        if np.random.rand() > 0.5:
            image_patch = np.fliplr(image_patch)


        return image_patch


   