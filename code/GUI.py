#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" GUI module

This modules contains a simple graphical user interface, developed using the 
PyGtk framework to visualize a two-layer network. It can be easily extended to 
visualize deeper networks and more information about the network.

"""

import sys
import pygtk
import gobject
import time
import threading
import numpy as np
import scipy as sp
import Image
import pylab as pl

import main
import convDBN
import utils
import config as cfg

import pycuda_convolution

import pycuda.driver as cuda

import main_thread


try:
    import pygtk
    pygtk.require('2.0')
    import gtk
except:
    print 'Please install pygtk,libgtk2.0 '
    import os
    os.exit(1)

# -----------------------------------------------------------------------------
class GUI(gtk.Window, gtk.ScrolledWindow):
    def __init__(self, parent=None):
        """
            Initializes the GTK application, in our case, create the window
            and other widgets, including control buttons, labels and images
            visualizing different aspects of the network.
            
        """
        # -- initialize the window
        gtk.Window.__init__(self)
        try:
            self.set_screen(parent.get_screen())
        except AttributeError:
            self.connect('destroy', lambda *w: gtk.main_quit())

        self.connect("destroy", self.destroy)

        self.set_border_width(8)
        self.set_title("Convolutional Deep Belief Networks")


        # list of control buttons
        self.buttons = {}

        # horizontal box contating the entire window
        hbox = gtk.HBox(False, 5)
        self.add(hbox)
        self.set_border_width(5)
        
        # ---------- create the vertical box with the control buttons ---------
        vbox_buttons = gtk.VBox(False, 8)
        vbox_buttons.set_border_width(8)
        hbox.pack_start(vbox_buttons, False, False, 0)


        # -- add the buttons to the vertical box and bind them to their
        # corresponding functions
        # create the start button
        self.add_button(label="Start", func=self.start_simulation, box=vbox_buttons)
        # create the save button
        self.add_button(label="Save all", func=self.save_all, box=vbox_buttons)
        # create the save button
        self.add_button(label="Load a saved network", func=self.load_openDialog, box=vbox_buttons)
        # create the plot button
        self.add_button(label="Plot error", func=self.make_plots, box=vbox_buttons)
        # create the exit button
        self.add_button(label="Exit", func=lambda wid: gtk.main_quit(), box=vbox_buttons)

        # -- dict of lists of labels for images to show
        self.txt_images_to_show = {\
            # -- Layer 1 info
            1: [
                "Current input image", \
                "Layer 1 Weights", \
                "Layer 1 output (pooling units)", \
                "Layer 1 positive activations", \
                "Layer 1 biases", \
                "Layer 1 negative data", \
               ],
            # -- Layer 2 info
            2: [
                "Layer 2 Weights", \
                "Layer 2 output (pooling units)", \
                "Layer 2 positive activations", \
                "Layer 2 biases", \
                "Layer 2 negative data", \
                "Layer 2 Filters" \
                ]
            }

        # lists of images to show
        self.images = {}
        # create the image holders according to the list in txt_images_to_show
        for txt_label in self.txt_images_to_show[1] + self.txt_images_to_show[2]:
              self.images[txt_label] = gtk.Image()
        
        # ---------- Vertical box containing the info about Layer 1 -----------
        vbox_layer1 = gtk.VBox(False, 8)
        vbox_layer1.set_border_width(8)
        # add vbox_layer1 to the horizontal box
        hbox.pack_start(vbox_layer1, False, False, 0)
        # bind the images to the vbox_layer1 
        for lbl in self.txt_images_to_show[1]:
            self.add_image_to_box(lbl, vbox_layer1)
            
        # ---------- Vertical box containing the info about Layer 2 -----------
        vbox_layer2 = gtk.VBox(False, 8)
        vbox_layer2.set_border_width(8)
        # add vbox_layer2 to the horizontal box
        hbox.pack_start(vbox_layer2, False, False, 0)
        # bind the images to the vbox_layer2
        for lbl in self.txt_images_to_show[2]:
            if lbl != "Layer 2 Filters":
                self.add_image_to_box(lbl, vbox_layer2)

        vbox_layer2_filts = gtk.VBox(False, 8)
        vbox_layer2_filts = gtk.VBox(False, 8)
        vbox_layer2_filts.set_border_width(8)
        # add vbox_layer2 to the horizontal box
        hbox.pack_start(vbox_layer2_filts, False, False, 0)
        # bind the images to the vbox_layer2
        self.add_image_to_box("Layer 2 Filters", vbox_layer2_filts)

        
        self.show_all() 

        # whether the network is loaded from a saved file
        self.network_loaded = False 

    def save_all(self, widget):
        """

        Save all the images shown and pickle the entire network into results_dir
        
        """
        
        for img in self.txt_images_to_show[1] +  self.txt_images_to_show[2]:
            buf = self.images[img].get_pixbuf()
            if (buf): 
                path = misc_params['results_dir']+img
                print "Saving %s in %s" %(img, path)
                buf.save(path, "png")
        
        print "\n** Pickling the entire network..\n"
        main.network.pickle(misc_params['results_dir'] + misc_params['pickle_fname'])
        
    def make_plots(self, widget):
        """

        Make plots using the pylab tools (currently only the error plot)

        """

        # -- open the error file and parse it
        errfile = open(misc_params['results_dir']+misc_params['err_fname'], 'r')
        errs =  map(float, errfile.readline().split())
        errfile.close()
        if len(errs) == 0:
            self.alert("No error has been recorded yet. Try again after a few epochs of training.")
            return
            
        print "Error in epochs:", errs
        pl.plot(range(len(errs)), errs)
        pl.xlabel('Epoch')
        pl.ylabel('error ||pos_data-neg_data||')
        pl.title('Decline of error in epochs')
        pl.grid(True)
        pl.show()
        
    def add_image_to_box(self, txt_label, box):
        """

        Add an image to a box

        Inputs:
            txt_label -- label of the image
            box -- the box to contain the image

        """
        
        frame = gtk.Frame(txt_label)
        frame.set_shadow_type(gtk.SHADOW_IN)
        align = gtk.Alignment(0.5, 0.5, 0, 0)
        align.add(frame)
        box.pack_start(align, False, False, 0)
        
        frame.add(self.images[txt_label])
    
    def refresh(self):
        """

        Refreshes the data shown in GUI (fetches them from the network) at each
        click of the refresh timer.

        """

        # -- Do not try to refresh GUI, if the network is not created or the
        # input image is not loaded yet
        if (not hasattr(main.network, 'layers')) or main.data.curr_input == None:  
            return True

        # visualize only the first channel
        # TODO: could be a linear sum or an RGB combination of the channels
        data = main.data.curr_input[:, :, 0]
        self.refresh_image("Current input image", data, 1)

        # ------------------- Refresh the info about Layer 1 ------------------
        tile_shape = (4, 6)  # how to arrange the bases in the layer
        tile_spacing = (1,1)  # distance between tiles
        
        data = main.network.layers[0].neg_data
        self.refresh_image("Layer 1 negative data", data, scale_factor=1)
        
        all_weights = main.network.layers[0].weights_for_visualization(channel=None, tile_shape=tile_shape)
        self.refresh_image("Layer 1 Weights", all_weights, scale_factor=2)
        
        all_outputs = main.network.layers[0].output_for_visualization(tile_shape, tile_spacing)
        self.refresh_image("Layer 1 output (pooling units)", all_outputs, scale_factor=1)
        
        if hasattr(main.network.layers[0], 'activations_for_visualization'):
            all_acts = main.network.layers[0].activations_for_visualization(tile_shape)
            self.refresh_image("Layer 1 positive activations", all_acts, scale_factor=10)
        
        all_biases = main.network.layers[0].biases_for_visualization(tile_shape)
        self.refresh_image("Layer 1 biases", all_biases, scale_factor=20)

        # ------------------- Refresh the info about Layer 1 ------------------
        # to speed up the visualization before Layer 2 has been learned at all
        if main.network.layer_to_learn > 0:
            tile_shape = (10,10)
            cnl = np.random.randint(main.network.layers[1].num_channels)
            # TEST
            cnl = 0
            all_weights = main.network.layers[1].weights_for_visualization(cnl, tile_shape)
            self.refresh_image("Layer 2 Weights", all_weights, scale_factor=2)
            
            data = main.network.layers[1].neg_data[:, :, cnl]
            self.refresh_image("Layer 2 negative data", data, scale_factor=4)

            all_outputs = main.network.layers[1].output_for_visualization(tile_shape, tile_spacing)
            self.refresh_image("Layer 2 output (pooling units)", all_outputs, scale_factor=2)
            
            all_acts = main.network.layers[1].activations_for_visualization(tile_shape)
            self.refresh_image("Layer 2 positive activations", all_acts, scale_factor=10)
            
            all_biases = main.network.layers[1].biases_for_visualization(tile_shape)
            self.refresh_image("Layer 2 biases", all_biases, scale_factor=10)

            all_weights = main.network.layers[1].avg_filters_for_visualization(tile_shape=(10,10))
            self.refresh_image("Layer 2 Filters", all_weights, scale_factor=2)
        
        return True
        
    def refresh_image(self, txt_label, data, scale_factor):
        """

        Preprocess a data array (scaling and normalizing) and load it to a Gtk
        image holder, pixbuf

        Input:
            txt_label -- text label of the image
            data -- data array to show
            scale_factor -- scale factor for resizing the image

        """

        data = utils.normalize_image(data)
        
        h = data.shape[0]
        w = data.shape[1]
        data.shape = (h, w, -1)
        pixbuf = gtk.gdk.Pixbuf(gtk.gdk.COLORSPACE_RGB, False, 8, w, h)
        pixbuf.pixel_array[:] = data
        pixbuf = pixbuf.scale_simple(w*scale_factor, h*scale_factor, gtk.gdk.INTERP_TILES)

        self.images[txt_label].set_from_pixbuf(pixbuf)

    def go(self, widget, data=None):
        """

        Starts the GUI refresh timer and calls the main to start the network.

        Input:

        """
        self.start_time = time.clock()
        timer_timeout = misc_params['GUI_refresh_timeout'] #ms
        gobject.timeout_add(timer_timeout, self.refresh)
        
        cuda.init()
        the_thread = main_thread.MainThread(False)
        the_thread.start()

        return
        
        
        # TEST
        #main.main(self.network_loaded)
        #main.network = convDBN.Network.unpickle('./results/savedAfter30/Saved_cdbn_net-natural_imgs.dump')
        #main.main(True)
        main.main(False)
        
 
    def start_simulation(self, widget, data=None):
        """

        Starts the thread for running the network.

        Input:

        """
        self.buttons['Start'].set_sensitive(False)
        # disable the load button
        self.buttons['Load a saved network'].set_sensitive(False)  
        main_thread = threading.Thread(target=self.go, args=(widget, data))
        #main_thread.setDaemon(True)
        main_thread.start()

    def add_button(self, label, func, box):
        """

        Adds a button with a given label to a box and connects it to a fucntion.

        Input:
            label -- label to show on the button
            func -- function to call on button click
            box -- the box containing the button

        """

        self.buttons[label] = gtk.Button(label)
        self.buttons[label].connect("clicked", func)
        frame = gtk.Frame()
        frame.set_shadow_type(gtk.SHADOW_IN)
        align = gtk.Alignment(0.5, 0.5, 0, 0)
        align.add(frame)
        box.pack_start(align, False, False, 0)
        frame.add(self.buttons[label])


    def start_window(self):
        """
            This function starts GTK drawing the GUI and responding to events
            such as button clicks.
        """
 
        gtk.main()
 
    def destroy(self, widget, data=None):
        """
            This function exits the application when the window is closed.
        """
 
        gtk.main_quit()

    def alert(self, message = "Alert"):
        """

        Show an alert message.

        Input:
            message -- the message to show_all

        """
        
        alert = gtk.MessageDialog(
            None,
            gtk.DIALOG_MODAL | gtk.DIALOG_DESTROY_WITH_PARENT,
            gtk.MESSAGE_QUESTION,
            gtk.BUTTONS_OK,
            None)
        alert.set_markup(message)
        alert.show_all()
        
        alert.run()
        alert.destroy()
        
    def load_openDialog(self, widget):
        """

        shows an open dialog box to choose a pickeled file to load the network
        from

        """
        
        chooser = gtk.FileChooserDialog(title=None, \
                                          action=gtk.FILE_CHOOSER_ACTION_OPEN, \
                                          buttons=(gtk.STOCK_CANCEL,gtk.RESPONSE_CANCEL, \
                                          gtk.STOCK_OPEN,gtk.RESPONSE_OK))
        dialog = gtk.FileChooserDialog("Open saved model..",
                                      None,
                                      gtk.FILE_CHOOSER_ACTION_OPEN,
                                      (gtk.STOCK_CANCEL, gtk.RESPONSE_CANCEL,
                                        gtk.STOCK_OPEN, gtk.RESPONSE_OK))
        dialog.set_default_response(gtk.RESPONSE_OK)
        response = dialog.run()
        if response == gtk.RESPONSE_OK:
            fname = dialog.get_filename()
            print fname, 'was selected to be loaded ...'
        elif response == gtk.RESPONSE_CANCEL:
            print 'Closed, no files selected!'
        dialog.destroy()


        main.network = convDBN.Network.unpickle(fname)

        self.alert("Learned network was successfuly loaded from %s. \n\
        The network won\'t learn in this mode, but rather just sample the hidden layers" % fname)

        self.network_loaded = True

        
if __name__ == "__main__":
  
    # load the parameters file
    exec(compile(open(cfg.params_filename).read(), cfg.params_filename, 'exec'))
    # Create an instance of our GTK application
    gtk.gdk.threads_init()
    app = GUI()
    gtk.gdk.threads_enter()
    app.start_window()
    gtk.gdk.threads_leave()