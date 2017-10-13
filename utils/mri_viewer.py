import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.widgets import Slider

class MRIViewer:

    def __init__(self, im1, im2, cmap, title1='Image 1', title2='Image 2'):
        self.i = 0

        self.im1 = im1
        self.vmin1 = self.im1.min()
        self.vmax1 = self.im1.max()
        self.title1 = title1

        self.im2 = im2
        self.vmin2 = self.im2.min()
        self.vmax2 = self.im2.max()
        self.title2 = title2

        [self.depth, _, _] = self.im1.shape

        self.cmap = cmap

        self.fig = plt.figure()
        
        gs = gridspec.GridSpec(2, 2, height_ratios=[10, 1])

        self.ax1 = plt.subplot(gs[0, 0])
        self.ax1.grid(False)
        self.ax1.set_xticklabels([])
        self.ax1.set_yticklabels([])
        self.ax1.set_title(title1)

        self.ax2 = plt.subplot(gs[0, 1])
        self.ax2.grid(False)
        self.ax2.set_xticklabels([])
        self.ax2.set_yticklabels([])
        self.ax2.set_title(title2)

        self.slax = plt.subplot(gs[1, :])
        self.slider = Slider(self.slax, 'Slice',
                      0.01, self.depth - 0.01, valinit=0, valfmt='%d')

        def update(val):
            self.i = int(val)
            self.draw()
            self.fig.canvas.draw()

        self.slider.on_changed(update)

        self.draw()
        self.fig.canvas.draw()

        plt.ion()
        plt.show()

    def draw(self):
        im1 = self.im1[self.i, :, :]
        self.ax1.imshow(im1, vmin=self.vmin1, vmax=self.vmax1,
                        cmap=self.cmap, interpolation=None)
        
        im2 = self.im2[self.i, :, :]
        self.ax2.imshow(im2, vmin=self.vmin2, vmax=self.vmax2,
                        cmap=self.cmap, interpolation=None)
