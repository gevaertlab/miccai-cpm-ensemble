import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.widgets import Slider

class MRIViewer:

    def __init__(self, im1, im2, im3, im4, cmap, title1='Image 1', title2='Image 2', title3='Image3', title4='Image4'):
        self.i = 0

        self.im1 = im1
        self.vmin1 = self.im1.min()
        self.vmax1 = self.im1.max()
        self.title1 = title1

        self.im2 = im2
        self.vmin2 = self.im2.min()
        self.vmax2 = self.im2.max()
        self.title2 = title2

        self.im3 = im3
        self.vmin3 = self.im3.min()
        self.vmax3 = self.im3.max()
        self.title3 = title3

        self.im4 = im4
        self.vmin4 = self.im4.min()
        self.vmax4 = self.im4.max()
        self.title4 = title4

        [self.depth, _, _] = self.im1.shape

        self.cmap = cmap

        self.fig = plt.figure()
        
        gs = gridspec.GridSpec(4, 2, height_ratios=[10, 1, 10, 1])

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

        self.ax3 = plt.subplot(gs[2, 0])
        self.ax3.grid(False)
        self.ax3.set_xticklabels([])
        self.ax3.set_yticklabels([])
        self.ax3.set_title(title3)

        self.ax4 = plt.subplot(gs[2, 1])
        self.ax4.grid(False)
        self.ax4.set_xticklabels([])
        self.ax4.set_yticklabels([])
        self.ax4.set_title(title4)

        self.slax = plt.subplot(gs[3, :])
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

        im3 = self.im3[self.i, :, :]
        self.ax3.imshow(im3, vmin=self.vmin3, vmax=self.vmax3,
                        cmap=self.cmap, interpolation=None)

        im4 = self.im4[self.i, :, :]
        self.ax4.imshow(im4, vmin=self.vmin4, vmax=self.vmax4,
                        cmap=self.cmap, interpolation=None)