import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

class SliderPlot():
    def __init__(self, ax, I, **kwargs):
        assert I.ndim == 2

        title = kwargs.get('title', None)
        pos = ax.get_position()
        rect = kwargs.get('rect', [pos.x0, pos.y0-0.03, pos.width, 0.03])
        slider_name = kwargs.get('legend', None)

        self.I = I
        N, pixels = self.I.shape
        self.size = int(pixels**(0.5))
        self.im = ax.imshow(I[0].reshape((self.size, -1)).T, cmap='gray')
        ax.set_title(title)
        
        ax_slider = plt.axes(rect)
        self.slider = Slider(ax_slider, slider_name, 0, N-1, valinit=0, valstep=1, valfmt="%.f")
        self.slider.on_changed(self.update_img)
    
    def update_img(self, val):
        r = int(self.slider.val)
        self.im.set_data(self.I[r].reshape((self.size, -1)).T)
        # fig.canvas.draw_idle()
