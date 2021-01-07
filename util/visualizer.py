import matplotlib as mlt
import matplotlib.pyplot as plt
import numpy as np

class visualizer(object):
    def __init__(self, **kwargs):
        plt.figure()

    def line_between(self, a, b, col):
        plt.plot([a[0], b[0]], [a[1], b[1]], color=col, alpha=.5)
    def show(self):
        plt.grid("on")
        plt.show()

    def get_colormap(self, colmap):
        cmap = plt.get_cmap(colmap).colors
        return np.array(cmap)
    def show_traj(self, points):
        cmap = self.get_colormap("plasma")
        col_idx = np.floor(np.linspace(0, np.size(cmap, 0) - 1, np.size(points, 0)))
        col_idx_int = [int(wlt) for wlt in col_idx]

        for idx in range(0, len(points)-1):
            act_point=points[idx]
            next_point = points[idx+1]
            x_pos = act_point[0]
            y_pos = act_point[1]
            x_dif=next_point[0]-x_pos
            y_dif=next_point[1]-y_pos
            scale = 1
            act_col=cmap[col_idx_int[idx], :]
            plt.arrow(x_pos, y_pos, scale * x_dif, scale * y_dif,
                          fc=act_col, ec="black", alpha=.65, width=.4,
                          head_width=1.4, head_length=1)
        plt.axis([-11.5, 11.5, -11.5, 11.5])
        plt.legend()
        plt.xlabel('x')
        plt.ylabel('y')
