import matplotlib.pyplot as plt
import numpy as np

class PlotterAnalyzer:

    def __init__(self) -> None:
        

        #define default plot parameters:
        self.font_size_axis_labels = 12
        self.font_size_title = 15
        self.font_size_ticks = 12
        self.font_size_legend = 12
        self.plot_x_max = 10
        self.plot_y_max = 20
        self.marker_size = 10

        return
    
    def plot_error_histogram(self,
                             errors:np.ndarray,
                             bins=10,
                             show=False,
                             ax=None):
        if not ax:
            fig,ax = plt.subplots()

        # Compute the histogram
        counts, bin_edges = np.histogram(errors, bins=bins)

        # Convert counts to percentages
        percentages = (counts / len(errors)) * 100

        print("Max: {}, sum: {}".format(
            np.max(percentages),
            np.sum(percentages)
        ))

        # Plot the histogram with percentage on the y-axis
        ax.bar(
            bin_edges[:-1],
            percentages,
            width=np.diff(bin_edges),
            edgecolor='black',
            align='edge')
        

        ax.set_title("Error Histogram",
                     fontsize=self.font_size_title)
        ax.set_xlabel("Error (m)",fontsize=self.font_size_axis_labels)
        ax.set_ylabel("Percent (%)",fontsize=self.font_size_axis_labels)
        ax.tick_params(labelsize=self.font_size_ticks)

        if show:
            plt.show()

        return