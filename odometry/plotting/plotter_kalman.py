import matplotlib.pyplot as plt
import numpy as np

class PlotterKalman:

    def __init__(self) -> None:
        
        #define default plot parameters:
        self.font_size_axis_labels = 12
        self.font_size_title = 15
        self.font_size_ticks = 12
        self.font_size_legend = 12
        self.plot_x_max = 10
        self.plot_y_max = 20

        return
    
    def plot_kalman_result_history(
            self,
            x_history:np.ndarray,
            p_history:np.ndarray=None,
            axs:plt.Axes=[],
            idx:int = 0
    ):
        if len(axs) == 0:
            fig, axs = plt.subplots(1, 7, figsize=(20, 5))
        if idx == 0:
            idx = -1
        
        #plot position history
        axs[0].plot(x_history[:idx,0],x_history[:idx,1])
        axs[0].set_title("Position",
                         fontsize=self.font_size_title)
        
        #plot the states individually
        for i, title in zip(range(1, 5), ["X", "Y", "Phi", "Speed"]):
            axs[i].plot(x_history[:idx,i - 1])
            if p_history is not None:
                axs[i].plot(x_history[:idx, i - 1] + p_history[:idx, i - 1], "r--")
                axs[i].plot(x_history[:idx, i - 1] - p_history[:idx, i - 1], "r--")
            axs[i].set_title(title)
        
        #plot biases if available
        if len(x_history[0, :]) > 4:
            for i, title in zip(range(5, 7), ["Gyro Bias", "Encoder Bias"]):
                axs[i].plot(x_history[:idx, i - 1])
                if p_history is not None:
                    axs[i].plot(x_history[:idx, i - 1] + p_history[:idx, i - 1], "r--")
                    axs[i].plot(x_history[:idx, i - 1] - p_history[:idx, i - 1], "r--")
                axs[i].set_title(title)
        
        plt.show()
    
    def plot_chi_2_resp(self,
                        g_thresh:int,
                        g_hist:np.ndarray,
                        idx:int=0,
                        ax:plt.Axes = None):
        
        if not ax:
            fig,ax = plt.subplots()
        
        if idx == 0:
            ax.plot(g_hist,label="test statistic")
        else:
            ax.plot(g_hist[:idx],label="test statistic")
        
        ax.axhline(
            g_thresh,
            color="green",
            linestyle="--",
            label="chi2 threshold")
        
        ax.legend()
        plt.show()
    
    def plot_residual_hist(
            self,
            y_hist:list,
            idx:int=0,
            ax:plt.Axes = None):

        if not ax:
            fig,ax = plt.subplots()
            
        if idx == 0:
            idx = -1

        ax.hist([y[0] for y in y_hist],bins=40,alpha=0.5)
        ax.hist([y[1] for y in y_hist], bins=40,alpha=0.5)
