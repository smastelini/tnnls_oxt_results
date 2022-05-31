import os
import matplotlib.pyplot as plt
import pandas as pd

from utils import OUT_PATH

if not os.path.exists(f"{OUT_PATH}/saturation/plots"):
    os.makedirs(f"{OUT_PATH}/saturation/plots")

def plot_saturation_curves(dataset_name):
    for n_rf in range(10, 110, 10):
        m_rf = pd.read_csv(
            f"{OUT_PATH}/saturation/final/mean_{dataset_name}_ARF-abs_t{n_rf}.csv"
        )
        s_rf = pd.read_csv(
            f"{OUT_PATH}/saturation/final/std_{dataset_name}_ARF-abs_t{n_rf}.csv"
        )
        for n_xt in range(10, 110, 10):
            m_xt = pd.read_csv(
                f"{OUT_PATH}/saturation/final/mean_{dataset_name}_XT_t{n_xt}.csv"
            )
            s_xt = pd.read_csv(
                f"{OUT_PATH}/saturation/final/std_{dataset_name}_XT_t{n_xt}.csv"
            )
            
            fig, ax = plt.subplots(figsize=(5, 10), nrows=4, dpi=600)
            ax[0].plot(m_rf["n_samples"], m_rf["RMSE"], c="blue", marker="s", label="ARF", markevery=0.1)
            ax[0].plot(m_xt["n_samples"], m_xt["RMSE"], c="black", marker="X", label="OXT", markevery=0.1)
            ax[1].plot(m_rf["n_samples"], m_rf["R2"], c="blue", marker="s", markevery=0.1)
            ax[1].plot(m_xt["n_samples"], m_xt["R2"], c="black", marker="X", markevery=0.1)
            ax[2].plot(m_rf["n_samples"], m_rf["memory"], c="blue", marker="s", markevery=0.1)
            ax[2].plot(m_xt["n_samples"], m_xt["memory"], c="black", marker="X", markevery=0.1)
            ax[3].plot(m_rf["n_samples"], m_rf["time"], c="blue", marker="s", markevery=0.1)
            ax[3].plot(m_xt["n_samples"], m_xt["time"], c="black", marker="X", markevery=0.1)
            
            ax[0].fill_between(
                m_rf["n_samples"], m_rf["RMSE"] - s_rf["RMSE"], m_rf["RMSE"] + s_rf["RMSE"],
                 color='blue', alpha=0.2
            )
            ax[0].fill_between(
                m_xt["n_samples"], m_xt["RMSE"] - s_xt["RMSE"], m_xt["RMSE"] + s_xt["RMSE"],
                 color='black', alpha=0.2
            )
            ax[1].fill_between(
                m_rf["n_samples"], m_rf["R2"] - s_rf["R2"], m_rf["R2"] + s_rf["R2"],
                 color='blue', alpha=0.2
            )
            ax[1].fill_between(
                m_xt["n_samples"], m_xt["R2"] - s_xt["R2"], m_xt["R2"] + s_xt["R2"],
                 color='black', alpha=0.2
            )
            ax[2].fill_between(
                m_rf["n_samples"], m_rf["memory"] - s_rf["memory"], m_rf["memory"] + s_rf["memory"],
                 color='blue', alpha=0.2
            )
            ax[2].fill_between(
                m_xt["n_samples"], m_xt["memory"] - s_xt["memory"], m_xt["memory"] + s_xt["memory"],
                 color='black', alpha=0.2
            )
            ax[3].fill_between(
                m_rf["n_samples"], m_rf["time"] - s_rf["time"], m_rf["time"] + s_rf["time"],
                 color='blue', alpha=0.2
            )
            ax[3].fill_between(
                m_xt["n_samples"], m_xt["time"] - s_xt["time"], m_xt["time"] + s_xt["time"],
                 color='black', alpha=0.2
            )
            
            ax[0].xaxis.set_ticklabels([])
            ax[1].xaxis.set_ticklabels([])
            ax[2].xaxis.set_ticklabels([])
            
            ax[3].set_xlabel("Instances")
            ax[0].set_ylabel("RMSE")
            ax[1].set_ylabel("R$^2$")
            ax[2].set_ylabel("Memory (MB)")
            ax[3].set_ylabel("Time (s)")
            
            ax[0].legend()
            ax[0].set_title(f"ARF$_{{{n_rf}}}$ x XT$_{{{n_xt}}}$")
            for tick in ax[3].get_xticklabels():
                tick.set_rotation(30)
                tick.set_ha("right")          
 
            plt.savefig(
                f"{OUT_PATH}/saturation/plots/{dataset_name}_arf_{n_rf:03d}_xt_{n_xt:03d}.png",
                bbox_inches="tight"
            )
            plt.close()
            

if __name__ == "__main__":
    plot_saturation_curves("cal_housing")
    plot_saturation_curves("elevators")
    plot_saturation_curves("friedman")
    plot_saturation_curves("friedman_lea")
    plot_saturation_curves("friedman_gra")
    plot_saturation_curves("friedman_gsg")
    