import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.special import j0

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

from generative.data.corazza import colored_gaussian_jakes
from generative.utils.seed import make_rng


def _acf_fft(x):
    n=len(x)
    n_fft=1<<(2*n-1).bit_length()
    X=np.fft.fft(x,n_fft)
    acf=np.fft.ifft(X*np.conj(X)).real[:n]
    return acf/acf[0]

def main():
    fs=1000.0
    ratios=[0.01,0.05,0.10]
    colors = [ "#2ca02c" , "#1f77b4" , "#d62728" ]
    n=50000
    n_lag=200
    rng=make_rng(2024)

    fig,ax=plt.subplots(figsize=(9,4.5))
    tau=np.arange(n_lag)/fs

    for ratio,color in zip(ratios,colors):
        fd=ratio*fs
        x=colored_gaussian_jakes(n,fs,fd,rng)
        acf_emp=_acf_fft(x)
        ax.plot(tau*1000,acf_emp[:n_lag],color=color,ls='--',label=f'经验 f_d/f_s={ratio}')

        acf_theory=j0(2*np.pi*fd*tau)
        ax.plot(tau*1000,acf_theory,color=color,ls='-',label=f'理论 f_d/f_s={ratio}')

    ax.axhline(np.exp(-1),color='gray',ls=':',label='e^-1')
    ax.legend(fontsize=8)
    ax.set_xlabel('滞后 (ms)')
    ax.set_ylabel('归一化自相关')
    ax.set_title('ACF of Colored Gaussian Jakes')
    plt.tight_layout()
    plt.savefig(ROOT/"figs"/"day3_fd_temporal.png",dpi=120)
if __name__=='__main__':
    main()
