# Code for computational model of frequency modulation by NDNF interneurons.
# author: Laura Bella Naumann
# in collaboration with Henning Sprekeler, Belen Pardi and Johannes Letzkus

import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
plt.style.use('pretty')


def generate_spikes(t, rate, tau_ref=0.005):
    """
    Generates a Poisson spike train.
    :param t:       time vector
    :param rate:    mean rate of generated Poisson train
    :param tau_ref: refractory time constant
    :return:        array of spike times
    """

    isi_mean = 1/rate  # mean of ISI distribution is 1 / firing rate
    sp = t[0] + np.maximum(np.random.exponential(isi_mean), tau_ref)  # first spike time
    spike_times = []
    while sp <= t[-1]:  # while the next spike time is within simulation time, append spike time
        spike_times.append(sp.copy())
        sp += np.maximum(np.random.exponential(isi_mean), tau_ref)
    return np.array(spike_times)


def simulate_neuron(t, rate_ndnf, spikes_01, tauf=0.2, taur=0.2):
    """
    Simulates the EPSC of a neuron with short-term plasticity.
    :param t:            time vector
    :param rate_ndnf:    NDNF activity (determines initial release)
    :param spikes_01:    array of binary spike times (for every timestep 0 (no spike) or 1 (spike))
    :param tauf:         facilitation time constant
    :param taur:         recovery time constant
    :return:             array of EPSCs over time period t
    """

    dt = t[1]-t[0]

    # default parameters
    taus = 0.02  # synaptic time constant
    frac = 0.1  # facilitation fraction
    A = 1  # amplitude factor

    # initial release probability depends on NDNF rate
    p_init = pre_inh_func(rate_ndnf)

    # initialise variables and tracking array
    u = p_init
    x = 1
    I = 0
    input_current = []

    # time integration
    for i, ti in enumerate(t):

        # Tsodyks-Markram short term plasticity dynamics
        du = -(u-p_init)/tauf + (1-u)*frac*spikes_01[i]/dt
        dx = (1-x)/taur - u*x*spikes_01[i]/dt

        # Integration of spikes to postsynaptic current
        dI = -I/taus + A*u*x*spikes_01[i]/dt

        # Euler step
        u = u + du*dt
        x = x + dx*dt
        I = I + dI*dt

        input_current.append(I)

    return np.array(input_current)


def pre_inh_func(r, c=5, beta=0.45):
    """
    Presynaptic inhibition function. Decreasing sigmoid between 0 and 1.
    :param r:       array of rates
    :param c:       center of sigmoid
    :param beta:    slope parameter
    :return:        returns array of initial release probabilities
    """
    return 1/(1+np.exp(beta*(r-c)))


def example_epsc(rate_ndnf, taur=0.2, tauf=0.2):
    """
    Get example EPSC trace of 200 ms for given NDNF activity.
    :param rate_ndnf:    NDNF activity level
    :param taur:         recovery time constant
    :param tauf:         facilitation time constant
    :return:             returns trace of epsc as array
    """
    t = np.arange(0, 0.2, 0.001)
    freq = 25  # input frequency of example train
    spikes = np.zeros(len(t))
    spikes[30:-30:int(1/freq/0.001)] = 1
    epsc = simulate_neuron(t, rate_ndnf, spikes, taur=taur, tauf=tauf)

    return epsc


def make_ax_box(axx):
    """
    Plot helper function.
    Takes an axis object (axx) and sets all spines (left, bottom, right, top) to visible but thin to create a box.
    """
    axx.set(xticks=[], yticks=[])
    for x in ['top', 'bottom', 'left', 'right']:
        axx.spines[x].set_linewidth(0.5)
        axx.spines[x].set_visible(True)


def plot_panel_ndnf_modulation(plot_supp=False):
    """
    Plot figure panel of computational model:
    - Left: Spike trains of HO-MG afferents
    - Center: Dependence of initial release probability on NDNF activity. Insets show example EPSC for 25 Hz spikes.
    - Right: Mean EPSC amplitude normalised to the first spike as a function of spike frequency and NDNF activity.

    :param plot_supp: Whether to plot the supplementary figure as well.
    """

    # --------------- #
    #  generate data  #
    # --------------- #

    # simulation parameters
    duration = 240  # length of experiment in seconds
    dt = 0.001  # integration timestep
    t = np.arange(0, duration, dt)

    # range of NDNF rates and frequency bins to test
    NDNFrates = np.arange(0, 10.1, 1)
    freq_bins = np.array([ff for ff in np.arange(0, 51, 2.5)])
    n_bins = len(freq_bins)-1

    # generate Poisson spike train and get frequencies of spikes
    Poisson_rate = 10
    spike_times = generate_spikes(t, Poisson_rate)
    freqs = 1/np.diff(spike_times)

    # build binary array of spike times
    spikes_idx = (spike_times/dt).astype('int')
    spikes_01 = np.zeros(len(t))
    spikes_01[spikes_idx] = 1

    # initialise arrays
    response_binned = np.zeros((n_bins, len(NDNFrates)))
    response_std = np.zeros((n_bins, len(NDNFrates)))
    current_all = []

    # loop over NDNF rates, simulate inputs to neuron and bin responses according to frequencies
    for i, ndnfr in enumerate(NDNFrates):

        current = simulate_neuron(t, ndnfr, spikes_01) # simulate neuron with given parameters
        epsc_amplitudes = current[spikes_idx[1:]]/current[spikes_idx[0]]  # normalised epsc amplitudes
        current_all.append(current)

        # loop over frequency bins
        for f in range(n_bins):

            epsc_in_bin = epsc_amplitudes[(freqs >= freq_bins[f]) & (freqs < freq_bins[f+1])]  # filter epscs
            response_binned[f, i] = np.mean(epsc_in_bin)  # get mean..
            response_std[f, i] = np.std(epsc_in_bin)  # ..and std of epsc amplitudes within bin

    # ---------- #
    #  plotting  #
    # ---------- #

    # set up figure grid
    fig_panel = plt.figure(figsize=(7, 2.5), dpi=300)
    gs = gridspec.GridSpec(1, 2, width_ratios=[2, 1])
    gs.update(wspace=0.1, hspace=0, top=0.95, bottom=0.2, left=0.1, right=0.9)
    subgs = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[0], wspace=0.5)
    ax0 = plt.subplot(subgs[0])
    ax1 = plt.subplot(subgs[1])
    ax3d = plt.subplot(gs[1], projection='3d')
    ax3d.set_position((0.62, 0.15, 0.33, 0.9))

    # left: plot inputs as raster
    spike_y = np.floor(spike_times)
    ax0.vlines(spike_times-spike_y, spike_y, spike_y+0.9, color='k', lw=0.8)
    ax0.set(xlim=[0, 1], ylim=[0, 50], xticks=[0, 1], yticks=[0, 25, 50], ylabel='Trials', xlabel='Time (s)')

    # center: plot transfer function of NDNF rate to release probability
    p_init_list = pre_inh_func(NDNFrates)
    ax1.plot(NDNFrates, p_init_list, '.-', c='#077E36', lw=1.5, ms=6)
    ax1.set(xlim=[-0.5, 10.5], ylim=[0, 1], xlabel='NDNF rate (1/s)', ylabel='Initial release prob. ',
            xticks=[0, 5, 10], yticks=[0, 0.5, 1])
    # inset axis showing example EPSCs
    axin1 = inset_axes(ax1, width="40%", height="20%", loc=1)
    axin2 = inset_axes(ax1, width="40%", height="20%", loc=3, borderpad=0.5)
    axin1.plot(-example_epsc(NDNFrates[1]), 'k', lw=1)
    axin2.plot(-example_epsc(NDNFrates[-2]), 'k', lw=1)
    make_ax_box(axin1)
    make_ax_box(axin2)
    ax1.annotate("", xy=(NDNFrates[1], p_init_list[1]), xytext=(6, p_init_list[1]), arrowprops=dict(arrowstyle="->"))
    ax1.annotate("", xy=(NDNFrates[-2], p_init_list[-2]), xytext=(4, p_init_list[-2]),
                 arrowprops=dict(arrowstyle="->"))

    # right: 3D plot of responses over NDNF rates and frequencies
    xx, yy = np.meshgrid((freq_bins[:-1]+freq_bins[1:])/2, NDNFrates)
    ax3d.plot_surface(xx, yy, response_binned.T, cmap='RdYlGn', alpha=0.7, vmin=0, vmax=2, edgecolor='k', linewidth=0.4,
                      cstride=2, rstride=1)
    ax3d.set(xlim=[0, 50], ylim=[0, 10], zlim=[0, 2.3], yticks=[0, 5, 10], zticks=[1, 2],
             xlabel='Frequency (Hz)', ylabel='NDNF rate (1/s)', zlabel='Response (rel.)')
    ax3d.set_xticklabels(np.arange(0, 51, 10), verticalalignment='baseline', horizontalalignment='center')
    ax3d.view_init(20, -80)

    # save figure in current directory as pdf (alternatively as png or eps)
    fig_panel.savefig('panel_freq_mod.pdf', dpi=300)

    if plot_supp:
        plot_supp_ndnf_modulation(t, spikes_idx, freqs, freq_bins, current_all, response_binned, response_std)


def plot_supp_ndnf_modulation(t, spikes_idx, freqs, freq_bins, current_all, response_binned, response_std):
    """
    Plot extended data for computational model.

    :param t:               time array
    :param spikes_idx:      indices of spike times
    :param freqs:           frequencies of spike times (1/ISI)
    :param freq_bins:       bin edges for frequency bins
    :param current_all:     list containing arrays of EPSCs for different NDNF rates
    :param response_binned: mean EPSC responses binned by frequency of corresponding input spike
    :param response_std:    standard deviation of EPSC responses for given frequency bin
    """

    fig = plt.figure(figsize=(8, 2.5), dpi=400)
    gs = gridspec.GridSpec(2, 4, bottom=0.2, right=0.95, width_ratios=[1, 1.2, 1, 1.5])
    gs.update(wspace=0.6)
    subgs1 = gridspec.GridSpecFromSubplotSpec(2, 1, gs[:, 1], hspace=0.5)
    subgs2 = gridspec.GridSpecFromSubplotSpec(3, 1, gs[:, 2], height_ratios=[1, 1, 0.5])

    # a: schematic and ISI -> freq
    ax0 = plt.subplot(gs[:, 0])
    ax0.set_position((0, 0, 0.25, 1))
    # im = plt.imread('supp_schema.png')  # requires png of schematic, will fail if file not found.
    # ax0.imshow(im)
    plt.axis('off')

    # b: frequency distribution
    ax11 = plt.subplot(subgs1[:])
    ax11.hist(freqs, bins=freq_bins, color='gray')
    ax11.set(xlim=[0, 50], xlabel='Freq. (Hz)', ylabel='count')

    # c: example EPSCs
    ax20 = plt.subplot(subgs2[0])
    ax21 = plt.subplot(subgs2[1])
    ax22 = plt.subplot(subgs2[2])
    ax20.plot(t[:1000], -current_all[0][:1000] / current_all[0][spikes_idx[0]], c='k', lw=1)
    ax21.plot(t[:1000], -current_all[-1][:1000] / current_all[-1][spikes_idx[0]], c='#077E36', lw=1)
    ax22.vlines(t[spikes_idx], 0, 1, lw=1, color='cornflowerblue')
    ax20.spines['bottom'].set_visible(False)
    ax21.spines['bottom'].set_visible(False)
    ax22.spines['left'].set_visible(False)
    ax20.set(xlim=[0, 1], xticks=[], ylabel='EPSC (norm.)')
    ax20.yaxis.set_label_coords(-0.4, -0.1)
    ax21.set(xlim=[0, 1], xticks=[])
    ax22.set(xlim=[0, 1], xlabel='Time (s)', xticks=[0, 1], yticks=[])
    ax20.text(0.2, 0.1, 'NDNF off', color='k')
    ax21.text(0.2, 0.03, 'NDNF on', color='#077E36')

    # d: 2d response plot for high and low NDNF activity

    ax3 = plt.subplot(gs[:, 3])
    ax3.plot(freq_bins[:-1], response_binned[:, 0], '.-', c='k', label='NDNF off')
    ax3.errorbar(freq_bins[:-1], response_binned[:, 0], response_std[:, 0], c='k', capsize=4)
    ax3.plot(freq_bins[:-1], response_binned[:, -1], '.-', c='#077E36', label='NDNF on')
    ax3.legend(loc='best', frameon=False)
    ax3.errorbar(freq_bins[:-1], response_binned[:, -1], response_std[:, -1], c='#077E36', capsize=4)
    ax3.set(xlim=[0, 50], ylim=[0, 2.5], xlabel='Freq. (Hz)', ylabel='Relative response', yticks=[0, 1, 2])

    # panel labels
    ax0.text(0.05, 1, 'a', transform=ax0.transAxes, weight='bold', fontsize=12)
    ax11.text(-0.2, 1.05, 'b', transform=ax11.transAxes, weight='bold', fontsize=12)
    ax20.text(-0.55, 1.15, 'c', transform=ax20.transAxes, weight='bold', fontsize=12)
    ax3.text(-0.25, 1.05, 'd', transform=ax3.transAxes, weight='bold', fontsize=12)

    #
    fig.savefig('supp_model_ex.pdf', dpi=400)


if __name__ in "__main__":

    plot_panel_ndnf_modulation(plot_supp=True)


