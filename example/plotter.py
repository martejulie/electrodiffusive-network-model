import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from set_style import set_style
import matplotlib.gridspec as gridspec

np_load_old = np.load
np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)

def return_value_alpha(alpha, feature, n_unit):
    if feature == 'alpha_sn':
        return alpha[:,0,0,n_unit]
    if feature == 'alpha_dn':
        return alpha[:,1,0,n_unit]
    if feature == 'alpha_sg':
        return alpha[:,0,1,n_unit]
    if feature == 'alpha_dg':
        return alpha[:,1,1,n_unit]
    if feature == 'alpha_se':
        return 1 - (alpha[:,0,0,n_unit] + alpha[:,0,1,n_unit])
    if feature == 'alpha_de':
        return 1 - (alpha[:,1,0,n_unit] + alpha[:,1,1,n_unit])

def return_value_c(c, feature,n_unit):
    if feature == 'Na_sn':
        return c[:,0,0,0,n_unit]
    if feature == 'Na_sg':
        return c[:,0,0,1,n_unit]
    if feature == 'Na_se':
        return c[:,0,0,2,n_unit]
    if feature == 'Na_dn':
        return c[:,1,0,0,n_unit]
    if feature == 'Na_dg':
        return c[:,1,0,1,n_unit]
    if feature == 'Na_de':
        return c[:,1,0,2,n_unit]
    if feature == 'K_sn':
        return c[:,0,1,0,n_unit]
    if feature == 'K_sg':
        return c[:,0,1,1,n_unit]
    if feature == 'K_se':
        return c[:,0,1,2,n_unit]
    if feature == 'K_dn':
        return c[:,1,1,0,n_unit]
    if feature == 'K_dg':
        return c[:,1,1,1,n_unit]
    if feature == 'K_de':
        return c[:,1,1,2,n_unit]
    if feature == 'Cl_sn':
        return c[:,0,2,0,n_unit]
    if feature == 'Cl_sg':
        return c[:,0,2,1,n_unit]
    if feature == 'Cl_se':
        return c[:,0,2,2,n_unit]
    if feature == 'Cl_dn':
        return c[:,1,2,0,n_unit]
    if feature == 'Cl_dg':
        return c[:,1,2,1,n_unit]
    if feature == 'Cl_de':
        return c[:,1,2,2,n_unit]
    if feature == 'Ca_sn':
        return c[:,0,3,0,n_unit]
    if feature == 'Ca_sg':
        return c[:,0,3,1,n_unit]
    if feature == 'Ca_se':
        return c[:,0,3,2,n_unit]
    if feature == 'Ca_dn':
        return c[:,1,3,0,n_unit]
    if feature == 'Ca_dg':
        return c[:,1,3,1,n_unit]
    if feature == 'Ca_de':
        return c[:,1,3,2,n_unit]

def return_value_phi(phi, feature, n_unit):
    if feature == 'phi_sn':
        return phi[:,0,0,n_unit]
    if feature == 'phi_dn':
        return phi[:,1,0,n_unit]
    if feature == 'phi_sg':
        return phi[:,0,1,n_unit]
    if feature == 'phi_dg':
        return phi[:,1,1,n_unit]
    if feature == 'phi_se':
        return phi[:,0,2,n_unit]
    if feature == 'phi_de':
        return phi[:,1,2,n_unit]

def return_value_ss(ss, feature, n_unit):
    if feature == 'h':
        return ss[:,0,n_unit]
    if feature == 'n':
        return ss[:,1,n_unit]
    if feature == 's':
        return ss[:,2,n_unit]
    if feature == 'c':
        return ss[:,3,n_unit]
    if feature == 'q':
        return ss[:,4,n_unit]
    if feature == 'z':
        return ss[:,5,n_unit]

def load_data(path_data, path_figures, n_unit):
    """ Load data for unit number n_unit from file. """

    data = np.load(path_data + 'data.npz')

    N_units = data['N_units']
    t = data['t']
    dt = data['dt']
    alpha = data['alpha']
    c = data['c']
    phi = data['phi']
    ss = data['ss']
    t_AP = data['t_AP'][:,n_unit]

    alpha_sn = return_value_alpha(alpha, 'alpha_sn', n_unit)
    alpha_dn = return_value_alpha(alpha, 'alpha_dn', n_unit)
    alpha_sg = return_value_alpha(alpha, 'alpha_sg', n_unit)
    alpha_dg = return_value_alpha(alpha, 'alpha_dg', n_unit)
    alpha_se = return_value_alpha(alpha, 'alpha_se', n_unit)
    alpha_de = return_value_alpha(alpha, 'alpha_de', n_unit)
    Na_sn = return_value_c(c, 'Na_sn', n_unit)
    Na_sg = return_value_c(c, 'Na_sg', n_unit)
    Na_se = return_value_c(c, 'Na_se', n_unit)
    Na_dn = return_value_c(c, 'Na_dn', n_unit)
    Na_dg = return_value_c(c, 'Na_dg', n_unit)
    Na_de = return_value_c(c, 'Na_de', n_unit)
    K_sn = return_value_c(c, 'K_sn', n_unit)
    K_sg = return_value_c(c, 'K_sg', n_unit)
    K_se = return_value_c(c, 'K_se', n_unit)
    K_dn = return_value_c(c, 'K_dn', n_unit)
    K_dg = return_value_c(c, 'K_dg', n_unit)
    K_de = return_value_c(c, 'K_de', n_unit)
    Cl_sn = return_value_c(c, 'Cl_sn', n_unit)
    Cl_sg = return_value_c(c, 'Cl_sg', n_unit)
    Cl_se = return_value_c(c, 'Cl_se', n_unit)
    Cl_dn = return_value_c(c, 'Cl_dn', n_unit)
    Cl_dg = return_value_c(c, 'Cl_dg', n_unit)
    Cl_de = return_value_c(c, 'Cl_de', n_unit)
    Ca_sn = return_value_c(c, 'Ca_sn', n_unit)
    Ca_sg = return_value_c(c, 'Ca_sg', n_unit)
    Ca_se = return_value_c(c, 'Ca_se', n_unit)
    Ca_dn = return_value_c(c, 'Ca_dn', n_unit)
    Ca_dg = return_value_c(c, 'Ca_dg', n_unit)
    Ca_de = return_value_c(c, 'Ca_de', n_unit)
    phi_sn = return_value_phi(phi, 'phi_sn', n_unit)
    phi_dn = return_value_phi(phi, 'phi_dn', n_unit)
    phi_sg = return_value_phi(phi, 'phi_sg', n_unit)
    phi_dg = return_value_phi(phi, 'phi_dg', n_unit)
    phi_se = return_value_phi(phi, 'phi_se', n_unit)
    phi_de = return_value_phi(phi, 'phi_de', n_unit)
    n = return_value_ss(ss, 'n', n_unit)
    h = return_value_ss(ss, 'h', n_unit)
    s = return_value_ss(ss, 's', n_unit)
    c = return_value_ss(ss, 'c', n_unit)
    q = return_value_ss(ss, 'q', n_unit)
    z = return_value_ss(ss, 'z', n_unit)

    phi_msn = phi_sn - phi_se
    phi_mdn = phi_dn - phi_de
    phi_msg = phi_sg - phi_se
    phi_mdg = phi_dg - phi_de

    data_dict = {
            "t": t,
            "dt":dt,
            "alpha_sn":  alpha_sn,
            "alpha_dn":  alpha_dn,
            "alpha_sg":  alpha_sg,
            "alpha_dg":  alpha_dg,
            "alpha_se":  alpha_se,
            "alpha_de":  alpha_de,
            "Na_sn":  Na_sn,
            "Na_sg":  Na_sg,
            "Na_se":  Na_se,
            "Na_dn":  Na_dn,
            "Na_dg":  Na_dg,
            "Na_de":  Na_de,
            "K_sn":  K_sn,
            "K_sg":  K_sg,
            "K_se":  K_se,
            "K_dn":  K_dn,
            "K_dg":  K_dg,
            "K_de":  K_de,
            "Cl_sn":  Cl_sn,
            "Cl_sg":  Cl_sg,
            "Cl_se":  Cl_se,
            "Cl_dn":  Cl_dn,
            "Cl_dg":  Cl_dg,
            "Cl_de":  Cl_de,
            "Ca_sn":  Ca_sn,
            "Ca_sg":  Ca_sg,
            "Ca_se":  Ca_se,
            "Ca_dn":  Ca_dn,
            "Ca_dg":  Ca_dg,
            "Ca_de":  Ca_de,
            "phi_sn":  phi_sn,
            "phi_sg":  phi_sg,
            "phi_se":  phi_se,
            "phi_dn":  phi_dn,
            "phi_dg":  phi_dg,
            "phi_de":  phi_de,
            "n": n,
            "h": h,
            "s": s,
            "c": c,
            "q": q,
            "z": z,
            "phi_msn": phi_msn,
            "phi_msg": phi_msg,
            "phi_mdn": phi_mdn,
            "phi_mdg": phi_mdg,
            "n_unit": n_unit,
            "N_units": N_units, 
            "t_AP": t_AP,
            "path_figures": path_figures
            }

    data.close()

    return data_dict

def plot_phi(path_data, path_figures, N_units):
    """ Plot neuronal membrane potentials (soma), extracellular potentials (soma layer),
    and glial membrane potentials (soma layer). """

    if not os.path.isdir(path_figures):
        os.makedirs(path_figures)

    set_style('default', w=1, h=4)

    # set axes
    fig = plt.figure()
    gs = gridspec.GridSpec(6,2)
    axA = fig.add_subplot(gs[0,:])
    axB = fig.add_subplot(gs[1,0])
    axC = fig.add_subplot(gs[1,1])
    axD = fig.add_subplot(gs[2,:])
    axE = fig.add_subplot(gs[3,0])
    axF = fig.add_subplot(gs[3,1])
    axG = fig.add_subplot(gs[4,:])
    axH = fig.add_subplot(gs[5,0])
    axI = fig.add_subplot(gs[5,1])

    # set colors
    colors = sns.color_palette('muted')

    # get t array
    data_dict = load_data(path_data, path_figures, 0)
    t = data_dict['t']

    # panel A and B
    for n_unit in range(N_units):
        data_dict = load_data(path_data, path_figures, n_unit)
        axA.plot(t, data_dict['phi_sn'] - data_dict['phi_se'], color=colors[n_unit], label=str(n_unit+1))
        axB.plot(t, data_dict['phi_sn'] - data_dict['phi_se'], color=colors[n_unit])

    # panel C
    n_unit = 0
    data_dict = load_data(path_data, path_figures, n_unit)
    l1 = axC.plot(t, data_dict['phi_sn'] - data_dict['phi_se'], color=colors[n_unit])[0]
    l2 = axC.plot(t, data_dict['phi_dn'] - data_dict['phi_de'], color=colors[n_unit], linestyle='--')[0]
    fig.legend([l1, l2], ['soma', 'dendrite'], \
            loc=(0.70,0.75), ncol=1, fontsize='small', handlelength=0.8, handletextpad=0.4)

    axinC = axC.inset_axes([0.7, 0.5, 0.3, 0.5], xlim=[0.1185, 0.12], ylim=[-77.5, -72.5], xticklabels=[], yticklabels=[])
    axinC.plot(t, data_dict['phi_sn'] - data_dict['phi_se'], color=colors[n_unit])
    axinC.plot(t, data_dict['phi_dn'] - data_dict['phi_de'], color=colors[n_unit], linestyle='--')
    axC.indicate_inset_zoom(axinC, edgecolor='black')

    # panel D and E
    for n_unit in range(N_units):
        data_dict = load_data(path_data, path_figures, n_unit)
        axD.plot(t, data_dict['phi_se'], color=colors[n_unit])
        axE.plot(t, data_dict['phi_se'], color=colors[n_unit])

    # panel F
    n_unit = 0
    data_dict = load_data(path_data, path_figures, n_unit)
    axF.plot(t, data_dict['phi_se'], label='soma',  color=colors[n_unit])
    axF.plot(t, data_dict['phi_de'], label='dendrite', color=colors[n_unit], linestyle='--')

    # panel G and H
    for n_unit in range(N_units):
        data_dict = load_data(path_data, path_figures, n_unit)
        axG.plot(t, data_dict['phi_sg'] - data_dict['phi_se'], color=colors[n_unit])
        axH.plot(t, data_dict['phi_sg'] - data_dict['phi_se'], color=colors[n_unit])

    # panel I
    n_unit = 0
    data_dict = load_data(path_data, path_figures, n_unit)
    axI.plot(t, data_dict['phi_sg'] - data_dict['phi_se'], label='soma',  color=colors[n_unit])
    axI.plot(t, data_dict['phi_dg'] - data_dict['phi_de'], label='dendrite', color=colors[n_unit], linestyle='--')

    # adjust axes
    axH.set_ylim([-85, -83])
    axI.set_ylim([-85, -83])
    for ax in [axA, axD, axG]:
        ax.set_xlim([0, t[-1]])
    for ax in [axB, axC, axE, axF, axH, axI]:
        ax.set_xlim([0.1, 0.14])

    # set titles labels
    axA.set_title('Neuronal membrane potentials', fontsize=11)
    axA.set_ylabel('$\phi_\mathrm{m,ns}$ [mV]')
    axB.set_ylabel('$\phi_\mathrm{m,ns}$ [mV]')
    axC.set_ylabel('$\phi_\mathrm{m,n}$ [mV]')
    axD.set_title('Extracellular potentials', fontsize=11)
    axD.set_ylabel('$\phi_\mathrm{es}$ [mV]')
    axE.set_ylabel('$\phi_\mathrm{es}$ [mV]')
    axF.set_ylabel('$\phi_\mathrm{e}$ [mV]')
    axG.set_title('Glial membrane potentials', fontsize=11)
    axG.set_ylabel('$\phi_\mathrm{m,gs}$ [mV]')
    axH.set_ylabel('$\phi_\mathrm{m,gs}$ [mV]')
    axI.set_ylabel('$\phi_\mathrm{m,g}$ [mV]')

    # make pretty
    for ax in [axA, axB, axC, axD, axE, axF, axG, axH, axI]:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_xlabel('time [s]')

    panel = np.array(['A', 'D', 'G'])
    i = 0
    for ax in [axA, axD, axG]:
        ax.text(-0.065, 1.06, panel[i], transform=ax.transAxes, fontsize=12, fontweight='bold', va='top', ha='right')
        i += 1
    panel = np.array(['B', 'C', 'E', 'F', 'H', 'I'])
    i = 0
    for ax in [axB, axC, axE, axF, axH, axI]:
        ax.text(-0.17, 1.06, panel[i], transform=ax.transAxes, fontsize=12, fontweight='bold', va='top', ha='right')
        i += 1

    fig.tight_layout(pad=0.4)
    fig.align_labels()

    plt.savefig(path_figures + 'phi.pdf', dpi=300)
    plt.close()

def plot_c_alpha(path_data, path_figures, N_units):
    """ Plot extracellular, neuronal, and glial potassium concentrations (soma layer), 
    and changes in extracellular volume fractions (soma layer). """

    if not os.path.isdir(path_figures):
        os.makedirs(path_figures)

    set_style('default', w=1, h=4)

    # set axes
    fig = plt.figure()
    gs = gridspec.GridSpec(6,2)
    
    axA = fig.add_subplot(gs[0,:])
    axB = fig.add_subplot(gs[1,0])
    axC = fig.add_subplot(gs[1,1])
    axD = fig.add_subplot(gs[2,:])
    axE = fig.add_subplot(gs[3,0])
    axF = fig.add_subplot(gs[3,1])
    axG = fig.add_subplot(gs[4,:])
    axH = fig.add_subplot(gs[5,:])

    # set colors
    colors = sns.color_palette('muted')

    # get t array
    data_dict = load_data(path_data, path_figures, 0)
    t = data_dict['t']

    # panel A and B
    for n_unit in range(N_units):
        data_dict = load_data(path_data, path_figures, n_unit)
        axA.plot(t, data_dict['K_se']-data_dict['K_se'][0], color=colors[n_unit], label=str(n_unit+1))
        axB.plot(t, data_dict['K_se']-data_dict['K_se'][0], color=colors[n_unit])

    # panel C
    n_unit = 0
    data_dict = load_data(path_data, path_figures, n_unit)
    l1 = axC.plot(t, data_dict['K_se']-data_dict['K_se'][0], color=colors[n_unit])[0]
    l2 = axC.plot(t, data_dict['K_de']-data_dict['K_de'][0], color=colors[n_unit], linestyle='--')[0]
    fig.legend([l1, l2], ['soma', 'dendrite'], \
            loc=(0.75,0.77), ncol=1, fontsize='small', handlelength=0.8, handletextpad=0.4)

    # panel D and E
    for n_unit in range(N_units):
        data_dict = load_data(path_data, path_figures, n_unit)
        axD.plot(t, data_dict['K_sn']-data_dict['K_sn'][0], color=colors[n_unit])
        axE.plot(t, data_dict['K_sn']-data_dict['K_sn'][0], color=colors[n_unit])

    # panel F
    n_unit = 0
    data_dict = load_data(path_data, path_figures, n_unit)
    axF.plot(t, data_dict['K_sn']-data_dict['K_sn'][0], label='soma',  color=colors[n_unit])
    axF.plot(t, data_dict['K_dn']-data_dict['K_dn'][0], label='dendrite', color=colors[n_unit], linestyle='--')

    # panel G
    for n_unit in range(N_units):
        data_dict = load_data(path_data, path_figures, n_unit)
        axG.plot(t, data_dict['K_sg']-data_dict['K_sg'][0], color=colors[n_unit])

    # panel H
    for n_unit in range(N_units):
        data_dict = load_data(path_data, path_figures, n_unit)
        axH.plot(t, ((data_dict['alpha_se']-data_dict['alpha_se'][0])/data_dict['alpha_se'][0])*100, color=colors[n_unit])

    # adjust axes
    axB.set_ylim([-0.01, 0.15])
    axC.set_ylim([-0.01, 0.15])
    axD.set_ylim([-0.4, 0.01])
    axE.set_ylim([-0.1, 0.01])
    axF.set_ylim([-0.1, 0.01])
    for ax in [axA, axD, axG, axH]:
        ax.set_xlim([0, t[-1]])
    for ax in [axB, axC, axE, axF]:
        ax.set_xlim([0.1, 0.14])

    # set titles labels
    axA.set_title('Extracellular $\Delta$[K$^+$]', fontsize=11)
    axA.set_ylabel(r'$\Delta$[K$^+$]$_\mathrm{es}$ [mM]')
    axB.set_ylabel(r'$\Delta$[K$^+$]$_\mathrm{es}$ [mM]')
    axC.set_ylabel(r'$\Delta$[K$^+$]$_\mathrm{e}$ [mM]')
    axD.set_title('Neuronal $\Delta$[K$^+$]', fontsize=11)
    axD.set_ylabel(r'$\Delta$[K$^+$]$_\mathrm{ns}$ [mM]')
    axE.set_ylabel(r'$\Delta$[K$^+$]$_\mathrm{ns}$ [mM]')
    axF.set_ylabel(r'$\Delta$[K$^+$]$_\mathrm{n}$ [mM]')
    axG.set_title('Glial $\Delta$[K$^+$]', fontsize=11)
    axG.set_ylabel(r'$\Delta$[K$^+$]$_\mathrm{gs}$ [mM]')
    axH.set_title(r'Extracellular $\Delta\alpha$', fontsize=11)
    axH.set_ylabel(r'$\Delta\alpha_\mathrm{es} [\%]$')

    # make pretty
    for ax in [axA, axB, axC, axD, axE, axF, axG, axH]:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_xlabel('time [s]')

    panel = np.array(['A', 'D', 'G', 'H'])
    i = 0
    for ax in [axA, axD, axG, axH]:
        ax.text(-0.05, 1.3, panel[i], transform=ax.transAxes, fontsize=12, fontweight='bold', va='top', ha='right')
        i += 1
    panel = np.array(['B', 'C', 'E', 'F', 'H'])
    i = 0
    for ax in [axB, axC, axE, axF]:
        ax.text(-0.14, 1.3, panel[i], transform=ax.transAxes, fontsize=12, fontweight='bold', va='top', ha='right')
        i += 1

    fig.tight_layout(pad=0.4)
    fig.align_labels()

    plt.savefig(path_figures + 'c_alpha.pdf', dpi=300)
    plt.close()


def plot_phi_m(path_data, path_figures, N_units):
    " Plot neuronal membrane potentials (soma) "

    if not os.path.isdir(path_figures):
        os.makedirs(path_figures)

    set_style('default', w=1, h=1.2)

    # set axis
    fig = plt.figure()
    gs = gridspec.GridSpec(1,1)
    ax = fig.add_subplot(gs[0,:])

    # set colors
    colors = sns.color_palette('muted')
    
    # get t array
    data_dict = load_data(path_data, path_figures, 0)
    t = data_dict['t']

    for n_unit in range(N_units):
        data_dict = load_data(path_data, path_figures, n_unit)
        ax.plot(t, data_dict['phi_sn'] - data_dict['phi_se'], color=colors[n_unit], label=str(n_unit+1))

    fig.legend(title='unit \#')

    # set titles labels
    ax.set_ylabel(r'$\phi_\mathrm{m,ns}$ [mV]')

    # make pretty
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xlabel('time [s]')
    ax.set_xlim([0, t[-1]])

    fig.tight_layout()

    plt.savefig(path_figures + 'phi_m.pdf', dpi=300)
    plt.close()

if __name__ == "__main__":

    path_data = 'results/data/'
    path_figures = 'results/figures/'

    plot_phi(path_data, path_figures, 10)
    plot_c_alpha(path_data, path_figures, 10)
    plot_phi_m(path_data, path_figures, 10)
