import numpy as np
from scipy.sparse.linalg import spsolve
from scipy.sparse import coo_array
from numpy.linalg import norm
import copy
from ednm.aot_residual import residual
from ednm.aot_jacobian import jacobian

#spikes = np.load('spike_train.npz')['spike_train']
spikes = np.linspace(0, 2, 10)

def alpha_c_phi(model, dt, alpha, c, phi, ss_, t, t_AP, j_stim, N_stim, verbose=True):
    """
    Updates alpha, c, and phi using Newton's method.

    Arguments:
        model (class): network model
        dt (float): time step
        alpha (3D array): volume fractions from previous time step [layer (s,d)][domain (n,g)][unit]
        c (4D array): ion concentrations from previous time step [layer (s,d)][ion (Na, K, Cl, Ca)][domain (n,g,e)][unit]
        phi (3D array): electrical potentials from previous time step [layer (s,d,e)][domain (n,g)][unit]
        ss_ (2D array): gating variables from previous time step [variable][unit]
        t (float): time
        t_AP (2D array): time of action potentials
        j_stim (float): stimulus flux
        N_stim (int): number of cells receiving external input
    Returns: 
        alpha (3D array): volume fractions at time t
        c (4D array): ion concentrations at time t
        phi (3D array): electrical potentials at time t
    """

    # get parameters
    synapses = model.synapses                   # (0: no synapses, 1: synapses)
    bc = model.bc                               # boundary condition (0: closed, 1: open)
    N_domains = model.N_domains                 # number of domains
    N_units = model.N_units                     # number of units
    N_ions = model.N_ions                       # number of ions
    N_unknowns_tot = model.N_unknowns_tot       # number of unknowns
    N_unknowns_unit = model.N_unknowns_unit     # number of unknowns within each unit
    # parameters defining the ordering of the unknowns
    jc_s = model.jc_s
    jc_d = model.jc_d
    jphi_s = model.jphi_s
    jphi_d = model.jphi_d
    jalpha_s = model.jalpha_s
    jalpha_d = model.jalpha_d

    # parameters for Newton iteration
    itermax = 10            # maximum Newton iterations allowed
    tol = 1e-12             # tolerance
    rsd = tol + 1           # initiate residual
    N_iter = 0              # iteration counter

    # split arrays and prepare volume fractions
    alpha_s = np.zeros((N_domains, N_units), dtype=np.float64)
    alpha_d = np.zeros((N_domains, N_units), dtype=np.float64)
    alpha_s[:-1], alpha_d[:-1] = alpha
    alpha_s[-1] = 1 - np.sum(alpha[0], 0)
    alpha_d[-1] = 1 - np.sum(alpha[1], 0)
    c_s, c_d = c
    phi_s, phi_d = phi
    # store past values
    alpha_s_ = copy.deepcopy(alpha_s)
    alpha_d_ = copy.deepcopy(alpha_d)
    c_s_, c_d_ = copy.deepcopy(c)
    phi_s_, phi_d_ = copy.deepcopy(phi)

    # newton solver
    while rsd > tol:

        # compute residual
        Res = residual(dt, N_units, synapses, bc, alpha_s, alpha_d, c_s, c_d, phi_s, phi_d,
                alpha_s_, alpha_d_, c_s_, c_d_, phi_s_, phi_d_, ss_, t, t_AP, j_stim, N_stim, spikes)
        rsd = norm(Res, np.inf)

        if verbose:
            print('N_iter: %d, rsd: %e, phi_de: %e' % (N_iter, rsd, phi_d[-1][0]))

        # if residual is small or iteration count is large, exit loop
        if rsd < tol or N_iter >= itermax:
            break

        N_iter += 1

        # compute Jacobian
        irow, icol, Avals = jacobian(dt, N_units, synapses, bc, alpha_s, alpha_d, c_s, c_d, phi_s, phi_d,
                                alpha_s_, alpha_d_, c_s_, c_d_, phi_s_, phi_d_, ss_, t, t_AP, spikes)

        # create sparse matrix
        A = coo_array((Avals, (irow.astype(int), icol.astype(int))), shape=(N_unknowns_tot, N_unknowns_tot)).tocsc()

        # solve (J(x_) dx = -f(x_)
        Q = spsolve(A,-Res)

        # update alpha, k, and phi 
        for r in range(N_domains):
            phi_s[r] = phi_s[r] + Q[jphi_s[r]:N_unknowns_tot:N_unknowns_unit]
            phi_d[r] = phi_d[r] + Q[jphi_d[r]:N_unknowns_tot:N_unknowns_unit]
            for k in range(N_ions):
                c_s[k][r] = c_s[k][r] + Q[jc_s[k][r]:N_unknowns_tot:N_unknowns_unit]
                c_d[k][r] = c_d[k][r] + Q[jc_d[k][r]:N_unknowns_tot:N_unknowns_unit]
        for r in range(N_domains-1):
                alpha_s[r] = alpha_s[r] + Q[jalpha_s[r]:N_unknowns_tot:N_unknowns_unit]
                alpha_d[r] = alpha_d[r] + Q[jalpha_d[r]:N_unknowns_tot:N_unknowns_unit]
        alpha_s[-1] = 1 - alpha_s[0] - alpha_s[1]
        alpha_d[-1] = 1 - alpha_d[0] - alpha_d[1]

    alpha = [alpha_s[0:-1], alpha_d[0:-1]]
    c = [c_s, c_d]
    phi = [phi_s, phi_d]

    return alpha, c, phi

def gating_variables(model, dt, ss_, c, phi):

    # split arrays
    h_, n_, s_, c_, q_, z_ = ss_
    c_s, c_d = c
    phi_s, phi_d = phi

    # get parameters
    Ca_dn = c_d[-1][0]

    # calculate membrane potentials
    phi_mns = phi_s[0] - phi_s[-1]
    phi_mnd = phi_d[0] - phi_d[-1]

    # solve
    h = (h_ + dt * model.alpha_h(phi_mns)) / (1.0 + dt * model.alpha_h(phi_mns) + dt * model.beta_h(phi_mns))
    n = (n_ + dt * model.alpha_n(phi_mns)) / (1.0 + dt * model.alpha_n(phi_mns) + dt * model.beta_n(phi_mns))
    s = (s_ + dt * model.alpha_s(phi_mnd)) / (1.0 + dt * model.alpha_s(phi_mnd) + dt * model.beta_s(phi_mnd))
    c = (c_ + dt * model.alpha_c(phi_mnd)) / (1.0 + dt * model.alpha_c(phi_mnd) + dt * model.beta_c(phi_mnd))
    q = (q_ + dt * model.alpha_q(Ca_dn)) / (1.0 + dt * model.alpha_q(Ca_dn) + dt * model.beta_q())
    z = (z_ + (dt/model.tau_z)*model.z_inf(phi_mnd)) / (1.0 + dt/model.tau_z)

    # collect solutions
    ss = np.zeros((6, model.N_units), dtype=np.float64)
    ss[0][:] = h 
    ss[1][:] = n
    ss[2][:] = s
    ss[3][:] = c
    ss[4][:] = q
    ss[5][:] = z

    return ss

def solve_system(model, path_results, Tstop, stimulus_protocol):

    # get stimulus parameters
    j_stim = stimulus_protocol["j_stim"]
    stim_start = stimulus_protocol["stim_start"]
    stim_end = stimulus_protocol["stim_end"]
    N_stim = stimulus_protocol["N_stim"]

    # get initial conditions
    alpha_n_init = model.alpha_sn_init
    alpha_g_init = model.alpha_sg_init
    alpha_l_ = np.array([alpha_n_init, alpha_g_init], np.float64)

    Na_sn_init = model.Na_sn_init
    Na_sg_init = model.Na_sg_init
    Na_se_init = model.Na_se_init
    K_sn_init = model.K_sn_init
    K_sg_init = model.K_sg_init
    K_se_init = model.K_se_init
    Cl_sn_init = model.Cl_sn_init
    Cl_sg_init = model.Cl_sg_init
    Cl_se_init = model.Cl_se_init
    Ca_sn_init = model.Ca_sn_init
    Ca_sg_init = model.Ca_sg_init
    Ca_se_init = model.Ca_se_init
    c_s_ = np.array([[Na_sn_init, Na_sg_init, Na_se_init],
            [K_sn_init, K_sg_init, K_se_init],
            [Cl_sn_init, Cl_sg_init, Cl_se_init],
            [Ca_sn_init, Ca_sg_init, Ca_se_init]], dtype=np.float64)
    Na_dn_init = model.Na_dn_init
    Na_dg_init = model.Na_dg_init
    Na_de_init = model.Na_de_init
    K_dn_init = model.K_dn_init
    K_dg_init = model.K_dg_init
    K_de_init = model.K_de_init
    Cl_dn_init = model.Cl_dn_init
    Cl_dg_init = model.Cl_dg_init
    Cl_de_init = model.Cl_de_init
    Ca_dn_init = model.Ca_dn_init
    Ca_dg_init = model.Ca_dg_init
    Ca_de_init = model.Ca_de_init
    c_d_ = np.array([[Na_dn_init, Na_dg_init, Na_de_init],
            [K_dn_init, K_dg_init, K_de_init],
            [Cl_dn_init, Cl_dg_init, Cl_de_init],
            [Ca_dn_init, Ca_dg_init, Ca_de_init]], dtype=np.float64)

    phi_sn_init = model.phi_sn_init
    phi_sg_init = model.phi_sg_init
    phi_se_init = model.phi_se_init
    phi_s_ = np.array([phi_sn_init, phi_sg_init, phi_se_init], dtype=np.float64)
    phi_dn_init = model.phi_dn_init
    phi_dg_init = model.phi_dg_init
    phi_de_init = model.phi_de_init #+ 1
    phi_d_ = np.array([phi_dn_init, phi_dg_init, phi_de_init], dtype=np.float64)

    h_init = model.h_init
    n_init = model.n_init
    s_init = model.s_init
    c_init = model.c_init
    q_init = model.q_init
    z_init = model.z_init

    alpha_ = np.array([alpha_l_, alpha_l_], dtype=np.float64)
    c_ = np.array([c_s_, c_d_], dtype=np.float64)
    phi_ = np.array([phi_s_, phi_d_], dtype=np.float64)
    ss_ = np.array([h_init, n_init, s_init, c_init, q_init, z_init], dtype=np.float64)

    q_n = model.F*(Na_sn_init + K_sn_init - Cl_sn_init + 2*Ca_sn_init - model.a_s[0]/alpha_n_init)*alpha_n_init
    q_g = model.F*(Na_sg_init + K_sg_init - Cl_sg_init - model.a_s[1]/alpha_g_init)*alpha_g_init
    q_e = model.F*(Na_se_init + K_se_init - Cl_se_init + 2*Ca_se_init - model.a_s[2]/0.2)*0.2
    print('------------------------------------')
    print('----soma----')
    print('q_n+q_g, unit 0:', q_n[0]+q_g[0])
    print('q_e, unit 0:', q_e[0])
    print('q_tot, unit 0:', (q_e[0]+q_n[0]+q_g[0]))
    q_n = model.F*(Na_dn_init + K_dn_init - Cl_dn_init + 2*Ca_dn_init - model.a_d[0]/alpha_n_init)*alpha_n_init
    q_g = model.F*(Na_dg_init + K_dg_init - Cl_dg_init - model.a_d[1]/alpha_g_init)*alpha_g_init
    q_e = model.F*(Na_de_init + K_de_init - Cl_de_init + 2*Ca_de_init - model.a_d[2]/0.2)*0.2
    print('----dendrite----')
    print('q_n+q_g, unit 0:', q_n[0]+q_g[0])
    print('q_e, unit 0:', q_e[0])
    print('q_tot, unit 0:', (q_e[0]+q_n[0]+q_g[0]))
    print('------------------------------------')

    # maximum number of time steps
    dt = 5e-5       # time step
    N_t = int(Tstop/dt) + 3

    # create arrays to store results
    t_array = np.zeros(N_t)
    dt_array = np.zeros(N_t)
    alpha_array = np.zeros((N_t, model.N_layers, model.N_domains-1, model.N_units))
    c_array = np.zeros((N_t, model.N_layers, model.N_ions, model.N_domains, model.N_units))
    phi_array = np.zeros((N_t, model.N_layers, model.N_domains, model.N_units))
    ss_array = np.zeros((N_t, len(ss_), model.N_units))

    # allocate array to save spike times
    t_AP = np.zeros((N_t, model.N_units), dtype=np.float64)
    # mark neurons as not spiking
    spiking = np.array(np.zeros(model.N_units), dtype=bool)

    # save initial conditions / initial guesses
    t_array[0] = - dt
    dt_array[0] = dt
    alpha_array[0] = alpha_
    c_array[0] = c_ 
    phi_array[0] = phi_
    ss_array[0] = ss_

    # solve
    k = 1
    t = 0
    while t < Tstop:

        print('*************************************')
        print('Current time:', t, 's')
        print('-------------------------------------')

        # solve
        if t > stim_start and t < stim_end:
            alpha, c, phi = alpha_c_phi(model, dt, alpha_, c_, phi_, ss_, t, t_AP, j_stim, N_stim)
        else:
            alpha, c, phi = alpha_c_phi(model, dt, alpha_, c_, phi_, ss_, t, t_AP, 0.0, N_stim)
        ss = gating_variables(model, dt, ss_, c, phi)
       
        # store solutions 
        t_array[k] = t
        dt_array[k] = dt
        alpha_array[k][:][:][:] = alpha
        c_array[k][:][:][:][:] = c
        phi_array[k][:][:][:] = phi
        ss_array[k][:][:] = ss 

        # update solution
        alpha_ = alpha
        c_ = c
        phi_ = phi
        ss_ = ss

        # save spike times
        phi_ms = phi[0][0] - phi[0][-1]
        above_th = np.greater_equal(phi_ms, -20)
        condition = above_th > spiking
        spiking[condition] = True
        t_AP[k][condition] = t
        below_th = np.less(phi_ms, -40)
        spiking[below_th] = False

        # update time
        t += dt
        # update k
        k += 1

    # remove empty stuff
    t_AP = t_AP[0:k]
    t_array = t_array[0:k]
    dt_array = dt_array[0:k]
    alpha_array = alpha_array[0:k][:][:][:]
    c_array = c_array[0:k][:][:][:][:]
    phi_array = phi_array[0:k][:][:][:]
    ss_array = ss_array[0:k][:][:]

    # save results to file
    filename = path_results + 'data.npz'
    np.savez(filename, N_units=model.N_units, t=t_array, dt=dt_array, alpha=alpha_array, c=c_array, phi=phi_array, ss=ss_array, t_AP=t_AP)

    return
