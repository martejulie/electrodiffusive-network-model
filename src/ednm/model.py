import numpy as np
from numba import int32, float64
from numba.experimental import jitclass

spec = [
        ('synapses', int32),
        ('bc', int32),
        ('N_units', int32),
        ('N_domains', int32),
        ('N_layers', int32),
        ('N_ions', int32),
        ('N_unknowns_unit', int32),
        ('N_unknowns_tot', int32),
        ('alpha_sn_init', float64[:]),
        ('alpha_sg_init', float64[:]),
        ('alpha_se_init', float64[:]),
        ('alpha_dn_init', float64[:]),
        ('alpha_dg_init', float64[:]),
        ('alpha_de_init', float64[:]),
        ('Na_sn_init', float64[:]),
        ('Na_sg_init', float64[:]),
        ('Na_se_init', float64[:]),
        ('K_sn_init', float64[:]),
        ('K_sg_init', float64[:]),
        ('K_se_init', float64[:]),
        ('Cl_sn_init', float64[:]),
        ('Cl_sg_init', float64[:]),
        ('Cl_se_init', float64[:]),
        ('Ca_sn_init', float64[:]),
        ('Ca_sg_init', float64[:]),
        ('Ca_se_init', float64[:]),
        ('Na_dn_init', float64[:]),
        ('Na_dg_init', float64[:]),
        ('Na_de_init', float64[:]),
        ('K_dn_init', float64[:]),
        ('K_dg_init', float64[:]),
        ('K_de_init', float64[:]),
        ('Cl_dn_init', float64[:]),
        ('Cl_dg_init', float64[:]),
        ('Cl_de_init', float64[:]),
        ('Ca_dn_init', float64[:]),
        ('Ca_dg_init', float64[:]),
        ('Ca_de_init', float64[:]),
        ('phi_sn_init', float64[:]),
        ('phi_sg_init', float64[:]),
        ('phi_se_init', float64[:]),
        ('phi_dn_init', float64[:]),
        ('phi_dg_init', float64[:]),
        ('phi_de_init', float64[:]),
        ('n_init', float64[:]),
        ('h_init', float64[:]),
        ('s_init', float64[:]),
        ('c_init', float64[:]),
        ('q_init', float64[:]),
        ('z_init', float64[:]),
        ('T', float64),
        ('F', float64),
        ('R', float64),
        ('psi', float64),
        ('Na_threshold', float64),
        ('K_threshold', float64),
        ('C_m', float64),
        ('gamma_m', float64),
        ('dxl', float64),
        ('dxu', float64),
        ('theta', float64[:]),
        ('D', float64[:]),
        ('upsilon', float64[:, :]),
        ('lambdas', float64[:]),
        ('z', float64[:]),
        ('g_Na_leak_n', float64),
        ('g_K_leak_n', float64),
        ('g_Cl_leak_n', float64),
        ('g_Na', float64),
        ('g_DR', float64),
        ('g_Ca', float64),
        ('g_AHP', float64),
        ('g_C', float64),
        ('g_Na_leak_g', float64),
        ('g_K_IR', float64),
        ('g_Cl_leak_g', float64),
        ('g_nmda_Na', float64),
        ('g_nmda_K', float64),
        ('g_nmda_Ca', float64),
        ('tau_z', float64),
        ('rho_n', float64),
        ('U_kcc2', float64),
        ('U_nkcc1', float64),
        ('U_Cadec', float64),
        ('rho_g', float64),
        ('eta_m', float64[:]),
        ('bK_sg', float64),
        ('bK_se', float64),
        ('bK_dg', float64),
        ('bK_de', float64),
        ('bCa_sn', float64),
        ('bCa_dn', float64),
        ('bE_K_sg', float64),
        ('bE_K_dg', float64),
        ('a_s', float64[:, :]),
        ('a_d', float64[:, :]),
        ('delta_p_s', float64[:, :]),
        ('delta_p_d', float64[:, :]),
        ('dt', float64),
        ('irow', int32[:]),
        ('icol', int32[:]),
        ('Avals', float64[:]),
        ('jc_s', int32[:, :]),
        ('jc_d', int32[:, :]),
        ('jphi_s', int32[:]),
        ('jphi_d', int32[:]),
        ('jalpha_s', int32[:]),
        ('jalpha_d', int32[:])
       ]


@jitclass(spec)
class Model(object):
    """  """

    def __init__(self, N_units, dxu, synapses, bc):
       
        self.N_units = N_units      # number of units
        self.N_domains = 3          # number of domains (neuron + ECS + glia)
        self.N_layers = 2           # number of layers (soma + dendrite)
        self.N_ions = 4             # number of ionic species (Na + K + Cl + Ca)
        self.synapses = synapses    # (0: no synapses, 1: synapse model one, 2: synapse model two)
        self.bc = bc                # boundary condition (0: closed, 1: periodic)
        self.dxu = dxu              # distance between units [cm]

        # set parameters and initial conditions
        self.set_initial_conditions()
        self.set_parameters()

        # prepare solver
        self.prepare_solver()

    def set_parameters(self):
        """ Set model parameters. """

        self.T = 309.14                     # temperature [K]
        self.F = 96.48                      # Faraday's constant [C/mmol]
        self.R = 8.314                      # gas constant [J/mol/K]
        self.psi = self.R*self.T/self.F

        # threshold concentrations for the glial Na/K pump [mM]
        self.Na_threshold = 10.0 
        self.K_threshold = 1.5

        # membrane capacitance [F/cm**2]
        self.C_m = 3e-6
        # membrane area per tissue volume [1/cm]
        self.gamma_m = 1.714683e3
        # distance between layers [cm]
        self.dxl = 6.67e-2
        
        # layer coupling
        th = 350
        self.theta = np.array([th, th, th], dtype=np.float64)

        # diffusion constants (cm**2/s)
        D_Na = 1.33e-5
        D_K = 1.96e-5
        D_Cl = 2.03e-5
        D_Ca = 0.71e-5
        self.D = np.array([D_Na, D_K, D_Cl, D_Ca], dtype=np.float64)

        # mobility fraction 
        self.upsilon = np.array([[1.0, 1.0, 1.0],
                       [1.0, 1.0, 1.0],
                       [1.0, 1.0, 1.0],
                       [0.01, 0.0, 1.0]], dtype=np.float64)

        # tortuosities
        lambda_n = 3.2
        lambda_g = 3.2
        lambda_e = 1.6
        self.lambdas = np.array([lambda_n, lambda_g, lambda_e], dtype=np.float64)

        # valencies
        z_Na = 1
        z_K = 1
        z_Cl = -1
        z_Ca = 2
        z_0 = -1
        self.z = np.array([z_Na, z_K, z_Cl, z_Ca, z_0], dtype=np.float64)

        # conductances [S/cm**2]
        self.g_Na_leak_n = 0.246e-4
        self.g_K_leak_n = 0.245e-4  
        self.g_Cl_leak_n = 1.0e-4
        self.g_Na = 300e-4
        self.g_DR = 150e-4
        self.g_Ca = 118e-4
        self.g_AHP = 8e-4
        self.g_C = 150e-4
        self.g_Na_leak_g = 1.0e-4
        self.g_K_IR = 16.96e-4
        self.g_Cl_leak_g = 0.5e-4
        self.g_nmda_Na = 1.623e-4
        self.g_nmda_K = 3.084e-4
        self.g_nmda_Ca = 10.55e-7

        # time constant [s]
        self.tau_z = 1
        
        # exchanger strengths 
        self.rho_n = 1.87e-4       # [mM cm / s]
        self.U_kcc2 = 1.49e-5      # [mM cm / s] 
        self.U_nkcc1 = 2.33e-5     # [mM cm / s]
        self.U_Cadec = 75.         # [1/s]
        self.rho_g = 1.12e-4       # [mM cm / s]

        # water permeabilities (cm/Pa/s)
        eta_n = 3.25e-12
        eta_g = 8.12e-12
        self.eta_m = np.array([eta_n, eta_g], dtype=np.float64)

        # baseline concentrations
        self.bK_sg = 99.959
        self.bK_se = 3.082 
        self.bK_dg = 99.959
        self.bK_de = 3.082
        self.bCa_sn = 0.01
        self.bCa_dn = 0.01
        # baseline reversal potentials (mV)
        self.bE_K_sg, dummy, dummy = self.nernst_potential(self.z[1], self.upsilon[1][1], self.bK_sg, self.bK_se)
        self.bE_K_dg, dummy, dummy = self.nernst_potential(self.z[1], self.upsilon[1][1], self.bK_dg, self.bK_de)

        # calculate and set immobile ions
        self.set_immobile_ions()
        # calculate and set hydrostatic pressure
        self.set_hydrostatic_pressure()
        
        return

    def set_immobile_ions(self):
        """ calculate and set amount of immobile ions """

        # get parameters
        F = self.F
        gamma_m = self.gamma_m
        C_m = self.C_m
        z_Na = self.z[0]
        z_K = self.z[1]
        z_Cl = self.z[2]
        z_Ca = self.z[3]
        z_0 = self.z[4]

        # get initial membrane potentials
        phi_msn_init = self.phi_sn_init - self.phi_se_init
        phi_msg_init = self.phi_sg_init - self.phi_se_init
        phi_mdn_init = self.phi_dn_init - self.phi_de_init
        phi_mdg_init = self.phi_dg_init - self.phi_de_init

        # set amount of immobile ions - [mol/m**3]
        a_sn = gamma_m*C_m/(z_0*F)*phi_msn_init \
                - self.alpha_sn_init/z_0*(z_Na*self.Na_sn_init
                + z_K*self.K_sn_init
                + z_Cl*self.Cl_sn_init
                + z_Ca*self.Ca_sn_init) 
        a_sg = gamma_m*C_m/(z_0*F)*phi_msg_init \
                - self.alpha_sg_init/z_0*(z_Na*self.Na_sg_init
                + z_K*self.K_sg_init
                + z_Cl*self.Cl_sg_init)
        a_se = - gamma_m*C_m/(z_0*F)*(phi_msn_init + phi_msg_init) \
                - self.alpha_se_init/z_0*(z_Na*self.Na_se_init
                + z_K*self.K_se_init
                + z_Cl*self.Cl_se_init
                + z_Ca*self.Ca_se_init) 
        a_dn = gamma_m*C_m/(z_0*F)*phi_mdn_init \
                - self.alpha_dn_init/z_0*(z_Na*self.Na_dn_init
                + z_K*self.K_dn_init
                + z_Cl*self.Cl_dn_init
                + z_Ca*self.Ca_dn_init) 
        a_dg = gamma_m*C_m/(z_0*F)*phi_mdg_init \
                - self.alpha_dg_init/z_0*(z_Na*self.Na_dg_init
                + z_K*self.K_dg_init
                + z_Cl*self.Cl_dg_init)
        a_de = - gamma_m*C_m/(z_0*F)*(phi_mdn_init + phi_mdg_init) \
                - self.alpha_de_init/z_0*(z_Na*self.Na_de_init
                + z_K*self.K_de_init
                + z_Cl*self.Cl_de_init
                + z_Ca*self.Ca_de_init) 

        self.a_s = np.zeros((self.N_domains, self.N_units), dtype=np.float64)
        self.a_d = np.zeros((self.N_domains, self.N_units), dtype=np.float64)
        self.a_s[0] = a_sn; self.a_s[1] = a_sg; self.a_s[2] = a_se
        self.a_d[0] = a_dn; self.a_d[1] = a_dg; self.a_d[2] = a_de

        return

    def set_hydrostatic_pressure(self):
        """ Calculate and set initial hydrostatic pressure ensuring zero fluid flow at t = 0. """

        # transmembrane hydrostatic pressure - neuron
        delta_p_sn = - self.R*self.T*(self.a_s[-1]/self.alpha_se_init
                   + self.upsilon[0][-1]*self.Na_se_init
                   + self.upsilon[1][-1]*self.K_se_init
                   + self.upsilon[2][-1]*self.Cl_se_init
                   + self.upsilon[3][-1]*self.Ca_se_init
                   - self.a_s[0]/self.alpha_sn_init
                   - (self.upsilon[0][0]*self.Na_sn_init
                   + self.upsilon[1][0]*self.K_sn_init
                   + self.upsilon[2][0]*self.Cl_sn_init
                   + self.upsilon[3][0]*self.Ca_sn_init))
        delta_p_dn = - self.R*self.T*(self.a_d[-1]/self.alpha_de_init
                   + self.upsilon[0][-1]*self.Na_de_init
                   + self.upsilon[1][-1]*self.K_de_init
                   + self.upsilon[2][-1]*self.Cl_de_init
                   + self.upsilon[3][-1]*self.Ca_de_init
                   - self.a_d[0]/self.alpha_dn_init
                   - (self.upsilon[0][0]*self.Na_dn_init
                   + self.upsilon[1][0]*self.K_dn_init
                   + self.upsilon[2][0]*self.Cl_dn_init
                   + self.upsilon[3][0]*self.Ca_dn_init))

        # transmembrane hydrostatic pressure - glia 
        delta_p_sg = - self.R*self.T*(self.a_s[-1]/self.alpha_se_init
                   + self.upsilon[0][-1]*self.Na_se_init
                   + self.upsilon[1][-1]*self.K_se_init
                   + self.upsilon[2][-1]*self.Cl_se_init
                   + self.upsilon[3][-1]*self.Ca_se_init
                   - self.a_s[1]/self.alpha_sg_init
                   - (self.upsilon[0][1]*self.Na_sg_init
                   + self.upsilon[1][1]*self.K_sg_init
                   + self.upsilon[2][1]*self.Cl_sg_init
                   + self.upsilon[3][1]*self.Ca_sg_init))
        delta_p_dg = - self.R*self.T*(self.a_d[-1]/self.alpha_de_init
                   + self.upsilon[0][-1]*self.Na_de_init
                   + self.upsilon[1][-1]*self.K_de_init
                   + self.upsilon[2][-1]*self.Cl_de_init
                   + self.upsilon[3][-1]*self.Ca_de_init
                   - self.a_s[1]/self.alpha_dg_init
                   - (self.upsilon[0][1]*self.Na_dg_init
                   + self.upsilon[1][1]*self.K_dg_init
                   + self.upsilon[2][1]*self.Cl_dg_init
                   + self.upsilon[3][1]*self.Ca_dg_init))

        # collect and set transmembrane hydrostatic pressure
        self.delta_p_s = np.zeros((self.N_domains-1, self.N_units), dtype=np.float64)
        self.delta_p_d = np.zeros((self.N_domains-1, self.N_units), dtype=np.float64)
        self.delta_p_s[0] = delta_p_sn; self.delta_p_s[1] = delta_p_sg
        self.delta_p_d[0] = delta_p_dn; self.delta_p_d[1] = delta_p_dg

        return

    def set_initial_conditions(self):
        """ set the system's initial conditions """

        # allocate arrays for initial conditions
        # volume fractions
        self.alpha_sn_init = np.zeros(self.N_units, dtype=np.float64)
        self.alpha_sg_init = np.zeros(self.N_units, dtype=np.float64)
        self.alpha_se_init = np.zeros(self.N_units, dtype=np.float64)
        self.alpha_dn_init = np.zeros(self.N_units, dtype=np.float64)
        self.alpha_dg_init = np.zeros(self.N_units, dtype=np.float64)
        self.alpha_de_init = np.zeros(self.N_units, dtype=np.float64)

        # ion concentrations [mM]
        self.Na_sn_init = np.zeros(self.N_units, dtype=np.float64)
        self.Na_sg_init = np.zeros(self.N_units, dtype=np.float64)
        self.Na_se_init = np.zeros(self.N_units, dtype=np.float64)
        self.K_sn_init = np.zeros(self.N_units, dtype=np.float64)
        self.K_sg_init = np.zeros(self.N_units, dtype=np.float64)
        self.K_se_init = np.zeros(self.N_units, dtype=np.float64)
        self.Cl_sn_init = np.zeros(self.N_units, dtype=np.float64)
        self.Cl_sg_init = np.zeros(self.N_units, dtype=np.float64)
        self.Cl_se_init = np.zeros(self.N_units, dtype=np.float64)
        self.Ca_sn_init = np.zeros(self.N_units, dtype=np.float64)
        self.Ca_sg_init = np.zeros(self.N_units, dtype=np.float64)
        self.Ca_se_init = np.zeros(self.N_units, dtype=np.float64)
        
        self.Na_dn_init = np.zeros(self.N_units, dtype=np.float64)
        self.Na_dg_init = np.zeros(self.N_units, dtype=np.float64)
        self.Na_de_init = np.zeros(self.N_units, dtype=np.float64)
        self.K_dn_init = np.zeros(self.N_units, dtype=np.float64)
        self.K_dg_init = np.zeros(self.N_units, dtype=np.float64)
        self.K_de_init = np.zeros(self.N_units, dtype=np.float64)
        self.Cl_dn_init = np.zeros(self.N_units, dtype=np.float64)
        self.Cl_dg_init = np.zeros(self.N_units, dtype=np.float64)
        self.Cl_de_init = np.zeros(self.N_units, dtype=np.float64)
        self.Ca_dn_init = np.zeros(self.N_units, dtype=np.float64)
        self.Ca_dg_init = np.zeros(self.N_units, dtype=np.float64)
        self.Ca_de_init = np.zeros(self.N_units, dtype=np.float64)

        # membrane potential [mV]
        self.phi_sn_init = np.zeros(self.N_units, dtype=np.float64)
        self.phi_sg_init = np.zeros(self.N_units, dtype=np.float64)
        self.phi_se_init = np.zeros(self.N_units, dtype=np.float64)
        self.phi_dn_init = np.zeros(self.N_units, dtype=np.float64)
        self.phi_dg_init = np.zeros(self.N_units, dtype=np.float64)
        self.phi_de_init = np.zeros(self.N_units, dtype=np.float64)

        # gating variables
        self.n_init = np.zeros(self.N_units, dtype=np.float64)
        self.h_init = np.zeros(self.N_units, dtype=np.float64)
        self.s_init = np.zeros(self.N_units, dtype=np.float64)
        self.c_init = np.zeros(self.N_units, dtype=np.float64)
        self.q_init = np.zeros(self.N_units, dtype=np.float64)
        self.z_init = np.zeros(self.N_units, dtype=np.float64)

        # volume fractions
        self.alpha_sn_init[:] = 0.4
        self.alpha_sg_init[:] = 0.4
        self.alpha_se_init[:] = 0.2
        self.alpha_dn_init[:] = 0.4
        self.alpha_dg_init[:] = 0.4
        self.alpha_de_init[:] = 0.2
        
        # ion concentrations [mM]
        self.Na_sn_init[:] = 18.736726421059462
        self.Na_sg_init[:] = 14.482994016900394 
        self.Na_se_init[:] = 142.2432115550127 
        self.K_sn_init[:] = 138.09996293477718 
        self.K_sg_init[:] = 101.18465990039368 
        self.K_se_init[:] = 3.543992191091376 
        self.Cl_sn_init[:] = 7.1367807975347 
        self.Cl_sg_init[:] = 5.667645067417762 
        self.Cl_se_init[:] = 131.88695497178296 
        self.Ca_sn_init[:] = 0.010000000361272177 
        self.Ca_sg_init[:] = 0.0 
        self.Ca_se_init[:] = 1.0999582039386777
        
        self.Na_dn_init[:] = 18.749629739539404 
        self.Na_dg_init[:] = 14.481533094846167 
        self.Na_de_init[:] = 142.23786070566686 
        self.K_dn_init[:] = 138.087060160187 
        self.K_dg_init[:] = 101.18636609798413 
        self.K_de_init[:] = 3.5487536199534273 
        self.Cl_dn_init[:] = 7.136859444530827 
        self.Cl_dg_init[:] = 5.667890176782432 
        self.Cl_de_init[:] = 131.88638090271596 
        self.Ca_dn_init[:] = 0.010039023740848025 
        self.Ca_dg_init[:] = 0.0
        self.Ca_de_init[:] = 1.0999657699842467
         
        # membrane potential [mV]
        self.phi_sn_init[:] = -66.95238678754431
        self.phi_sg_init[:] = -83.89643848278573 
        self.phi_se_init[:] = 9.404342525113351e-05
        self.phi_dn_init[:] = -66.95290356270945
        self.phi_dg_init[:] = -83.89528586360711 
        self.phi_de_init[:] = 0.0

        # gating variables
        self.h_init[:] = 0.9993107753531558
        self.n_init[:] = 0.0003052605509823481
        self.s_init[:] = 0.007648596924219629
        self.c_init[:] = 0.0056423193653572885
        self.q_init[:] = 0.011666649776873634
        self.z_init[:] = 1.0

        return

    def interlayer_flux(self, k, r, c_s, c_d, phi_s, phi_d):

        # get parameters
        R = self.R
        T = self.T
        F = self.F
        psi = R*T/F
        D = self.D             
        lambdas = self.lambdas 
        upsilon = self.upsilon
        dxl = self.dxl
        z = self.z
        theta = self.theta

        # calculate flux and derivatives
        f_1 = theta[r] * D[k] * upsilon[k][r] / (lambdas[r]**2 * dxl)
        f_2 = f_1 * z[k] / (2 * psi)

        j_interlayer = - f_1 * (c_d[k][r] - c_s[k][r]) \
                - f_2 * (c_s[k][r] + c_d[k][r]) * (phi_d[r]-phi_s[r])
        djl_dcs = f_1 - f_2 * (phi_d[r] - phi_s[r])
        djl_dcd = - f_1 - f_2 * (phi_d[r] - phi_s[r])
        djl_dphis = f_2*(c_s[k][r] + c_d[k][r]) 
        djl_dphid = - djl_dphis

        return j_interlayer, djl_dcs, djl_dcd, djl_dphis, djl_dphid
    
    def interunits_flux(self, k, r, c_l, phi_l):

        # get parameters
        psi = self.psi 
        D = self.D             
        lambdas = self.lambdas 
        upsilon = self.upsilon
        dxu = self.dxu
        z = self.z

        # calculate flux and derivatives
        j_interunits = np.zeros(self.N_units)
        dju_dc0 = np.zeros(self.N_units)
        dju_dc1 = np.zeros(self.N_units)
        dju_dphi0 = np.zeros(self.N_units)
        dju_dphi1 = np.zeros(self.N_units)

        f_1 = D[k] * upsilon[k][r] / (lambdas[r]**2 * dxu)
        f_2 = f_1 * z[k] / (2 * psi)
        cc = (c_l[k][r][:-1] + c_l[k][r][1:])
        pp = (phi_l[r][1:] - phi_l[r][:-1])

        if r != 0:
            j_interunits[:-1] = - f_1 * (c_l[k][r][1:] - c_l[k][r][:-1]) \
                    - f_2 * cc * pp
            dju_dc0[:-1] = f_1 - f_2 * pp
            dju_dc1[:-1] = - f_1 - f_2 * pp
            dju_dphi0[:-1] = f_2 * cc
            dju_dphi1[:-1] = - dju_dphi0[:-1]

            if self.bc == 1:
                j_interunits[-1] = - f_1 * (c_l[k][r][0] - c_l[k][r][-1]) \
                        - f_2 * (c_l[k][r][-1] + c_l[k][r][0]) \
                        * (phi_l[r][0] - phi_l[r][-1])
                dju_dc0[-1] = f_1 - f_2 * (phi_l[r][0] - phi_l[r][-1])
                dju_dc1[-1] = - f_1 - f_2 * (phi_l[r][0] - phi_l[r][-1])
                dju_dphi0[-1] = f_2 * (c_l[k][r][-1] + c_l[k][r][0])
                dju_dphi1[-1] = - dju_dphi0[-1]

        return j_interunits, dju_dc0, dju_dc1, dju_dphi0, dju_dphi1

    def nernst_potential(self, z, upsilon_i, c_i, c_e):
        E = self.psi / z * np.log(c_e / (upsilon_i*c_i))
        dE_dci = - self.psi / (z * c_i)
        dE_dce = self.psi / (z * c_e)
        return E, dE_dci, dE_dce

    def j_pump_n(self, Na_n, K_e):
        j = (self.rho_n / (1.0 + np.exp((25.0 - Na_n)/3.0))) * (1.0 / (1.0 + np.exp(3.5 - K_e)))
        return j

    def j_pump_g(self, Na_g, K_e):
        j = self.rho_g * (Na_g**1.5 / (Na_g**1.5 + self.Na_threshold**1.5)) * (K_e / (K_e + self.K_threshold))
        return j

    def j_kcc2(self, K_n, K_e, Cl_n, Cl_e):
        j = self.U_kcc2 * np.log(K_n*Cl_n/(K_e*Cl_e))
        return j
    
    def j_nkcc1(self, Na_n, Na_e, K_n, K_e, Cl_n, Cl_e):
        j = self.U_nkcc1 * (1.0 / (1.0 + np.exp(16.0 - K_e))) * (np.log(K_n*Cl_n/(K_e*Cl_e)) + np.log(Na_n*Cl_n/(Na_e*Cl_e)))
        return j

    def g_syn_one(self, t, t_AP):
        """ Each neurons connetcs to its right neighbor from soma to dendrite 
        (neuron U connects to neuron 0), and neuron 0 receives a stimulus at t = 0.1 s. """
           
        tau_1 = 3.0e-3   # [s]
        tau_2 = 1.0e-3   # [s]

        S = 14.0          # synaptic strength (unitless)
        t_delay = .01     # synaptic delay [s]

        t_s = np.zeros(np.shape(t_AP), dtype=np.float64)
        t_s[:,1:] = t_AP[:,:-1]
        t_s[:,0] = t_AP[:,-1]

        # initiate activity
        t_s[0,0] = 0.09
       
        # calculate conductance
        g = np.zeros(self.N_units, dtype=np.float64)

        for u in range(self.N_units):
            condition_1 = t_s[:,u] > 0.0
            t_s_ = np.extract(condition_1, t_s[:,u])
            if np.sum(t_s_) > 0.0:
                condition_2 = t_s_ + t_delay < t
                t_s_ = np.extract(condition_2, t_s_)
                g_ = S*(np.exp(-(t-(t_s_+t_delay))/tau_1) - np.exp(-(t-(t_s_+t_delay))/tau_2))
                g[u] = sum(g_)
            else:
                g[u] = 0.0

        return g

    def g_syn_two(self, t, spike_train):
        """ Neuron 0 is stimulated by a spike train. """
           
        tau_1 = 3.0e-3   # [s]
        tau_2 = 1.0e-3   # [s]

        g = np.zeros(self.N_units, dtype=np.float64)
        
        condition = spike_train <= t
        t_s = np.extract(condition, spike_train)
        if sum(t_s)>0:
            G = np.exp(-(t-t_s)/tau_1) - np.exp(-(t-t_s)/tau_2)
            g[0] += sum(G)

        return g

    def set_membrane_fluxes_s(self, alpha_s, c_s, phi_s, alpha_s_, c_s_, phi_s_, ss_):
        """ set the models transmembrane ion fluxes - soma layer """

        # split unknowns
        [alpha_n, alpga_g, alpha_e] = alpha_s
        [[Na_n, Na_g, Na_e], [K_n, K_g, K_e], \
                [Cl_n, Cl_g, Cl_e], [Ca_n, Ca_g, Ca_e]] = c_s
        [phi_n, phi_g, phi_e] = phi_s

        # split solution from previous time step 
        [alpha_n_, alpga_g_, alpha_e_] = alpha_s_
        [[Na_n_, Na_g_, Na_e_], [K_n_, K_g_, K_e_], \
                [Cl_n_, Cl_g_, Cl_e_], [Ca_n_, Ca_g_, Ca_e_]] = c_s_
        [phi_n_, phi_g_, phi_e_] = phi_s_
        [h_, n_, s_, c_, q_, z_] = ss_
        
        # set parameters
        z_Na = self.z[0]
        z_K = self.z[1]
        z_Cl = self.z[2]
        bCa_n = self.bCa_sn

        # calculate membrane potentials
        phi_mn = phi_n - phi_e
        phi_mg = phi_g - phi_e
        # calculate membrane potentials from previous time step
        phi_mn_ = phi_n_ - phi_e_
        phi_mg_ = phi_g_ - phi_e_
        
        # calculate reversal potentials - neuron
        E_Na_n, dE_Na_dci_n, dE_Na_dce_n = self.nernst_potential(self.z[0], self.upsilon[0][0], Na_n, Na_e)
        E_K_n, dE_K_dci_n, dE_K_dce_n = self.nernst_potential(self.z[1], self.upsilon[1][0], K_n, K_e)
        E_Cl_n, dE_Cl_dci_n, dE_Cl_dce_n = self.nernst_potential(self.z[2], self.upsilon[2][0], Cl_n, Cl_e)
        E_Ca_n, dE_Ca_dci_n, dE_Ca_dce_n = self.nernst_potential(self.z[3], self.upsilon[3][0], Ca_n, Ca_e)
      
        # calculate reversal potentials - glia  
        E_Na_g, dE_Na_dci_g, dE_Na_dce_g = self.nernst_potential(self.z[0], self.upsilon[0][1], Na_g, Na_e)
        E_K_g, dE_K_dci_g, dE_K_dce_g = self.nernst_potential(self.z[1], self.upsilon[1][1], K_g, K_e)
        E_Cl_g, dE_Cl_dci_g, dE_Cl_dce_g = self.nernst_potential(self.z[2], self.upsilon[2][1], Cl_g, Cl_e)
  
        # neuronal membrane flux - sodium
        sum_g_Na_n = (self.g_Na_leak_n + self.g_Na * self.m_inf(phi_mn_)**2 * h_)/(self.F*z_Na)
        j_Na_mn = sum_g_Na_n * (phi_mn - E_Na_n) \
                + 3.0*self.j_pump_n(Na_n_, K_e_) \
                + self.j_nkcc1(Na_n_, Na_e_, K_n_, K_e_, Cl_n_, Cl_e_) \
                - 2.0*self.U_Cadec*(Ca_n_ - bCa_n)*alpha_n_/self.gamma_m 
        djNa_dci_n = - sum_g_Na_n * dE_Na_dci_n
        djNa_dce_n = - sum_g_Na_n * dE_Na_dce_n
        djNa_dphii_n = sum_g_Na_n
        djNa_dphie_n = - djNa_dphii_n

        # neuronal membrane flux - potassium 
        sum_g_K_n = (self.g_K_leak_n + self.g_DR * n_) / (self.F*z_K) 
        j_K_mn = sum_g_K_n * (phi_mn - E_K_n) \
                - 2.0*self.j_pump_n(Na_n_, K_e_) \
                + self.j_kcc2(K_n_, K_e_, Cl_n_, Cl_e_) \
                + self.j_nkcc1(Na_n_, Na_e_, K_n_, K_e_, Cl_n_, Cl_e_)
        djK_dci_n = - sum_g_K_n * dE_K_dci_n
        djK_dce_n = - sum_g_K_n * dE_K_dce_n
        djK_dphii_n = sum_g_K_n
        djK_dphie_n = - djK_dphii_n

        # neuronal membrane flux - chloride
        sum_g_Cl_n = self.g_Cl_leak_n / (self.F*z_Cl)
        j_Cl_mn = sum_g_Cl_n * (phi_mn - E_Cl_n) \
                + self.j_kcc2(K_n_, K_e_, Cl_n_, Cl_e_) \
                + 2.0*self.j_nkcc1(Na_n_, Na_e_, K_n_, K_e_, Cl_n_, Cl_e_)
        djCl_dci_n = - sum_g_Cl_n * dE_Cl_dci_n
        djCl_dce_n = - sum_g_Cl_n * dE_Cl_dce_n
        djCl_dphii_n = sum_g_Cl_n
        djCl_dphie_n = - djCl_dphii_n

        # neuronal membrane flux - calsium
        j_Ca_mn =  self.U_Cadec * (Ca_n_ - bCa_n)*alpha_n_/self.gamma_m 

        # glial membrane flux - sodium
        sum_g_Na_g = self.g_Na_leak_g / (self.F*z_Na)
        j_Na_mg = sum_g_Na_g * (phi_mg - E_Na_g) \
                + 3.0*self.j_pump_g(Na_g_, K_e_)
        djNa_dci_g = - sum_g_Na_g * dE_Na_dci_g
        djNa_dce_g = - sum_g_Na_g * dE_Na_dce_g
        djNa_dphii_g = sum_g_Na_g
        djNa_dphie_g = - djNa_dphii_g

        # glial membrane flux - potassium 
        E_K_g_, dE_K_dci_g_, dE_K_dce_g_ = self.nernst_potential(self.z[1], self.upsilon[1][1], K_g_, K_e_)
        dphi = (phi_mg_ - E_K_g_)
        fact1 = (1.0 + np.exp(18.4/42.4))/(1.0 + np.exp((dphi + 18.5)/42.5))
        fact2 = (1.0 + np.exp(-(118.6+self.bE_K_sg)/44.1))/(1.0+np.exp(-(118.6+phi_mg_)/44.1))
        f = np.sqrt(K_e_/self.bK_se) * fact1 * fact2 
        sum_g_K_g = (self.g_K_IR * f) / (self.F*z_K) 
        j_K_mg = sum_g_K_g * (phi_mg - E_K_g) \
               - 2.0 * self.j_pump_g(Na_g_, K_e_)
        djK_dci_g = - sum_g_K_g * dE_K_dci_g
        djK_dce_g = - sum_g_K_g * dE_K_dce_g
        djK_dphii_g = sum_g_K_g
        djK_dphie_g = - djK_dphii_g

        # glial membrane flux - chloride
        sum_g_Cl_g = self.g_Cl_leak_g / (self.F*z_Cl) 
        j_Cl_mg = sum_g_Cl_g * (phi_mg - E_Cl_g)
        djCl_dci_g = - sum_g_Cl_g * dE_Cl_dci_g
        djCl_dce_g = - sum_g_Cl_g * dE_Cl_dce_g
        djCl_dphii_g = sum_g_Cl_g
        djCl_dphie_g = - djCl_dphii_g

        # collect membrane fluxes
        j_m_s = np.zeros((self.N_ions, self.N_domains-1, self.N_units), dtype=np.float64)
        j_m_s[0][0][:] = j_Na_mn; j_m_s[0][1][:] = j_Na_mg
        j_m_s[1][0][:] = j_K_mn; j_m_s[1][1][:] = j_K_mg
        j_m_s[2][0][:] = j_Cl_mn; j_m_s[2][1][:] = j_Cl_mg
        j_m_s[3][0][:] = j_Ca_mn;

        # collect derivatives
        djm_dci_s = np.zeros((self.N_ions, self.N_domains-1, self.N_units), dtype=np.float64)
        djm_dci_s[0][0][:] = djNa_dci_n; djm_dci_s[0][1][:] = djNa_dci_g
        djm_dci_s[1][0][:] = djK_dci_n; djm_dci_s[1][1][:] = djK_dci_g
        djm_dci_s[2][0][:] = djCl_dci_n; djm_dci_s[2][1][:] = djCl_dci_g
        
        djm_dce_s = np.zeros((self.N_ions, self.N_domains-1, self.N_units), dtype=np.float64)
        djm_dce_s[0][0][:] = djNa_dce_n; djm_dce_s[0][1][:] = djNa_dce_g
        djm_dce_s[1][0][:] = djK_dce_n; djm_dce_s[1][1][:] = djK_dce_g
        djm_dce_s[2][0][:] = djCl_dce_n; djm_dce_s[2][1][:] = djCl_dce_g

        djm_dphii_s = np.zeros((self.N_ions, self.N_domains-1, self.N_units), dtype=np.float64)
        djm_dphii_s[0][0][:] = djNa_dphii_n; djm_dphii_s[0][1][:] = djNa_dphii_g
        djm_dphii_s[1][0][:] = djK_dphii_n; djm_dphii_s[1][1][:] = djK_dphii_g
        djm_dphii_s[2][0][:] = djCl_dphii_n; djm_dphii_s[2][1][:] = djCl_dphii_g
        
        djm_dphie_s = np.zeros((self.N_ions, self.N_domains-1, self.N_units), dtype=np.float64)
        djm_dphie_s[0][0][:] = djNa_dphie_n; djm_dphie_s[0][1][:] = djNa_dphie_g
        djm_dphie_s[1][0][:] = djK_dphie_n; djm_dphie_s[1][1][:] = djK_dphie_g
        djm_dphie_s[2][0][:] = djCl_dphie_n; djm_dphie_s[2][1][:] = djCl_dphie_g

        return j_m_s, djm_dci_s, djm_dce_s, djm_dphii_s, djm_dphie_s

    def set_membrane_fluxes_d(self, alpha_d, c_d, phi_d, alpha_d_, c_d_, phi_d_, ss_, t, t_AP, spike_train):
        """ set the models transmembrane ion fluxes - dendrite layer """

        # split unknowns
        [alpha_n, alpha_g, alpha_e] = alpha_d
        [[Na_n, Na_g, Na_e], [K_n, K_g, K_e], \
                [Cl_n, Cl_g, Cl_e], [Ca_n, Ca_g, Ca_e]] = c_d
        [phi_n, phi_g, phi_e] = phi_d

        # split solution from previous time step 
        [alpha_n_, alpha_g_, alpha_e_] = alpha_d
        [[Na_n_, Na_g_, Na_e_], [K_n_, K_g_, K_e_], \
                [Cl_n_, Cl_g_, Cl_e_], [Ca_n_, Ca_g_, Ca_e_]] = c_d_
        [phi_n_, phi_g_, phi_e_] = phi_d_
        [h_, n_, s_, c_, q_, z_] = ss_
    
        # set parameters
        z_Na = self.z[0]
        z_K = self.z[1]
        z_Cl = self.z[2]
        z_Ca = self.z[3]
        bCa_n = self.bCa_dn 
        if self.synapses == 1:
            g_synapse = self.g_syn_one(t, t_AP)
        elif self.synapses == 2:
            g_synapse = self.g_syn_two(t, spike_train)
        else: 
            g_synapse = np.zeros(self.N_units, dtype=np.float64)

        # calculate membrane potentials
        phi_mn = phi_n - phi_e
        phi_mg = phi_g - phi_e
        # calculate membrane potentials from previous time step
        phi_mg_ = phi_g_ - phi_e_

        # calculate reversal potentials - neuron
        E_Na_n, dE_Na_dci_n, dE_Na_dce_n = self.nernst_potential(self.z[0], self.upsilon[0][0], Na_n, Na_e)
        E_K_n, dE_K_dci_n, dE_K_dce_n = self.nernst_potential(self.z[1], self.upsilon[1][0], K_n, K_e)
        E_Cl_n, dE_Cl_dci_n, dE_Cl_dce_n = self.nernst_potential(self.z[2], self.upsilon[2][0], Cl_n, Cl_e)
        E_Ca_n, dE_Ca_dci_n, dE_Ca_dce_n = self.nernst_potential(self.z[3], self.upsilon[3][0], Ca_n, Ca_e)
        
        # calculate reversal potentials - glia  
        E_Na_g, dE_Na_dci_g, dE_Na_dce_g = self.nernst_potential(self.z[0], self.upsilon[0][1], Na_g, Na_e)
        E_K_g, dE_K_dci_g, dE_K_dce_g = self.nernst_potential(self.z[1], self.upsilon[1][1], K_g, K_e)
        E_Cl_g, dE_Cl_dci_g, dE_Cl_dce_g = self.nernst_potential(self.z[2], self.upsilon[2][1], Cl_g, Cl_e)

        # neuronal membrane flux - sodium
        sum_g_Na_n = (self.g_Na_leak_n + self.g_nmda_Na*g_synapse)/(self.F*z_Na)
        j_Na_mn = sum_g_Na_n*(phi_mn - E_Na_n) \
                + 3.0*self.j_pump_n(Na_n_, K_e_) \
                + self.j_nkcc1(Na_n_, Na_e_, K_n_, K_e_, Cl_n_, Cl_e_) \
                - 2.0*self.U_Cadec*(Ca_n_ - bCa_n)*alpha_n_/self.gamma_m
        djNa_dci_n = - sum_g_Na_n * dE_Na_dci_n
        djNa_dce_n = - sum_g_Na_n * dE_Na_dce_n
        djNa_dphii_n = sum_g_Na_n
        djNa_dphie_n = - sum_g_Na_n
        
        # neuronal membrane flux - potassium
        sum_g_K_n = (self.g_K_leak_n + self.g_AHP * q_ \
                + self.g_C * c_ * self.chi(Ca_n_) \
                + self.g_nmda_K*g_synapse) / (self.F*z_K) 
        j_K_mn = sum_g_K_n*(phi_mn - E_K_n) \
                - 2.0*self.j_pump_n(Na_n_, K_e_) \
                + self.j_kcc2(K_n_, K_e_, Cl_n_, Cl_e_) \
                + self.j_nkcc1(Na_n_, Na_e_, K_n_, K_e_, Cl_n_, Cl_e_)
        djK_dci_n = - sum_g_K_n * dE_K_dci_n
        djK_dce_n = - sum_g_K_n * dE_K_dce_n
        djK_dphii_n = sum_g_K_n
        djK_dphie_n = - sum_g_K_n

        # neuronal membrane flux - chloride 
        sum_g_Cl_n = self.g_Cl_leak_n / (self.F*z_Cl)
        j_Cl_mn = sum_g_Cl_n*(phi_mn - E_Cl_n) \
                + self.j_kcc2(K_n_, K_e_, Cl_n_, Cl_e_) \
                + 2.0*self.j_nkcc1(Na_n_, Na_e_, K_n_, K_e_, Cl_n_, Cl_e_)
        djCl_dci_n = - sum_g_Cl_n * dE_Cl_dci_n
        djCl_dce_n = - sum_g_Cl_n * dE_Cl_dce_n
        djCl_dphii_n = sum_g_Cl_n
        djCl_dphie_n = - sum_g_Cl_n
        
        # neuronal membrane flux - calsium
        sum_g_Ca_n = (self.g_Ca * s_**2.0 * z_ + self.g_nmda_Ca*g_synapse) / (self.F*z_Ca) 
        j_Ca_mn = sum_g_Ca_n * (phi_mn - E_Ca_n) \
                + self.U_Cadec*(Ca_n_ - bCa_n)*alpha_n_/self.gamma_m
        djCa_dci_n = - sum_g_Ca_n * dE_Ca_dci_n
        djCa_dce_n = - sum_g_Ca_n * dE_Ca_dce_n
        djCa_dphii_n = sum_g_Ca_n
        djCa_dphie_n = - sum_g_Ca_n

        # glial membrane flux - sodium
        sum_g_Na_g = self.g_Na_leak_g / (self.F*z_Na)
        j_Na_mg = sum_g_Na_g * (phi_mg - E_Na_g) \
                + 3.0*self.j_pump_g(Na_g_, K_e_)
        djNa_dci_g = - sum_g_Na_g * dE_Na_dci_g
        djNa_dce_g = - sum_g_Na_g * dE_Na_dce_g
        djNa_dphii_g = sum_g_Na_g
        djNa_dphie_g = - sum_g_Na_g

        # glial membrane flux - potassium
        E_K_g_, dE_K_dci_g_, dE_K_dce_g_ = self.nernst_potential(self.z[1], self.upsilon[1][1], K_g_, K_e_)
        dphi = (phi_mg_ - E_K_g_)
        fact1 = (1.0 + np.exp(18.4/42.4))/(1.0 + np.exp((dphi + 18.5)/42.5))
        fact2 = (1.0 + np.exp(-(118.6+self.bE_K_sg)/44.1))/(1.0+np.exp(-(118.6+phi_mg_)/44.1))
        f = np.sqrt(K_e_/self.bK_se) * fact1 * fact2 
        sum_g_K_g = (self.g_K_IR * f) / (self.F*z_K) 
        j_K_mg = sum_g_K_g * (phi_mg - E_K_g) \
               - 2.0 * self.j_pump_g(Na_g_, K_e_)
        djK_dci_g = - sum_g_K_g * dE_K_dci_g
        djK_dce_g = - sum_g_K_g * dE_K_dce_g
        djK_dphii_g = sum_g_K_g
        djK_dphie_g = - sum_g_K_g

        # glial membrane flux - chloride
        sum_g_Cl_g = self.g_Cl_leak_g / (self.F*z_Cl) 
        j_Cl_mg = sum_g_Cl_g * (phi_mg - E_Cl_g)
        djCl_dci_g = - sum_g_Cl_g * dE_Cl_dci_g
        djCl_dce_g = - sum_g_Cl_g * dE_Cl_dce_g
        djCl_dphii_g = sum_g_Cl_g
        djCl_dphie_g = - sum_g_Cl_g

        # collect membrane fluxes
        j_m_d = np.zeros((self.N_ions, self.N_domains-1, self.N_units), dtype=np.float64)
        j_m_d[0][0][:] = j_Na_mn; j_m_d[0][1][:] = j_Na_mg
        j_m_d[1][0][:] = j_K_mn; j_m_d[1][1][:] = j_K_mg
        j_m_d[2][0][:] = j_Cl_mn; j_m_d[2][1][:] = j_Cl_mg
        j_m_d[3][0][:] = j_Ca_mn;

        # collect derivatives
        djm_dci_d = np.zeros((self.N_ions, self.N_domains-1, self.N_units), dtype=np.float64)
        djm_dci_d[0][0][:] = djNa_dci_n; djm_dci_d[0][1][:] = djNa_dci_g
        djm_dci_d[1][0][:] = djK_dci_n; djm_dci_d[1][1][:] = djK_dci_g
        djm_dci_d[2][0][:] = djCl_dci_n; djm_dci_d[2][1][:] = djCl_dci_g
        djm_dci_d[3][0][:] = djCa_dci_n;
        
        djm_dce_d = np.zeros((self.N_ions, self.N_domains-1, self.N_units), dtype=np.float64)
        djm_dce_d[0][0][:] = djNa_dce_n; djm_dce_d[0][1][:] = djNa_dce_g
        djm_dce_d[1][0][:] = djK_dce_n; djm_dce_d[1][1][:] = djK_dce_g
        djm_dce_d[2][0][:] = djCl_dce_n; djm_dce_d[2][1][:] = djCl_dce_g
        djm_dce_d[3][0][:] = djCa_dce_n;

        djm_dphii_d = np.zeros((self.N_ions, self.N_domains-1, self.N_units), dtype=np.float64)
        djm_dphii_d[0][0][:] = djNa_dphii_n; djm_dphii_d[0][1][:] = djNa_dphii_g
        djm_dphii_d[1][0][:] = djK_dphii_n; djm_dphii_d[1][1][:] = djK_dphii_g
        djm_dphii_d[2][0][:] = djCl_dphii_n; djm_dphii_d[2][1][:] = djCl_dphii_g
        djm_dphii_d[3][0][:] = djCa_dphii_n;
        
        djm_dphie_d = np.zeros((self.N_ions, self.N_domains-1, self.N_units), dtype=np.float64)
        djm_dphie_d[0][0][:] = djNa_dphie_n; djm_dphie_d[0][1][:] = djNa_dphie_g
        djm_dphie_d[1][0][:] = djK_dphie_n; djm_dphie_d[1][1][:] = djK_dphie_g
        djm_dphie_d[2][0][:] = djCl_dphie_n; djm_dphie_d[2][1][:] = djCl_dphie_g
        djm_dphie_d[3][0][:] = djCa_dphie_n;

        return j_m_d, djm_dci_d, djm_dce_d, djm_dphii_d, djm_dphie_d

    def set_water_flow(self, alpha_s, alpha_d, c_s, c_d):
        
        w_m_s = np.zeros((2, self.N_units), dtype=np.float64)
        w_m_d = np.zeros((2, self.N_units), dtype=np.float64)

        for r in range(self.N_domains-1): # n, g
            
            # add contribution from immobile ions to w_m 
            w_m_s[r][:] += self.a_s[-1][:]/alpha_s[-1][:] - self.a_s[r][:]/alpha_s[r][:]
            w_m_d[r][:] += self.a_d[-1][:]/alpha_d[-1][:] - self.a_d[r][:]/alpha_d[r][:]

            # add contribution from Na, K, Cl, and Ca ions to w_m 
            for k in range(self.N_ions):
                w_m_s[r][:] += self.upsilon[k][-1]*c_s[k][-1][:] - self.upsilon[k][r]*c_s[k][r][:]
                w_m_d[r][:] += self.upsilon[k][-1]*c_d[k][-1][:] - self.upsilon[k][r]*c_d[k][r][:]

        # contributions from R and T to w_m
        w_m_s = self.R*self.T*w_m_s
        w_m_d = self.R*self.T*w_m_d

        for r in range(self.N_domains-1):
            # contribution for hydrostatic pressure to w_m
            w_m_s[r][:] += self.delta_p_s[r][:]
            w_m_d[r][:] += self.delta_p_d[r][:]
            # contribution from water permeability eta_m to w_m
            w_m_s[r][:] *= self.eta_m[r]
            w_m_d[r][:] *= self.eta_m[r]

        # derivatives - volume fractions
        dw_dalpha_s = np.zeros((2, self.N_units), dtype=np.float64)
        dw_dalpha_d = np.zeros((2, self.N_units), dtype=np.float64)
        for r in range(self.N_domains-1):
            dw_dalpha_s[r][:] = self.eta_m[r] * self.R * self.T * self.a_s[r][:] / alpha_s[r][:]**2
            dw_dalpha_d[r][:] = self.eta_m[r] * self.R * self.T * self.a_d[r][:] / alpha_d[r][:]**2
            
        # derivatives - intracellular ion concentrations 
        dwn_dci_s = np.zeros((self.N_ions, self.N_units), dtype=np.float64)
        dwn_dci_d = np.zeros((self.N_ions, self.N_units), dtype=np.float64)
        dwg_dci_s = np.zeros((self.N_ions, self.N_units), dtype=np.float64)
        dwg_dci_d = np.zeros((self.N_ions, self.N_units), dtype=np.float64)
        dw_dci_s = [dwn_dci_s, dwg_dci_s]
        dw_dci_d = [dwn_dci_d, dwg_dci_d]
        for r in range(self.N_domains-1):
            for k in range(self.N_ions):
                dw_dci_s[r][k][:] = - self.eta_m[r] * self.R * self.T * self.upsilon[k][r]
                dw_dci_d[r][k][:] = - self.eta_m[r] * self.R * self.T * self.upsilon[k][r]
        
        # derivatives - extracellular ion concentrations 
        dwn_dce_s = np.zeros((self.N_ions, self.N_units), dtype=np.float64)
        dwn_dce_d = np.zeros((self.N_ions, self.N_units), dtype=np.float64)
        dwg_dce_s = np.zeros((self.N_ions, self.N_units), dtype=np.float64)
        dwg_dce_d = np.zeros((self.N_ions, self.N_units), dtype=np.float64)
        dw_dce_s = [dwn_dce_s, dwg_dce_s]
        dw_dce_d = [dwn_dce_d, dwg_dce_d]
        for r in range(self.N_domains-1):
            for k in range(self.N_ions):
                dw_dce_s[r][k][:] = self.eta_m[r] * self.R * self.T * self.upsilon[k][-1]
                dw_dce_d[r][k][:] = self.eta_m[r] * self.R * self.T * self.upsilon[k][-1]
                
        return w_m_s, w_m_d, dw_dalpha_s, dw_dalpha_d, dw_dci_s, dw_dci_d, dw_dce_s, dw_dce_d

    def alpha_m(self, phi_m):
        phi_1 = phi_m + 46.9
        alpha = -3.2e2 * phi_1 / (np.exp(-phi_1 / 4.0) - 1.0)
        return alpha

    def beta_m(self, phi_m):
        phi_2 = phi_m + 19.9
        beta = 2.8e2 * phi_2 / (np.exp(phi_2 / 5.0) - 1.0)
        return beta

    def alpha_h(self, phi_m):
        alpha = 128.0 * np.exp((-43.0 - phi_m) / 18.0)
        return alpha

    def beta_h(self, phi_m):
        phi_3 = phi_m + 20.0
        beta = 4000.0 / (1.0 + np.exp(-phi_3 / 5.0))
        return beta

    def alpha_n(self, phi_m):
        phi_4 = phi_m + 24.9
        alpha = - 16.0 * phi_4 / (np.exp(-phi_4 / 5.0) - 1.0)
        return alpha

    def beta_n(self, phi_m):
        phi_5 = phi_m + 40.0
        beta = 250.0 * np.exp(-phi_5 / 40.0)
        return beta

    def alpha_s(self, phi_m):
        alpha = 1600.0 / (1.0 + np.exp(-0.072 * (phi_m - 5.0)))
        return alpha

    def beta_s(self, phi_m):
        phi_6 = phi_m + 8.9
        beta = 20.0 * phi_6 / (np.exp(phi_6 / 5.0) - 1.0)
        return beta

    def alpha_c(self, phi_m):
        phi_8 = phi_m + 50.0
        phi_9 = phi_m + 53.5
        alpha = np.where(phi_m <= -10.0, 52.7 * np.exp(phi_8/11.0 - phi_9/27.0), 2000.0 * np.exp(-phi_9 / 27.0))
        return alpha

    def beta_c(self, phi_m):
        phi_9 = phi_m + 53.5
        beta = np.where(phi_m <= -10.0, 2000.0 * np.exp(-phi_9 / 27.0) - self.alpha_c(phi_m), 0.0)
        return beta

    def chi(self, Ca_n):
        return np.minimum((self.upsilon[-1][0]*Ca_n-99.8e-6)/2.5e-4, 1.0)

    def alpha_q(self, Ca_n):
        return np.minimum(2e4*(self.upsilon[-1][0]*Ca_n-99.8e-6), 10.0) 

    def beta_q(self):
        return 1.0

    def m_inf(self, phi_m):
        return self.alpha_m(phi_m) / (self.alpha_m(phi_m) + self.beta_m(phi_m))

    def z_inf(self, phi_m):
        phi_7 = phi_m + 30.0
        return 1.0/(1.0 + np.exp(phi_7))

    def prepare_solver(self):
      
        # get parameters
        N_units = self.N_units
        N_domains = self.N_domains
        N_layers = self.N_layers
        N_ions = self.N_ions

        # number of unknowns to be solved at each unit
        self.N_unknowns_unit = (N_ions + 2)*N_domains*N_layers - 2
        # total number of unknowns
        self.N_unknowns_tot = self.N_unknowns_unit*N_units

        # allocate arrays needed to build sparse matrix
        N_A = (N_domains-1)*N_units*N_layers*(1 + 2*N_ions) + \
              N_ions*N_domains*N_units*N_layers*4 + \
              N_ions*(N_domains-1)*N_units*N_layers*17 + \
              N_domains*N_ions*N_units*N_layers + \
              (N_domains-1)*N_units*N_layers*(4 + N_ions) + 1
        self.irow = np.zeros(N_A, dtype=np.int32) 
        self.icol = np.zeros(N_A, dtype=np.int32)
        self.Avals = np.zeros(N_A, dtype=np.float64)

        # specify ordering of variables
        self.jc_s = np.empty((N_ions, N_domains), dtype=np.int32)
        self.jc_d = np.empty((N_ions, N_domains), dtype=np.int32)
        for k in np.arange(N_ions):
            self.jc_s[k,:] = k + np.arange(N_domains)*(N_ions+2)
            self.jc_d[k,:] = self.jc_s[k,:] + (N_ions+2)*N_domains - 1

        self.jphi_s = np.empty(N_domains, dtype=np.int32)
        self.jphi_d = np.empty(N_domains, dtype=np.int32)
        self.jphi_s[:] = N_ions + 1 + np.arange(N_domains)*(N_ions+2) - 1
        self.jphi_d[:] = self.jphi_s + (N_ions+2)*N_domains - 1

        self.jalpha_s = np.empty(N_domains-1, dtype=np.int32)
        self.jalpha_d = np.empty(N_domains-1, dtype=np.int32)
        self.jalpha_s[:] = np.arange(1,N_domains)*(N_ions+2) - 1
        self.jalpha_d[:] = self.jalpha_s[:] + (N_ions+2)*N_domains - 1

        return 

    def residual(self, dt, alpha_s, alpha_d, c_s, c_d, phi_s, phi_d, alpha_s_, alpha_d_, c_s_, c_d_, phi_s_, phi_d_, ss_, t, t_AP, j_stim, N_stim, spike_train):
        """ Calculate residual for Newton's method. """

        # get parameters, solver
        N_unknowns_unit = self.N_unknowns_unit
        N_unknowns_tot = self.N_unknowns_tot
        jc_s = self.jc_s
        jc_d = self.jc_d
        jphi_s = self.jphi_s
        jphi_d = self.jphi_d
        jalpha_s = self.jalpha_s
        jalpha_d = self.jalpha_d
        
        # get parameters, model 
        N_domains = self.N_domains
        N_ions = self.N_ions
        gamma_m = self.gamma_m
        C_m = self.C_m
        dxl = self.dxl
        dxu = self.dxu
        a_s = self.a_s
        a_d = self.a_d
        z = self.z
        F = self.F

        ### compute residual ###
        Res = np.zeros(N_unknowns_tot, dtype=np.float64)

        # calculate transmembrane fluxes
        j_m_s, djm_dci_s, djm_dce_s, djm_dphii_s, djm_dphie_s = self.set_membrane_fluxes_s(alpha_s, c_s, phi_s, alpha_s_, c_s_, phi_s_, ss_)
        j_m_d, djm_dci_d, djm_dce_d, djm_dphii_d, djm_dphie_d = self.set_membrane_fluxes_d(alpha_d, c_d, phi_d, alpha_d_, c_d_, phi_d_, ss_, t, t_AP, spike_train)

        # calculate transmembrane water flow
        w_m_s, w_m_d, dw_dalpha_s, dw_dalpha_d, dw_dci_s, dw_dci_d, dw_dce_s, dw_dce_d = self.set_water_flow(alpha_s, alpha_d, c_s, c_d)

        ## residual for volume fractions ##
        for r in range(N_domains-1):
            Res[jalpha_s[r]:N_unknowns_tot:N_unknowns_unit] += - alpha_s[r] + alpha_s_[r] - dt*gamma_m*w_m_s[r]
            Res[jalpha_d[r]:N_unknowns_tot:N_unknowns_unit] += - alpha_d[r] + alpha_d_[r] - dt*gamma_m*w_m_d[r]

        ## residual for ion concentrations ##
        for r in range(N_domains):
            for k in range(N_ions):
                
                j_interlayer, djl_dcs, djl_dcd, djl_dphis, djl_dphid = self.interlayer_flux(k, r, c_s, c_d, phi_s, phi_d)
                f_j_interlayer = dt*alpha_s_[r]/(dxl*gamma_m)*j_interlayer

                Res[jc_s[k][r]:N_unknowns_tot:N_unknowns_unit] += (- alpha_s[r]*c_s[k][r] + alpha_s_[r]*c_s_[k][r])/gamma_m - f_j_interlayer
                Res[jc_d[k][r]:N_unknowns_tot:N_unknowns_unit] += (- alpha_d[r]*c_d[k][r] + alpha_d_[r]*c_d_[k][r])/gamma_m + f_j_interlayer

        # contributions from membrane currents
        for r in range(N_domains-1):
            for k in range(N_ions):
                f_jm_s = dt*j_m_s[k][r]
                f_jm_d = dt*j_m_d[k][r]

                Res[jc_s[k][r]:N_unknowns_tot:N_unknowns_unit] -= f_jm_s
                Res[jc_s[k][-1]:N_unknowns_tot:N_unknowns_unit] += f_jm_s
                Res[jc_d[k][r]:N_unknowns_tot:N_unknowns_unit] -= f_jm_d
                Res[jc_d[k][-1]:N_unknowns_tot:N_unknowns_unit] += f_jm_d

        # contributions from interunit fluxes
        if self.bc == 1:
            for r in range(1,N_domains):
                for k in range(N_ions):
                    # soma layer
                    j_interunits_s, dju_dc0_s, dju_dc1_s, dju_dphi0_s, dju_dphi1_s = self.interunits_flux(k, r, c_s, phi_s)
                    Res[jc_s[k][r]] += dt*alpha_s_[r][-1] / (dxu*gamma_m) * j_interunits_s[-1]
                    Res[jc_s[k][r]+N_unknowns_unit:N_unknowns_tot:N_unknowns_unit] += dt*alpha_s_[r][:-1] / (dxu*gamma_m) * j_interunits_s[:-1]
                    Res[jc_s[k][r]:N_unknowns_tot:N_unknowns_unit] -= dt*alpha_s_[r] / (dxu*gamma_m) * j_interunits_s
                    # dendrite layer
                    j_interunits_d, dju_dc0_d, dju_dc1_d, dju_dphi0_d, dju_dphi1_d = self.interunits_flux(k, r, c_d, phi_d)
                    Res[jc_d[k][r]] += dt*alpha_d_[r][-1] / (dxu*gamma_m) * j_interunits_d[-1]
                    Res[jc_d[k][r]+N_unknowns_unit:N_unknowns_tot:N_unknowns_unit] += dt*alpha_d_[r][:-1] / (dxu*gamma_m) * j_interunits_d[:-1]
                    Res[jc_d[k][r]:N_unknowns_tot:N_unknowns_unit] -= dt*alpha_d_[r] / (dxu*gamma_m) * j_interunits_d
        else:
            for r in range(1,N_domains):
                for k in range(N_ions):
                    # soma layer
                    j_interunits_s, dju_dc0_s, dju_dc1_s, dju_dphi0_s, dju_dphi1_s = self.interunits_flux(k, r, c_s, phi_s)
                    Res[jc_s[k][r]+N_unknowns_unit:N_unknowns_tot:N_unknowns_unit] += dt*alpha_s_[r][:-1] / (dxu*gamma_m) * j_interunits_s[:-1]
                    Res[jc_s[k][r]:N_unknowns_tot-N_unknowns_unit:N_unknowns_unit] -= dt*alpha_s_[r][:-1] / (dxu*gamma_m) * j_interunits_s[:-1]
                    # dendrite layer
                    j_interunits_d, dju_dc0_d, dju_dc1_d, dju_dphi0_d, dju_dphi1_d = self.interunits_flux(k, r, c_d, phi_d)
                    Res[jc_d[k][r]+N_unknowns_unit:N_unknowns_tot:N_unknowns_unit] += dt*alpha_d_[r][:-1] / (dxu*gamma_m) * j_interunits_d[:-1]
                    Res[jc_d[k][r]:N_unknowns_tot-N_unknowns_unit:N_unknowns_unit] -= dt*alpha_d_[r][:-1] / (dxu*gamma_m) * j_interunits_d[:-1]

        # stimulate
        Res[jc_s[1][0]:N_unknowns_unit*N_stim:N_unknowns_unit] += dt*j_stim 
        Res[jc_s[1][-1]:N_unknowns_unit*N_stim:N_unknowns_unit] -= dt*j_stim 

        # electroneutrality conditons
        for r in range(N_domains-1): # n, g

            # add contribution from ICS immobile ions to electroneutrality condition
            Res[jphi_s[r]:N_unknowns_tot:N_unknowns_unit] += z[-1]*a_s[r]
            Res[jphi_d[r]:N_unknowns_tot:N_unknowns_unit] += z[-1]*a_d[r]

            # add contribution from phi_m to ICS electroneutrality condition
            Res[jphi_s[r]:N_unknowns_tot:N_unknowns_unit] += - gamma_m*C_m*(phi_s[r] - phi_s[-1])/F
            Res[jphi_d[r]:N_unknowns_tot:N_unknowns_unit] += - gamma_m*C_m*(phi_d[r] - phi_d[-1])/F

            # add contribution from ICS ions to electroneutrality condition 
            for k in range(N_ions):
                Res[jphi_s[r]:N_unknowns_tot:N_unknowns_unit] += alpha_s[r]*z[k]*c_s[k][r]
                Res[jphi_d[r]:N_unknowns_tot:N_unknowns_unit] += alpha_d[r]*z[k]*c_d[k][r]
        
        # contribution from phi to ECS electroneutrality conditions
        Res[jphi_s[-1]:N_unknowns_tot:N_unknowns_unit] += gamma_m*C_m*(phi_s[0] + phi_s[1] - 2*phi_s[-1])/F
        Res[jphi_d[-1]:N_unknowns_tot:N_unknowns_unit] += gamma_m*C_m*(phi_d[0] + phi_d[1] - 2*phi_d[-1])/F
        # contribution from immobile ions to ECS electroneutrality conditions
        Res[jphi_s[-1]:N_unknowns_tot:N_unknowns_unit] += z[-1]*a_s[-1]
        Res[jphi_d[-1]:N_unknowns_tot:N_unknowns_unit] += z[-1]*a_d[-1]

        # contributions from ions to ECS electroneutrality condition
        for k in range(N_ions):
            Res[jphi_s[-1]:N_unknowns_tot:N_unknowns_unit] += alpha_s[-1]*z[k]*c_s[k][-1]
            Res[jphi_d[-1]:N_unknowns_tot:N_unknowns_unit] += alpha_d[-1]*z[k]*c_d[k][-1]

        # set phi_de to 0
        Res[jphi_d[-1]] += -1*phi_d[-1][0]    

        return Res

    def jacobian(self, dt, alpha_s, alpha_d, c_s, c_d, phi_s, phi_d, alpha_s_, alpha_d_, c_s_, c_d_, phi_s_, phi_d_, ss_, t, t_AP, spike_train):
        """ Calculate Jacobian for Newton's method. """

        # get parameters, solver
        irow = self.irow
        icol = self.icol
        Avals = self.Avals
        N_unknowns_tot = self.N_unknowns_tot
        N_unknowns_unit = self.N_unknowns_unit
        jc_s = self.jc_s
        jc_d = self.jc_d
        jphi_s = self.jphi_s
        jphi_d = self.jphi_d
        jalpha_s = self.jalpha_s
        jalpha_d = self.jalpha_d
        
        # get parameters, model 
        N_units = self.N_units
        N_domains = self.N_domains
        N_ions = self.N_ions
        gamma_m = self.gamma_m
        C_m = self.C_m
        dxl = self.dxl
        dxu = self.dxu
        z = self.z
        F = self.F

        # calculate transmembrane fluxes
        j_m_s, djm_dci_s, djm_dce_s, djm_dphii_s, djm_dphie_s = self.set_membrane_fluxes_s(alpha_s, c_s, phi_s, alpha_s_, c_s_, phi_s_, ss_)
        j_m_d, djm_dci_d, djm_dce_d, djm_dphii_d, djm_dphie_d = self.set_membrane_fluxes_d(alpha_d, c_d, phi_d, alpha_d_, c_d_, phi_d_, ss_, t, t_AP, spike_train)

        # calculate transmembrane water flow
        w_m_s, w_m_d, dw_dalpha_s, dw_dalpha_d, dw_dci_s, dw_dci_d, dw_dce_s, dw_dce_d = self.set_water_flow(alpha_s, alpha_d, c_s, c_d)

        ### Compute Jacobian matrix ###
        ind = 0
        indend = ind+N_units

        ### volume fraction equations ###
        for r in range(N_domains-1):
            # alpha-alpha entries
            # soma layer
            irow[ind:indend] = np.arange(jalpha_s[r], N_unknowns_tot, N_unknowns_unit)
            icol[ind:indend] = np.arange(jalpha_s[r], N_unknowns_tot, N_unknowns_unit)
            Avals[ind:indend] = - 1 - dt * gamma_m * dw_dalpha_s[r]
            ind = indend
            indend = ind+N_units
            # dendrite layer
            irow[ind:indend] = np.arange(jalpha_d[r], N_unknowns_tot, N_unknowns_unit)
            icol[ind:indend] = np.arange(jalpha_d[r], N_unknowns_tot, N_unknowns_unit)
            Avals[ind:indend] = - 1 - dt * gamma_m * dw_dalpha_d[r]
            ind = indend
            indend = ind+N_units
            # alpha-concentration entries
            for k in range(N_ions):
                # soma layer
                irow[ind:indend] = np.arange(jalpha_s[r], N_unknowns_tot, N_unknowns_unit)
                icol[ind:indend] = np.arange(jc_s[k][r], N_unknowns_tot, N_unknowns_unit)
                Avals[ind:indend] = - dt * dw_dci_s[r][k] * gamma_m
                ind = indend
                indend = ind+N_units
                irow[ind:indend] = np.arange(jalpha_s[r], N_unknowns_tot, N_unknowns_unit)
                icol[ind:indend] = np.arange(jc_s[k][-1], N_unknowns_tot, N_unknowns_unit)
                Avals[ind:indend] = - dt * dw_dce_s[r][k] * gamma_m
                ind = indend
                indend = ind+N_units
                # dendrite layer
                irow[ind:indend] = np.arange(jalpha_d[r], N_unknowns_tot, N_unknowns_unit)
                icol[ind:indend] = np.arange(jc_d[k][r], N_unknowns_tot, N_unknowns_unit)
                Avals[ind:indend] = - dt * dw_dci_d[r][k] * gamma_m
                ind = indend
                indend = ind+N_units
                irow[ind:indend] = np.arange(jalpha_d[r], N_unknowns_tot, N_unknowns_unit)
                icol[ind:indend] = np.arange(jc_d[k][-1], N_unknowns_tot, N_unknowns_unit)
                Avals[ind:indend] = - dt * dw_dce_d[r][k] * gamma_m
                ind = indend
                indend = ind+N_units

        ### ionic concentration equations ###
        for k in range(N_ions):
            # concentration-concentration and concentration-voltage entries
            for r in range(N_domains):
                # contributions from j_interlayer
                j_interlayer, djl_dcs, djl_dcd, djl_dphis, djl_dphid =  self.interlayer_flux(k, r, c_s, c_d, phi_s, phi_d)
                f = dt * alpha_s_[r] / (dxl * gamma_m)
                # soma entries
                irow[ind:indend] = np.arange(jc_s[k][r], N_unknowns_tot, N_unknowns_unit)
                icol[ind:indend] = np.arange(jc_s[k][r], N_unknowns_tot, N_unknowns_unit)
                Avals[ind:indend] = - alpha_s[r]/gamma_m - f * djl_dcs
                ind = indend
                indend = ind+N_units
                irow[ind:indend] = np.arange(jc_s[k][r], N_unknowns_tot, N_unknowns_unit)
                icol[ind:indend] = np.arange(jc_d[k][r], N_unknowns_tot, N_unknowns_unit)
                Avals[ind:indend] = - f * djl_dcd
                ind = indend
                indend = ind+N_units
                irow[ind:indend] = np.arange(jc_s[k][r], N_unknowns_tot, N_unknowns_unit)
                icol[ind:indend] = np.arange(jphi_s[r], N_unknowns_tot, N_unknowns_unit)
                Avals[ind:indend] = - f * djl_dphis
                ind = indend
                indend = ind+N_units
                irow[ind:indend] = np.arange(jc_s[k][r], N_unknowns_tot, N_unknowns_unit) 
                icol[ind:indend] = np.arange(jphi_d[r], N_unknowns_tot, N_unknowns_unit)
                Avals[ind:indend] = - f * djl_dphid
                ind = indend
                indend = ind+N_units
                # dendrite entries
                irow[ind:indend] = np.arange(jc_d[k][r], N_unknowns_tot, N_unknowns_unit) 
                icol[ind:indend] = np.arange(jc_s[k][r], N_unknowns_tot, N_unknowns_unit)
                Avals[ind:indend] = f * djl_dcs
                ind = indend
                indend = ind+N_units
                irow[ind:indend] = np.arange(jc_d[k][r], N_unknowns_tot, N_unknowns_unit)
                icol[ind:indend] = np.arange(jc_d[k][r], N_unknowns_tot, N_unknowns_unit)
                Avals[ind:indend] = - alpha_d[r]/gamma_m + f * djl_dcd
                ind = indend
                indend = ind+N_units
                irow[ind:indend] = np.arange(jc_d[k][r], N_unknowns_tot, N_unknowns_unit) 
                icol[ind:indend] = np.arange(jphi_s[r], N_unknowns_tot, N_unknowns_unit)
                Avals[ind:indend] = f * djl_dphis
                ind = indend
                indend = ind+N_units
                irow[ind:indend] = np.arange(jc_d[k][r], N_unknowns_tot, N_unknowns_unit) 
                icol[ind:indend] = np.arange(jphi_d[r], N_unknowns_tot, N_unknowns_unit)
                Avals[ind:indend] = f * djl_dphid
                ind = indend
                indend = ind+N_units

            # contribution from j_interunits
            if self.bc == 1:
                for r in range(1, N_domains):
                    j_interunits_s, dju_dc0_s, dju_dc1_s, dju_dphi0_s, dju_dphi1_s = self.interunits_flux(k, r, c_s, phi_s)
                    f = dt / (dxu * gamma_m)
                    # soma layer
                    # concentration-concentration entries
                    irow[ind] = jc_s[k][r] 
                    icol[ind] = N_unknowns_tot-N_unknowns_unit+jc_s[k][r]
                    Avals[ind] = f * alpha_s_[r][-1] * dju_dc0_s[-1]
                    irow[ind+1:indend] = np.arange(jc_s[k][r]+N_unknowns_unit, N_unknowns_tot, N_unknowns_unit) 
                    icol[ind+1:indend] = np.arange(jc_s[k][r], N_unknowns_tot-N_unknowns_unit, N_unknowns_unit) 
                    Avals[ind+1:indend] = f * alpha_s_[r][:-1] * dju_dc0_s[:-1]
                    ind = indend
                    indend = ind+N_units
                    irow[ind] = jc_s[k][r] 
                    icol[ind] = jc_s[k][r]
                    Avals[ind] = f * alpha_s_[r][-1] * dju_dc1_s[-1]
                    irow[ind+1:indend] = np.arange(jc_s[k][r]+N_unknowns_unit, N_unknowns_tot, N_unknowns_unit) 
                    icol[ind+1:indend] = np.arange(jc_s[k][r]+N_unknowns_unit, N_unknowns_tot, N_unknowns_unit) 
                    Avals[ind+1:indend] = f * alpha_s_[r][:-1] * dju_dc1_s[:-1]
                    ind = indend
                    indend = ind+N_units
                    irow[ind:indend] = np.arange(jc_s[k][r], N_unknowns_tot, N_unknowns_unit) 
                    icol[ind:indend] = np.arange(jc_s[k][r], N_unknowns_tot, N_unknowns_unit) 
                    Avals[ind:indend] = - f * alpha_s_[r] * dju_dc0_s
                    ind = indend
                    indend = ind+N_units
                    irow[ind] = N_unknowns_tot-N_unknowns_unit+jc_s[k][r]
                    icol[ind] = jc_s[k][r]
                    Avals[ind] = - f * alpha_s_[r][-1] * dju_dc1_s[-1]
                    irow[ind+1:indend] = np.arange(jc_s[k][r], N_unknowns_tot-N_unknowns_unit, N_unknowns_unit) 
                    icol[ind+1:indend] = np.arange(jc_s[k][r]+N_unknowns_unit, N_unknowns_tot, N_unknowns_unit)
                    Avals[ind+1:indend] = - f * alpha_s_[r][:-1] * dju_dc1_s[:-1]
                    ind = indend
                    indend = ind+N_units
                    # concentration-voltage entries
                    irow[ind] = jc_s[k][r] 
                    icol[ind] = N_unknowns_tot-N_unknowns_unit+jphi_s[r]
                    Avals[ind] = f * alpha_s_[r][-1] * dju_dphi0_s[-1]
                    irow[ind+1:indend] = np.arange(jc_s[k][r]+N_unknowns_unit, N_unknowns_tot, N_unknowns_unit) 
                    icol[ind+1:indend] = np.arange(jphi_s[r], N_unknowns_tot-N_unknowns_unit, N_unknowns_unit)
                    Avals[ind+1:indend] = f * alpha_s_[r][:-1] * dju_dphi0_s[:-1]
                    ind = indend
                    indend = ind+N_units
                    irow[ind] = jc_s[k][r] 
                    icol[ind] = jphi_s[r]
                    Avals[ind] = f * alpha_s_[r][-1] * dju_dphi1_s[-1]
                    irow[ind+1:indend] = np.arange(jc_s[k][r]+N_unknowns_unit, N_unknowns_tot, N_unknowns_unit) 
                    icol[ind+1:indend] = np.arange(jphi_s[r]+N_unknowns_unit, N_unknowns_tot, N_unknowns_unit) 
                    Avals[ind+1:indend] = f * alpha_s_[r][:-1] * dju_dphi1_s[:-1]
                    ind = indend
                    indend = ind+N_units
                    irow[ind:indend] = np.arange(jc_s[k][r], N_unknowns_tot, N_unknowns_unit) 
                    icol[ind:indend] = np.arange(jphi_s[r], N_unknowns_tot, N_unknowns_unit) 
                    Avals[ind:indend] = - f * alpha_s_[r] * dju_dphi0_s
                    ind = indend
                    indend = ind+N_units
                    irow[ind] = N_unknowns_tot-N_unknowns_unit+jc_s[k][r] 
                    icol[ind] = jphi_s[r]
                    Avals[ind] = - f * alpha_s_[r][-1] * dju_dphi1_s[-1]
                    irow[ind+1:indend] = np.arange(jc_s[k][r], N_unknowns_tot-N_unknowns_unit, N_unknowns_unit) 
                    icol[ind+1:indend] = np.arange(jphi_s[r]+N_unknowns_unit, N_unknowns_tot, N_unknowns_unit)
                    Avals[ind+1:indend] = - f * alpha_s_[r][:-1] * dju_dphi1_s[:-1]
                    ind = indend
                    indend = ind+N_units
                    # dendrite layer
                    # concentration-concentration entries
                    j_interunits_d, dju_dc0_d, dju_dc1_d, dju_dphi0_d, dju_dphi1_d = self.interunits_flux(k, r, c_d, phi_d)
                    irow[ind] = jc_d[k][r] 
                    icol[ind] = N_unknowns_tot-N_unknowns_unit+jc_d[k][r]
                    Avals[ind] = f * alpha_d_[r][-1] * dju_dc0_d[-1]
                    irow[ind+1:indend] = np.arange(jc_d[k][r]+N_unknowns_unit, N_unknowns_tot, N_unknowns_unit) 
                    icol[ind+1:indend] = np.arange(jc_d[k][r], N_unknowns_tot-N_unknowns_unit, N_unknowns_unit)
                    Avals[ind+1:indend] = f * alpha_d_[r][:-1] * dju_dc0_d[-1]
                    ind = indend
                    indend = ind+N_units
                    irow[ind] = jc_d[k][r]
                    icol[ind] = jc_d[k][r]
                    Avals[ind] = f * alpha_d_[r][-1] * dju_dc1_d[-1]
                    irow[ind+1:indend] = np.arange(jc_d[k][r]+N_unknowns_unit, N_unknowns_tot, N_unknowns_unit) 
                    icol[ind+1:indend] = np.arange(jc_d[k][r]+N_unknowns_unit, N_unknowns_tot, N_unknowns_unit) 
                    Avals[ind+1:indend] = f * alpha_d_[r][:-1] * dju_dc1_d[:-1]
                    ind = indend
                    indend = ind+N_units
                    irow[ind:indend] = np.arange(jc_d[k][r], N_unknowns_tot, N_unknowns_unit) 
                    icol[ind:indend] = np.arange(jc_d[k][r], N_unknowns_tot, N_unknowns_unit) 
                    Avals[ind:indend] = - f * alpha_d_[r] * dju_dc0_d
                    ind = indend
                    indend = ind+N_units
                    irow[ind] = N_unknowns_tot-N_unknowns_unit+jc_d[k][r]
                    icol[ind] = jc_d[k][r]
                    Avals[ind] = - f * alpha_d_[r][-1] * dju_dc1_d[-1]
                    irow[ind+1:indend] = np.arange(jc_d[k][r], N_unknowns_tot-N_unknowns_unit, N_unknowns_unit) 
                    icol[ind+1:indend] = np.arange(jc_d[k][r]+N_unknowns_unit, N_unknowns_tot, N_unknowns_unit)
                    Avals[ind+1:indend] = - f * alpha_d_[r][:-1] * dju_dc1_d[:-1]
                    ind = indend
                    indend = ind+N_units
                    # concentration-voltage entries
                    irow[ind] = jc_d[k][r] 
                    icol[ind] = N_unknowns_tot-N_unknowns_unit+jphi_d[r]
                    Avals[ind] = f * alpha_d_[r][-1] * dju_dphi0_d[-1]
                    irow[ind+1:indend] = np.arange(jc_d[k][r]+N_unknowns_unit, N_unknowns_tot, N_unknowns_unit) 
                    icol[ind+1:indend] = np.arange(jphi_d[r], N_unknowns_tot-N_unknowns_unit, N_unknowns_unit)
                    Avals[ind+1:indend] = f * alpha_d_[r][:-1] * dju_dphi0_d[:-1]
                    ind = indend
                    indend = ind+N_units
                    irow[ind] = jc_d[k][r] 
                    icol[ind] = jphi_d[r]
                    Avals[ind] = f * alpha_d_[r][-1] * dju_dphi1_d[-1]
                    irow[ind+1:indend] = np.arange(jc_d[k][r]+N_unknowns_unit, N_unknowns_tot, N_unknowns_unit) 
                    icol[ind+1:indend] = np.arange(jphi_d[r]+N_unknowns_unit, N_unknowns_tot, N_unknowns_unit) 
                    Avals[ind+1:indend] = f * alpha_d_[r][:-1] * dju_dphi1_d[:-1]
                    ind = indend
                    indend = ind+N_units
                    irow[ind:indend] = np.arange(jc_d[k][r], N_unknowns_tot, N_unknowns_unit) 
                    icol[ind:indend] = np.arange(jphi_d[r], N_unknowns_tot, N_unknowns_unit) 
                    Avals[ind:indend] = - f * alpha_d_[r] * dju_dphi0_d
                    ind = indend
                    indend = ind+N_units
                    irow[ind] = N_unknowns_tot-N_unknowns_unit+jc_d[k][r] 
                    icol[ind] = jphi_d[r]
                    Avals[ind] = - f * alpha_d_[r][-1] * dju_dphi1_d[-1]
                    irow[ind+1:indend] = np.arange(jc_d[k][r], N_unknowns_tot-N_unknowns_unit, N_unknowns_unit) 
                    icol[ind+1:indend] = np.arange(jphi_d[r]+N_unknowns_unit, N_unknowns_tot, N_unknowns_unit)
                    Avals[ind+1:indend] = - f * alpha_d_[r][:-1] * dju_dphi1_d[:-1]
                    ind = indend
                    indend = ind+N_units
            else:
                indend -= 1
                for r in range(1, N_domains):
                    j_interunits_s, dju_dc0_s, dju_dc1_s, dju_dphi0_s, dju_dphi1_s = self.interunits_flux(k, r, c_s, phi_s)
                    f = dt / (dxu * gamma_m)
                    # soma layer
                    # concentration-concentration entries
                    irow[ind:indend] = np.arange(jc_s[k][r]+N_unknowns_unit, N_unknowns_tot, N_unknowns_unit) 
                    icol[ind:indend] = np.arange(jc_s[k][r], N_unknowns_tot-N_unknowns_unit, N_unknowns_unit) 
                    Avals[ind:indend] = f * alpha_s_[r][:-1] * dju_dc0_s[:-1]
                    ind = indend
                    indend = ind+N_units-1
                    irow[ind:indend] = np.arange(jc_s[k][r]+N_unknowns_unit, N_unknowns_tot, N_unknowns_unit) 
                    icol[ind:indend] = np.arange(jc_s[k][r]+N_unknowns_unit, N_unknowns_tot, N_unknowns_unit) 
                    Avals[ind:indend] = f * alpha_s_[r][:-1] * dju_dc1_s[:-1]
                    ind = indend
                    indend = ind+N_units-1
                    irow[ind:indend] = np.arange(jc_s[k][r], N_unknowns_tot-N_unknowns_unit, N_unknowns_unit) 
                    icol[ind:indend] = np.arange(jc_s[k][r], N_unknowns_tot-N_unknowns_unit, N_unknowns_unit) 
                    Avals[ind:indend] = - f * alpha_s_[r][:-1] * dju_dc0_s[:-1]
                    ind = indend
                    indend = ind+N_units-1
                    irow[ind:indend] = np.arange(jc_s[k][r], N_unknowns_tot-N_unknowns_unit, N_unknowns_unit) 
                    icol[ind:indend] = np.arange(jc_s[k][r]+N_unknowns_unit, N_unknowns_tot, N_unknowns_unit)
                    Avals[ind:indend] = - f * alpha_s_[r][:-1] * dju_dc1_s[:-1]
                    ind = indend
                    indend = ind+N_units-1
                    # concentration-voltage entries
                    irow[ind:indend] = np.arange(jc_s[k][r]+N_unknowns_unit, N_unknowns_tot, N_unknowns_unit) 
                    icol[ind:indend] = np.arange(jphi_s[r], N_unknowns_tot-N_unknowns_unit, N_unknowns_unit)
                    Avals[ind:indend] = f * alpha_s_[r][:-1] * dju_dphi0_s[:-1]
                    ind = indend
                    indend = ind+N_units-1
                    irow[ind:indend] = np.arange(jc_s[k][r]+N_unknowns_unit, N_unknowns_tot, N_unknowns_unit) 
                    icol[ind:indend] = np.arange(jphi_s[r]+N_unknowns_unit, N_unknowns_tot, N_unknowns_unit) 
                    Avals[ind:indend] = f * alpha_s_[r][:-1] * dju_dphi1_s[:-1]
                    ind = indend
                    indend = ind+N_units-1
                    irow[ind:indend] = np.arange(jc_s[k][r], N_unknowns_tot-N_unknowns_unit, N_unknowns_unit) 
                    icol[ind:indend] = np.arange(jphi_s[r], N_unknowns_tot-N_unknowns_unit, N_unknowns_unit) 
                    Avals[ind:indend] = - f * alpha_s_[r][:-1] * dju_dphi0_s[:-1]
                    ind = indend
                    indend = ind+N_units-1
                    irow[ind:indend] = np.arange(jc_s[k][r], N_unknowns_tot-N_unknowns_unit, N_unknowns_unit) 
                    icol[ind:indend] = np.arange(jphi_s[r]+N_unknowns_unit, N_unknowns_tot, N_unknowns_unit)
                    Avals[ind:indend] = - f * alpha_s_[r][:-1] * dju_dphi1_s[:-1]
                    ind = indend
                    indend = ind+N_units-1
                    # dendrite layer
                    j_interunits_d, dju_dc0_d, dju_dc1_d, dju_dphi0_d, dju_dphi1_d = self.interunits_flux(k, r, c_d, phi_d)
                    # concentration-concentration entries
                    irow[ind:indend] = np.arange(jc_d[k][r]+N_unknowns_unit, N_unknowns_tot, N_unknowns_unit) 
                    icol[ind:indend] = np.arange(jc_d[k][r], N_unknowns_tot-N_unknowns_unit, N_unknowns_unit) 
                    Avals[ind:indend] = f * alpha_d_[r][:-1] * dju_dc0_d[:-1]
                    ind = indend
                    indend = ind+N_units-1
                    irow[ind:indend] = np.arange(jc_d[k][r]+N_unknowns_unit, N_unknowns_tot, N_unknowns_unit) 
                    icol[ind:indend] = np.arange(jc_d[k][r]+N_unknowns_unit, N_unknowns_tot, N_unknowns_unit) 
                    Avals[ind:indend] = f * alpha_d_[r][:-1] * dju_dc1_d[:-1]
                    ind = indend
                    indend = ind+N_units-1
                    irow[ind:indend] = np.arange(jc_d[k][r], N_unknowns_tot-N_unknowns_unit, N_unknowns_unit) 
                    icol[ind:indend] = np.arange(jc_d[k][r], N_unknowns_tot-N_unknowns_unit, N_unknowns_unit) 
                    Avals[ind:indend] = - f * alpha_d_[r][:-1] * dju_dc0_d[:-1]
                    ind = indend
                    indend = ind+N_units-1
                    irow[ind:indend] = np.arange(jc_d[k][r], N_unknowns_tot-N_unknowns_unit, N_unknowns_unit) 
                    icol[ind:indend] = np.arange(jc_d[k][r]+N_unknowns_unit, N_unknowns_tot, N_unknowns_unit)
                    Avals[ind:indend] = - f * alpha_d_[r][:-1] * dju_dc1_d[:-1]
                    ind = indend
                    indend = ind+N_units-1
                    # concentration-voltage entries
                    irow[ind:indend] = np.arange(jc_d[k][r]+N_unknowns_unit, N_unknowns_tot, N_unknowns_unit) 
                    icol[ind:indend] = np.arange(jphi_d[r], N_unknowns_tot-N_unknowns_unit, N_unknowns_unit)
                    Avals[ind:indend] = f * alpha_d_[r][:-1] * dju_dphi0_d[:-1]
                    ind = indend
                    indend = ind+N_units-1
                    irow[ind:indend] = np.arange(jc_d[k][r]+N_unknowns_unit, N_unknowns_tot, N_unknowns_unit) 
                    icol[ind:indend] = np.arange(jphi_d[r]+N_unknowns_unit, N_unknowns_tot, N_unknowns_unit) 
                    Avals[ind:indend] = f * alpha_d_[r][:-1] * dju_dphi1_d[:-1]
                    ind = indend
                    indend = ind+N_units-1
                    irow[ind:indend] = np.arange(jc_d[k][r], N_unknowns_tot-N_unknowns_unit, N_unknowns_unit) 
                    icol[ind:indend] = np.arange(jphi_d[r], N_unknowns_tot-N_unknowns_unit, N_unknowns_unit) 
                    Avals[ind:indend] = - f * alpha_d_[r][:-1] * dju_dphi0_d[:-1]
                    ind = indend
                    indend = ind+N_units-1
                    irow[ind:indend] = np.arange(jc_d[k][r], N_unknowns_tot-N_unknowns_unit, N_unknowns_unit) 
                    icol[ind:indend] = np.arange(jphi_d[r]+N_unknowns_unit, N_unknowns_tot, N_unknowns_unit)
                    Avals[ind:indend] = - f * alpha_d_[r][:-1] * dju_dphi1_d[:-1]
                    ind = indend
                    indend = ind+N_units-1
                indend += 1

            # contributions from membrane currents
            for r in range(N_domains-1):
                # intracellular concentrations - soma layer
                irow[ind:indend] = np.arange(jc_s[k][r], N_unknowns_tot, N_unknowns_unit)
                icol[ind:indend] = np.arange(jc_s[k][r], N_unknowns_tot, N_unknowns_unit)
                Avals[ind:indend] = - dt * djm_dci_s[k][r]
                ind = indend
                indend = ind+N_units
                irow[ind:indend] = np.arange(jc_s[k][r], N_unknowns_tot, N_unknowns_unit)
                icol[ind:indend] = np.arange(jc_s[k][-1], N_unknowns_tot, N_unknowns_unit) 
                Avals[ind:indend] = - dt * djm_dce_s[k][r]
                ind = indend
                indend = ind+N_units
                irow[ind:indend] = np.arange(jc_s[k][r], N_unknowns_tot, N_unknowns_unit)
                icol[ind:indend] = np.arange(jphi_s[r], N_unknowns_tot, N_unknowns_unit)
                Avals[ind:indend] = - dt * djm_dphii_s[k][r]
                ind = indend
                indend = ind+N_units
                irow[ind:indend] = np.arange(jc_s[k][r], N_unknowns_tot, N_unknowns_unit)
                icol[ind:indend] = np.arange(jphi_s[-1], N_unknowns_tot, N_unknowns_unit)
                Avals[ind:indend] = - dt * djm_dphie_s[k][r]
                ind = indend
                indend = ind+N_units
                # extracellular concentrations - soma layer
                irow[ind:indend] = np.arange(jc_s[k][-1], N_unknowns_tot, N_unknowns_unit)
                icol[ind:indend] = np.arange(jc_s[k][r], N_unknowns_tot, N_unknowns_unit)
                Avals[ind:indend] = dt * djm_dci_s[k][r]
                ind = indend
                indend = ind+N_units
                irow[ind:indend] = np.arange(jc_s[k][-1], N_unknowns_tot, N_unknowns_unit)
                icol[ind:indend] = np.arange(jc_s[k][-1], N_unknowns_tot, N_unknowns_unit) 
                Avals[ind:indend] = dt * djm_dce_s[k][r]
                ind = indend
                indend = ind+N_units
                irow[ind:indend] = np.arange(jc_s[k][-1], N_unknowns_tot, N_unknowns_unit)
                icol[ind:indend] = np.arange(jphi_s[r], N_unknowns_tot, N_unknowns_unit)
                Avals[ind:indend] = dt * djm_dphii_s[k][r]
                ind = indend
                indend = ind+N_units
                irow[ind:indend] = np.arange(jc_s[k][-1], N_unknowns_tot, N_unknowns_unit)
                icol[ind:indend] = np.arange(jphi_s[-1], N_unknowns_tot, N_unknowns_unit)
                Avals[ind:indend] = dt * djm_dphie_s[k][r]
                ind = indend
                indend = ind+N_units
                # intracellular concentrations - dendrite layer
                irow[ind:indend] = np.arange(jc_d[k][r], N_unknowns_tot, N_unknowns_unit)
                icol[ind:indend] = np.arange(jc_d[k][r], N_unknowns_tot, N_unknowns_unit)
                Avals[ind:indend] = - dt * djm_dci_d[k][r]
                ind = indend
                indend = ind+N_units
                irow[ind:indend] = np.arange(jc_d[k][r], N_unknowns_tot, N_unknowns_unit)
                icol[ind:indend] = np.arange(jc_d[k][-1], N_unknowns_tot, N_unknowns_unit)
                Avals[ind:indend] = - dt * djm_dce_d[k][r]
                ind = indend
                indend = ind+N_units
                irow[ind:indend] = np.arange(jc_d[k][r], N_unknowns_tot, N_unknowns_unit)
                icol[ind:indend] = np.arange(jphi_d[r], N_unknowns_tot, N_unknowns_unit) 
                Avals[ind:indend] = - dt * djm_dphii_d[k][r]
                ind = indend
                indend = ind+N_units
                irow[ind:indend] = np.arange(jc_d[k][r], N_unknowns_tot, N_unknowns_unit)
                icol[ind:indend] = np.arange(jphi_d[-1], N_unknowns_tot, N_unknowns_unit) 
                Avals[ind:indend] = - dt * djm_dphie_d[k][r]
                ind = indend
                indend = ind+N_units
                # extracellular concentrations - dendrite layer
                irow[ind:indend] = np.arange(jc_d[k][-1], N_unknowns_tot, N_unknowns_unit)
                icol[ind:indend] = np.arange(jc_d[k][r], N_unknowns_tot, N_unknowns_unit)
                Avals[ind:indend] = dt * djm_dci_d[k][r]
                ind = indend
                indend = ind+N_units
                irow[ind:indend] = np.arange(jc_d[k][-1], N_unknowns_tot, N_unknowns_unit)
                icol[ind:indend] = np.arange(jc_d[k][-1], N_unknowns_tot, N_unknowns_unit) 
                Avals[ind:indend] = dt * djm_dce_d[k][r]
                ind = indend
                indend = ind+N_units
                irow[ind:indend] = np.arange(jc_d[k][-1], N_unknowns_tot, N_unknowns_unit)
                icol[ind:indend] = np.arange(jphi_d[r], N_unknowns_tot, N_unknowns_unit)
                Avals[ind:indend] = dt * djm_dphii_d[k][r]
                ind = indend
                indend = ind+N_units
                irow[ind:indend] = np.arange(jc_d[k][-1], N_unknowns_tot, N_unknowns_unit)
                icol[ind:indend] = np.arange(jphi_d[-1], N_unknowns_tot, N_unknowns_unit) 
                Avals[ind:indend] = dt * djm_dphie_d[k][r]
                ind = indend
                indend = ind+N_units

        # concentration-alpha entries
        for k in range(N_ions):
            for r in range(N_domains-1):
                irow[ind:indend] = np.arange(jc_s[k][r], N_unknowns_tot, N_unknowns_unit)
                icol[ind:indend] = np.arange(jalpha_s[r], N_unknowns_tot, N_unknowns_unit)
                Avals[ind:indend] = - c_s[k][r]/gamma_m
                ind = indend
                indend = ind+N_units
                irow[ind:indend] = np.arange(jc_d[k][r], N_unknowns_tot, N_unknowns_unit)
                icol[ind:indend] = np.arange(jalpha_d[r], N_unknowns_tot, N_unknowns_unit)
                Avals[ind:indend] = - c_d[k][r]/gamma_m
                ind = indend
                indend = ind+N_units

        ### electrical potential equations ###
        # voltage-concentration entries
        for r in range(N_domains):
            for k in range(N_ions):
                # soma entries
                irow[ind:indend] = np.arange(jphi_s[r], N_unknowns_tot, N_unknowns_unit)
                icol[ind:indend] = np.arange(jc_s[k][r], N_unknowns_tot, N_unknowns_unit)
                Avals[ind:indend] = alpha_s[r] * z[k]
                ind = indend
                indend = ind+N_units
                # dendrite entries
                irow[ind:indend] = np.arange(jphi_d[r], N_unknowns_tot, N_unknowns_unit)
                icol[ind:indend] = np.arange(jc_d[k][r], N_unknowns_tot, N_unknowns_unit)
                Avals[ind:indend] = alpha_d[r] * z[k]
                ind = indend
                indend = ind+N_units

        gC = gamma_m * C_m / F
        for r in range(N_domains-1):
                # voltage-voltage entries for ICS potential
                # soma entries
                irow[ind:indend] = np.arange(jphi_s[r], N_unknowns_tot, N_unknowns_unit)
                icol[ind:indend] = np.arange(jphi_s[r], N_unknowns_tot, N_unknowns_unit)
                Avals[ind:indend] = - gC
                ind = indend
                indend = ind+N_units
                irow[ind:indend] = np.arange(jphi_s[r], N_unknowns_tot, N_unknowns_unit)
                icol[ind:indend] = np.arange(jphi_s[-1], N_unknowns_tot, N_unknowns_unit)
                Avals[ind:indend] = gC
                ind = indend
                indend = ind+N_units
                # dendrite entries
                irow[ind:indend] = np.arange(jphi_d[r], N_unknowns_tot, N_unknowns_unit)
                icol[ind:indend] = np.arange(jphi_d[r], N_unknowns_tot, N_unknowns_unit)
                Avals[ind:indend] = - gC
                ind = indend
                indend = ind+N_units
                irow[ind:indend] = np.arange(jphi_d[r], N_unknowns_tot, N_unknowns_unit)
                icol[ind:indend] = np.arange(jphi_d[-1], N_unknowns_tot, N_unknowns_unit)
                Avals[ind:indend] = gC
                ind = indend
                indend = ind+N_units
                # voltage-voltage entry for ECS potential
                # soma entries
                irow[ind:indend] = np.arange(jphi_s[-1], N_unknowns_tot, N_unknowns_unit)
                icol[ind:indend] = np.arange(jphi_s[r], N_unknowns_tot, N_unknowns_unit)
                Avals[ind:indend] = gC
                ind = indend
                indend = ind+N_units
                irow[ind:indend] = np.arange(jphi_s[-1], N_unknowns_tot, N_unknowns_unit)
                icol[ind:indend] = np.arange(jphi_s[-1], N_unknowns_tot, N_unknowns_unit)
                Avals[ind:indend] = - gC
                ind = indend
                indend = ind+N_units
                # dendrite entries
                irow[ind:indend] = np.arange(jphi_d[-1], N_unknowns_tot, N_unknowns_unit)
                icol[ind:indend] = np.arange(jphi_d[r], N_unknowns_tot, N_unknowns_unit)
                Avals[ind:indend] = gC
                ind = indend
                indend = ind+N_units
                irow[ind:indend] = np.arange(jphi_d[-1], N_unknowns_tot, N_unknowns_unit)
                icol[ind:indend] = np.arange(jphi_d[-1], N_unknowns_tot, N_unknowns_unit)
                Avals[ind:indend] = - gC
                ind = indend
                indend = ind+N_units

        # voltage-alpha entries
        for r in range(N_domains-1):
            for k in range(N_ions):
                # soma entries
                irow[ind:indend] = np.arange(jphi_s[r], N_unknowns_tot, N_unknowns_unit)
                icol[ind:indend] = np.arange(jalpha_s[r], N_unknowns_tot, N_unknowns_unit)
                Avals[ind:indend] = z[k] * c_s[k][r]
                ind = indend
                indend = ind+N_units
                # dendrite entries
                irow[ind:indend] = np.arange(jphi_d[r], N_unknowns_tot, N_unknowns_unit)
                icol[ind:indend] = np.arange(jalpha_d[r], N_unknowns_tot, N_unknowns_unit)
                Avals[ind:indend] = z[k] * c_d[k][r]
                ind = indend
                indend = ind+N_units

        # set phi_de to 0
        irow[ind] = jphi_d[-1]
        icol[ind] = jphi_d[-1]
        Avals[ind] = -1

        return irow, icol, Avals
