# this file must be run if changes are made to the model.py file

from numba.pycc import CC
from ednm.model import Model

cc = CC('aot_residual')

@cc.export('residual', '(float64, int32, int32, int32, float64[:,:], float64[:,:], float64[:,:,:], float64[:,:,:], float64[:,:], float64[:,:], \
        float64[:,:], float64[:,:], float64[:,:,:], float64[:,:,:], float64[:,:], float64[:,:], float64[:,:], float64, float64[:,:], float64, int32, float64[:])')
def get_residual(dt, N_units, synapses, bc, alpha_s, alpha_d, c_s, c_d, phi_s, phi_d, alpha_s_, alpha_d_, c_s_, c_d_, phi_s_, phi_d_, ss_, t, t_AP, j_in, N_in, spikes):
    model = Model(N_units, synapses, bc)
    Res = model.residual(dt, alpha_s, alpha_d, c_s, c_d, phi_s, phi_d, alpha_s_, alpha_d_, c_s_, c_d_, phi_s_, phi_d_, ss_, t, t_AP, j_in, N_in, spikes)
    return Res

if __name__ == '__main__':
    cc.compile()
