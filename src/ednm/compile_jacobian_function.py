# this file must be run if changes are made to the model.py file

from numba.pycc import CC
from ednm.model import Model

cc = CC('aot_jacobian')

@cc.export('jacobian', '(float64, int32, float64, int32, int32, float64[:,:], float64[:,:], float64[:,:,:], float64[:,:,:], float64[:, :], float64[:, :], \
            float64[:,:], float64[:,:], float64[:,:,:], float64[:,:,:], float64[:, :], float64[:, :], float64[:,:], float64, float64[:,:], float64[:])')
def get_jacobian(dt, N_units, dxu, synapses, bc, alpha_s, alpha_d, c_s, c_d, phi_s, phi_d, alpha_s_, alpha_d_, c_s_, c_d_, phi_s_, phi_d_, ss_, t, t_AP, spikes):
    model = Model(N_units, dxu, synapses, bc)
    return model.jacobian(dt, alpha_s, alpha_d, c_s, c_d, phi_s, phi_d, alpha_s_, alpha_d_, c_s_, c_d_, phi_s_, phi_d_, ss_, t, t_AP, spikes)

if __name__ == '__main__':
    cc.compile()

