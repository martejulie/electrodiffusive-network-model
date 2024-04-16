import os
import time

# set path to solver
from ednm import model, solver 

if __name__ == '__main__':

    # synapses (0: off, 1: on)
    synapses = 0

    # boundary condition (0: closed, 1: open)
    bc = 0

    # time variables
    Tstop = 1

    # stimulus
    stim_start = 0.1
    stim_end = 10
    #j_stim = 45e-5    
    #j_stim = 4.5e-5    
    j_stim = 20e-5    
    
    stimulus_protocol = {
            "j_stim": j_stim,
            "stim_start": stim_start,
            "stim_end": stim_end,
            "N_stim": 1}  # number of cells receiving external input

    # number of cells
    N_units = 2

    # check that directory for results (data) and figures exist, if not create
    path_data = 'results/data/'

    if not os.path.isdir(path_data):
        os.makedirs(path_data)

    start_ = time.time()

    model = model.Model(N_units, synapses, bc)
    
    end_ = time.time()
    seconds_ = end_-start_
    
    # solve system
    start = time.time()
    
    solver.solve_system(model, path_data, Tstop, stimulus_protocol)

    end = time.time()
    seconds = end-start

    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    print('time spent to set up the model:', seconds_, 'sec')
    print('simulation time:', h, 'h,', m, 'min, and', s, 'sec')
