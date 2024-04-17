import os
import time
from plotter import plot_phi, plot_c_alpha, plot_phi_m
from ednm import model, solver 

# synapses (0: off, 1: on)
# When turned on, each neuron connects to its right neighbour from soma to dendrite (neuron U connects to 0), 
# and neuron 0 receives a stimulus at t = 0.1 s.
synapses = 1

# boundary condition (0: closed, 1: periodic)
bc = 1

# time variables
Tstop = 1.5

# stimulus
stim_start = 0
stim_end = 0
j_stim = 0    
N_stim = 0 # number of cells receiving constant external input

stimulus_protocol = {
        "j_stim": j_stim,
        "stim_start": stim_start,
        "stim_end": stim_end,
        "N_stim": N_stim}

# number of cells
N_units = 10

# check that directory for results (data) exists, if not create
path_data = 'results/data/'
if not os.path.isdir(path_data):
    os.makedirs(path_data)

# build model
start_ = time.time()

model = model.Model(N_units, synapses, bc)

end_ = time.time()
seconds_ = end_-start_

# solve system
start = time.time()

solver.solve_system(model, path_data, Tstop, stimulus_protocol)

end = time.time()
seconds = end-start

# print time
m, s = divmod(seconds, 60)
h, m = divmod(m, 60)
print('time spent to set up the model:', seconds_, 'sec')
print('simulation time:', h, 'h,', m, 'min, and', s, 'sec')

# plot data
path_figures = 'results/figures/'
plot_phi(path_data, path_figures, N_units)
plot_c_alpha(path_data, path_figures, N_units)
plot_phi_m(path_data, path_figures, N_units)
