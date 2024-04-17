import matplotlib.pyplot as plt

def set_style(style='default', w=1, h=1):
   sdict = {
       'default': {
           'figure.figsize' : (4.98 * w, 2 * h),
           'lines.linewidth': 1,
           'lines.markersize': 1.5,
           'font.size'      : 10,
           'legend.frameon' : True,
           'font.family'    : 'serif',
           'text.usetex'    : True,
           'xtick.direction': 'in',
           'ytick.direction': 'in'
       }, 
       'poster': {
           'figure.figsize' : (4.98 * w, 2 * h),
           'lines.linewidth': 1.5,
           'lines.markersize': 1.5,
           'font.size'      : 20,
           'legend.frameon' : True,
           'font.family'    : 'serif',
           'text.usetex'    : True,
           'xtick.direction': 'in',
           'ytick.direction': 'in'
       } 
       }
   rc = sdict[style]
   plt.rcParams.update(rc)
