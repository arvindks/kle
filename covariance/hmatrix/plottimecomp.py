from scipy.io import loadmat
import numpy as np
from matplotlib import pyplot as plt
import matplotlib

plt.close('all')

matplotlib.rcParams['xtick.labelsize'] = 18.
matplotlib.rcParams['ytick.labelsize'] = 18.

N = np.array([ 500, 1000, 5000, 10000, 50000, 100000]) #, 500000])
data = loadmat('time.mat')
multtime = data['multtime']
setuptime = data['setuptime']


plt.figure()
plt.loglog(N,multtime, '.-k', linewidth = 2., label = r'${\cal H}$matrix')
plt.loglog(N, 0.7*multtime[0]*N/N[0], 'k--' , linewidth = 2., label = r'${\cal O} (N)$')
plt.loglog(N, 4.*multtime[0]*(N/N[0])**2., 'k-.' , linewidth = 2., label = r'${\cal O} (N^2)$')
plt.xlabel('system size', fontsize = 16)
plt.ylabel('Time[s]', fontsize = 16)
plt.title(r'Matvec time for ${\cal H}$matrix', fontsize = 20)
#plt.legend(loc = 'upper left')
plt.text(N[4], 0.25*multtime[0]*N[4]/N[0], r'${\cal O}(N)$', fontsize = 24)
plt.text(0.4*N[3], 15.*multtime[0]*(N[3]/N[0])**2., r'${\cal O}(N^2)$', fontsize = 24)
plt.savefig('figs/matvectime.png')

plt.figure()
plt.loglog(N,setuptime, '.-k', linewidth = 2.,label = r'${\cal H}$matrix')
plt.loglog(N, 0.9*setuptime[0]*N/N[0], 'k--' , linewidth = 2., label = r'${\cal O} (N)$')
plt.loglog(N, 1.5*setuptime[0]*(N/N[0])**2., 'k-.' , linewidth = 2., label = r'${\cal O} (N^2)$')
plt.xlabel('system size', fontsize = 16)
plt.xlabel('system size', fontsize = 16)
plt.ylabel('Time[s]', fontsize = 16)
plt.title(r'Setup time for ${\cal H}$matrix', fontsize = 20)
#plt.legend(loc = 'upper left')
plt.text(N[4], 0.25*setuptime[0]*N[4]/N[0], r'${\cal O}(N)$', fontsize = 24)
plt.text(0.4*N[3], 15.*setuptime[0]*(N[3]/N[0])**2., r'${\cal O}(N^2)$', fontsize = 24)
plt.savefig('figs/setuptime.png')
plt.show()
