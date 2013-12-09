import numpy as np
from matplotlib import pyplot as plt
import matplotlib

from scipy.io import loadmat

plt.close('all')

matplotlib.rcParams['xtick.labelsize'] = 18.
matplotlib.rcParams['ytick.labelsize'] = 18.

N = np.array([ 500, 1000, 5000, 10000, 50000, 100000]) 

nst = ['3', '6', '9']

for n in nst:
    data = loadmat('time52_eps_' + n + '.mat')
    multtime = data['multtime']
    setuptime = data['setuptime']
    
    plt.figure(1)
    plt.loglog(N,setuptime,  linewidth = 2., label = r'$\varepsilon = 10^{-%s}$'%(n))
    
    plt.figure(2)
    plt.loglog(N,multtime,  linewidth = 2., label = r'$\varepsilon = 10^{-%s}$'%(n))
    

plt.figure(1)
plt.xlabel('system size', fontsize = 16)
plt.ylabel('Time[s]', fontsize = 16)
plt.title(r'Setup time for ${\cal H}$matrix', fontsize = 20)
    

plt.loglog(N, 0.9*setuptime[0]*N/N[0], 'k--' , linewidth = 2.,)
plt.loglog(N, 1.5*setuptime[0]*(N/N[0])**2., 'k-.' , linewidth = 2.)

plt.text(N[3], 0.25*setuptime[0]*N[3]/N[0], r'${\cal O}(N)$', fontsize = 24)
plt.text(0.4*N[3], 15.*setuptime[0]*(N[3]/N[0])**2., r'${\cal O}(N^2)$', fontsize = 24)
plt.legend()
plt.savefig('figs/setuptimeepscomp.png')
#cplt.show()


plt.figure(2)
plt.loglog(N, 0.7*multtime[0]*N/N[0], 'k--' , linewidth = 2.)
plt.loglog(N, 4.*multtime[0]*(N/N[0])**2., 'k-.' , linewidth = 2.)

plt.xlabel('system size', fontsize = 16)
plt.ylabel('Time[s]', fontsize = 16)
plt.title(r'Matvec time for ${\cal H}$matrix', fontsize = 20)


plt.text(N[3], 0.25*multtime[0]*N[3]/N[0], r'${\cal O}(N)$', fontsize = 24)
plt.text(0.4*N[3], 15.*multtime[0]*(N[3]/N[0])**2., r'${\cal O}(N^2)$', fontsize = 24)
plt.legend()
plt.savefig('figs/matvectimeepscomp.png')
plt.show()