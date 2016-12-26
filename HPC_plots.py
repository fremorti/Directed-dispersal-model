import matplotlib.pyplot as plt
import numpy as np
import os
default_path = os.getcwd()




directed_ = {0: 'random dispersal', 1: 'directed dispersal'}
costs = (0.0, 10.0, 20.0, 50.0, 100.0, 150.0)
disps = np.arange(0.25, 5.25, 0.25)
m = 'density dependent'
m_ = {'density dependent': 'dd_data', 'density independent': 'di_data'}
mode = 'dispersal'
varis = {'dispersal':np.arange(0.25, 5.25, 0.25), 'varT': np.arange(0.025, 0.525, 0.025)}
directed = 0



def plot(y, costs, pos, densdep, mode, directed, title, ylab=''):
    bp0 = plt.plot(pos, np.transpose(y))
    plt.title(title)
    plt.xlabel(mode) 
    axes = plt.gca() 
    axes.set_xlim([0,pos[-1]+pos[0]])
    plt.ylabel(ylab)
    if not os.path.exists(default_path + "/plots/{}/LH_{}/{}".format(densdep, mode, str(directed))):
        os.makedirs(default_path + "/plots/{}/LH_{}/{}".format(densdep, mode, str(directed)))
    plt.savefig( default_path + "/plots/{}/LH_{}/{}/{}"
                    .format(densdep, mode, str(directed), title))
    plt.clf()
    
    
def costplot():
    
    
#for m in ['density dependent', 'density independent']
#for mode in ['dispersal', 'varT', 'both']

    #col = ['r', 'g', 'blue', 'orange', 'darkturquoise', 'purple', 'brown']
    for directed in (0, 1):
        directed = 0
        
        y = np.zeros((8, len(costs), len(disps)))
        
        for o,cost in enumerate(costs):
           
            
            for n, disp in enumerate(disps):
                a = np.array([np.load(default_path + '/hpcdata/{}/{}/{}/{}/{}/rep{}.npy'
                                      .format(m_[m], mode, str(directed), str(cost), str(disp), str(rep))) for rep in range(30)])
                avs = np.average(a, axis = 0)
                for b in range(8):
                    y[b, o, n] = avs[b]
            
            
        plot(y[0], costs, varis[mode], m_[m], mode, directed, 'muT')
            
        if mode != "varT":
            plot(y[1], costs, varis[mode], m_[m], mode, directed, 'niche width')
     
        if mode != "dispersal":
            plot(y[1], costs, varis[mode], m_[m], mode, directed, 'dispersal threshold')
        plot(y[2], costs, varis[mode], m_[m],  mode, directed, 'habitat mismatch')
        plot(y[3], costs, varis[mode], m_[m], mode, directed, 'metapopulation size', 'individuals')
        plot(y[4], costs, varis[mode], m_[m], mode, directed, 'dispersal probability')
        plot(y[5], costs, varis[mode], m_[m], mode, directed, 'local population variability', 'alpha variability')
        plot(y[6], costs, varis[mode], m_[m], mode, directed, 'metapopulation variability', 'gamma variability')
        plot(y[7], costs, varis[mode], m_[m], mode, directed, 'metapopulation asynchrony', 'beta variability')

def show_vars(var, mode, directed, pos, iters, cost):
    data = np.zeros((len(pos), iters))
    for x, val in enumerate(pos):
        for rep in range(iters):
            data[x, rep] = np.load(default_path + '/hpcdata/{}/{}/{}/{}/{}/rep{}.npy'.format('dd_data', mode, str(directed), str(cost), str(val), str(rep)))[var]
    print(data)
    
show_vars(1, 'dispersal', 0, disps, 30, 0.0)