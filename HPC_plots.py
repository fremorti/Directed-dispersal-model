import matplotlib.pyplot as plt
import numpy as np
import sys
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
reps = 30



def plot(y, kind_of_plot, densdep, mode, directed, costs, pos, title, ylab=''):
    '''
    Helper function that plots a set of ys in the prefered lay-out and saves them in the prefered folder structure
    '''
    
    bp0 = plt.plot(pos, np.transpose(y))
    plt.title(title)
    plt.xlabel(mode) 
    axes = plt.gca() 
    axes.set_xlim([0,pos[-1]+pos[0]])
    plt.ylabel(ylab)
    if not os.path.exists(default_path + "/plots/{}/{}/LH_{}/{}".format(kind_of_plot, densdep, mode, str(directed))):
        os.makedirs(default_path + "/plots/{}/{}/LH_{}/{}".format(kind_of_plot, densdep, mode, str(directed)))
    plt.savefig( default_path + "/plots/{}/{}/LH_{}/{}/{}"
                    .format(kind_of_plot, densdep, mode, str(directed), title))
    plt.clf()

def barplot(y, kind_of_plot, densdep, mode, directed, costs, pos, title, ylab=''):
    bp0 = plt.boxplot(y, positions = pos, sym='.', widths = 0.5*(pos[1]-pos[0]), patch_artist=1, manage_xticks=0)
    plt.setp(bp0['boxes'], color='red', alpha = 0.3)
    plt.setp(bp0['whiskers'], color='red')
    plt.setp(bp0['fliers'], color='red')
    plt.setp(bp0['medians'], color='red')
    plt.title(title)
    plt.xlabel(mode) 
    axes = plt.gca() 
    axes.set_xlim([0,pos[0]+pos[-1]])
    plt.ylabel(ylab)
    if not os.path.exists(default_path + "/plots/{}/{}/LH_{}/{}".format(kind_of_plot, densdep, mode, str(directed))):
        os.makedirs(default_path + "/plots/{}/{}/LH_{}/{}".format(kind_of_plot, densdep, mode, str(directed)))
    plt.savefig( default_path + "/plots/{}/{}/LH_{}/{}/{}"
                    .format(kind_of_plot, densdep, mode, str(directed), title))
    plt.clf()

def plotall(y, kind_of_plot, densdep, mode, directed, costs, pos):
    '''
    Helper function that requests a plot for each separate metric that is followed
    '''
    
    plot(y[0], kind_of_plot, densdep, mode, directed, costs, pos, 'muT')
        
    if mode != "varT":
        plot(y[1],  kind_of_plot, densdep, mode, directed, costs, pos, 'niche width')
 
    if mode != "dispersal":
        plot(y[1],  kind_of_plot, densdep, mode, directed, costs, pos, 'dispersal threshold')
    plot(y[2],  kind_of_plot, densdep, mode, directed, costs, pos, 'habitat mismatch')
    plot(y[3],  kind_of_plot, densdep, mode, directed, costs, pos, 'metapopulation size', 'individuals')
    plot(y[4],  kind_of_plot, densdep, mode, directed, costs, pos, 'dispersal probability')
    plot(y[5],  kind_of_plot, densdep, mode, directed, costs, pos, 'local population variability', 'alpha variability')
    plot(y[6],  kind_of_plot, densdep, mode, directed, costs, pos, 'metapopulation variability', 'gamma variability')
    plot(y[7],  kind_of_plot, densdep, mode, directed, costs, pos, 'metapopulation asynchrony', 'beta variability')



def LHplot(cost):
    #for m in ['density dependent', 'density independent']
#for mode in ['dispersal', 'varT', 'both']

    #col = ['r', 'g', 'blue', 'orange', 'darkturquoise', 'purple', 'brown']
    for directed in (0, 1):
        directed = 0
        
        a = np.array([[np.load(default_path + '/hpcdata/{}/{}/{}/{}/{}/rep{}.npy'
                              .format(m_[m], mode, str(directed), str(float(cost)), str(disp), str(rep))) for rep in range(reps)] for disp in disps])
        plotall(np.transpose(a), __name__ , m_[m], mode, directed, costs, varis[mode])

def costplot():
    
    
#for m in ['density dependent', 'density independent']
#for mode in ['dispersal', 'varT', 'both']

    #col = ['r', 'g', 'blue', 'orange', 'darkturquoise', 'purple', 'brown']
    for directed in (0, 1):
        directed = 0

        a = np.array([[[np.load(default_path + '/hpcdata/{}/{}/{}/{}/{}/rep{}.npy'
                              .format(m_[m], mode, str(directed), str(float(cost)), str(disp), str(rep))) 
                        for rep in range(reps)]
                       for cost in costs] 
                      for disp in disps])
        b = np.average(a, axis = 2)    
        plotall(np.transpose(b), sys._getframe().f_code.co_name , m_[m], mode, directed, costs, varis[mode])    
        



def show_vars(var, mode, directed, pos, iters, cost):
    data = np.zeros((len(pos), iters))
    for x, val in enumerate(pos):
        for rep in range(iters):
            data[x, rep] = np.load(default_path + '/hpcdata/{}/{}/{}/{}/{}/rep{}.npy'.format('dd_data', mode, str(directed), str(cost), str(val), str(rep)))[var]
    print(data)
    
costplot()