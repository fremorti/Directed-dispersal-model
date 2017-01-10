import matplotlib.pyplot as plt
import numpy as np
import sys
import os
default_path = os.getcwd()




directed_ = {0: 'random dispersal', 1: 'directed dispersal'}
costs = (0.0, 10.0, 20.0, 50.0, 100.0, 150.0)
disps = {'dispersal': np.arange(0.25, 5.25, 0.25), 'varT': np.arange(0.025, 0.525, 0.025)}
dd = {'density dependent': 'dd_data', 'density independent': 'di_data'}
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
    plt.legend((x for x in bp0), ('0', '1/30', '1/15', '1/6', '1/3', '1/2'), loc = 0, fontsize = 'small', title = 'cost:')      
    if not os.path.exists(default_path + "/plots/{}/{}/LH_{}/{}".format(kind_of_plot, densdep, mode, str(directed))):
        os.makedirs(default_path + "/plots/{}/{}/LH_{}/{}".format(kind_of_plot, densdep, mode, str(directed)))
    plt.savefig( default_path + "/plots/{}/{}/LH_{}/{}/{}"
                    .format(kind_of_plot, densdep, mode, str(directed), title))
    plt.clf()

def barplot(y, kind_of_plot, densdep, mode, directed, costs, pos, title, ylab=''):
    bp0 = plt.boxplot(y, positions = pos, sym='.', widths = 0.5*(pos[1]-pos[0]), patch_artist=1, manage_xticks=0)
    plt.setp(bp0['boxes'], color='red', alpha = 0.3)
    plt.setp(bp0['fliers'], markerfacecolor='red', markeredgecolor = 'red')
    plt.setp(bp0['medians'], color='red')
    plt.setp(bp0['whiskers'], color='black')
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

def cplot(y, kind_of_plot, densdep, mode, directed, costs, pos, title, ylab=''):
    bp0 = plt.boxplot(y[0], positions = pos, sym='.', widths = 0.5*(pos[1]-pos[0]), patch_artist=1, manage_xticks=0)
    bp1 = plt.boxplot(y[1], positions = pos, sym='.', widths = 0.5*(pos[1]-pos[0]), patch_artist=1, manage_xticks=0)
    plt.setp(bp0['boxes'], color='red', alpha = 0.3)
    plt.setp(bp0['fliers'], markerfacecolor='red', markeredgecolor = 'red')
    plt.setp(bp0['medians'], color='red')
    plt.setp(bp0['whiskers'], color='black')
    plt.setp(bp1['boxes'], color='green', alpha = 0.3)
    plt.setp(bp1['fliers'], markerfacecolor='green', markeredgecolor = 'green')
    plt.setp(bp1['medians'], color='green')
    plt.setp(bp1['whiskers'], color='black')
    plt.title(title)
    plt.legend((bp0['boxes'][0], bp1['boxes'][0]), ('random', 'directed'), loc = 0)
    plt.xlabel(mode) 
    axes = plt.gca() 
    axes.set_xlim([0,pos[0]+pos[-1]])
    plt.ylabel(ylab)
    if not os.path.exists(default_path + "/plots/{}/{}/LH_{}".format(kind_of_plot, densdep, mode)):
        os.makedirs(default_path + "/plots/{}/{}/LH_{}".format(kind_of_plot, densdep, mode))
    plt.savefig( default_path + "/plots/{}/{}/LH_{}/{}"
                    .format(kind_of_plot, densdep, mode, title))
    plt.clf()
    
def plotall(y, kind_of_plot, densdep, mode, directed, costs, pos):
    '''
    Helper function that requests a plot for each separate metric that is followed
    '''
    func = {'costplot': plot, 'LHplot': barplot, 'combinedplot': cplot}
    func[kind_of_plot](y[0], kind_of_plot, densdep, mode, directed, costs, pos, 'muT')        
    if mode != "varT":
        func[kind_of_plot](y[1],  kind_of_plot, densdep, mode, directed, costs, pos, 'niche width')
    if mode != "dispersal":
        func[kind_of_plot](y[1],  kind_of_plot, densdep, mode, directed, costs, pos, 'dispersal threshold')
    func[kind_of_plot](y[2],  kind_of_plot, densdep, mode, directed, costs, pos, 'habitat mismatch')
    func[kind_of_plot](y[3],  kind_of_plot, densdep, mode, directed, costs, pos, 'metapopulation size', 'individuals')
    func[kind_of_plot](y[4],  kind_of_plot, densdep, mode, directed, costs, pos, 'dispersal probability')
    func[kind_of_plot](y[5],  kind_of_plot, densdep, mode, directed, costs, pos, 'local population variability', 'alpha variability')
    func[kind_of_plot](y[6],  kind_of_plot, densdep, mode, directed, costs, pos, 'metapopulation variability', 'gamma variability')
    func[kind_of_plot](y[7],  kind_of_plot, densdep, mode, directed, costs, pos, 'metapopulation asynchrony', 'beta variability')


def LHplot(cost):

    for _, m_ in dd.items():
        for mode in ['dispersal', 'varT']:
            for directed in (0, 1):
                
                a = np.array([[np.load(default_path + '/hpcdata/{}/{}/{}/{}/{}/rep{}.npy'
                                      .format(m_, mode, str(directed), str(float(cost)), str(disp), str(rep))) 
                               for rep in range(reps)] 
                              for disp in disps[mode]])
                
                print(a[0, ..., 1])
                plotall(np.transpose(a), sys._getframe().f_code.co_name, m_, mode, directed, costs, varis[mode])


def costplot(): 
    dd = {'density dependent': 'dd_data'}
    for _, m_ in dd.items():
        for mode in ['dispersal', 'varT']:
            for directed in (0, 1):
        
                a = np.array([[[np.load(default_path + '/hpcdata/{}/{}/{}/{}/{}/rep{}.npy'
                                      .format(m_, mode, str(directed), str(float(cost)), str(disp), str(rep))) 
                                for rep in range(reps)]
                               for cost in costs] 
                              for disp in disps[mode]])
                b = np.average(a, axis = 2)    
                plotall(np.transpose(b), sys._getframe().f_code.co_name , m_, mode, directed, costs, varis[mode])    

def combinedplot(cost):
    dd = {'density dependent': 'dd_data'}
    for _, m_ in dd.items():
        for mode in ['dispersal', 'varT']:
            a = np.array([[[np.load(default_path + '/hpcdata/{}/{}/{}/{}/{}/rep{}.npy'
                                  .format(m_, mode, str(directed), str(float(cost)), str(disp), str(rep)))
                             for directed in (0, 1)]
                            for rep in range(reps)]
                           for disp in disps[mode]])   
            plotall(np.transpose(a), sys._getframe().f_code.co_name , m_, mode, directed, costs, varis[mode])    

                
costplot()