import matplotlib.pyplot as plt
import numpy as np
import sys
import os
from mpl_toolkits.mplot3d import Axes3D
default_path = os.getcwd()




directed_ = {0: 'random dispersal', 1: 'niche width'}
costs = (0.0, 10.0, 20.0, 50.0, 100.0, 150.0)
disps = {'dispersal': np.arange(0.25, 5.25, 0.25), 'varT': np.arange(0.025, 0.525, 0.025)}
dd = {'density dependent': 'dd_data', 'density independent': 'di_data'}
varis = {1: np.arange(0.25, 5.25, 0.25), 0: np.arange(0.025, 0.525, 0.025)}
directed, departure = 0, 0
reps = 10
    
def bplot(y, kind_of_plot, departure, mode, directed, costs, pos, title, ylab='', ylim = 0):

    bp0 = plt.boxplot(np.transpose(y), positions = pos, sym='.', widths = 0.5*(pos[1]-pos[0]), patch_artist=1, manage_xticks=0)
    plt.setp(bp0['boxes'][0], color='red', alpha = 0.3)
    plt.setp(bp0['whiskers'][0], color='red')
    plt.setp(bp0['fliers'][0], color='red')
    plt.setp(bp0['medians'][0], color='red')
    plt.setp(bp0['boxes'][1], color='green', alpha = 0.3)
    plt.setp(bp0['whiskers'][1], color='green')
    plt.setp(bp0['fliers'][1], color='green')
    plt.setp(bp0['medians'][1], color='green')
    plt.title(title, fontsize = 26)
    plt.legend((bp0['boxes'][0], bp0['boxes'][1]), ('random', 'directed'))
    plt.xticks((0, 1), ["random movement", "informed movement"], fontsize = 22)
    plt.ylabel(ylab, fontsize = 22)
    if ylim:
        plt.ylim(ymin = 0 if ylim == 1 else ylim)
    if not os.path.exists(default_path + "/plots/{}/{}/LH_{}/{}".format(kind_of_plot, departure, mode, str(directed))):
        os.makedirs(default_path + "/plots/{}/{}/LH_{}/{}".format(kind_of_plot, departure, mode, str(directed)))
    plt.savefig( default_path + "/plots/{}/{}/LH_{}/{}/{}"
                    .format(kind_of_plot, departure, mode, str(directed), title))
    plt.clf()
    
def pplot(y, kind_of_plot, departure, mode, directed, costs, pos, title, ylab='', ylim = 0):
    xlab = {'dispersal': 'dispersal propensity', 'varT': 'niche width'}
    pos_ = [[pos[x, w, z] for w in range(10) for z in range(20)] for x in range(3)] if mode == 'dispersal' else [[x for _ in range(10) for x in pos] for _ in range(3)]
    y_ = [[y[x, w, z] for w in range(10) for z in range(20)] for x in range(3)]
    
    bp0 = plt.scatter(pos_[0], y_[0], color = '#d95f02', alpha = 0.3)
    bp1 = plt.scatter(pos_[1], y_[1], color = '#1b9e77', alpha = 0.3)
    bp2 = plt.scatter(pos_[2], y_[2], color = '#7570b3', alpha = 0.3)
    axes = plt.gca() 
    plt.xlabel(xlab[mode] , fontsize = 22) 
    #axes.set_ylim(bottom = 0)
    plt.xlim(0, 1.05*max(max(pos_[0]), max(pos_[1])))
    plt.ylabel(ylab, fontsize = 22)
    if ylim:
        plt.ylim(ymin = 0 if ylim == 1 else ylim)
    #plt.ylim(ymin = 0 if ylim == 1 else ylim, ymax = 0.602)
    line1 = plt.Line2D(range(0), range(0), color="white", marker='o', markerfacecolor="#d95f02")
    line2 = plt.Line2D(range(0), range(0), color="white", marker='o',markerfacecolor="#1b9e77")
    line3 = plt.Line2D(range(0), range(0), color="white", marker='o', markerfacecolor="#7570b3")
    plt.title(title, fontsize = 26)
    plt.legend((line1, line3, line2), ('random settlement', 'imperfect settlement choice (0.5)', 'perfect settlement choice'), loc = 2, numpoints = 1, fontsize = 10)
        
    if not os.path.exists(default_path + "/plots/{}/departure{}/LH_{}".format(kind_of_plot, departure, mode)):
        os.makedirs(default_path + "/plots/{}/departure{}/LH_{}".format(kind_of_plot, departure, mode))
    plt.savefig( default_path + "/plots/{}/departure{}/LH_{}/{}"
                    .format(kind_of_plot, departure, mode, title))
    plt.clf()

def pall(y, kind_of_plot, departure, mode, directed, costs, pos, title, ylab='', ylim = 0):
    xlab = {'dispersal': 'dispersal propensity', 'varT': 'niche width'}
    pos_0 = [[pos[0, x, w, z] for w in range(10) for z in range(20)] for x in range(2)] if mode == 'dispersal' else [[x for _ in range(10) for x in np.arange(0.025, 0.525, 0.025)] for _ in range(2)]
    pos_1 = [[pos[1, x, w, z] for w in range(10) for z in range(20)] for x in range(2)] if mode == 'dispersal' else [[x for _ in range(10) for x in np.arange(0.025, 0.525, 0.025)] for _ in range(2)]
    y_0 = [[y[0,x, w, z] for w in range(10) for z in range(20)] for x in range(2)]
    y_1 = [[y[1,x, w, z] for w in range(10) for z in range(20)] for x in range(2)]
    
    
    bp00 = plt.scatter(pos_0[0], y_0[0], color = '#fc8205', alpha = 0.5, marker="x")
    bp01 = plt.scatter(pos_0[1], y_0[1], color = '#018571', alpha = 0.5, marker="x")
    bp10 = plt.scatter(pos_1[0], y_1[0], color = '#fc8205', alpha = 0.3)
    bp11 = plt.scatter(pos_1[1], y_1[1], color = '#018571', alpha = 0.3)
    axes = plt.gca() 
    plt.xlabel(xlab[mode] , fontsize = 22) 
    #axes.set_ylim(bottom = 0)
    plt.xlim(0, 1.05*max(max(pos_0[0]), max(pos_1[0]), max(pos_0[1]), max(pos_1[1])))
    if ylim:
        plt.ylim(ymin = 0 if ylim == 1 else ylim)
    line1 = plt.Line2D(range(0), range(0), color="white", marker='x', markerfacecolor='#a6611a', mec = '#a6611a')
    line2 = plt.Line2D(range(0), range(0), color="white", marker='x',markerfacecolor='#018571', mec = '#018571')
    line3 = plt.Line2D(range(0), range(0), color="white", marker='o', markerfacecolor='#fc8205', alpha = 0.5, mec = '#a6611a')
    line4 = plt.Line2D(range(0), range(0), color="white", marker='o',markerfacecolor='#018571', alpha = 0.5, mec = '#018571')
    
    
    plt.title(title, fontsize = 26)
    plt.ylabel(ylab, fontsize = 22)
    #plt.gcf().subplots_adjust(left=0.15)
    plt.legend((line1, line3, line2, line4), ('random dispersal', 'departure choice', 'settlement choice', 'combined choice'), loc = 2, fontsize = 10, numpoints = 1)
    #plt.legend((line1, line3), ('random dispersal', 'departure choice'), loc = 3, fontsize = 10, numpoints = 1)
    plt.tight_layout()    
    if not os.path.exists(default_path + "/plots/{}/LH_{}".format(kind_of_plot, mode)):
        os.makedirs(default_path + "/plots/{}/LH_{}".format(kind_of_plot, mode))
    plt.savefig( default_path + "/plots/{}/LH_{}/{}"
                    .format(kind_of_plot, mode, title))
    plt.clf()
      
def plotall(y, kind_of_plot, departure, mode, directed, costs, pos = None):
    '''
    Helper function that requests a plot for each separate metric that is followed
    '''
    func = {'pointplot': pplot, 'pointboth': bplot, 'pointall': pall}
    pos = y[5] if mode == 'dispersal' else np.arange(0.025, 0.525, 0.025)
    func[kind_of_plot](y[0], kind_of_plot, departure, mode, directed, costs, pos, 'muT', ylim = 0)        
    if mode != "varT":
        func[kind_of_plot](y[2],  kind_of_plot, departure, mode, directed, costs, pos, 'evolved niche width', 'niche width', ylim = 0)
    if mode != "dispersal":
        func[kind_of_plot](y[1],  kind_of_plot, departure, mode, directed, costs, pos, 'evolved dispersal', 'dispersal', ylim = 1)
    func[kind_of_plot](y[3],  kind_of_plot, departure, mode, directed, costs, pos, 'habitat mismatch', ylim = 1)
    
    func[kind_of_plot](y[4],  kind_of_plot, departure, mode, directed, costs, pos, 'metapopulation size', 'metapopulation size', ylim = 0)
    func[kind_of_plot](y[5],  kind_of_plot, departure, mode, directed, costs, pos, 'dispersal propensity','proportion dispersing', ylim = 1)
    func[kind_of_plot](y[6],  kind_of_plot, departure, mode, directed, costs, pos, 'prospecting propensity', ylim = 1)
    func[kind_of_plot](y[7],  kind_of_plot, departure, mode, directed, costs, pos, 'local population variability', r'$\mathrm{\mathsf{\alpha\ variability}}$', ylim = 0.009)
    func[kind_of_plot](y[8],  kind_of_plot, departure, mode, directed, costs, pos, 'metapopulation variability',r'$\mathrm{\mathsf{\gamma\ variability}}$', ylim = 1)
    func[kind_of_plot](y[9],  kind_of_plot, departure, mode, directed, costs, pos, 'metapopulation asynchrony', r'$\mathrm{\mathsf{\beta\ variability}}$', ylim = 1)
    func[kind_of_plot](y[10],  kind_of_plot, departure, mode, directed, costs, pos, 'unconsumed resources', '')

def pointplot(cost):
    gg = np.load(default_path + '/hpcdata/LH_dispersal/departure0.5/settlement0/0.0/0.25/rep0.npy')
    for departure in (0, 1):
        for mode in [ 'varT', 'dispersal']:
            treats = np.arange(0.05, 1.05, 0.05)
            a = np.array([[[np.load(default_path + '/hpcdata/LH_{}/departure{}/settlement{}/{}/{}/rep{}.npy'
                                  .format(mode, str(departure), str(directed), str(float(cost)), str(disp*(1 if mode == 'varT' else 5 if departure else 1)), str(rep)))
                             for directed in [0, 1, 0.5]]
                            for rep in range(reps)]
                           for disp in treats])   
            plotall(np.transpose(a), sys._getframe().f_code.co_name , departure, mode, directed, costs, treats)

def pointall(cost):
    for mode in ['dispersal', 'varT']:
        a = np.transpose(np.array([[[[np.load(default_path + '/hpcdata/LH_{}/departure{}/settlement{}/{}/{:.10}/rep{}.npy'
                              .format(mode, str(departure), str(directed), str(float(cost)), 0.5*disp if mode == 'varT' else disp*(5 if departure else 1), str(rep)))
                         for departure in (0, 1)]
                        for directed in (0, 1)]
                       for rep in range(reps)]
                      for disp in np.arange(0.05, 1.05, 0.05)]
                                   ))  
        plotall(a, sys._getframe().f_code.co_name , 1, mode, directed, costs)    


def pointboth(cost):
    for departure in (0, 1):
        
        a = np.array([[np.load(default_path + '/hpcdata/LH_{}/departure{}/settlement{}/{}/10/rep{}.npy'
                              .format('both', str(departure), str(directed), str(float(cost)), str(rep)))
                         for directed in (0, 1)]
                        for rep in range(reps)])   
        plotall(np.transpose(a), sys._getframe().f_code.co_name , departure, 'both', directed, costs, (0, 1))    

        
pointall(0)