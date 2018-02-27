'''
Created on 19 mei 2017

@author: fremorti
'''
import matplotlib.pyplot as plt
import numpy as np
import sys
import os
default_path = os.getcwd()







def plot(y, kind_of_plot, densdep, mode, directed, costs, pos, title, ylab='', ylim = 0):
    '''
    Helper function that plots a set of ys in the prefered lay-out and saves them in the prefered folder structure
    '''
    for rep in range(replicates):
        bp0 = plt.plot(pos, y[rep])
    plt.title(title)
    axes = plt.gca() 
    axes.set_xlim([0,pos[-1]+pos[1]-pos[0]])
    plt.xlabel('time (generations)') 
    plt.ylabel(ylab)
    #plt.legend((x for x in bp0), ('0', '1/30', '1/15', '1/6', '1/3', '1/2'), loc = 0, fontsize = 'small', title = 'cost:')      
    if not os.path.exists(default_path + "/eqplots/{}/{}/{}/{}/{}".format(kind_of_plot, densdep, mode, str(directed), str(disp))):
        os.makedirs(default_path + "/eqplots/{}/{}/{}/{}/{}".format(kind_of_plot, densdep, mode, str(directed), str(disp)))
    plt.savefig( default_path + "/eqplots/{}/{}/{}/{}/{}/{}"
                    .format(kind_of_plot, densdep, mode, str(directed), str(disp), title))
    plt.clf()


        
def plotall(data, kind_of_plot, densdep, mode, directed, costs, pos):
    '''
    Helper function that requests a plot for each separate metric that is followed
    '''
    plot([y[0] for y in data], kind_of_plot, densdep, mode, directed, costs, range(pos), 'muT', ylim = 0)        
    if mode != "varT":
        plot([y[2] for y in data],  kind_of_plot, densdep, mode, directed, costs, range(pos), 'evolved niche width', 'niche width', ylim = 1)
    if mode != "dispersal":
        plot([y[1] for y in data],  kind_of_plot, densdep, mode, directed, costs, range(pos), 'evolved dispersal', 'dispersal', ylim = 1)
    plot([y[3] for y in data],  kind_of_plot, densdep, mode, directed, costs, range(pos), 'habitat mismatch', ylim = 1)
    
    plot([y[4] for y in data],  kind_of_plot, densdep, mode, directed, costs, range(pos), 'metapopulation size', 'metapopulation size', ylim = 0)
    plot([y[5] for y in data],  kind_of_plot, densdep, mode, directed, costs, range(pos), 'dispersal probability', ylim = 1)
    plot([y[6] for y in data],  kind_of_plot, densdep, mode, directed, costs, range(pos), 'prospecting probability', ylim = 1)
    plot([y[7] for y in data],  kind_of_plot, densdep, mode, directed, costs, range(pos), 'local population variability', 'alpha variability', ylim = 1)
    plot([y[8] for y in data],  kind_of_plot, densdep, mode, directed, costs, range(pos), 'metapopulation variability', 'gamma variability', ylim = 1)
    plot([y[9] for y in data],  kind_of_plot, densdep, mode, directed, costs, range(pos), 'metapopulation asynchrony', 'beta variability', ylim = 1)
  

def pointplot(departure, mode, settlement, cost, disp, timer):
               
    a = [np.load(default_path + '/eqdata/{}/{}/{}/{}/{}/{}.npy'
                 .format(departure, mode, str(settlement), str(float(cost)), str(disp), str(rep))) for rep in range(replicates)]                   
    plotall(a, sys._getframe().f_code.co_name , departure, mode, settlement, cost, timer)    

departure = 1
settlement = 1
mode = 'LH_varT'
cost = 0
disp = 1
timer = 1000
replicates = 10
pointplot(departure, mode, settlement, cost, disp, timer)