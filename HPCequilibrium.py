'''
Created on 19 mei 2017

@author: fremorti
'''

contmuT = 0
from Adapt import Metapopulation as Metapopulation
import numpy as np
import os
import sys
default_path = os.getcwd()



def LH_dispersal(disp, cost = 0):
    runall(disp, None, 0, 1, sys._getframe().f_code.co_name)
    
def LH_varT(disp, cost = 0):
    runall(None, disp, 1, 0, sys._getframe().f_code.co_name)

def LH_both(cost = 0):
    runall(None, None, 1, 1, sys._getframe().f_code.co_name)

def runall(initialthreshold, initialvarT, mutable_threshold, mutable_variability, mode):
    meta = Metapopulation(dim,dim,R_res,K_res, initialmaxd, initialvarT, initialthreshold, mutable_threshold, mutable_variability, departure, directed, cost)
    
    meta.loadlandscape()
    data = np.zeros((10, MAXTIME))
    for timer in range(MAXTIME):
        print('generation ',timer)
    
        meta.lifecycle()
        print ("popsize: {}\n".format(len(meta.population)))
        
    
        diversity = [ind.muT for ind in meta.population]
        thresholds = [ind.threshold for ind in meta.population]
        nichebr = [ind.varT for ind in meta.population]
        if contmuT:
            habitatmatch = [min(abs(ind.muT-meta.environment[ind.x][ind.y]), 1-abs(ind.muT-meta.environment[ind.x][ind.y])) for ind in meta.population]
        else:
            habitatmatch = [abs(ind.muT-meta.environment[ind.x][ind.y]) for ind in meta.population]
            
        data[0, timer] = sum(diversity)/len(diversity)
        data[1, timer] = sum(thresholds)/len(thresholds)
        data[2, timer] = sum(nichebr)/len(nichebr)
        data[3, timer] = abs(sum(habitatmatch)/len(habitatmatch))
        data[4, timer] = len(meta.population)
        data[5, timer] = meta.disp_prop
        data[6, timer] = meta.pros_prop
    
        localstddev = [[pow(np.var([size[x][y] for size in meta.localsizes[-5:]]), 0.5) for y in range(dim)] for x in range(dim)]
        localmean = [[np.mean([size[x][y] for size in meta.localsizes[-5:]]) for y in range(dim)] for x in range(dim)]
        globalvar = np.var([np.sum(size) for size in meta.localsizes[-5:]])
        globalmean = np.mean([np.sum(size) for size in meta.localsizes[-5:]])
        data[7, timer] = pow(np.sum(localstddev)/np.sum(localmean), 2)
        data[8, timer] = globalvar/pow(globalmean, 2)
        data[9, timer] = pow(np.sum(localstddev), 2)/globalvar

    if not os.path.exists('{}/eqdata/{}/{}/{}/{}/{}'.format(default_path, departure, mode, str(directed), str(float(cost)), str(disp))):
        os.makedirs('{}/eqdata/{}/{}/{}/{}/{}'.format(default_path, departure, mode, str(directed), str(float(cost)), str(disp)))
    np.save('{}/eqdata/{}/{}/{}/{}/{}/{}'.format(default_path, departure, mode, str(directed), str(float(cost)), str(disp), str(rep)), data)
    
'''
Default PARAMETERS
'''

MAXTIME=1000  #50
dim = 32
R_res = 0.25
K_res = 1  
initialmaxd = 2
departure = 1
directed = 0 #int(sys.argv[4]) 
cost = 0#float(sys.argv[1]) #cost of directed dispersal
disp = 0.5#float(sys.argv[2])
rep = 1#sys.argv[1]

LH_varT(disp, cost)