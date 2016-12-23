'''
Created on 13 dec. 2016

@author: fremorti
'''


from Adapt import Metapopulation as Metapopulation
import numpy as np
import os
import sys
default_path = os.getcwd()



def LH_dispersal(disp, rep, cost = 0):
    '''
    Generate a run with a set of tested fixed dispersal values and save the life-history data in files
    '''
    
    mutable_threshold = 0
    mutable_variability = 1
    meta = run(MAXTIME, dim,R_res,K_res, initialmaxd, initialvarT, initialthreshold, mutable_threshold, mutable_variability, directed, cost)
    
    data = np.zeros(8)
    diversity = [ind.muT for ind in meta.population]
    nichebr = [ind.varT for ind in meta.population]
    habitatmatch = [ind.muT-meta.environment[ind.y][ind.x] for ind in meta.population]
    data[0] = sum(diversity)/len(diversity)
    data[1] = sum(nichebr)/len(nichebr)
    data[2] = abs(sum(habitatmatch)/len(habitatmatch))
    data[3] = len(meta.population)
    data[4] = meta.disp_prop

    localstddev = [[pow(np.var([size[x][y] for size in meta.localsizes[-5:]]), 0.5) for y in range(dim)] for x in range(dim)]
    localmean = [[np.mean([size[x][y] for size in meta.localsizes[-5:]]) for y in range(dim)] for x in range(dim)]
    globalvar = np.var([np.sum(size) for size in meta.localsizes[-5:]])
    globalmean = np.mean([np.sum(size) for size in meta.localsizes[-5:]])
    data[5] = pow(np.sum(localstddev)/np.sum(localmean), 2)
    data[6] = globalvar/pow(globalmean, 2)
    data[7] = pow(np.sum(localstddev), 2)/globalvar
            
    
    
    
    if not os.path.exists(default_path + '/data/dispersal/'+ str(directed) + '/'+ str(cost) + '/'+ str(disp)):
        os.makedirs(default_path + '/data/dispersal/'+ str(directed) + '/'+ str(cost) + '/'+ str(disp))
    np.save(default_path + '/data/dispersal/'+ str(directed) + '/'+ str(cost) + '/'+ str(disp) + '/rep'+ str(rep), data)

      
    
def LH_varT(var, rep, cost = 0):
    '''
    Generate a run with a set of tested niche widths and save the life-history data in files
    '''
    mutable_threshold = 1
    mutable_variability = 0

    
    data = np.zeros(8)

    meta = run(MAXTIME, dim,R_res,K_res, initialmaxd, initialvarT, initialthreshold, mutable_threshold, mutable_variability, directed, cost)
    diversity = [ind.muT for ind in meta.population]
    thresholds = [ind.threshold for ind in meta.population]
    habitatmatch = [ind.muT-meta.environment[ind.y][ind.x] for ind in meta.population]

    data[0] = sum(diversity)/len(diversity)
    data[1] = sum(thresholds)/len(thresholds)
    data[2] = abs(sum(habitatmatch)/len(habitatmatch))
    data[3] = len(meta.population)
    data[4] = meta.disp_prop
    
    localstddev = [[pow(np.var([size[x][y] for size in meta.localsizes[-5:]]), 0.5) for y in range(dim)] for x in range(dim)]
    localmean = [[np.mean([size[x][y] for size in meta.localsizes[-5:]]) for y in range(dim)] for x in range(dim)]
    globalvar = np.var([np.sum(size) for size in meta.localsizes[-5:]])
    globalmean = np.mean([np.sum(size) for size in meta.localsizes[-5:]])
    data[5] = pow(np.sum(localstddev)/np.sum(localmean), 2)
    data[6] = globalvar/pow(globalmean, 2)
    data[7] = pow(np.sum(localstddev), 2)/globalvar
    
    if not os.path.exists(default_path + '/data/varT/'+ str(directed) + '/'+ str(cost) + '/'+ str(var)):
        os.makedirs(default_path + '/data/varT/'+ str(directed) + '/'+ str(cost) + '/'+ str(var))
    np.save(default_path + '/data/varT/'+ str(directed) + '/'+ str(cost) + '/'+ str(var) + '/rep'+ str(rep), data)
       


def run(MAXTIME, dim,R_res,K_res, initialmaxd, initialvarT, initialthreshold, mutable_dispersal, mutable_variability, directed, cost):
    '''
    helper function that initiates a metapopulation and let it evolve for a given amount of generations
    dispersal and niche width are either evolvable or not, with initial values passed (important for fixed traits), dispersal is directed or not
    '''
    
    meta = Metapopulation(dim,dim,R_res,K_res, initialmaxd, initialvarT, initialthreshold, mutable_dispersal, mutable_variability, directed, cost)
    
    meta.loadlandscape()
    
    for timer in range(MAXTIME):
        print('generation ',timer)
    
        meta.lifecycle()
        print ("popsize: {}\n".format(len(meta.population)))
    return(meta)
    
    
def LH_both(directed, rep, cost = 0):
    '''
    evolved values of metapopulation and life history parameters regressed for varT
    '''    
    mutable_threshold, mutable_variability = 1, 1
    data = np.zeros(9)
    meta = run(MAXTIME, dim,R_res,K_res, initialmaxd, initialvarT, initialthreshold, mutable_threshold, mutable_variability,directed, cost)
    diversity = [ind.muT for ind in meta.population]
    thresholds = [ind.threshold for ind in meta.population]
    nichebr = [ind.varT for ind in meta.population]
    habitatmatch = [ind.muT-meta.environment[ind.y][ind.x] for ind in meta.population]
    data[0] = sum(diversity)/len(diversity)
    data[1] = sum(thresholds)/len(thresholds)
    data[2] = sum(nichebr)/len(nichebr)
    data[3] = abs(sum(habitatmatch)/len(habitatmatch))
    data[4] = len(meta.population)
    data[5] = meta.disp_prop
    
    localstddev = [[pow(np.var([size[x][y] for size in meta.localsizes[-5:]]), 0.5) for y in range(dim)] for x in range(dim)]
    localmean = [[np.mean([size[x][y] for size in meta.localsizes[-5:]]) for y in range(dim)] for x in range(dim)]
    globalvar = np.var([np.sum(size) for size in meta.localsizes[-5:]])
    globalmean = np.mean([np.sum(size) for size in meta.localsizes[-5:]])
    data[6] = pow(np.sum(localstddev)/np.sum(localmean), 2)
    data[7] = globalvar/pow(globalmean, 2)
    data[8] = pow(np.sum(localstddev), 2)/globalvar
        
    if not os.path.exists(default_path + '/data/both/' + str(directed) + '/'+ str(cost)):
        os.makedirs(default_path + '/data/both/' + str(directed) + '/'+ str(cost))
    np.save(default_path + '/data/both/' + str(directed) + '/'+ str(cost)+ '/' + str(rep), data)
        
    
    
'''
Default PARAMETERS
'''

MAXTIME=50  #50
dim = 32
R_res = 0.25
K_res = 1  
initialmaxd = 2
initialvarT = 0.05
initialthreshold = 0.1
directed = 1    
cost = float(sys.argv[1]) #cost of directed dispersal
disp = float(sys.argv[2])
rep  = sys.argv[3]


#LHcostplots(start, end, step, atype, directed, costs)
LH_dispersal(disp, rep, cost)
#LHcombinedplot(np.arange(start, end+step, step), 'dispersal', step/2)