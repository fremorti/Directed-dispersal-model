'''
Created on Oct 29, 2016

@author: frederik
Seperate runs from visualisation
STEP2:  Save datasets generated during simulation
        Call seperate function that generates plots
        
'''




from Adapt import Metapopulation as Metapopulation
import matplotlib.pyplot as plt
from itertools import repeat
import numpy as np
import os
default_path = os.getcwd()


def LHplot(pos, atype, directed, widths, cost):
    '''
    plot life-history (LH) data of one run
    '''
       
    alpha    = np.load(default_path + '/data/'+ atype + str(directed)+'/alpha.npy')
    gamma    = np.load(default_path + '/data/'+ atype + str(directed)+'/gamma.npy')
    beta     = np.load(default_path + '/data/'+ atype + str(directed)+'/beta.npy')
    div      = np.load(default_path + '/data/'+ atype + str(directed)+'/div.npy')
    HM       = np.load(default_path + '/data/'+ atype + str(directed)+'/HM.npy')
    popsizes = np.load(default_path + '/data/'+ atype + str(directed)+'/popsizes.npy')
    dispprop = np.load(default_path + '/data/'+ atype + str(directed)+'/dispprop.npy')
    
    if not os.path.exists(default_path + "/plots/LH_"+ atype + str(directed) +'/' + str(cost)):
        os.makedirs(default_path + "/plots/LH_"+ atype + str(directed) +'/' + str(cost))
    def plot(y, title, ylab=''):
        bp0 = plt.boxplot(y, positions = pos, sym='.', widths = widths, patch_artist=1, manage_xticks=0)
        plt.setp(bp0['boxes'], color='red', alpha = 0.3)
        plt.setp(bp0['whiskers'], color='red')
        plt.setp(bp0['fliers'], color='red')
        plt.setp(bp0['medians'], color='red')
        plt.title(title)
        plt.xlabel(atype) 
        axes = plt.gca() 
        axes.set_xlim([0,end + step])
        plt.ylabel(ylab)
        plt.savefig( default_path + "/plots/LH_"+ atype + str(directed) +'/' + str(cost) + "/{}".format(title))
        plt.clf()
    
    plot(alpha, 'local population variability', 'alpha variability')
    plot(gamma, 'metapopulation variability', 'gamma variability')
    plot(beta, 'metapopulation asynchrony', 'beta variability')
    plot(div, 'muT')
    plot(HM, 'habitat mismatch')
    plot(popsizes, 'metapopulation size', 'individuals')
    plot(dispprop, 'dispersal probability')
    
    if atype != "varT":
        NB = np.load(default_path + '/data/'+ atype + str(directed)  + '/NB.npy')
        plot(NB, 'niche width')
 
    if atype != "dispersal":
        TH = np.load(default_path + '/data/'+ atype + str(directed) + '/TH.npy')
        plot(TH, 'dispersal threshold')



def LHcombinedplot(pos, atype, widths):
    '''
    Plot the LH data of informed and random dispersal runs of the model together
    '''
    
    widths *= (2/3)
    alpha0    = np.load(default_path + '/data/'+ atype + '0/alpha.npy')
    gamma0    = np.load(default_path + '/data/'+ atype + '0/gamma.npy')
    beta0     = np.load(default_path + '/data/'+ atype + '0/beta.npy')
    div0      = np.load(default_path + '/data/'+ atype + '0/div.npy')
    HM0       = np.load(default_path + '/data/'+ atype + '0/HM.npy')
    popsizes0 = np.load(default_path + '/data/'+ atype + '0/popsizes.npy')
    dispprop0 = np.load(default_path + '/data/'+ atype + '0/dispprop.npy')
    alpha1    = np.load(default_path + '/data/'+ atype + '1/alpha.npy')
    gamma1    = np.load(default_path + '/data/'+ atype + '1/gamma.npy')
    beta1     = np.load(default_path + '/data/'+ atype + '1/beta.npy')
    div1      = np.load(default_path + '/data/'+ atype + '1/div.npy')
    HM1       = np.load(default_path + '/data/'+ atype + '1/HM.npy')
    popsizes1 = np.load(default_path + '/data/'+ atype + '1/popsizes.npy')
    dispprop1 = np.load(default_path + '/data/'+ atype + '1/dispprop.npy')
    
    if not os.path.exists(default_path + "/combinedplots/LH_"+ atype):
        os.makedirs(default_path + "/combinedplots/LH_"+ atype)
    def combinedplot(y0, y1, title, ylab):
        bp0 = plt.boxplot(y0, positions = pos-0.5*widths, sym='.', widths = widths, patch_artist=1, manage_xticks=0)
        bp1 = plt.boxplot(y1, positions = pos+0.5*widths, sym='.', widths = widths, patch_artist=1, manage_xticks=0)
        plt.setp(bp0['boxes'], color='red', alpha = 0.3)
        plt.setp(bp1['boxes'], color='green', alpha = 0.3)
        plt.setp(bp0['fliers'], color='red')
        plt.setp(bp1['fliers'], color='green')
        plt.setp(bp0['whiskers'], color='red')
        plt.setp(bp1['whiskers'], color='green')
        plt.setp(bp0['medians'], color='red')
        plt.setp(bp1['medians'], color='green')
        plt.legend((bp0['boxes'][0], bp1['boxes'][0]), ('random', 'directed'))
        plt.title(title)
        plt.xlabel(atype)
        if atype == "both":
            plt.xticks(pos, ["random movement", "informed movement"])
        else:
            axes = plt.gca() 
            axes.set_xlim([0,end + step])
        plt.ylabel(ylab)
        plt.savefig( default_path + "/combinedplots/LH_"+ atype + "/{}".format(title))
        plt.clf()
    
    combinedplot(alpha0, alpha1, 'local population variability', 'alpha variability')
    combinedplot(gamma0, gamma1, 'metapopulation variability', 'gamma variability')
    combinedplot(beta0, beta1, 'metapopulation asynchrony', 'beta variability')
    combinedplot(div0, div1, 'muT')
    combinedplot(HM0, HM1, 'habitat mismatch')
    combinedplot(popsizes0, popsizes1, 'metapopulation size', 'individuals')
    combinedplot(dispprop0, dispprop1, 'dispersal probability')
    
    if atype != "varT":
        NB0 = np.load(default_path + '/data/'+ atype + '0/NB.npy')
        NB1 = np.load(default_path + '/data/'+ atype + '1/NB.npy')
        combinedplot(NB0, NB1, 'niche width')
 
    if atype != "dispersal":
        TH0 = np.load(default_path + '/data/'+ atype + '0/TH.npy')
        TH1 = np.load(default_path + '/data/'+ atype + '1/TH.npy')
        combinedplot(TH0, TH1, 'dispersal threshold')
    
    
       
def LH_dispersal(iters, start, end, step, cost = 0):
    '''
    Generate a run with a set of tested fixed dispersal values and save the life-history data in files
    '''
    
    mutable_threshold = 0
    mutable_variability = 1

    
    disps = np.arange(start, end+step, step) 
    div = np.zeros([iters, len(disps)])                 #Dictionary of mean diversity (muT) for each run
    NB = np.zeros([iters, len(disps)])                    #same for niche breadth (varT)
    HM = np.zeros([iters, len(disps)])
    TH = np.zeros([iters, len(disps)])  
    alpha = np.zeros([iters, len(disps)])  
    gamma = np.zeros([iters, len(disps)])   
    beta = np.zeros([iters, len(disps)]) 
    popsizes = np.zeros([iters, len(disps)])
    dispprop =  np.zeros([iters, len(disps)])
    
    for n, initialthreshold in enumerate(disps):    
        
        for m in range(iters):
            meta = run(MAXTIME, dim,R_res,K_res, initialmaxd, initialvarT, initialthreshold, mutable_threshold, mutable_variability, directed, cost)
            if m == 0 and initialmaxd == start:
                meta.visual.canvas.update()
            diversity = [ind.muT for ind in meta.population]
            nichebr = [ind.varT for ind in meta.population]
            habitatmatch = [ind.muT-meta.environment[ind.y][ind.x] for ind in meta.population]
            thresholds = [ind.threshold for ind in meta.population]
            div[m][n] = sum(diversity)/len(diversity)
            TH[m][n] = sum(thresholds)/len(thresholds)
            NB[m][n] = sum(nichebr)/len(nichebr)
            HM[m][n] = abs(sum(habitatmatch)/len(habitatmatch))
            popsizes[m][n] = len(meta.population)
            dispprop [m][n] = meta.disp_prop
            
            localstddev = [[pow(np.var([size[x][y] for size in meta.localsizes[-5:]]), 0.5) for y in range(dim)] for x in range(dim)]
            localmean = [[np.mean([size[x][y] for size in meta.localsizes[-5:]]) for y in range(dim)] for x in range(dim)]
            globalvar = np.var([np.sum(size) for size in meta.localsizes[-5:]])
            globalmean = np.mean([np.sum(size) for size in meta.localsizes[-5:]])
            alpha[m][n] = pow(np.sum(localstddev)/np.sum(localmean), 2)
            gamma[m][n] = globalvar/pow(globalmean, 2)
            beta[m][n] = pow(np.sum(localstddev), 2)/globalvar
            
        print(initialmaxd)
    
    
    if not os.path.exists(default_path + '/data/dispersal'+ str(directed)):
        os.makedirs(default_path + '/data/dispersal'+ str(directed))
    np.save(default_path + '/data/dispersal'+ str(directed) +'/alpha', alpha)
    np.save(default_path + '/data/dispersal'+ str(directed) +'/gamma', gamma)
    np.save(default_path + '/data/dispersal'+ str(directed) +'/beta', beta)
    np.save(default_path + '/data/dispersal'+ str(directed) +'/div', div)
    np.save(default_path + '/data/dispersal'+ str(directed) +'/NB', NB)
    np.save(default_path + '/data/dispersal'+ str(directed) +'/HM', HM)
    np.save(default_path + '/data/dispersal'+ str(directed) +'/popsizes', popsizes)
    np.save(default_path + '/data/dispersal'+ str(directed) +'/dispprop', dispprop)
    
    LHplot(disps, "dispersal", directed, step/2, cost)
      
    
def LH_varT(iters, start, end, step, cost = 0):
    '''
    Generate a run with a set of tested niche widths and save the life-history data in files
    '''
    mutable_threshold = 1
    mutable_variability = 0
    
    
    
    varTs = np.arange(start, end+step, step)  
    div = np.zeros([iters, len(varTs)])                 #Dictionary of mean diversity (muT) for each run
    HM = np.zeros([iters, len(varTs)])    
    TH = np.zeros([iters, len(varTs)])    
    alpha = np.zeros([iters, len(varTs)])  
    gamma = np.zeros([iters, len(varTs)])   
    beta = np.zeros([iters, len(varTs)])  
    popsizes = np.zeros([iters, len(varTs)])  
    dispprop =  np.zeros([iters, len(varTs)])
    
    for n, initialvarT in enumerate(varTs):    
        for m in range(iters):
            
            meta = run(MAXTIME, dim,R_res,K_res, initialmaxd, initialvarT, initialthreshold, mutable_threshold, mutable_variability, directed, cost)
            if m == 0 and initialmaxd == start:
                meta.visual.canvas.update()
            diversity = [ind.muT for ind in meta.population]
            thresholds = [ind.threshold for ind in meta.population]
            habitatmatch = [ind.muT-meta.environment[ind.y][ind.x] for ind in meta.population]

            div[m][n] = sum(diversity)/len(diversity)
            TH[m][n] = sum(thresholds)/len(thresholds)
            HM[m][n] = abs(sum(habitatmatch)/len(habitatmatch))
            popsizes[m][n] = len(meta.population)
            dispprop [m][n] = meta.disp_prop
            
            localstddev = [[pow(np.var([size[x][y] for size in meta.localsizes[-5:]]), 0.5) for y in range(dim)] for x in range(dim)]
            localmean = [[np.mean([size[x][y] for size in meta.localsizes[-5:]]) for y in range(dim)] for x in range(dim)]
            globalvar = np.var([np.sum(size) for size in meta.localsizes[-5:]])
            globalmean = np.mean([np.sum(size) for size in meta.localsizes[-5:]])
            alpha[m][n] = pow(np.sum(localstddev)/np.sum(localmean), 2)
            gamma[m][n] = globalvar/pow(globalmean, 2)
            beta[m][n] = pow(np.sum(localstddev), 2)/globalvar
        print(initialmaxd)
    
    if not os.path.exists(default_path + '/data/varT'+ str(directed)):
        os.makedirs(default_path + '/data/varT'+ str(directed))
    np.save(default_path + '/data/varT'+ str(directed) +'/alpha', alpha)
    np.save(default_path + '/data/varT'+ str(directed) +'/gamma', gamma)
    np.save(default_path + '/data/varT'+ str(directed) +'/beta', beta)
    np.save(default_path + '/data/varT'+ str(directed) +'/div', div)
    np.save(default_path + '/data/varT'+ str(directed) +'/TH', TH)
    np.save(default_path + '/data/varT'+ str(directed) +'/HM', HM)
    np.save(default_path + '/data/varT'+ str(directed) +'/popsizes', popsizes)
    np.save(default_path + '/data/varT'+ str(directed) +'/dispprop', dispprop)
    
    LHplot(varTs, "varT", directed, step/2, cost)
       
def resource_dispersal(iters, start, end, step):
    '''
    How much resources are left in the environment
    '''
    mutable_threshold, mutable_variability = 1, 1    
    mean_res = {}
    xs = range(start, end+1, step)
    
    for initialmaxd in xs:
        
        for _ in range(iters):
            meta = run(MAXTIME, dim,R_res,K_res, initialmaxd, initialvarT, initialthreshold, mutable_threshold, mutable_variability, directed, cost)
            
            mean_res[initialmaxd] = mean_res.get(initialmaxd, []) + [np.average(meta.resources)]
    
    for k in mean_res:
        plt.scatter([n for n in repeat(k, len(mean_res[k]))], mean_res[k])
    plt.title('Amount of resources left')
    plt.xlim([0, end+1])
    plt.show()

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
    
    
def LH_both(iters, cost = 0):
    '''
    evolved values of metapopulation and life history parameters regressed for varT
    '''    

    dispmode = (0, 1) 
    div = np.zeros([iters, len(dispmode)])                 #Dictionary of mean diversity (muT) for each run
    HM = np.zeros([iters, len(dispmode)])    
    NB = np.zeros([iters, len(dispmode)])
    TH = np.zeros([iters, len(dispmode)])    
    alpha = np.zeros([iters, len(dispmode)])  
    gamma = np.zeros([iters, len(dispmode)])   
    beta = np.zeros([iters, len(dispmode)])  
    popsizes = np.zeros([iters, len(dispmode)])
    dispprop =  np.zeros([iters, len(dispmode)])
    mutable_threshold, mutable_variability = 1 , 1
    
    
    for n in dispmode:    
        for m in range(iters):
            
            meta = run(MAXTIME, dim,R_res,K_res, initialmaxd, initialvarT, initialthreshold, mutable_threshold, mutable_variability, n, cost)
            if m == 0 and initialmaxd == start:
                meta.visual.canvas.update()
            diversity = [ind.muT for ind in meta.population]
            thresholds = [ind.threshold for ind in meta.population]
            nichebr = [ind.varT for ind in meta.population]
            habitatmatch = [ind.muT-meta.environment[ind.y][ind.x] for ind in meta.population]
            div[m][n] = sum(diversity)/len(diversity)
            TH[m][n] = sum(thresholds)/len(thresholds)
            NB[m][n] = sum(nichebr)/len(nichebr)
            HM[m][n] = abs(sum(habitatmatch)/len(habitatmatch))
            popsizes[m][n] = len(meta.population)
            dispprop [m][n] = meta.disp_prop
            
            localstddev = [[pow(np.var([size[x][y] for size in meta.localsizes[-5:]]), 0.5) for y in range(dim)] for x in range(dim)]
            localmean = [[np.mean([size[x][y] for size in meta.localsizes[-5:]]) for y in range(dim)] for x in range(dim)]
            globalvar = np.var([np.sum(size) for size in meta.localsizes[-5:]])
            globalmean = np.mean([np.sum(size) for size in meta.localsizes[-5:]])
            alpha[m][n] = pow(np.sum(localstddev)/np.sum(localmean), 2)
            gamma[m][n] = globalvar/pow(globalmean, 2)
            beta[m][n] = pow(np.sum(localstddev), 2)/globalvar
        print(initialmaxd)
    if not os.path.exists(default_path + '/data/both0'):
        os.makedirs(default_path + '/data/both0')
    np.save(default_path + '/data/both0' +'/alpha', alpha)
    np.save(default_path + '/data/both0' +'/gamma', gamma)
    np.save(default_path + '/data/both0' +'/beta', beta)
    np.save(default_path + '/data/both0' +'/div', div)
    np.save(default_path + '/data/both0' +'/TH', TH)
    np.save(default_path + '/data/both0' +'/NB', NB)
    np.save(default_path + '/data/both0' +'/HM', HM)
    np.save(default_path + '/data/both0' +'/popsizes', popsizes)
    np.save(default_path + '/data/both0' +'/dispprop', dispprop)
    
    LHbothplot(cost)
    
        
def LHbothplot():
    
    alpha    = np.load(default_path + '/data/both0/alpha.npy')
    gamma    = np.load(default_path + '/data/both0/gamma.npy')
    beta     = np.load(default_path + '/data/both0/beta.npy')
    div      = np.load(default_path + '/data/both0/div.npy')
    HM       = np.load(default_path + '/data/both0/HM.npy')
    popsizes = np.load(default_path + '/data/both0/popsizes.npy')
    dispprop = np.load(default_path + '/data/both0/dispprop.npy')
    NB       = np.load(default_path + '/data/both0/NB.npy')
    TH       = np.load(default_path + '/data/both0/TH.npy')
    
    
    if not os.path.exists(default_path + "/plots/LH_both0/" + str(cost)):
        os.makedirs(default_path + "/plots/LH_both0/" + str(cost))
    def bothplot(y, title, ylab = ''):
        bp0 = plt.boxplot(y, positions = (1, 2), sym='.', widths = 0.5, patch_artist=1, manage_xticks=0)
        plt.setp(bp0['boxes'][0], color='red', alpha = 0.3)
        plt.setp(bp0['whiskers'][0], color='red')
        plt.setp(bp0['fliers'][0], color='red')
        plt.setp(bp0['medians'][0], color='red')
        plt.setp(bp0['boxes'][1], color='green', alpha = 0.3)
        plt.setp(bp0['whiskers'][1], color='green')
        plt.setp(bp0['fliers'][1], color='green')
        plt.setp(bp0['medians'][1], color='green')
        plt.title(title)
        plt.legend((bp0['boxes'][0], bp0['boxes'][1]), ('random', 'directed'))
        plt.xticks((1, 2), ["random movement", "informed movement"])
        plt.ylabel(ylab)
        plt.savefig(default_path + "/plots/LH_both0/" + str(cost)+ "/{}".format(title))
        plt.clf()
    
    bothplot(alpha, 'local population variability', 'alpha variability')
    bothplot(gamma, 'metapopulation variability', 'gamma variability')
    bothplot(beta, 'metapopulation asynchrony', 'beta variability')
    bothplot(div, 'muT')
    bothplot(HM, 'habitat mismatch')
    bothplot(popsizes, 'metapopulation size', 'individuals')
    bothplot(dispprop, 'dispersal probability')
    bothplot(NB, 'niche width')
    bothplot(TH, 'dispersal threshold')

def LH_cost(iters, start, end, step, function, costs, directed):
    
    types = {LH_dispersal:'dispersal', LH_varT:'varT', LH_both:'both'}
    
    
    xvals = np.arange(start, end+step, step) 
    div_ = np.zeros([len(costs), len(xvals)])                 #Dictionary of mean diversity (muT) for each run
    HM_ = np.zeros([len(costs), len(xvals)])
    alpha_ =  np.zeros([len(costs), len(xvals)]) 
    gamma_ =  np.zeros([len(costs), len(xvals)])
    beta_ =  np.zeros([len(costs), len(xvals)])
    popsizes_ =  np.zeros([len(costs), len(xvals)])
    dispprop_ =   np.zeros([len(costs), len(xvals)])
    if function != LH_dispersal:
        TH_ =  np.zeros([len(costs), len(xvals)])
    if function != LH_varT:
        NB_ = np.zeros([len(costs), len(xvals)])                 #same for niche breadth (varT)
    
     
    for n,cost in enumerate(costs):
        function(iters, start, end, step, cost)
        alpha    = np.load(default_path + '/data/'+ types[function] + str(directed)+'/alpha.npy')
        gamma    = np.load(default_path + '/data/'+ types[function] + str(directed)+'/gamma.npy')
        beta     = np.load(default_path + '/data/'+ types[function] + str(directed)+'/beta.npy')
        div      = np.load(default_path + '/data/'+ types[function] + str(directed)+'/div.npy')
        HM       = np.load(default_path + '/data/'+ types[function] + str(directed)+'/HM.npy')
        popsizes = np.load(default_path + '/data/'+ types[function] + str(directed)+'/popsizes.npy')
        dispprop = np.load(default_path + '/data/'+ types[function] + str(directed)+'/dispprop.npy')
        div_[n] = [np.average([div[x][i] for x in range(iters)]) for i in range(len(xvals))]
        HM_[n] = [np.average([HM[x][i] for x in range(iters)]) for i in range(len(xvals))]
        alpha_[n] = [np.average([alpha[x][i] for x in range(iters)]) for i in range(len(xvals))]
        gamma_[n] = [np.average([gamma[x][i] for x in range(iters)]) for i in range(len(xvals))]
        beta_[n] = [np.average([beta[x][i] for x in range(iters)]) for i in range(len(xvals))]
        popsizes_[n] = [np.average([popsizes[x][i] for x in range(iters)]) for i in range(len(xvals))]
        dispprop_[n] = [np.average([dispprop[x][i] for x in range(iters)]) for i in range(len(xvals))]
        if function != LH_dispersal:
            TH = np.load(default_path + '/data/'+ types[function] + str(directed) + '/TH.npy')
            TH_ = np.load(default_path + '/data/'+ types[function] + str(directed)+'/TH.npy')
            TH_[n] = [np.average([TH[x][i] for x in range(iters)]) for i in range(len(xvals))]
        if function != LH_varT:
            NB = np.load(default_path + '/data/'+ types[function] + str(directed) + '/NB.npy')
            NB_ = np.load(default_path + '/data/'+ types[function] + str(directed)+'/NB.npy')
            NB_[n] = [np.average([NB[x][i] for x in range(iters)]) for i in range(len(xvals))]

    if not os.path.exists(default_path + '/data/cost/'+ types[function]+ str(directed)):
        os.makedirs(default_path + '/data/cost/'+ types[function]+ str(directed))
    np.save(default_path + '/data/cost/'+ types[function]+ str(directed) +'/alpha', alpha_)
    np.save(default_path + '/data/cost/'+ types[function]+ str(directed) +'/gamma', gamma_)
    np.save(default_path + '/data/cost/'+ types[function]+ str(directed) +'/beta', beta_)
    np.save(default_path + '/data/cost/'+ types[function]+ str(directed) +'/div', div_) 
    np.save(default_path + '/data/cost/'+ types[function]+ str(directed) +'/HM', HM_)
    np.save(default_path + '/data/cost/'+ types[function]+ str(directed) +'/popsizes', popsizes_)
    np.save(default_path + '/data/cost/'+ types[function]+ str(directed) +'/dispprop', dispprop_)
    if function != LH_dispersal:
        np.save(default_path + '/data/cost/'+ types[function]+ str(directed) +'/TH', TH_)
    if function != LH_varT:
        np.save(default_path + '/data/cost/'+ types[function]+ str(directed) +'/NB', NB_)
    
    LHcostplots(start, end, step, atype, directed, costs)
        
def LHcostplots(start, end, step, atype, directed, costs):
    alpha    = np.load(default_path + '/data/cost/'+ atype + str(directed)+'/alpha.npy')
    gamma    = np.load(default_path + '/data/cost/'+ atype + str(directed)+'/gamma.npy')
    beta     = np.load(default_path + '/data/cost/'+ atype + str(directed)+'/beta.npy')
    div      = np.load(default_path + '/data/cost/'+ atype + str(directed)+'/div.npy')
    HM       = np.load(default_path + '/data/cost/'+ atype + str(directed)+'/HM.npy')
    popsizes = np.load(default_path + '/data/cost/'+ atype + str(directed)+'/popsizes.npy')
    dispprop = np.load(default_path + '/data/cost/'+ atype + str(directed)+'/dispprop.npy')
    #cols = ['b', 'r', 'g', 'coral', 'darkmagenta', 'steelblue', 'brown']
    xs = np.arange(start, end+step, step)
    
    if not os.path.exists(default_path + "/plots/cost/LH_"+ atype + str(directed)):
        os.makedirs(default_path + "/plots/cost/LH_"+ atype + str(directed))
    def plot(y, title, ylab=''):
        pl0 = plt.plot(xs, np.transpose(y))
        plt.title(title)
        plt.xlabel(atype) 
        axes = plt.gca() 
        axes.set_xlim([0,end + step])
        plt.ylabel(ylab)
        plt.savefig( default_path + "/plots/cost/LH_"+ atype + str(directed) + "/{}".format(title))
        plt.clf()
    
    plot(alpha, 'local population variability', 'alpha variability')
    plot(gamma, 'metapopulation variability', 'gamma variability')
    plot(beta, 'metapopulation asynchrony', 'beta variability')
    plot(div, 'muT')
    plot(HM, 'habitat mismatch')
    plot(popsizes, 'metapopulation size', 'individuals')
    plot(dispprop, 'dispersal probability')
    
    if atype != "varT":
        NB = np.load(default_path + '/data/cost/'+ atype + str(directed)+'/NB.npy')
        plot(NB, 'niche width')
 
    if atype != "dispersal":
        TH = np.load(default_path + '/data/cost/'+ atype + str(directed)+ '/TH.npy')
        plot(TH, 'dispersal threshold')
    
        
    
    
'''
Default PARAMETERS
'''

MAXTIME=10  #50
dim = 32
R_res = 0.25
K_res = 1  
initialmaxd = 2
initialvarT = 0.05
initialthreshold = 0.1
directed = 0
cost = 0 #cost of directed dispersal
costs = [0, 1]
iters = 5      #30   
start = 0.1   #20 steps in total start-stop
end = 0.5
step = 0.1
atype = 'dispersal'


#LHcostplots(start, end, step, atype, directed, costs)
LH_dispersal(iters, start, end, step, cost)
#LHcombinedplot(np.arange(start, end+step, step), 'dispersal', step/2)