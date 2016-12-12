'''
Created on Oct 29, 2016

@author: frederik
Seperate runs from visualisation
STEP2:  Save datasets generated during simulation
        Call seperate function that generates plots
        
'''




'''
Created on Oct 29, 2016

@author: frederik
Seperate runs from visualisation
STEP2:  Save datasets generated during simulation
        Call seperate function that generates plots
        
'''




from DensIndep_Adapt import Metapopulation as Metapopulation
import matplotlib.pyplot as plt
from itertools import repeat
import numpy as np
default_path = "C:/Users/frederik/Documents/Doctoraat Gent/Model"


def LHplot(pos, atype, directed, widths):
       
    alpha    = np.load(default_path + '/data/'+ atype + str(directed)+'/alpha.npy')
    gamma    = np.load(default_path + '/data/'+ atype + str(directed)+'/gamma.npy')
    beta     = np.load(default_path + '/data/'+ atype + str(directed)+'/beta.npy')
    div      = np.load(default_path + '/data/'+ atype + str(directed)+'/div.npy')
    HM       = np.load(default_path + '/data/'+ atype + str(directed)+'/HM.npy')
    popsizes = np.load(default_path + '/data/'+ atype + str(directed)+'/popsizes.npy')
    dispprop = np.load(default_path + '/data/'+ atype + str(directed)+'/dispprop.npy')
    
    bp0 = plt.boxplot(alpha, positions = pos, sym='.', widths = widths, patch_artist=1, manage_xticks=0)
    plt.setp(bp0['boxes'], color='red', alpha = 0.3)
    plt.setp(bp0['whiskers'], color='red')
    plt.setp(bp0['fliers'], color='red')
    plt.setp(bp0['medians'], color='red')
    plt.title('local population variability')
    plt.xlabel(atype) 
    axes = plt.gca() 
    axes.set_xlim([0,end + step])
    plt.ylabel('alpha variability')
    plt.savefig( default_path + "/plots/LH_"+ atype + str(directed) + "/_Alpha Variability")
    plt.clf()
    
    
    bp0 = plt.boxplot(gamma, positions = pos, sym='.', widths = widths, patch_artist=1, manage_xticks=0)
    plt.setp(bp0['fliers'], color='red')
    plt.setp(bp0['boxes'], color='red', alpha = 0.3)
    plt.setp(bp0['whiskers'], color='red')
    plt.setp(bp0['medians'], color='red')
    plt.title('metapopulation variability') 
    plt.xlabel(atype) 
    axes = plt.gca() 
    axes.set_xlim([0,end + step])
    plt.ylabel('gamma variability')
    plt.savefig(default_path + "/plots/LH_"+ atype + str(directed) + "/_Gamma Variability")
    plt.clf()
    
    bp0 = plt.boxplot(beta, positions = pos, sym='.', widths = widths, patch_artist=1, manage_xticks=0)
    plt.setp(bp0['fliers'], color='red')
    plt.setp(bp0['boxes'], color='red', alpha = 0.3)
    plt.setp(bp0['whiskers'], color='red')
    plt.setp(bp0['medians'], color='red')
    plt.title('metapopulation asynchrony') 
    plt.xlabel(atype) 
    axes = plt.gca() 
    axes.set_xlim([0,end + step])
    plt.ylabel('beta variability')
    plt.savefig(default_path + "/plots/LH_"+ atype + str(directed) + "/_Metapop Asynchony")
    plt.clf()
    
    
    bp0 = plt.boxplot(div, positions = pos, sym='.', widths = widths, patch_artist=1, manage_xticks=0)
    plt.setp(bp0['fliers'], color='red')
    plt.setp(bp0['boxes'], color='red', alpha = 0.3)
    plt.setp(bp0['whiskers'], color='red')
    plt.setp(bp0['medians'], color='red')
    plt.title('muT') 
    plt.xlabel(atype) 
    axes = plt.gca() 
    axes.set_xlim([0,end + step])
    plt.savefig(default_path + "/plots/LH_"+ atype + str(directed) + "/Trait Value")
    plt.clf()
    
    bp0 = plt.boxplot(HM, positions = pos, sym='.', widths = widths, patch_artist=1, manage_xticks=0)
    plt.setp(bp0['fliers'], color='red')
    plt.setp(bp0['boxes'], color='red', alpha = 0.3)
    plt.setp(bp0['whiskers'], color='red')
    plt.setp(bp0['medians'], color='red')
    plt.title('habitat mismatch')
    plt.xlabel(atype) 
    axes = plt.gca() 
    axes.set_xlim([0,end + step])
    plt.ylabel('habitat mismatch')
    plt.savefig(default_path + "/plots/LH_"+ atype + str(directed) + "/Habitat Mismatch")
    plt.clf()
    
    bp0 = plt.boxplot(popsizes, positions = pos, sym='.', widths = widths, patch_artist=1, manage_xticks=0)
    plt.setp(bp0['fliers'], color='red')
    plt.setp(bp0['boxes'], color='red', alpha = 0.3)
    plt.setp(bp0['whiskers'], color='red')
    plt.setp(bp0['medians'], color='red')
    plt.title('metapopulation size') 
    plt.xlabel(atype) 
    axes = plt.gca() 
    axes.set_xlim([0,end + step])
    plt.ylabel('population size') 
    plt.savefig(default_path + "/plots/LH_"+ atype + str(directed) + "/Population Sizes")
    plt.clf()
    
    
    bp0 = plt.boxplot(dispprop, positions = pos, sym='.', widths = widths, patch_artist=1, manage_xticks=0)
    plt.setp(bp0['fliers'], color='red')
    plt.setp(bp0['boxes'], color='red', alpha = 0.3)
    plt.setp(bp0['whiskers'], color='red')
    plt.setp(bp0['medians'], color='red')
    plt.title('dispersal propensity')
    plt.xlabel(atype) 
    axes = plt.gca() 
    axes.set_xlim([0,end + step])
    plt.ylabel('dispersal propensity')
    plt.savefig( default_path + "/plots/LH_"+ atype + str(directed) + "/Dispersal_propensity")
    plt.clf()
    
    
    if atype != "varT":
        NB = np.load(default_path + '/data/'+ atype + str(directed) + '/NB.npy')
        bp0 = plt.boxplot(NB, positions = pos, sym='.', widths = widths, patch_artist=1, manage_xticks=0)
        plt.setp(bp0['fliers'], color='red')
        plt.setp(bp0['boxes'], color='red', alpha = 0.3)
        plt.setp(bp0['whiskers'], color='red')
        plt.setp(bp0['medians'], color='red')
        plt.title('niche width')
        plt.xlabel(atype) 
        axes = plt.gca() 
        axes.set_xlim([0,end + step])
        plt.ylabel('niche width')
        plt.savefig(default_path + "/plots/LH_"+ atype + str(directed) + "/Niche Breadth")   
        plt.clf()
 
    if atype != "dispersal":
        TH = np.load(default_path + '/data/'+ atype + str(directed) + '/TH.npy')
        bp0 = plt.boxplot(TH, positions = pos, sym='.', widths = widths, patch_artist=1, manage_xticks=0)
        plt.setp(bp0['fliers'], color='red', alpha = 0.3)
        plt.setp(bp0['boxes'], color='red', alpha = 0.3)
        plt.setp(bp0['whiskers'], color='red')
        plt.setp(bp0['medians'], color='red')
        plt.title('dispersal threshold')
        plt.ylabel('dispersal threshold')
        plt.xlabel(atype) 
        axes = plt.gca() 
        axes.set_xlim([0,end + step])
        
        plt.savefig(default_path + "/plots/LH_"+ atype + str(directed) + "/Threshold")   
        plt.clf() 

def LHcombinedplot(pos, atype, widths):
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
    
    
    bp0 = plt.boxplot(alpha0, positions = pos-0.5*widths, sym='.', widths = widths, patch_artist=1, manage_xticks=0)
    bp1 = plt.boxplot(alpha1, positions = pos+0.5*widths, sym='.', widths = widths, patch_artist=1, manage_xticks=0)
    plt.setp(bp0['boxes'], color='red', alpha = 0.3)
    plt.setp(bp1['boxes'], color='green', alpha = 0.3)
    plt.setp(bp0['fliers'], color='red')
    plt.setp(bp1['fliers'], color='green')
    plt.setp(bp0['whiskers'], color='red')
    plt.setp(bp1['whiskers'], color='green')
    plt.setp(bp0['medians'], color='red')
    plt.setp(bp1['medians'], color='green')
    plt.legend((bp0['boxes'][0], bp1['boxes'][0]), ('random', 'directed'))
    plt.title('local population variability')
    plt.xlabel(atype) 
    if atype == "both":
        plt.xticks(pos, ["random movement", "informed movement"])
    else:
        axes = plt.gca() 
        axes.set_xlim([0,end + step])
    plt.ylabel('alpha variability')
    plt.savefig( default_path + "/combinedplots/LH_"+ atype + "/_Alpha Variability")
    plt.clf()
    
    
    bp0 = plt.boxplot(gamma0, positions = pos-0.5*widths, sym='.', widths = widths, patch_artist=1, manage_xticks=0)
    bp1 = plt.boxplot(gamma1, positions = pos+0.5*widths, sym='.', widths = widths, patch_artist=1, manage_xticks=0)
    plt.setp(bp0['boxes'], color='red', alpha = 0.3)
    plt.setp(bp1['boxes'], color='green', alpha = 0.3)
    plt.setp(bp0['fliers'], color='red')
    plt.setp(bp1['fliers'], color='green')
    plt.setp(bp0['whiskers'], color='red')
    plt.setp(bp1['whiskers'], color='green')
    plt.setp(bp0['medians'], color='red')
    plt.setp(bp1['medians'], color='green')
    plt.title('metapopulation variability') 
    plt.legend((bp0['boxes'][0], bp1['boxes'][0]), ('random', 'directed'))
    plt.xlabel(atype) 
    if atype == "both":
        plt.xticks(pos, ["random movement", "informed movement"])
    else:
        axes = plt.gca() 
        axes.set_xlim([0,end + step])
    plt.ylabel('gamma variability')
    plt.savefig(default_path + "/combinedplots/LH_"+ atype + "/_Gamma Variability")
    plt.clf()
    
    bp0 = plt.boxplot(beta0, positions = pos-0.5*widths, sym='.', widths = widths, patch_artist=1, manage_xticks=0)
    bp1 = plt.boxplot(beta1, positions = pos+0.5*widths, sym='.', widths = widths, patch_artist=1, manage_xticks=0)
    plt.setp(bp0['boxes'], color='red', alpha = 0.3)
    plt.setp(bp1['boxes'], color='green', alpha = 0.3)
    plt.setp(bp0['fliers'], color='red')
    plt.setp(bp1['fliers'], color='green')
    plt.setp(bp0['whiskers'], color='red')
    plt.setp(bp1['whiskers'], color='green')
    plt.setp(bp0['medians'], color='red')
    plt.setp(bp1['medians'], color='green')
    plt.title('metapopulation asynchrony') 
    plt.legend((bp0['boxes'][0], bp1['boxes'][0]), ('random', 'directed'))
    if atype == "both":
        plt.xticks(pos, ["random movement", "informed movement"])
    else:
        plt.xlabel(atype) 
        axes = plt.gca() 
        axes.set_xlim([0,end + step])
    plt.ylabel('beta variability')
    plt.savefig(default_path + "/combinedplots/LH_"+ atype + "/_Metapop Asynchony")
    plt.clf()
    
    
    bp0 = plt.boxplot(div0, positions = pos-0.5*widths, sym='.', widths = widths, patch_artist=1, manage_xticks=0)
    bp1 = plt.boxplot(div1, positions = pos+0.5*widths, sym='.', widths = widths, patch_artist=1, manage_xticks=0)
    plt.setp(bp0['boxes'], color='red', alpha = 0.3)
    plt.setp(bp1['boxes'], color='green', alpha = 0.3)
    plt.setp(bp0['fliers'], color='red')
    plt.setp(bp1['fliers'], color='green')
    plt.setp(bp0['whiskers'], color='red')
    plt.setp(bp1['whiskers'], color='green')
    plt.setp(bp0['medians'], color='red')
    plt.setp(bp1['medians'], color='green')
    plt.title('muT') 
    plt.legend((bp0['boxes'][0], bp1['boxes'][0]), ('random', 'directed'))
    if atype == "both":
        plt.xticks(pos, ["random movement", "informed movement"])
    else:
        plt.xlabel(atype) 
        axes = plt.gca() 
        axes.set_xlim([0,end + step])
    plt.savefig(default_path + "/combinedplots/LH_"+ atype + "/Trait Value")
    plt.clf()
    
    bp0 = plt.boxplot(HM0, positions = pos-0.5*widths, sym='.', widths = widths, patch_artist=1, manage_xticks=0)
    bp1 = plt.boxplot(HM1, positions = pos+0.5*widths, sym='.', widths = widths, patch_artist=1, manage_xticks=0)
    plt.setp(bp0['boxes'], color='red', alpha = 0.3)
    plt.setp(bp1['boxes'], color='green', alpha = 0.3)
    plt.setp(bp0['fliers'], color='red')
    plt.setp(bp1['fliers'], color='green')
    plt.setp(bp0['whiskers'], color='red')
    plt.setp(bp1['whiskers'], color='green')
    plt.setp(bp0['medians'], color='red')
    plt.setp(bp1['medians'], color='green')
    plt.title('habitat mismatch')
    plt.legend((bp0['boxes'][0], bp1['boxes'][0]), ('random', 'directed'))
    if atype == "both":
        plt.xticks(pos, ["random movement", "informed movement"])
    else:
        plt.xlabel(atype) 
        axes = plt.gca() 
        axes.set_xlim([0,end + step])
    plt.ylabel('habitat mismatch')
    plt.savefig(default_path + "/combinedplots/LH_"+ atype + "/Habitat Mismatch")
    plt.clf()
    
    bp0 = plt.boxplot(popsizes0, positions = pos-0.5*widths, sym='.', widths = widths, patch_artist=1, manage_xticks=0)
    bp1 = plt.boxplot(popsizes1, positions = pos+0.5*widths, sym='.', widths = widths, patch_artist=1, manage_xticks=0)
    plt.setp(bp0['fliers'], color='red')
    plt.setp(bp1['fliers'], color='green')
    plt.setp(bp0['boxes'], color='red', alpha = 0.3)
    plt.setp(bp1['boxes'], color='green', alpha = 0.3)
    plt.setp(bp0['whiskers'], color='red')
    plt.setp(bp1['whiskers'], color='green')
    plt.setp(bp0['medians'], color='red')
    plt.setp(bp1['medians'], color='green')
    plt.title('metapopulation size') 
    plt.legend((bp0['boxes'][0], bp1['boxes'][0]), ('random', 'directed'))
    if atype == "both":
        plt.xticks(pos, ["random movement", "informed movement"])
    else:
        plt.xlabel(atype) 
        axes = plt.gca() 
        axes.set_xlim([0,end + step])
    plt.ylabel('population size') 
    plt.savefig(default_path + "/combinedplots/LH_"+ atype + "/Population Sizes")
    plt.clf()
    
    bp0 = plt.boxplot(dispprop0, positions = pos-0.5*widths, sym='.', widths = widths, patch_artist=1, manage_xticks=0)
    bp1 = plt.boxplot(dispprop1, positions = pos+0.5*widths, sym='.', widths = widths, patch_artist=1, manage_xticks=0)
    plt.setp(bp0['fliers'], color='red')
    plt.setp(bp1['fliers'], color='green')
    plt.setp(bp0['boxes'], color='red', alpha = 0.3)
    plt.setp(bp1['boxes'], color='green', alpha = 0.3)
    plt.setp(bp0['whiskers'], color='red')
    plt.setp(bp1['whiskers'], color='green')
    plt.setp(bp0['medians'], color='red')
    plt.setp(bp1['medians'], color='green')
    plt.title('dispersal propensity')
    plt.legend((bp0['boxes'][0], bp1['boxes'][0]), ('random', 'directed'))
    plt.xlabel(atype) 
    if atype == "both":
        plt.xticks(pos, ["random movement", "informed movement"])
    else:
        axes = plt.gca() 
        axes.set_xlim([0,end + step])
    plt.ylabel('dispersal propensity')
    plt.savefig( default_path + "/combinedplots/LH_"+ atype + "/Dispersal_propensity")
    plt.clf()
    
    
    if atype != "varT":
        NB0 = np.load(default_path + '/data/'+ atype + '0/NB.npy')
        NB1 = np.load(default_path + '/data/'+ atype + '1/NB.npy')
        bp0 = plt.boxplot(NB0, positions = pos-0.5*widths, sym='.', widths = widths, patch_artist=1, manage_xticks=0)
        bp1 = plt.boxplot(NB1, positions = pos+0.5*widths, sym='.', widths = widths, patch_artist=1, manage_xticks=0)
        plt.setp(bp0['fliers'], color='red')
        plt.setp(bp1['fliers'], color='green')
        plt.setp(bp0['boxes'], color='red', alpha = 0.3)
        plt.setp(bp1['boxes'], color='green', alpha = 0.3)
        plt.setp(bp0['whiskers'], color='red')
        plt.setp(bp1['whiskers'], color='green')
        plt.setp(bp0['medians'], color='red')
        plt.setp(bp1['medians'], color='green')
        plt.title('niche width')
        plt.legend((bp0['boxes'][0], bp1['boxes'][0]), ('random', 'directed'))
        if atype == "both":
            plt.xticks(pos, ["random movement", "informed movement"])
        else:
            plt.xlabel(atype) 
            axes = plt.gca() 
            axes.set_xlim([0,end + step])
        plt.ylabel('niche width')
        plt.savefig(default_path + "/combinedplots/LH_"+ atype + "/Niche Breadth")   
        plt.clf()
 
    if atype != "dispersal":
        TH0 = np.load(default_path + '/data/'+ atype + '0/TH.npy')
        TH1 = np.load(default_path + '/data/'+ atype + '1/TH.npy')
        bp0 = plt.boxplot(TH0, positions = pos-0.5*widths, sym='.', widths = widths, patch_artist=1, manage_xticks=0)
        bp1 = plt.boxplot(TH1, positions = pos+0.5*widths, sym='.', widths = widths, patch_artist=1, manage_xticks=0)
        plt.setp(bp0['fliers'], color='red')
        plt.setp(bp1['fliers'], color='green')
        plt.setp(bp0['boxes'], color='red', alpha = 0.3)
        plt.setp(bp1['boxes'], color='green', alpha = 0.3)
        plt.setp(bp0['whiskers'], color='red')
        plt.setp(bp1['whiskers'], color='green')
        plt.setp(bp0['medians'], color='red')
        plt.setp(bp1['medians'], color='green')
        plt.title('dispersal threshold')
        plt.ylabel('dispersal threshold')
        plt.legend((bp0['boxes'][0], bp1['boxes'][0]), ('random', 'directed'))
        if atype == "both":
            plt.xticks(pos, ["random movement", "informed movement"])
        else:
            plt.xlabel(atype) 
            axes = plt.gca() 
            axes.set_xlim([0,end + step])
        
        plt.savefig(default_path + "/combinedplots/LH_"+ atype + "/Threshold")   
        plt.clf() 

    
def LH_dispersal(iters, start, end, step):
    '''
    evolved values of life metapopulation and history parameters regressed for dispersal
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
    
    _, axes = plt.subplots(nrows=len(disps)//3+1, ncols=3)
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
        axes[n//3][n%3].hist(nichebr, bins = 20, range = (0,1.2))
        axes[n//3][n%3].set_title(str(initialthreshold))
    plt.show()
    
    
    
    np.save(default_path + '/data/dispersal'+ str(directed) +'/alpha', alpha)
    np.save(default_path + '/data/dispersal'+ str(directed) +'/gamma', gamma)
    np.save(default_path + '/data/dispersal'+ str(directed) +'/beta', beta)
    np.save(default_path + '/data/dispersal'+ str(directed) +'/div', div)
    np.save(default_path + '/data/dispersal'+ str(directed) +'/NB', NB)
    np.save(default_path + '/data/dispersal'+ str(directed) +'/HM', HM)
    np.save(default_path + '/data/dispersal'+ str(directed) +'/popsizes', popsizes)
    np.save(default_path + '/data/dispersal'+ str(directed) +'/dispprop', dispprop)
    
    LHplot(disps, "dispersal", directed, step/2)
      
    
def LH_varT(iters, start, end, step):
    '''
    evolved values of metapopulation and life history parameters regressed for varT
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
    
    _, axes = plt.subplots(nrows=len(varTs)//3+1, ncols=3)
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
        axes[n//3][n%3].hist(thresholds, bins = 20, range = (0,1.2))
        axes[n//3][n%3].set_title(str(initialvarT))
    plt.show()
    
    
    np.save(default_path + '/data/varT'+ str(directed) +'/alpha', alpha)
    np.save(default_path + '/data/varT'+ str(directed) +'/gamma', gamma)
    np.save(default_path + '/data/varT'+ str(directed) +'/beta', beta)
    np.save(default_path + '/data/varT'+ str(directed) +'/div', div)
    np.save(default_path + '/data/varT'+ str(directed) +'/TH', TH)
    np.save(default_path + '/data/varT'+ str(directed) +'/HM', HM)
    np.save(default_path + '/data/varT'+ str(directed) +'/popsizes', popsizes)
    np.save(default_path + '/data/varT'+ str(directed) +'/dispprop', dispprop)
    
    LHplot(varTs, "varT", directed, step/2)
       
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
    meta = Metapopulation(dim,dim,R_res,K_res, initialmaxd, initialvarT, initialthreshold, mutable_dispersal, mutable_variability, directed, cost)
    
    meta.loadlandscape()
    
    for timer in range(MAXTIME):
        print('generation ',timer)
    
        meta.lifecycle()
        print ("popsize: {}\n".format(len(meta.population)))
    return(meta)

def Trait_Distribution_Analyses():
    mutable_threshold, mutable_variability = 1, 1    
    meta = Metapopulation(dim,dim,R_res,K_res, initialmaxd, initialvarT, initialthreshold, mutable_threshold, mutable_variability, directed, cost)
    
    meta.loadlandscape()
    for timer in range(MAXTIME):
        print('generation ',timer)
    
        meta.lifecycle()
        print(len(meta.population))
    meta.Diversity_analysis()
    meta.Landscape_analysis()
    meta.Habitatmatch_analysis()
    if mutable_variability:
        meta.Niche_breadth_analysis()
    if mutable_threshold:
        meta.Dispersal_distance_analysis()
    

def Local_density_reg_in_time(x, y):
    mutable_threshold, mutable_variability = 1, 1    
    meta = Metapopulation(dim,dim,R_res,K_res, initialmaxd, initialvarT, initialthreshold, mutable_threshold, mutable_variability, directed, cost)
    ys = []
    meta.loadlandscape()
    for timer in range(MAXTIME):
        print('generation ',timer)
    
        meta.lifecycle()
        ys.append(np.log(meta.localsizes[-1][x][y])/np.log(meta.localsizes[-2][x][y]))
        print(len(meta.population))
        
    plt.plot(range(MAXTIME), ys)
    plt.title('density regulation at {}, {}'.format(x, y))
    plt.show()
    
def Local_dens_reg_adaptation(iters, start, end, step):
    xs = []
    xs_ = []
    ys = []
    
    mutable_threshold, mutable_variability = 1, 1    
    
    for initialmaxd in range(start, end+1, step):
        
        
        for n in range(iters):
            
            meta = run(MAXTIME, dim,R_res,K_res, initialmaxd, initialvarT, initialthreshold, mutable_threshold, mutable_variability, directed, cost)
            if n == 0 and initialmaxd == start:
                meta.visual.canvas.update()
            densreg = np.log(meta.localsizes[-1])/np.log(meta.localsizes[-2])
            for x in range(meta.max_x):
                for y in range(meta.max_y):
                    listadapt = meta.localadapt[x][y]
                    listadapt_ = meta.localadapt_[x] [y]
                    xs.append(sum(listadapt)/len(listadapt))
                    xs_.append(sum(listadapt_)/len(listadapt_))
                    ys.append(densreg[x, y])
        print(initialmaxd)
    
    plt.scatter(xs, ys)
    plt.title('density regulation (maladaptation of current generation')   
    plt.show()
    
    plt.scatter(xs_, ys)
    plt.title('density regulation (maladaptation of previous generation')   
    plt.show()
    
    
def LH_both(iters):
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
    
    np.save(default_path + '/data/both0' +'/alpha', alpha)
    np.save(default_path + '/data/both0' +'/gamma', gamma)
    np.save(default_path + '/data/both0' +'/beta', beta)
    np.save(default_path + '/data/both0' +'/div', div)
    np.save(default_path + '/data/both0' +'/TH', TH)
    np.save(default_path + '/data/both0' +'/NB', NB)
    np.save(default_path + '/data/both0' +'/HM', HM)
    np.save(default_path + '/data/both0' +'/popsizes', popsizes)
    np.save(default_path + '/data/both0' +'/dispprop', dispprop)
    
    LHbothplot()
    
        
def LHbothplot():
    
    alpha    = np.load(default_path + '/data/both0/alpha.npy')
    gamma    = np.load(default_path + '/data/both0/gamma.npy')
    beta     = np.load(default_path + '/data/both0/beta.npy')
    div      = np.load(default_path + '/data/both0/div.npy')
    HM       = np.load(default_path + '/data/both0/HM.npy')
    popsizes = np.load(default_path + '/data/both0/popsizes.npy')
    dispprop = np.load(default_path + '/data/both0/dispprop.npy')
    
    bp0 = plt.boxplot(alpha, positions = (1, 2), sym='.', widths = 0.5, patch_artist=1, manage_xticks=0)
    plt.setp(bp0['boxes'][0], color='red', alpha = 0.3)
    plt.setp(bp0['whiskers'][0], color='red')
    plt.setp(bp0['fliers'][0], color='red')
    plt.setp(bp0['medians'][0], color='red')
    plt.setp(bp0['boxes'][1], color='green', alpha = 0.3)
    plt.setp(bp0['whiskers'][1], color='green')
    plt.setp(bp0['fliers'][1], color='green')
    plt.setp(bp0['medians'][1], color='green')
    plt.title('local population variability')
    plt.legend((bp0['boxes'][0], bp0['boxes'][1]), ('random', 'directed'))
    plt.xticks((1, 2), ["random movement", "informed movement"])
    plt.ylabel('alpha variability')
    plt.savefig( default_path + "/plots/LH_both0/_Alpha Variability")
    plt.clf()
    
    
    bp0 = plt.boxplot(gamma, positions = (1, 2), sym='.', widths = 0.5, patch_artist=1, manage_xticks=0)
    plt.setp(bp0['boxes'][0], color='red', alpha = 0.3)
    plt.setp(bp0['whiskers'][0], color='red')
    plt.setp(bp0['fliers'][0], color='red')
    plt.setp(bp0['medians'][0], color='red')
    plt.setp(bp0['boxes'][1], color='green', alpha = 0.3)
    plt.setp(bp0['whiskers'][1], color='green')
    plt.setp(bp0['fliers'][1], color='green')
    plt.setp(bp0['medians'][1], color='green')
    plt.title('metapopulation variability') 
    plt.legend((bp0['boxes'][0], bp0['boxes'][1]), ('random', 'directed'))
    plt.xticks((1, 2), ["random movement", "informed movement"])
    plt.ylabel('gamma variability')
    plt.savefig(default_path + "/plots/LH_both0/_Gamma Variability")
    plt.clf()
    
    bp0 = plt.boxplot(beta, positions = (1, 2), sym='.', widths = 0.5, patch_artist=1, manage_xticks=0)
    plt.setp(bp0['boxes'][0], color='red', alpha = 0.3)
    plt.setp(bp0['whiskers'][0], color='red')
    plt.setp(bp0['fliers'][0], color='red')
    plt.setp(bp0['medians'][0], color='red')
    plt.setp(bp0['boxes'][1], color='green', alpha = 0.3)
    plt.setp(bp0['whiskers'][1], color='green')
    plt.setp(bp0['fliers'][1], color='green')
    plt.setp(bp0['medians'][1], color='green')
    plt.title('metapopulation asynchrony') 
    plt.legend((bp0['boxes'][0], bp0['boxes'][1]), ('random', 'directed'))
    plt.xticks((1, 2), ["random movement", "informed movement"])
    plt.ylabel('beta variability')
    plt.savefig(default_path + "/plots/LH_both0/_Metapop Asynchony")
    plt.clf()
    
    
    bp0 = plt.boxplot(div, positions = (1, 2), sym='.', widths = 0.5, patch_artist=1, manage_xticks=0)
    plt.setp(bp0['boxes'][0], color='red', alpha = 0.3)
    plt.setp(bp0['whiskers'][0], color='red')
    plt.setp(bp0['fliers'][0], color='red')
    plt.setp(bp0['medians'][0], color='red')
    plt.setp(bp0['boxes'][1], color='green', alpha = 0.3)
    plt.setp(bp0['whiskers'][1], color='green')
    plt.setp(bp0['fliers'][1], color='green')
    plt.setp(bp0['medians'][1], color='green')
    plt.title('muT')
    plt.legend((bp0['boxes'][0], bp0['boxes'][1]), ('random', 'directed'))
    plt.xticks((1, 2), ["random movement", "informed movement"])
    plt.savefig(default_path + "/plots/LH_both0/Trait Value")
    plt.clf()
    
    bp0 = plt.boxplot(HM, positions = (1, 2), sym='.', widths = 0.5, patch_artist=1, manage_xticks=0)
    plt.setp(bp0['boxes'][0], color='red', alpha = 0.3)
    plt.setp(bp0['whiskers'][0], color='red')
    plt.setp(bp0['fliers'][0], color='red')
    plt.setp(bp0['medians'][0], color='red')
    plt.setp(bp0['boxes'][1], color='green', alpha = 0.3)
    plt.setp(bp0['whiskers'][1], color='green')
    plt.setp(bp0['fliers'][1], color='green')
    plt.setp(bp0['medians'][1], color='green')
    plt.title('habitat mismatch')
    plt.legend((bp0['boxes'][0], bp0['boxes'][1]), ('random', 'directed'))
    plt.xticks((1, 2), ["random movement", "informed movement"])
    plt.ylabel('habitat mismatch')
    plt.savefig(default_path + "/plots/LH_both0/Habitat Mismatch")
    plt.clf()
    
    bp0 = plt.boxplot(popsizes, positions = (1, 2), sym='.', widths = 0.5, patch_artist=1, manage_xticks=0)
    plt.setp(bp0['boxes'][0], color='red', alpha = 0.3)
    plt.setp(bp0['whiskers'][0], color='red')
    plt.setp(bp0['fliers'][0], color='red')
    plt.setp(bp0['medians'][0], color='red')
    plt.setp(bp0['boxes'][1], color='green', alpha = 0.3)
    plt.setp(bp0['whiskers'][1], color='green')
    plt.setp(bp0['fliers'][1], color='green')
    plt.setp(bp0['medians'][1], color='green')
    plt.title('metapopulation size') 
    plt.xticks((1, 2), ["random movement", "informed movement"])
    plt.ylabel('population size') 
    plt.savefig(default_path + "/plots/LH_both0/Population Sizes")
    plt.clf()
    
    
    bp0 = plt.boxplot(dispprop, positions = (1, 2), sym='.', widths = 0.5, patch_artist=1, manage_xticks=0)
    plt.setp(bp0['boxes'][0], color='red', alpha = 0.3)
    plt.setp(bp0['whiskers'][0], color='red')
    plt.setp(bp0['fliers'][0], color='red')
    plt.setp(bp0['medians'][0], color='red')
    plt.setp(bp0['boxes'][1], color='green', alpha = 0.3)
    plt.setp(bp0['whiskers'][1], color='green')
    plt.setp(bp0['fliers'][1], color='green')
    plt.setp(bp0['medians'][1], color='green')
    plt.title('dispersal propensity')
    plt.xticks((1, 2), ["random movement", "informed movement"])
    plt.ylabel('dispersal propensity')
    plt.savefig( default_path + "/plots/LH_both0/Dispersal_propensity")
    plt.clf()
    
    
    NB = np.load(default_path + '/data/both0/NB.npy')
    bp0 = plt.boxplot(NB, positions = (1, 2), sym='.', widths = 0.5, patch_artist=1, manage_xticks=0)
    plt.setp(bp0['boxes'][0], color='red', alpha = 0.3)
    plt.setp(bp0['whiskers'][0], color='red')
    plt.setp(bp0['fliers'][0], color='red')
    plt.setp(bp0['medians'][0], color='red')
    plt.setp(bp0['boxes'][1], color='green', alpha = 0.3)
    plt.setp(bp0['whiskers'][1], color='green')
    plt.setp(bp0['fliers'][1], color='green')
    plt.setp(bp0['medians'][1], color='green')
    plt.title('niche width')
    plt.legend((bp0['boxes'][0], bp0['boxes'][1]), ('random', 'directed'))
    plt.xticks((1, 2), ["random movement", "informed movement"])
    plt.ylabel('niche width')
    plt.savefig(default_path + "/plots/LH_both0/Niche Breadth")   
    plt.clf()

    TH = np.load(default_path + '/data/both0/TH.npy')
    bp0 = plt.boxplot(TH, positions = (1, 2), sym='.', widths = 0.5, patch_artist=1, manage_xticks=0)
    plt.setp(bp0['boxes'], color='red', alpha = 0.3)
    plt.setp(bp0['whiskers'][0], color='red')
    plt.setp(bp0['fliers'][0], color='red')
    plt.setp(bp0['medians'][0], color='red')
    plt.setp(bp0['boxes'][1], color='green', alpha = 0.3)
    plt.setp(bp0['whiskers'][1], color='green')
    plt.setp(bp0['fliers'][1], color='green')
    plt.setp(bp0['medians'][1], color='green')
    plt.title('dispersal threshold')
    plt.legend((bp0['boxes'][0], bp0['boxes'][1]), ('random', 'directed'))
    plt.ylabel('dispersal threshold')
    plt.xticks((1, 2), ["random movement", "informed movement"])
    plt.savefig(default_path + "/plots/LH_both0/Threshold")   
    plt.clf() 
 
   
'''
Default PARAMETERS
'''

MAXTIME=50  #50
dim = 8
R_res = 0.25
K_res = 1  
initialmaxd = 2
initialvarT = 0.05
initialthreshold = 0.1
directed = 1
cost = 0 #cost of directed dispersal

iters = 30      #30   
start = 0.025   #20 steps in total start-stop
end = 0.5
step = 0.025


LH_both(iters)
#LHplot(np.arange(start, end+step, step), 'dispersal', directed, step/2)
#LHcombinedplot(np.arange(start, end+step, step), 'dispersal', step/2)