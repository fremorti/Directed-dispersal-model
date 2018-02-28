'''
Created on Aug 9, 2016

@author: frederik
added incomplete habitat choice
'''

import random as rnd
import numpy as np
import math as math

class Individual:
    '''Class that regulates individuals and their properties'''
    def __init__(self,
                 x,
                 y,
                 muT,
                 varT,
                 maxd,
                 threshold,
                 settlement,
                 cost_of_disp):
        '''Initialization'''
        self.x = x
        self.y = y
        self.muT=muT        #Habitat value of the optimal habitat?
        self.varT=varT      #Niche breadth, how much fitness declines away from optimality
        self.maxd = maxd    #max dispersion distance
        self.threshold = threshold
        self.amax=0.05    #maximum encounter rateoriginal 0.05 maybe correct as exp(ct*0.5)*0.5
        self.ct=1 #1 --> ct waarvoor de oppervlakte onder e^-(ct*varT) * e^-((x-0.5)^2/varT^2) voor x [0, 1] gelijk is voor varT 0.025 tot 0.5        #strength of the trade-off original 1
        self.h=0.2          #handling time, default:0.2
        self.settlement = settlement
        self.sigma=300 - settlement*cost_of_disp         #conversion factor
        
        
    def move(self, max_x, max_y,  env):
            
        g = [x%len(env) for x in range(self.x-self.maxd, self.x+self.maxd+1)]
        h = [y%len(env[0]) for y in range(self.y-self.maxd, self.y+self.maxd+1)]
        rnd.shuffle(g)
        rnd.shuffle(h)
        
        if self.settlement:
            diff = 1   
            for x in g:
                for y in h:
                    diff_ = abs(self.muT - env[x][y])
                    if diff_ < diff and (self.x != x or self.y != y):
                        diff = diff_
                        x_ = x
                        y_ = y
            self.x, self.y = x_, y_
        
        else:
            while self.x == g[0] and self.y == h[0]:
                rnd.shuffle(g)
                rnd.shuffle(h)
            self.x, self.y = g[0], h[0]
        

    def mutation(self,rate, md, mv):
        if rnd.random()<rate and md :
            self.threshold=abs(np.random.normal(self.threshold,0.1))
            self.threshold = 2-self.threshold if self.threshold > 1 else self.threshold
            
        if rnd.random()<rate:
            self.muT=np.random.normal(self.muT,0.1)
        if rnd.random()<rate and mv :
            self.varT=abs(np.random.normal(self.varT,0.1))
            self.varT = 2-self.varT if self.varT > 1 else self.varT
        
                                                                                                       
    def resource_use(self,localhabitat,R):
        Gamma=math.exp(-self.ct*self.varT)    #!!!aangepaste relatie met varT   
        Wij=Gamma*math.exp(-(math.pow((self.muT-localhabitat),2)/math.pow((self.varT),2)))  #, 3.2
        Alpha_ij=self.amax*Wij   #
        Ri=(Alpha_ij*R/(1+(self.h+self.h*Alpha_ij*R)))  #resources used, eq 3#    
        
        return Ri
        
                  
    def fitness(self,localhabitat,R):        
        #draw a random number of offspring with an average proportionate with the resources used
        return np.random.poisson(self.sigma*self.resource_use(localhabitat, R))
        
   
class Metapopulation:
    '''Contains the whole population, regulates daily affairs'''
    def __init__(self,
                 max_x,
                 max_y,
                 res_R,
                 res_K,
                 initialmaxd,
                 initialvarT,
                 initialthreshold,
                 mutable_dispersal,
                 mutable_var,
                 departure, 
                 settlement,
                 cod):
        '''Initialization'''           
        self.max_x = max_x
        self.max_y = max_y
        self.res_R = res_R  
        self.res_K = res_K    
        self.initialmaxd = initialmaxd
        self.initialthreshold = initialthreshold
        self.initvarT = initialvarT
        self.mv = mutable_var
        self.md = mutable_dispersal
        self.departure = departure
        self.settlement = settlement
        self.environment = np.zeros((self.max_x,self.max_y))
        self.resources = np.zeros((self.max_x,self.max_y))
        self.population = []
        self.localsizes = []   #list of population sizes at each location for each generation
        self.cod = cod #cost of settlement dispersal (binary: either a cost or no cost)
        self.disp_prop = 0
        self.pros_prop = 0
        self.initialize_pop()
        
    def initialize_pop(self):
        '''Initialize individuals'''
        startpop = 70000
        
        for _ in range(startpop):
            x, y, muT = rnd.randint(0,(self.max_x-1)), rnd.randint(0,(self.max_y-1)), rnd.random()
            self.population.append(Individual(x,
                                              y,
                                              muT,
                                              (self.initvarT if self.initvarT else 0.5*rnd.random()) , 
                                              self.initialmaxd,
                                              (self.initialthreshold if self.initialthreshold else (5 if self.departure else 1)*rnd.random()),
                                              self.settlement,
                                              self.cod))

                                             
    def lifecycle(self):   
        '''all actions during one generation for the metapopulation'''
        
        #resources grow
        self.resources += self.res_R*(1-self.resources/self.res_K)

        #replace generation with new one
        oldpop = self.population[:]
        del self.population[:]
        
        #randomize the order in which individuals will perfom their actions
        rnd.shuffle(oldpop)
        
        movenumber, prospectnumber = 0, 0
        oldpopsize = len(oldpop)        #old metapopulation size
        newlocalsizes= np.zeros((self.max_x,self.max_y))
        
        for ind in oldpop:
            
            #mutate
            ind.mutation(0.01, self.md, self.mv)
                     
            #move
            #calculate how much resources the individual needs to reproduce
            necessary_resources=ind.resource_use(self.environment[ind.x,ind.y],self.resources[ind.x, ind.y])
            #decide to move: according to available resources when there is a departure decision, random when there is not
            if (ind.sigma*necessary_resources if self.departure else rnd.random()) < ind.threshold:
                x_, y_ = ind.x, ind.y 
                ind.move(self.max_x, self.max_y, self.environment)
                prospectnumber += 1
                if not(ind.x == x_ and ind.y == y_):
                    movenumber += 1
                
            #reproduce
            necessary_resources=ind.resource_use(self.environment[ind.x,ind.y],self.resources[ind.x, ind.y])
            #if there are enough resources present locally...
            if necessary_resources<self.resources[ind.x,ind.y]:
                #...deplete resources
                self.resources[ind.x,ind.y]-=necessary_resources   
                #reproduce according to the fitness value
                Fitness=ind.fitness(self.environment[ind.x,ind.y],self.resources[ind.x, ind.y])
                newlocalsizes[ind.x, ind.y] += Fitness
                for _ in range(Fitness):
                    #add a new individual with the same traits as its parent to the new population
                    self.population.append(Individual(ind.x,
                                                      ind.y,
                                                      ind.muT,
                                                      ind.varT,
                                                      ind.maxd,
                                                      ind.threshold,
                                                      self.settlement,
                                                      self.cod))
    
            else:
                #deplete resources, but no reproduction (fitness dependent on environmnet only, not resources)
                self.resources[ind.x, ind.y] = 0
        self.disp_prop = movenumber/oldpopsize
        self.pros_prop = prospectnumber/oldpopsize
        #calculate local population sizes of this generation
        self.localsizes.append(newlocalsizes)
         
    def loadlandscape(self):
            for x in range(self.max_x):
                for y in range(self.max_y):
                    self.environment[x,y]=rnd.random()