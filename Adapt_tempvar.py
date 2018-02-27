'''
Created on 23 apr. 2017

@author: fremorti
'''
'''
Created on Aug 9, 2016

@author: frederik
Added a cost of developing the ability of directed movement as a lower conversion rate of resources into offspring
line 85: self.sigma=300                     without cost of directed dispersal
         self.sigma=300 - directed*100      with cost of directed dispersal
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
                 directed,
                 cost_of_disp):
        '''Initialization'''
        self.x = x
        self.y = y
        self.muT=muT        #Habitat value of the optimal habitat?
        self.varT=varT      #Niche breadth, how much fitness declines away from optimality
        self.maxd = maxd    #max dispersion distance
        self.threshold = threshold
        self.amax=0.05      #maximum encounter rate
        self.ct=1           #strength of the trade-off
        self.h=0.2          #handling time
        self.directed = directed
        self.sigma=300 - directed*cost_of_disp         #conversion factor
        
        
    def move(self, max_x, max_y,  env):    
        #generate x and y values in range of the individual    
        g = [x%len(env) for x in range(self.x-self.maxd, self.x+self.maxd+1)]
        h = [y%len(env[0]) for y in range(self.y-self.maxd, self.y+self.maxd+1)]
        rnd.shuffle(g)
        rnd.shuffle(h)
        
        if self.directed :
            #look for the location in the range with the lowest difference between environment and muT
            diff = abs(self.muT - env[self.x][self.y])   
            for x in g:
                for y in h:
                    diff_ = abs(self.muT - env[x][y])
                    if diff_ < diff:
                        diff = diff_
                        self.x = x
                        self.y = y
        
        else:
            self.x = g[0]
            self.y = h[0]
        

    def mutation(self,rate, md, mv):
        if rnd.random()<rate and md :
            self.threshold=abs(np.random.normal(self.threshold,0.1))
            self.threshold = 1 if self.threshold > 1 else self.threshold
            
        if rnd.random()<rate :
            self.muT=np.random.normal(self.muT,0.1)
        if rnd.random()<rate and mv :
            self.varT=rnd.random()
        
                                                                                                       
    def resource_use(self,localhabitat,R):
        Gamma=math.exp(-self.ct*self.varT)    #max fitness with trade off function 3.3   
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
                 directed,
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
        self.directed = directed
        self.environment = np.zeros((self.max_x,self.max_y))
        self.resources = np.zeros((self.max_x,self.max_y))
        self.population = []
        self.localsizes = []   #list of population sizes at each location for each generation
        self.cod = cod #cost of directed dispersal (binary: either a cost or no cost)
        self.disp_prop = 0
        self.pros_prop = 0
        self.initialize_pop()
        
    def initialize_pop(self):
        '''Initialize individuals'''
        startpop = 1000
        
        for _ in range(startpop):
            x, y, muT = rnd.randint(0,(self.max_x-1)), rnd.randint(0,(self.max_y-1)), rnd.random()
            varT = self.initvarT
            self.population.append(Individual(x,
                                              y,
                                              muT,
                                              varT, 
                                              self.initialmaxd,
                                              self.initialthreshold,
                                              self.directed,
                                              self.cod))

                                             
    def lifecycle(self):   
        
        for x in range(self.max_x):
                for y in range(self.max_y):
                    self.environment[x,y] =  (self.environment[x,y] + np.random.normal(0,0.1))%1
                        
        #resources grow
        self.resources += self.res_R*(1-self.resources/self.res_K)

        #replace generation with new one
        oldpop = self.population[:]
        del self.population[:]

        rnd.shuffle(oldpop)
        movenumber, prospectnumber = 0, 0
        oldpopsize = len(oldpop)
        for ind in oldpop:
            
            #mutate
            ind.mutation(0.01, self.md, self.mv)
                     
            
            #move
            #calculate how much resources the individual needs to reproduce
            necessary_resources=ind.resource_use(self.environment[ind.x,ind.y],self.resources[ind.x, ind.y])
            #decide to move according to available resources
            if (ind.sigma*necessary_resources if self.departure else rnd.random()) < ind.threshold:
                x_, y_ = ind.x, ind.y 
                ind.move(self.max_x, self.max_y, self.environment)
                prospectnumber += 1
                if not(ind.x == x_ and ind.y == y_):
                    movenumber += 1
            
            
            #reproduce
            #if there are enough resources present locally
            necessary_resources=ind.resource_use(self.environment[ind.x,ind.y],self.resources[ind.x, ind.y])
            if necessary_resources<self.resources[ind.x,ind.y]:
                #deplete resources if present
                self.resources[ind.x,ind.y]-=necessary_resources   
                #succesfully reproduce according to your fitness value
                Fitness=ind.fitness(self.environment[ind.x,ind.y],self.resources[ind.x, ind.y])
                for _ in range(Fitness):
                    self.population.append(Individual(ind.x,
                                                      ind.y,
                                                      ind.muT,
                                                      ind.varT,
                                                      #Pdisp
                                                      ind.maxd,
                                                      ind.threshold,
                                                      self.directed,
                                                      self.cod))
    
            else:
                #deplete resources, but no reproduction (fitness dependent on environmnet only, not resources)
                self.resources[ind.x, ind.y] = 0

        self.disp_prop = movenumber/oldpopsize
        self.pros_prop = prospectnumber/oldpopsize
        self.localsizes.append(np.array([[[(ind.x, ind.y) for ind in self.population].count((x, y)) for x in range(self.max_x)] for y in range(self.max_y)]))
 
    def loadlandscape(self):
            for x in range(self.max_x):
                for y in range(self.max_y):
                    self.environment[x,y]=rnd.random()