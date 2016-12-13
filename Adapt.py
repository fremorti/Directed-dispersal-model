'''
Created on Aug 9, 2016

@author: frederik
Added a cost of developing the ability of directed movement as a lower conversion rate of resources into offspring
line 85: self.sigma=300                     without cost of directed dispersal
         self.sigma=300 - directed*100      with cost of directed dispersal
'''

import random as rnd
import tkinter as tk
import numpy as np
import matplotlib.pyplot as plt
import math as math


class Visual:
    '''This class arranges the visual output.'''
    def __init__(self, max_x, max_y):
        '''Initialize the visual class'''
        self.zoom = 15
        self.max_x = max_x
        self.max_y = max_y
        self.root = tk.Tk()
        self.canvas = tk.Canvas(self.root, 
                                width =  self.max_x * self.zoom, 
                                height = self.max_y * self.zoom) #create window
        self.canvas.pack()
        self.canvas.config(background = 'white')
        self.squares = np.empty((self.max_x, self.max_y),dtype=object)
        self.initialize_squares()
                                       
        
    def color_square(self, resources, x, y):
        '''Changes the color of the square'''        
        color = (resources)/float(100)
        if color < 0:
            color = 0
        elif color > 1:
            color = 1  
        green = int(255 * color)
        red = 255 - green        
        blue = 0
        rgb = red, green, blue     
        hex_code = '#%02x%02x%02x' % rgb        
        self.canvas.itemconfigure(self.squares[x, y],fill=str(hex_code))
        
    def initialize_squares(self):
        '''returns a square (drawing object)'''
        for x in range(self.max_x):
            for y in range(self.max_y):
                self.squares[x, y] = self.canvas.create_rectangle(self.zoom * x,
                                                     self.zoom * y, 
                                                     self.zoom * x + self.zoom,
                                                     self.zoom * y + self.zoom,
                                                     outline = 'black', 
                                                     fill = 'black')
                


class Individual:
    '''Class that regulates individuals and their properties'''
    def __init__(self,
                 x,
                 y,
                 muT,
                 varT,
                 #Pdisp,
                 maxd,
                 threshold,
                 directed,
                 cost_of_disp):
        '''Initialization'''
        self.x = x
        self.y = y
        self.muT=muT        #Habitat value of the optimal habitat?
        self.varT=varT      #Niche breadth, how much fitness declines away from optimality
        #self.disp=Pdisp     #dispersion chance
        self.maxd = maxd    #max dispersion distance
        self.threshold = threshold
        self.amax=0.05      #maximum encounter rate
        self.ct=1           #strength of the trade-off
        self.h=0.2          #handling time
        self.directed = directed
           
        self.sigma=300 - directed*100*cost_of_disp         #conversion factor
        

        
    def move(self, max_x, max_y,  env):
        
        
        #print(self.disp)
        '''if rnd.random()<self.disp:
            dx=1
            dy=1
            if rnd.random()<0.5:
                dx=-1
                
            if rnd.random()<0.5:
                dy=-1'''
        
            
            
        g = [x for x in range(self.x-self.maxd, self.x+self.maxd+1) if 0<=x<len(env)]
        h = [y for y in range(self.y-self.maxd, self.y+self.maxd+1) if 0<=y<len(env)]
        rnd.shuffle(g)
        rnd.shuffle(h)
        
        if self.directed :
            diff = 1   
            for x in g:
                for y in h:
                    diff_ = abs(self.muT - env[x][y])
                    if diff_ < diff:
                        diff = diff_
                        self.x = x
                        self.y = y
        
        else:
                
            
            #unbound movement
            self.x = g[0]
            self.y = h[0]
        


        

    
    
    def mutation(self,rate, md, mv):
        if rnd.random()<rate and md : 
            self.threshold=abs(np.random.normal(self.muT,0.1))
            self.threshold = 1 if self.threshold > 1 else self.threshold
            
        if rnd.random()<rate :     
            self.muT=abs(np.random.normal(self.muT,0.1))
        if rnd.random()<rate and mv : 
            self.varT=rnd.random()
            '''!!!!!!'''
        
                                                    
                                                     
    def resource_use(self,localhabitat,R):
        Gamma=math.exp(-self.ct*self.varT)    #max fitness with trade off function 3.3
        Wij=Gamma*math.exp(-(math.pow((self.muT-localhabitat),2)/math.pow((self.varT),2)))  #, 3.2
        Alpha_ij=self.amax*Wij   #
        Ri=(Alpha_ij*R/(1+(self.h+self.h*Alpha_ij*R)))  #resources used, eq 3#    
        
        return Ri        
        
        
              
    def fitness(self,localhabitat,R):
         
        Ri = self.resource_use(localhabitat, R)
        
        Lambda=np.random.poisson(self.sigma*Ri)
       
        return Lambda            
        
   
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
                 directed,
                 cod):
        '''Initialization'''           
        self.max_x = max_x
        self.max_y = max_y
        self.visual = Visual(self.max_x, self.max_y)
        self.res_R = res_R  
        self.res_K = res_K    
        self.initialmaxd = initialmaxd
        self.initialthreshold = initialthreshold
        self.initvarT = initialvarT
        self.mv = mutable_var
        self.md = mutable_dispersal
        self.directed = directed
        self.competition = [] # list of resources/ind per lifecycle

        self.TemporalVariance=0.1
        self.environment = np.zeros((self.max_x,self.max_y))
        self.resources = np.zeros((self.max_x,self.max_y))
        self.localsizes = []   #list of population sizes at each location for each generation
        self.localadapt = 'C'
        self.localadapt_ = 'D'
        
        self.population = []
        self.cod = cod #cost of directed dispersal (binary: either a cost or no cost)
        
        self.initialize_pop()
        self.CV = []
        self.disp_prop = 0
        
    def initialize_pop(self):
        '''Initialize individuals'''
        startpop = 1000
        
        for _ in range(startpop):
            x = rnd.randint(0,(self.max_x-1))
            y = rnd.randint(0,(self.max_y-1))
            muT=rnd.random()
            varT = self.initvarT
            #Pdisp=0.1
            self.population.append(Individual(x,
                                              y,
                                              muT,
                                              varT,
                                              #Pdisp, 
                                              self.initialmaxd,
                                              self.initialthreshold,
                                              self.directed,
                                              self.cod))

                                             
    def lifecycle(self):   
            
        #resources grow
        self.resources += self.res_R*(1-self.resources/self.res_K)  
        
              
        #mutate
        for ind in self.population:
            ind.mutation(0.01, self.md, self.mv)     
        
        self.localadapt_ = self.localadapt
        self.localadapt = [[[] for _ in range(self.max_y)] for _ in range(self.max_x)]
        for ind in self.population:
            self.localadapt[ind.x][ind.y].append(abs(ind.muT - self.environment[ind.x,ind.y]))
        
        #individuals reproduce
        oldpop = self.population[:]
        del self.population[:]
        
      
        rnd.shuffle(oldpop)
        movenumber = 0
        oldpopsize = len(oldpop)
        for ind in oldpop:
            
            #In which habitat is the individual
            localenvironment=self.environment[ind.x,ind.y]
            #What is the resource density
            R = self.resources[ind.x, ind.y]
            #calculate how much resources the individual needs to reproduce
            necessary_resources=ind.resource_use(localenvironment,R)
            if ind.sigma*necessary_resources < ind.threshold:
                ind.move(self.max_x, self.max_y, self.environment)
                movenumber += 1
                
            localenvironment=self.environment[ind.x,ind.y]
            Fitness=ind.fitness(localenvironment,R)

           
            
            
            if necessary_resources<self.resources[ind.x,ind.y]:
                #deplete resources if present
                self.resources[ind.x,ind.y]-=necessary_resources   
                #survive and reproduce
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
        indcoords = [(ind.x, ind.y) for ind in self.population] #array with all coords of the individuals
        localpopsize = [indcoords.count((x, y)) for y in range(self.max_y) for x in range(self.max_x) if indcoords.count((x, y))]
        
        self.CV.append(np.var(localpopsize)/np.average(localpopsize))
                     
        self.localsizes.append(np.array([[[(ind.x, ind.y) for ind in self.population].count((x, y)) for x in range(self.max_x)] for y in range(self.max_y)]))
           
        #=======================================================================
        # for x in range(self.max_x):
        #             for y in range(self.max_y):
        #                 #self.visual.color_square(self.environment[x,y]*100, x, y)   
        #                 self.visual.color_square(self.resources[x,y]*50, x, y) 
        # print ('endpopulation',len(self.population))
        # self.visual.canvas.update() 
        #=======================================================================
        

        
    def Landscape_analysis(self):
        '''title='Landscape_output.txt'
       
        output=open(title,'w')
        output.write('x'+'\t'+'y'+'\t'+'habitatcolour'+'\t'+'Resources'+'\n')
        
        for x in range(self.max_x):
            for y in range(self.max_y):
                output.write(str(x)+'\t'+str(y)+'\t'+str(self.environment[x,y])+'\t'+str(self.resources[x,y])+'\n')
        
        output.close()'''
        
        data = [self.environment[y][x] for x in range(self.max_x) for y in range(self.max_y)]
        plt.hist(data, bins = np.arange(0, 1, 0.05))
        plt.title('habitat density')
        plt.savefig("C:/Users/frederik/Documents/Integrated research project/plots/Trait Distribution/Landscape Distribution.jpeg")
        plt.clf()
        
        
    def Diversity_analysis(self):
        '''title='IndividualTrait_output.txt'
       
        output=open(title,'w')
        output.write('x'+'\t'+'y'+'\t'+'habitatcolour'+'\t'+'Resources'+'\t'+'muT'+'\t'+'varT'+'\t'+'disp'+'\n')
        
        print(len(self.population))
        for ind in self.population:
            output.write(str(ind.x)+'\t'+str(ind.y)+'\t'+str(self.environment[ind.x,ind.y])+'\t'+str(self.resources[ind.x,ind.y])+'\t'+str(ind.muT)+'\t'+str(ind.varT)+'\t'+str(ind.disp)+'\n')
        
        output.close()'''    
        
        
        data = [ind.muT for ind in self.population]
        plt.hist(data, bins = np.arange(0, 1, 0.05))
        plt.title('muT density')
        plt.savefig("C:/Users/frederik/Documents/Integrated research project/plots/Trait Distribution/muT Distribution.jpeg")
        plt.clf()
    
    def Niche_breadth_analysis(self):
        
        data = [ind.varT for ind in self.population]
        plt.hist(data, bins = np.arange(0, 1, 0.05))
        plt.title('varT density')
        plt.savefig("C:/Users/frederik/Documents/Integrated research project/plots/Trait Distribution/Niche Breath Distribution.jpeg")
        plt.clf()
        
    def Dispersal_distance_analysis(self):
        
        data = [ind.maxd for ind in self.population]
        plt.hist(data, bins = range(0, max(data)+2, 1))
        plt.title('max. distance density')
        plt.savefig("C:/Users/frederik/Documents/Integrated research project/plots/Trait Distribution/Dispersal Distance Distribution.jpeg")
        plt.clf() 
    
   

    def Habitatmatch_analysis(self):
        
        data = np.array([ind.muT-self.environment[ind.y][ind.x] for ind in self.population])
        plt.hist(data, bins = np.arange(0, 1, 0.05))
        plt.title('habitat match')
        plt.savefig("C:/Users/frederik/Documents/Integrated research project/plots/Trait Distribution/Habitat Match Distribution.jpeg")
        plt.clf()
        
    
    def loadlandscape(self):
        rando = 1
        
        if rando:
            for x in range(self.max_x):
                for y in range(self.max_y):
                    self.environment[x,y]=rnd.random()
                    self.visual.color_square(self.environment[x,y]*100, x, y)
                    """
            for x in range(self.max_x):
                for y in range(self.max_y):
                    if rnd.random()>0.5:
                        self.environment[x,y] = 0.7
                    else:
                        self.environment[x,y] = 0.2
                    self.visual.color_square(self.environment[x,y]*100, x, y)"""
        
        else:    
            FileToLaoad='MatrixLandscape16.txt'
            
            table = np.loadtxt(FileToLaoad)
            print(table)
            
            for x in range(self.max_x):
                for y in range(self.max_y):
                    basicquality=table[x,y]
                    self.environment[x,y]=basicquality
                    self.visual.color_square(self.environment[x,y]*100, x, y)
                    
                    '''testtest
        test
        test
        test
        test
        test
        '''