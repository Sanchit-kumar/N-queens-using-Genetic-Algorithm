import random
import copy
import math
import numpy as np
import time
from queue import PriorityQueue
import matplotlib.pyplot as plt #for ploting graph

best_fitness=[]
crossover_prob=0.6   #Lets consider these as the default values
mutation_prob=0.03
#this function calculates number of attacking pairs
def fitness(individual): ##IN LINEAR TIME
    n = len(individual)
    board=[x-1 for x in individual]

    row_freq=n*[0]
    main_diag_freq = (2*n)*[0] 
    secondary_diag_freq = (2*n)*[0] 
    
    for i in range(n):
        row_freq[board[i]] += 1
        main_diag_freq[board[i] + i] += 1
        secondary_diag_freq[n - board[i] + i] += 1

    conflicts= 0
    for i in range(2*n):
        if i < n:
            conflicts += (row_freq[i] * (row_freq[i]-1)) / 2
        conflicts += (main_diag_freq[i] * (main_diag_freq[i]-1)) / 2
        conflicts += (secondary_diag_freq[i] * (secondary_diag_freq[i]-1)) / 2
    return int(conflicts)

def crossover(individual1, individual2): 
    n=len(individual1)
    partition=random.randint(0,n-1)   #new genese will randomly take few partion of parent 1 & remaining from parent2
    return individual1[0:partition] + individual2[partition:n]


#will manage latter for probability of 0.03 if required
def mutation(individual):
    n = len(individual)
    r_index =random.randint(0,n-1) #select random index for mutation
    r_value =random.randint(1,n) 
    individual[r_index]=r_value #changing random index with some random value
    return individual


def generate_individual(n):
    result = list(range(1, n + 1))
    np.random.shuffle(result)
    return result
    
def display_board(individual): #will display the board configuration
    n=len(individual)
    for i in range(n):
        print("")
        for j in range(n):
            if (i+1)==individual[j]:
                print('Q ',end=" ")
            else:
                print('x ',end=" ")
    print("")
z=0
class Genetic(object):

    def __init__(self, n ,pop_size):
        #initializing a random individuals with size of initial population entered by user
        self.fit=999999999999
        self.queens = []
        self.queens_fitness=PriorityQueue()
        for i in range(pop_size):
            self.queens.append(generate_individual(n))
            self.fit=min(self.fit,fitness(self.queens[i])) 

    #generating individuals for a single iteration of algorithm
    def generate_population(self, random_selections=5):
        global crossover_prob,mutation_prob
        global best_fitness,z
        candid_parents = []
        candid_fitness = []
        
        #getting individuals from queens randomly for an iteration
        for i in range(random_selections):
            candid_parents.append(self.queens[random.randint(0, len(self.queens) - 1)])
            candid_fitness.append(fitness(candid_parents[i]))
        
        
        pq=PriorityQueue()  #priority queue
        
        for i in range(random_selections):
           pq.put((candid_fitness[i],candid_parents[i]))
        
        #sorted_fitness = copy.deepcopy(candid_fitness)   #no reference, just copy
        #sort the fitnesses of individuals
       
       		#IN PRIORITY QUEUE, EVERYTHING IS ALREADY SORTED
       	
        #getting 2 first individuals(min attackings)
        
        item=pq.get()    #taking minimum conflicting individual from priority queue
        fittest=item[0]  #the minimum conflict pair amoung all
        individual1=item[1];  #fitness with minimum conflict pair will be prefered
        
        item=pq.get()
        individual2=item[1];
        
        #print("Parents",individual1,individual2)
        
        #crossover the two parents
        child=individual1  #if better child won't found, child will not be added to the populatoin
        if random.random() < crossover_prob:  #probability check for crossover
        	child=crossover(individual1, individual2)
        	#print("child",child)
        
        
        # mutation
        if random.random() < mutation_prob:   #Probability check for muattion
        	child=mutation(child)
        	#print("child after mutation:",child)
        #in code below check if each child is better than each one of queens individuals, set that individual the new child
        
        if fitness(child)<fittest:      #child will be better than all if it have minimum number of conflict pairs
        	self.queens.append(child)
        	self.fit=min(self.fit,fitness(child))
        #just for predicting the graph
        #best_fitness.append((min([fitness(i) for i in self.queens])))
        #z=self.queens_fitness.get()
        best_fitness.append(self.fit)
        #self.queens_fitness.put(z) #putting item back to the queue

    def finished(self): 
   	
        for i in self.queens:
            #we check if for each queen there is no attacking(cause this algorithm should work for n queen,
            # it was easier to use attacking pairs for fitness instead of non-attacking)
            conflicts=fitness(i)
            #print("Positions:",i,"Conflict:",conflicts)
            if conflicts==0:
            	break
        res=[]
        if conflicts==0:
        	res.append(1)
        else:
        	res.append(0)
        res.append(i)
        return res
            

    def start(self, random_selections=5): 
        #generate new population and start algorithm until number of attacking pairs is zero
        while not self.finished()[0]:
            self.generate_population(random_selections)
        final_state = self.finished()
        print(('Solution : ' + str(final_state[1])))
        display_board(final_state[1]) #added line for displaying the board
        


#******************** N-Queen Problem With GA Algorithm ***********************
n=(int)(input('Enter the number of queens of N:'))
initial_population=(int)(input('Enter population size:'))
crossover_prob=(float)(input('Enter crossover probability:'))
mutation_prob=(float)(input('Enter mutation probability:'))
"""
n=8
initial_population=20
crossover_prob=1
mutation_prob=0.03
"""
start_time=time.time()
algorithm = Genetic(n=n,pop_size=initial_population)
algorithm.start()
print("\nTotal time taken:",time.time()-start_time)
#print(best_fitness)
if len(best_fitness)!=0:
	plt.title("Generation Vs Fitness, Initial Population: %d \n Queens: %d Crossover Prob:%.3f Mutation Prob:%.3f"%(initial_population,n,crossover_prob,mutation_prob))
	plt.ylabel("Fitness (minimum conflict)")
	plt.xlabel("Generation")
	plt.plot(best_fitness)
	plt.show()	
else:
	print("\n Plot not generated as solution found on the present generation (1st generation parents)")
