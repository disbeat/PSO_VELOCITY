"""
my_pso.py
Implementation of the standard PSO (CEC 2007). Global neighborhood: each particle
has all the others as neighbors.
Ernesto Costa, March 2010.
"""

# imports
from copy import deepcopy
from math import sqrt, sin, cos, pi
from pylab import *
from random import uniform

filename = "out.csv"

OUTPUT = open(filename, 'w')


def saveResult(result, run, weight_inertia, phi_1, phi_2, type_problem):
    best = type_problem([generation[0] for generation in result])
    best_average = type_problem([generation[1] for generation in result])
    OUTPUT.write( ("run:;%i;inertia:;%f;phi_1:;%f;phi_2:;%f;best;%f;average;%f\n" % (run, weight_inertia, phi_1, phi_2, best, best_average)).replace(".",",") )
    line = ""
    for value in result:
        line = line + str(value[0]).replace(".", ",") + ";"
    OUTPUT.write( line + "\n" )
    line = ""
    for value in result:
        line = line + str(value[1]).replace(".", ",") + ";"
    OUTPUT.write( line + "\n" )

# colect and display
def run(numb_runs,numb_generations,numb_particles,weight_inertia_list, phi_1_list,phi_2_list,vel_max, domain, function,type_problem):
    global OUTPUT
    
    # Colect Data
    statistics_total = []
    print 'Wait, please '
    for i in range(numb_runs):
        
        # initialize population
        numb_dimensions = len(domain)
        particles = [[generate_particle(domain),0] for count in range(numb_particles)]
        velocities = [generate_velocity(vel_max, numb_dimensions) for count in range(numb_particles)]
        
        for weight_inertia in weight_inertia_list:
            for phi_1 in phi_1_list:
                for phi_2 in phi_2_list:
                    execution_result = pso(numb_generations, deepcopy(particles), deepcopy(velocities), weight_inertia, phi_1,phi_2,vel_max, domain, function,type_problem)
                    saveResult(execution_result, i, weight_inertia, phi_1, phi_2, type_problem)
                    statistics_total.append( execution_result )
    print "That's it!"
     
    
    
    # Process data: best and average by generation
    results = zip(*statistics_total)   
    best = [type_problem([result[0] for result in generation]) for generation in results]
    best_average = [sum([result[0] for result in generation])/float(numb_runs) for generation in results]
    average = [sum([indiv[1] for indiv in genera])/float(numb_runs) for genera in results]
    
    # Mostra
    ylabel('Fitness')
    xlabel('Generation')
    tit = 'Runs: %d , Phi: %0.2f, Vel: %0.2f' % (numb_runs,phi_1, vel_max)
    title(tit)
    axis= [0,numb_generations,0,len(domain)]
    #p1 = plot(best,'r-o',label="Best")
    p2 = plot(average,'g->',label="Average")
    p3 = plot(best_average, 'y-s',label="Average of Best")
    if type_problem == max:
        legend(loc='lower right')
    else:
        legend(loc='upper right')
    show()
    
# main program

def pso(numb_generations, particles, velocities, weight_inertia, phi_1, phi_2, vel_max, domain, function,type_problem):
    """
    num_generations = number of generations
    particles = initial particles (10 + 2 * sqrt(dimensions)) or [10,50]
    velocities = initial velocities
    weight_inertia (w) = to control oscilations (0.721 or [0,1[)
    phi_1, phi_2 = cognitive and social weights (1.193 or less than (12* w* (w-1) / (5*w -7)) or sum equal to 4)
    vel_max = maximum variation for move
    domain = [...,(inf_i,sup_i)...] domain values for each dimension
    function = to compute the fitness of a solution candidate
    type_problem = max or min, for maximization or minimization problems.
    
    Structures:
    particles = [...,[[...,p_i_j,...],fit_i],...], current position and fitness of particle i
    velocities = [..., [...,v_i_j,...],...]
    best_past = [...,[[...,p_i_j,...],fit_i],...], previous best position and fitness of particle i
    global_best = [[...,p_k_j,...],fit_k], position and fitness of the global best particle
    statistics_by_generation = [...,[best_fitness_gen_i, average_fitness_gen_i], ...]
    """
    
    numb_particles = len(particles)
    numb_dimensions = len(domain)
    
    # first evaluations
    particles = [[part, function(part)] for [part,fit] in particles]
    best_past = deepcopy(particles)
    
    # statistics
    statistics_by_generation = []
    
    # Run!
    for gen in range(numb_generations):
        # for each particle
        for part in range(numb_particles): 
            # compute the global best. Is here for if we want to
            # modify the neighborhood. For the case of the neighborhood
            # equal to the set of particles can be put out of this cicle
            global_best = find_global_best(best_past, type_problem)
            # for each dimension
            for dim in range(numb_dimensions):
                # update velocity
                velocities[part][dim] = weight_inertia * velocities[part][dim] 
                + phi_1 * (best_past[part][0][dim] - particles[part][0][dim])
                + phi_2 * (best_past[global_best][0][dim] - particles[part][0][dim])
                # update position
                particles[part][0][dim] = particles[part][0][dim] + velocities[part][dim]
                # clampling
                if particles[part][0][dim] < domain[dim][0]:
                    particles[part][0][dim] = domain[dim][0]
                    velocities[part][dim] = 0
                elif particles[part][0][dim] > domain[dim][1]:
                    particles[part][0][dim] = domain[dim][1]
                    velocities[part][dim] = 0
            # update fitness particle
            particles[part][1] = function(particles[part][0])
            # update best past
            if type_problem == max:
                # maximization situation
                if particles[part][1] > best_past[part][1]:
                    best_past[part] = deepcopy(particles[part])
                    # new global best?
                    if best_past[part][1] > best_past[global_best][1]:
                        global_best = part
            else: # minimization problem
                if particles[part][1] < best_past[part][1]:
                    best_past[part] = deepcopy(particles[part])
                    # new global best?
                    if best_past[part][1] < best_past[global_best][1]:
                        global_best = part
        # update statistics
        generation_average_fitness = sum([particle[1] for particle in best_past])/float(numb_particles)
        generation_best_fitness = best_past[global_best][1]
        statistics_by_generation.append([generation_best_fitness,generation_average_fitness])
    # give me the best!
    print 'Best Solution: %s\nFitness: %0.2f\n' % (best_past[global_best][0],best_past[global_best][1])
    return statistics_by_generation
    
    
    
    
    
# Utilities

def generate_particle(domain):
    """ randomly construct a particle."""
    particle = [uniform(inf,sup) for inf,sup in domain]
    return particle

def generate_velocity(vel_max, numb_dimensions):
    """ randomly define the velocity of a particle."""
    velocity = [uniform(-vel_max,vel_max) for count in range(numb_dimensions)]
    return velocity

def find_global_best(past_best,type_problem):
    """ index of the best (according to fitness)."""
    fitness = [fit for [part,fit] in past_best]
    best = type_problem(fitness)
    index = fitness.index(best)
    return index
  
# -----------------------------  Functions  -------------------------

# Sphere
def de_jong_f1(indiv):
    """Sphere. dominio [[-5.12,5.12],[-5.12,5.12],[-5.12,5.12]]"""
    # validate values. Clip if outside bounds.
    x = indiv[0]
    y = indiv[1]
    z = indiv[2]
    if (x < -5.12) or (x > 5.12) or (y < -5.12) or (y > 5.12) or (z < -5.12) or (z > 5.12):
        return 0
    else:
        f = indiv[0]**2 + indiv[1]**2 + indiv[2]**2
        return f

# Rosenbrock  
def de_jong_f2(indiv):
    """ Rosenbrock. dominio [[-2.048,2.048],[-2.048,2.048]]"""
    # validate values. Clip if outside bounds.
    x = indiv[0]
    y = indiv[1]
    if (x < -2.048) or (x > 2.048) or (y < -2.048) or (y >2.048):
        return 0
    else:
        f = 100* (indiv[0]**2 - indiv[1])**2 + (1 - indiv[0])**2
        return f
    
    
# -- Michaelewicz    
def michalewicz_1(indiv):
    x=indiv[0]
    if (x < -1) or (x >2):
        return 0
    else:
        f = x * sin(10 * pi * x) + 1.0
        return f
    

def michalewicz_2(indiv):
    """ Maximo = 38.85."""
    x=indiv[0]
    y=indiv[1]
    if (x < -3.0) or (x >12.1) or (y < 4.1) or (y > 5.8):
        return 0
    else:
        f= x * sin(4 * pi * x) + y* sin(20 *pi * y) + 21.5
        return f

# Rastringin   
def rastringin_3d(indiv):
    """ Rastringin. Dominio [(-5.12,5.12),(-5.12,5.12),(-5.12,5.12)]."""
    # validate values. Clip if outside bounds.
    x=indiv[0]
    y = indiv[1]
    z = indiv[2]
    if (x < -5.12) or (x > 5.12) or (y < -5.12) or (y > 5.12) or (z < -5.12) or (z > 5.12):
        return 0
    else:
        f = 3 * 10.0 + (x**2 - 10.0 * cos(2*pi*x)) + (y**2 - 10.0 * cos(2*pi*y)) + (z**2 - 10.0 * cos(2*pi*z))
        return f  

    

if __name__== '__main__':
    #pso(numb_generations,numb_particles,weight_inertia, phi_1,phi_2,vel_max, domain, function,type_problem)
    #print pso(3,10,1,2,2,0.3,[[-1,2]],michalewicz_1,min)
    #print pso(1000,20,1,2,2,1.5,[[-5.12,5.12],[-5.12,5.12],[-5.12,5.12]],rastringin_3d,min)
    #print pso(1000,20,1,2,2,0.8,[[-2.048,2.048],[-2.048,2.048]],de_jong_f2,max)
    
    """ run(numb_runs, numb_generations, numb_particles, weight_inertia, phi_1, phi_2, vel_max,                          domain,   function, type_problem) """
    run(           10,             100,             15,            [0.8],     [2],     [2],     0.8, [[-2.048,2.048],[-2.048,2.048]], de_jong_f2,          max)
