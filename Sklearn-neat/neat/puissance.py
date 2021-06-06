import numpy as np
import random

class Puissance:
    previous_generation_fitness = 0
    parentChildDict = {}

    # Mup, Much, psi_max, Sigma and Lambda possibly psi_r (look for optimizations)
    # Sigma is init 1. In paper chapter 4 it's 0.01, but we use it as a scalar for a gaussian.
    def __init__(self, psi_max = 50, p_re_eval = 0.3, p_rdm_mutate=0.3, sigma_min = 1.00, _lambda = 0.6):
        self.psi_max = psi_max
        
        self.p_re_eval = p_re_eval
        self.p_rdm_mutate = p_rdm_mutate

        self.sigma = sigma_min
        self.sigma_min = sigma_min

        self._lambda = _lambda

    def chooseWeight(psi):
        arr = []
        for i in psi:
            arr = np.append(arr, np.repeat(i['name'], i['psi_value']))
        return random.choice(arr)
    

    # If the mutations result in an ANN exhibiting better performance, the puissances of the weights that were mutated are replenished; else,
    # they are reduced.

    # The puissances of the other weights that remain untouched, however, decay over time.
    def adjustPuissance(self, fitness, child, parent):
        mup = abs(self.previous_generation_fitness - fitness)
        much = abs()
        psir = mup * much 
        psic = much * (self.psin / self.psimax)
        delta_psi = psir - psic

    def run(self):
        randomNumber = random.random(0,1)
        if random.random(0,1) > self.p_rdm_mutate:
            return
        else:
            challanger = mutate_puissance_Weights()    
    
    def get_dict(self):
        return self.parentChildDict
      
    # setter method
    def set_dict(self, key, value):
        self.parentChildDict[key] = value