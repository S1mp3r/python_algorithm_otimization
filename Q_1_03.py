import numpy as np
import matplotlib.pyplot as plt
import Local_Random_Search
import Hill_Climbing
import Global_Random_Search

def func(x1, x2):
    termo1 = -20 * np.exp(-0.2 * np.sqrt(0.5 * (x1**2 + x2**2)))
    termo2 = np.exp(0.5 * (np.cos(2 * np.pi * x1) + np.cos(2 * np.pi * x2)))
    termo3 = 20 + np.e
    
    return termo1 - termo2 + termo3

Hill_Climbing.hill_Climbing_init_(
                              rounds = 100,
                              limit_1 = -8,
                              limit_2 = 8,
                              limit_3 = -8,
                              limit_4 = 8,
                              f_apt = func,
                              minimization=True
                              )


Local_Random_Search.LRS_init_(
                              rounds = 100,
                              limit_1 = -8,
                              limit_2 = 8,
                              limit_3 = -8,
                              limit_4 = 8,
                              f_apt = func,
                              minimization=True
                              )

Global_Random_Search.GRS_init_(
                              rounds = 100,
                              limit_1 = -8,
                              limit_2 = 8,
                              limit_3 = -8,
                              limit_4 = 8,
                              f_apt = func,
                              minimization=True
                              )

bp = 1