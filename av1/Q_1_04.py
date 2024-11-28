import numpy as np
import matplotlib.pyplot as plt
import Local_Random_Search
import Hill_Climbing
import Global_Random_Search

def func(x1, x2):
    termo1 = x1**2 - 10 * np.cos(2 * np.pi * x1) + 10
    termo2 = x2**2 - 10 * np.cos(2 * np.pi * x2) + 10
    
    return termo1 + termo2

Hill_Climbing.hill_Climbing_init_(
                              rounds = 100,
                              limit_1 = -5.12,
                              limit_2 = 5.12,
                              limit_3 = -5.12,
                              limit_4 = 5.12,
                              f_apt = func,
                              minimization=True
                              )

Local_Random_Search.LRS_init_(
                              rounds = 100,
                              limit_1 = -5.12,
                              limit_2 = 5.12,
                              limit_3 = -5.12,
                              limit_4 = 5.12,
                              f_apt = func,
                              minimization=True
                              )

Global_Random_Search.GRS_init_(
                              rounds = 100,
                              limit_1 = -5.12,
                              limit_2 = 5.12,
                              limit_3 = -5.12,
                              limit_4 = 5.12,
                              f_apt = func,
                              minimization=True
                              )

bp = 1