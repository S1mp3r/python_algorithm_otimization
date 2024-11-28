import numpy as np
import matplotlib.pyplot as plt
import Local_Random_Search
import Hill_Climbing
import Global_Random_Search

def func(x1, x2):
    termo1 = - (x2 + 47) * np.sin(np.sqrt(np.abs((x1 / 2) + (x2 + 47))))
    termo2 = x1 * np.sin(np.sqrt(np.abs(x1 - (x2 + 47))))
    
    return termo1 - termo2

Hill_Climbing.hill_Climbing_init_(
                              rounds = 100,
                              limit_1 = -200,
                              limit_2 = 20,
                              limit_3 = -200,
                              limit_4 = 20,
                              f_apt = func,
                              minimization=True
                              )

Local_Random_Search.LRS_init_(
                              rounds = 100,
                              limit_1 = -200,
                              limit_2 = 20,
                              limit_3 = -200,
                              limit_4 = 20,
                              f_apt = func,
                              minimization=True
                              )

Global_Random_Search.GRS_init_(
                              rounds = 100,
                              limit_1 = -200,
                              limit_2 = 20,
                              limit_3 = -200,
                              limit_4 = 20,
                              f_apt = func,
                              minimization=True
                              )

bp = 1