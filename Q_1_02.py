import numpy as np
import matplotlib.pyplot as plt
import Local_Random_Search
import Hill_Climbing
import Global_Random_Search

def func(x1, x2):
    termo1 = np.exp(-((x1**2) + (x2**2)))
    termo2 = (2*np.exp(-(((x1-1.7)**2) + ((x2-1.7)**2))))

    return termo1 + termo2

Hill_Climbing.hill_Climbing_init_(
                              rounds = 100,
                              limit_1 = -2,
                              limit_2 = 4,
                              limit_3 = -2,
                              limit_4 = 5,
                              f_apt = func,
                              minimization=False
                              )

Local_Random_Search.LRS_init_(
                              rounds = 100,
                              limit_1 = -2,
                              limit_2 = 4,
                              limit_3 = -2,
                              limit_4 = 5,
                              f_apt = func,
                              minimization=False
                              )

Global_Random_Search.GRS_init_(
                              rounds = 100,
                              limit_1 = -2,
                              limit_2 = 4,
                              limit_3 = -2,
                              limit_4 = 5,
                              f_apt = func,
                              minimization=False
                              )

bp = 1