import numpy as np
import matplotlib.pyplot as plt
import Local_Random_Search
import Hill_Climbing
import Global_Random_Search

def func(x1, x2):
    termo1 = - np.sin(x1) * np.sin(x1**2 / np.pi)**(2 * 10)
    termo2 = np.sin(x2) * np.sin(2 * x2**2 / np.pi)**(2 * 10)
    
    return termo1 - termo2

Hill_Climbing.hill_Climbing_init_(
                              rounds = 100,
                              limit_1 = 0,
                              limit_2 = np.pi,
                              limit_3 = 0,
                              limit_4 = np.pi,
                              f_apt = func,
                              minimization=True
                              )

Local_Random_Search.LRS_init_(
                              rounds = 100,
                              limit_1 = 0,
                              limit_2 = np.pi,
                              limit_3 = 0,
                              limit_4 = np.pi,
                              f_apt = func,
                              minimization=True
                              )

Global_Random_Search.GRS_init_(
                              rounds = 100,
                              limit_1 = 0,
                              limit_2 = np.pi,
                              limit_3 = 0,
                              limit_4 = np.pi,
                              f_apt = func,
                              minimization=True
                              )

bp = 1