import numpy as np
import matplotlib.pyplot as plt
import Local_Random_Search
import Hill_Climbing
import Global_Random_Search

def func(x1, x2):
    termo1 = x1 * np.sin(4 * np.pi * x1)
    termo2 = x2 * np.sin((4 * np.pi * x2) + np.pi)

    return termo1 - termo2 + 1

Hill_Climbing.hill_Climbing_init_(
                              rounds = 100,
                              limit_1 = -1,
                              limit_2 = 3,
                              limit_3 = -1,
                              limit_4 = 3,
                              f_apt = func,
                              minimization=False
                              )

Local_Random_Search.LRS_init_(
                              rounds = 100,
                              limit_1 = -1,
                              limit_2 = 3,
                              limit_3 = -1,
                              limit_4 = 3,
                              f_apt = func,
                              minimization=False
                              )

Global_Random_Search.GRS_init_(
                              rounds = 100,
                              limit_1 = -1,
                              limit_2 = 3,
                              limit_3 = -1,
                              limit_4 = 3,
                              f_apt = func,
                              minimization=False
                              )

bp = 1