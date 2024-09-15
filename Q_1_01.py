import numpy as np
import matplotlib.pyplot as plt
import Local_Random_Search
import Hill_Climbing
import Global_Random_Search

def func(x1, x2):
    return (x1**2) + (x2**2)

Hill_Climbing.hill_Climbing_init_(
                              rounds = 2,
                              limit_1 = -100,
                              limit_2 = 100,
                              limit_3 = -100,
                              limit_4 = 100,
                              f_apt = func,
                              minimization=True
                              )

Local_Random_Search.LRS_init_(
                              rounds = 2,
                              limit_1 = -100,
                              limit_2 = 100,
                              limit_3 = -100,
                              limit_4 = 100,
                              f_apt = func,
                              minimization=True
                              )

Global_Random_Search.GRS_init_(
                              rounds = 2,
                              limit_1 = -100,
                              limit_2 = 100,
                              limit_3 = -100,
                              limit_4 = 100,
                              f_apt = func,
                              minimization=True
                              )

bp = 1