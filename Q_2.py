import numpy as np
import matplotlib.pyplot as plt
import time
# import imageio

def plot_chessboard_GIF(solutions, gif_name):
    """
    Função para gerar Gifs do tabuleiro de Xadrez com as 92 soluções .

    OBS: Para o uso dessa função, é necessário descomentar a linha 4 com o " import imageio " e ter a biblioteca instalada

    Parâmetros:
    - solutions: Lista de soluções das Oito Damas
    - gif_name: Nome do arquivo para ser salvo com a extensão ".gif"

    Retorna:
    - Gif

    Exemplo de Uso:
    - plot_chessboard_GIF(solutions, '8_queens.gif')
    """

    n = 8
    board_template = np.zeros((n, n, 3))  # Criacao do RGB do tabuleiro, padrao eh preto e branco
    images = []
    

    for i in range(n):
        for j in range(n):
            if (i + j) % 2 == 0:
                board_template[i, j] = [1, 1, 1]  # W
            else:
                board_template[i, j] = [0, 0, 0]  # K

    for solution in solutions:
        board = board_template.copy()

        
        for i in range(n):
            board[solution[i], i] = [1, 0, 0]  # R
        
        plt.imshow(board)
        plt.xticks(np.arange(n))
        plt.yticks(np.arange(n))
        plt.gca().set_xticklabels([])
        plt.gca().set_yticklabels([])

        plt.savefig('temp_image.png')
        images.append(imageio.imread('temp_image.png')) 
        plt.clf()  # Limpa a figura para o próximo gráfico
        

    imageio.mimsave(gif_name, images, fps=1) 

def plot_chessboard(solutions):
    n = 8
    
    board_template = np.zeros((n, n, 3))  # Criacao do RGB do tabuleiro, padrao eh preto e branco

    for i in range(n):
        for j in range(n):
            if (i + j) % 2 == 0:
                board_template[i, j] = [1, 1, 1]  # W
            else:
                board_template[i, j] = [0, 0, 0]  # K

    for solution in solutions:
        board = board_template.copy()

        
        for i in range(n):
            board[solution[i], i] = [1, 0, 0]  # R
        
        plt.imshow(board)
        plt.xticks(np.arange(n))
        plt.yticks(np.arange(n))
        plt.gca().set_xticklabels([])
        plt.gca().set_yticklabels([])

        plt.draw()  # Atualiza o gráfico
        plt.pause(0.5)
        plt.clf()  # Limpa a figura para o próximo gráfico

    plt.show() 

# Ideia do Gui
# def remRep(arr):
#     rest = [i for i in range(len(arr)) if i not in arr]
#     return rest

# Ideia do Gui
# def isRep(arr):
#     for i in range(len(arr)-1):
#         for j in range(i+1, len(arr)):
#             if arr[i] == arr[j]:
#                 return True, i
#     return False, None

# Ideia do Gui
def perturb(x):
    # rep, index = isRep(x)
    # if rep:
    #     substitute = np.random.permutation(remRep(x))[0]
    #     x[index] = substitute
    #     return x
    # else:
        i1, i2 = np.random.permutation(len(x))[0:2]
        x[i1], x[i2] = x[i2], x[i1]
        return x

def h(x):
    n = len(x)
    pares_atacantes = 0

    #Verifica os pares
    for i in range(n):
        for j in range(i + 1, n):
            if x[i] == x[j] or abs(x[i] - x[j]) == abs(i - j):
                pares_atacantes += 1

    return pares_atacantes

def f(x):
    return 28 - h(x)

def isNotInside(value, list):
    for optimal in list:
        if np.array_equal(optimal, value):
            return False
    return True


x_l = 0
x_u = 7

x_opt = np.random.permutation(8)
f_opt = f(x_opt)

it_max  = 1000000

T = 100

i = 0
xs_otimos = []

inicio = time.time()

while i < it_max and len(xs_otimos) < 92:
    x_cand = perturb(x_opt)
    f_cand = f(x_cand)

    p_ij = np.exp(-(f_cand-f_opt)/T)

    if f_cand > f_opt or p_ij >= np.random.uniform(0,1):
        x_opt = x_cand
        f_opt = f_cand

    i+=1
    # T *= .79
    # T = (T / (1.79 * np.sqrt(T)))
    T = T - ((100 - T) / i)
    
    if f_opt == 28:
        if isNotInside(x_opt, xs_otimos):
            xs_otimos.append(x_opt)

        x_opt = np.random.permutation(8)
        
        T = 100

fim = time.time()

tempo_execucao = fim - inicio
print(f"Tempo de execução: {tempo_execucao} segundos")

plot_chessboard(xs_otimos)

bp = 1