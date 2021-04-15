from PIL import Image
import numpy as np
import math
import random
import sys, getopt

def createFilter(sigma):  # создает матрицу для фильтра нужной нам размерности
# от сигма зависит степень размытия. чем больше сигма тем больше размытие

    diam = 6 * int(sigma) + 1
    rad = 3 * int(sigma)

    matr = np.zeros(shape = (diam, diam)) # матрица из 0 (diam, diam)
                                          # Для матрицы из n строк и m столбов, shape будет (n,m)

    for i in range(matr.shape[0]):
        for j in range(matr.shape[1]):
            x = i - rad
            y = j - rad
            matr[i, j] = math.exp(-(x ** 2 + (y ** 2)) / (2 * math.pi * sigma ** 2)) / (2 * sigma ** 2 * math.pi)
    s = np.sum(matr)
    matr /= s
    return matr


def GaususFilter(a, sigma):  # сам способ фильтрации, то что мы применяем к созданной нами матрице
    rad = 3 * int(sigma)
    b = np.zeros(shape=(a.shape[0] + 2 * rad, a.shape[1] + 2 * rad, a.shape[2]))
    b[rad: -rad, rad: -rad, :] = a
    c = np.zeros(shape=(a.shape[0] + 2 * rad, a.shape[1] + 2 * rad, a.shape[2]))

    filter = createFilter(sigma)
    filter_3D = np.stack((filter,) * 3, axis =- 1)
    for i in range(rad, a.shape[0] + rad):
        for j in range(rad, a.shape[1] + rad):
            matr = b[i - rad: i + rad + 1, j - rad: j + rad + 1, :]
            matr = matr * filter_3D
            c[i, j, 0] = np.sum(matr[:, :, 0])
            c[i, j, 1] = np.sum(matr[:, :, 1])
            c[i, j, 2] = np.sum(matr[:, :, 2])
    c = np.clip(c, 0, 255).astype(np.uint8)
    a = c[rad: -rad, rad: -rad, :]
    return a

    input_image = "/Users/u0da/Download/src.png"
    im = Image.open(input_image)
    im.show()
    data = np.array(im)         # массив чисел из которых сост изображение ?

    data_f = data.astype(np.float) # рассматриваем как float

    final = GaususFilter(data_f, 1)

    final = final.astype(np.uint8)
    omg = Image.fromarray(final)
    #omg = omg.save('/Users/u0da/Download/')
    omg.show()
