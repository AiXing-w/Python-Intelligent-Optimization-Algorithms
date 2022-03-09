from random import randint, random
import matplotlib.pyplot as plt
import numpy as np
from random import random
import os
plt.rcParams['font.sans-serif'] = ['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False #用来正常显示负号

def f(x, y):
   return np.sin(np.sqrt(x ** 2 + y ** 2))

def plt_image(bestPath, features):
    x = np.linspace(-6, 6, 30)
    y = np.linspace(-6, 6, 30)

    X, Y = np.meshgrid(x, y)
    Z = f(X, Y)
    ax = plt.axes(projection='3d')
    ax.contour3D(X, Y, Z, 25, c=Z)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.set_title('寻优过程')
    if len(bestPath) > 0:
        ax.plot3D(bestPath[:, 0], bestPath[:, 1], func(bestPath, features), c="r")
    plt.savefig(os.path.join('plt_path.png'))
    plt.show()

def func(x, features):
    X = x.reshape(-1, features).copy()
    return np.sin(np.sqrt(X[:, 0] ** 2 + X[:, 1] ** 2))


class Alternative_Evolutionary:
    def __init__(self, func, max_group, features, F0, CR, epochs, lower_range, upper_range):
        self.func = func
        self.max_group = max_group
        self.features = features
        self.F0 = F0
        self.CR = CR
        self.epochs = epochs
        self.lower_range = lower_range
        self.upper_range = upper_range
        self.groups = np.random.rand(max_group,features) * (upper_range - lower_range) + lower_range  # 随机生成族群
        self.v = []
        self.u = []
        self.bestPath = []
        self.bestScore = 0
        self.bestOne = None

    def variation(self, g):
        # 变异
        self.v.clear()
        for i in range(self.max_group):
            r1 = randint(0, self.max_group - 1)
            while r1 == i:
                r1 = randint(0, self.max_group - 1)

            r2 = randint(0, self.max_group - 1)
            while r2 == i or r2 == r1:
                r2 = randint(0, self.max_group - 1)

            r3 = randint(0, self.max_group - 1)
            while r3 == i or r3 == r1 or r3 == r2:
                r3 = randint(0, self.max_group - 1)

            x1 = self.groups[r1]
            x2 = self.groups[r2]
            x3 = self.groups[r3]
            labda = np.e * (1 - self.epochs / (self.epochs + 1 - g))
            F = self.F0 * 2 ** labda
            self.v.append(x1 + F * (x2 - x3))

    def cross(self):
        self.u.clear()
        r = randint(0, self.max_group - 1)
        for i in range(self.max_group):
            cr = random()
            if cr <= self.CR or i == r:
                self.u.append(self.v[i])
            else:
                self.u.append(self.groups[i])

    def select(self):
        for i in range(self.max_group):
            s1 = func(self.u[i].copy(), self.features)
            s2 = func(self.groups[i].copy(), self.features)
            if s1 > s2 and s1 > self.bestScore:
                self.groups[i] = self.u[i]
                for j in range(len(self.groups[i])):
                    if self.groups[i][j] < self.lower_range:
                        self.groups[i][j] = self.lower_range
                    if self.groups[i][j] > self.upper_range:
                        self.groups[i][j] = self.upper_range

                self.bestScore = s1
                self.bestOne = self.groups[i]
        if self.bestOne is not None:
            self.bestPath.append(self.bestOne.copy())

    def fit(self):
        for i in range(self.epochs):
            self.variation(i+1)
            self.cross()
            self.select()
            print("Epoch : ", i+1, " Score : ", self.bestScore)

        return self.bestPath

alt_evol = Alternative_Evolutionary(func=func, max_group=50, features=2, F0=0.25, CR=0.1, epochs=100, lower_range=-6, upper_range=6)
savePath = alt_evol.fit()
plt_image(np.array(savePath), 2)