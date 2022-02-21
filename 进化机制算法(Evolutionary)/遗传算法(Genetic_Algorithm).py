import numpy as np
from skimage.metrics import structural_similarity
import cv2
from imageio import imread
from random import random, choice, randint, sample
from PIL import Image
from PIL import ImageDraw
import os

class Genetic_Algorithm:
    def __init__(self, imgPath, saveName="temp", top=5, maxgroup=100, features=100, epochs=1000):
        self.orignal_img, self.type, self.row, self.col = self.OpenImg(imgPath)
        self.max_group = maxgroup
        self.top = top
        self.saveName = saveName
        self.groups = []
        self.features = features
        self.epochs = epochs

        if not os.path.exists(saveName):
            os.mkdir(saveName)
        print("初始化...")
        for i in range(self.max_group):
            g = []
            for j in range(self.features):
                tmp = [[choice(np.linspace(0, self.row, features)), choice(np.linspace(0, self.col, features))] for x in range(3)]
                tmp.append("#" + ''.join(choice('0123456789ABCDEF') for x in range(6)))

                g.append(tmp.copy())

            self.groups.append(g.copy())

        self.maxg = self.groups[0]
        print("初始化完成！")

    def OpenImg(self, imgPath):
        img = imread(imgPath)
        print(type(img))
        row, col = img.shape[0], img.shape[1]
        return img, imgPath.split(".")[-1], row, col

    def to_image(self, g):
        array = np.ndarray((self.orignal_img.shape[0], self.orignal_img.shape[1], self.orignal_img.shape[2]), np.uint8)
        array[:, :, 0] = 255
        array[:, :, 1] = 255
        array[:, :, 2] = 255
        newIm1 = Image.fromarray(array)
        draw = ImageDraw.Draw(newIm1)
        for d in g:
            draw.polygon((d[0][0], d[0][1], d[1][0], d[1][1], d[2][0], d[2][1]), d[3])

        return newIm1

    def getSimilar(self, g) -> float:
        # array = np.ndarray((self.orignal_img.shape[0], self.orignal_img.shape[1], self.orignal_img.shape[2]), np.uint8)
        # array[:, :, 0] = 255
        # array[:, :, 1] = 255
        # array[:, :, 2] = 255
        # newIm1 = Image.fromarray(array)
        # draw = ImageDraw.Draw(newIm1)
        # for d in g:
        #     draw.polygon((d[0][0], d[0][1], d[1][0], d[1][1], d[2][0], d[2][1]), d[3])
        newIm1 = self.to_image(g)
        ssim = structural_similarity(np.array(self.orignal_img), np.array(newIm1), multichannel=True)
        return ssim

    def draw_image(self, g, cur):
        image1 = self.to_image(g)
        image1.save(os.path.join(self.saveName, str(cur) + "." + self.type))

    def exchange(self, father, mother)->[]:
        # 交换
        # 随机生成互换个数
        min_locate = min(len(father), len(mother))
        n = randint(0, int(random() * min_locate))
        # 随机选出多个位置
        selected = sample(range(0, min_locate), n)
        # 交换内部
        for s in selected:
            father[s], mother[s] = mother[s], father[s]

        # 交换尾部
        locat = randint(0, min_locate)
        fhead = father[:locat].copy()
        mhead = mother[:locat].copy()

        ftail = father[locat:].copy()
        mtail = mother[locat:].copy()

        # print(fhead, ftail, mhead, mtail)
        fhead.extend(mtail)
        father = fhead
        mhead.extend(ftail)
        mother = mhead
        return [father, mother]

    def mutation(self, gen):
        # 突变
        # 随机生成变异个数
        n = randint(0, int(random() * len(gen)))
        selected = sample(range(0, len(gen)), n)

        for s in selected:
            tmp = [[choice(np.linspace(0, self.row, 100)), choice(np.linspace(0, self.col, 100))] for x in
                   range(3)]
            tmp.append("#" + ''.join(choice('0123456789ABCDEF') for x in range(6)))
            gen[s] = tmp

        return gen

    def move(self, gen):
        # 易位
        exchage = randint(0, self.features)
        for e in range(exchage):
            sec1 = randint(0, len(gen) - 1)
            sec2 = randint(0, len(gen) - 1)

            gen[sec1], gen[sec2] = gen[sec2], gen[sec1]

        return gen

    def add(self, gen):
        # 增加
        n = randint(0, int(self.features * random()))

        for s in range(n):
            tmp = [
                [choice(np.linspace(0, self.row, self.features)),
                 choice(np.linspace(0, self.col, self.features))]
                for x in range(3)]
            tmp.append("#" + ''.join(choice('0123456789ABCDEF') for x in range(6)))
            gen.append(tmp)

        return gen

    def cut(self, gen):
        # 减少
        n = randint(0, int(random() * len(gen)))
        selected = sample(range(0, len(gen)), n)

        g = []
        for gp in range(len(gen)):
            if gp not in selected:
                g.append(gen[gp])

        return g

    def variation(self, gen):
        # 变异
        gen = self.mutation(gen.copy())
        gen1 = self.move(gen.copy())
        gen2 = self.add(gen1.copy())
        gen3 = self.cut(gen2.copy())
        return [gen, gen1, gen2, gen3]

    def breeds(self, father, mother):
        # 繁殖
        new1, new2 = self.exchange(father.copy(), mother.copy())

        # 变异
        new3, new4, new5, new6 = self.variation(father.copy())
        new7, new8, new9, new10 = self.variation(mother.copy())

        return [new1, new2, new3, new4, new5, new6, new7, new8, new9, new10]

    def eliminated(self, groups):
        group_dict = dict()
        # print(0, self.getSimilar(groups[0]))
        for gp in range(len(groups)):
            group_dict[gp] = self.getSimilar(groups[gp])
        # print(1, self.getSimilar(groups[0]))
        # print(1, groups[0])
        group_dict = {key: value for key, value in
                           sorted(group_dict.items(), key=lambda item: item[1], reverse=True)}

        g = []
        for key in list(group_dict.keys())[:self.max_group]:
            g.append(groups[key].copy())

        groups = g.copy()
        return groups, list(group_dict.values())[0]

    def fit(self):
        for cur in range(self.epochs):
            # 繁殖过程
            breed_n = randint(int(self.max_group // 2), self.max_group)
            for i in range(breed_n):
                f = randint(0, self.max_group - 1)
                m = randint(0, self.max_group - 1)
                self.groups.extend(self.breeds(self.groups[f].copy(), self.groups[m].copy()))


            # 淘汰
            self.groups, acc = self.eliminated(self.groups.copy())
            print("Epochs :", cur+1, " Acc:", acc)
            last = self.groups[0]

            if cur % 100 == 0:
                self.draw_image(self.groups[0], cur)

            if acc >= 0.95:
                self.draw_image(self.groups[0], cur)
                break

if __name__=='__main__':
    path = input("输入路径:")
    savename = input("输入保存文件名:")
    groups = int(input("输入族群数量:"))
    GA = Genetic_Algorithm(path, savename, 5, groups, 50, 520 * 365)
    GA.fit()
