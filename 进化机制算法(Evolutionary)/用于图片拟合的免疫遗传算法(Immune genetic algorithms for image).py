import numpy as np
from skimage.metrics import structural_similarity
import cv2
from imageio import imread
from random import random, choice, randint, sample
from PIL import Image
from PIL import ImageDraw
import os

class Genetic_Algorithm:
    def __init__(self, imgPath, saveName="temp", alpha=5, beta=0.5, maxgroup=200, features=100, epochs=1000):
        self.orignal_img, self.type, self.row, self.col = self.OpenImg(imgPath)
        self.max_group = maxgroup
        self.saveName = saveName
        self.groups = []
        self.features = features
        self.epochs = epochs
        self.group_dict = dict()
        self.alpha = alpha
        self.beta = beta
        if not os.path.exists(saveName):
            os.mkdir(saveName)
        print("初始化...")
        for i in range(randint(self.max_group, self.max_group * 2)):
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
        # print(type(img))
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
        ssim = structural_similarity(np.array(self.orignal_img), np.array(g), multichannel=True)
        return ssim

    def get_antibody_similar(self, g1, g2):
        ssim = structural_similarity(np.array(g1), np.array(g2), multichannel=True)
        return ssim

    def draw_image(self, g, cur):
        image1 = self.to_image(g)
        image1.save(os.path.join(self.saveName, str(cur) + "." + self.type))

    def exchange(self, father, mother)->[]:
        # 交换
        # 随机生成互换个数
        min_locate = min(len(father), len(mother))
        n = randint(0, int(randint(25, 100) / 100 * min_locate))
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
        n = int(randint(1, 100) / 1000 * len(gen))

        selected = sample(range(0, len(gen)), n)

        for s in selected:
            tmp = [[choice(np.linspace(0, self.row, 100)), choice(np.linspace(0, self.col, 100))] for x in
                   range(3)]
            tmp.append("#" + ''.join(choice('0123456789ABCDEF') for x in range(6)))
            gen[s] = tmp

        return gen

    def move(self, gen):
        # 易位
        exchage = int(randint(1, 100) / 1000 * len(gen))
        for e in range(exchage):
            sec1 = randint(0, len(gen) - 1)
            sec2 = randint(0, len(gen) - 1)

            gen[sec1], gen[sec2] = gen[sec2], gen[sec1]

        return gen

    def add(self, gen):
        # 增加
        n = int(randint(1, 100) / 1000 * len(gen))

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
        n = int(randint(1, 100) / 1000 * len(gen))
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

    def get_aff(self, g_img):
        # 抗体浓度和抗体亲和度(抗体与抗体间的亲和度和抗体与抗原的亲和度)
        group_len = len(g_img)
        den = np.zeros((group_len, group_len))
        aff = np.zeros(group_len)
        for i in range(group_len):
            for j in range(i, group_len):
                if self.get_antibody_similar(g_img[i], g_img[j]) > 0.75:
                    den[i][j] += 1
                    den[j][i] += 1

            aff[i] = self.getSimilar(g_img[i])

        return aff, den.sum(axis=0) / group_len

    def eliminated(self, groups, epoch):
        g_img = []
        for g in groups:
            g_img.append(self.to_image(g.copy()))

        aff, den = self.get_aff(g_img)
        sim = aff - (np.e ** (np.log(self.alpha+1-self.beta) / (self.epochs - epoch + 1)) - 1 + self.beta) * den
        sim = sim.tolist()
        self.group_dict.clear()
        self.group_dict = {key: value for key, value in enumerate(sim)}
        self.group_dict = {key: value for key, value in
                         sorted(self.group_dict.items(), key=lambda item: item[1], reverse=True)}

        g = []
        Acc = 0
        Den = 0
        Sim = 0
        for key in list(self.group_dict.keys())[:self.max_group]:
            if Acc == 0 and Den == 0 and Sim == 0:
                Acc, Den, Sim = aff[key], den[key], sim[key]

            g.append(groups[key].copy())

        return g, Acc, Den, Sim

    def fit(self):
        self.groups, _, _, _ = self.eliminated(self.groups, 1)
        for cur in range(self.epochs):
            # 繁殖过程
            breed_n = randint(self.max_group // 2, self.max_group)
            g_f = np.abs(np.array(list(self.group_dict.values())))
            p = g_f / np.sum(g_f)
            for i in range(breed_n):
                f = np.random.choice(list(self.group_dict.keys()), p=p.ravel())
                m = np.random.choice(list(self.group_dict.keys()), p=p.ravel())
                if f < self.max_group and m < self.max_group:
                    self.groups.extend(self.breeds(self.groups[int(f)].copy(), self.groups[int(m)].copy()))


            for i in range(randint(0, self.max_group // 2)):
                g = []
                for j in range(self.features):
                    tmp = [[choice(np.linspace(0, self.row, self.features)), choice(np.linspace(0, self.col, self.features))] for
                           x in range(3)]
                    tmp.append("#" + ''.join(choice('0123456789ABCDEF') for x in range(6)))

                    g.append(tmp.copy())

                self.groups.append(g.copy())

            # 淘汰
            self.groups, Acc, Den, Sim = self.eliminated(self.groups.copy(), cur+1)
            print("Epochs :", cur+1, " Acc:", Acc, " Den:", Den, " Sim:", Sim)
            with open("Log.txt", 'a') as f:
                f.write(str(Acc))
                f.write(" ")
                f.write(str(Den))
                f.write(" ")
                f.write(str(Sim))
                f.write("\n")

            if cur % 100 == 0:
                self.draw_image(self.groups[0], cur)
            if Acc >= 0.95:
                break

        self.draw_image(self.groups[0], "End")


path = '../datasets/1.png' # input("输入路径:")  # '../datasets/ldjj.png'
savename = 'xl1'  # input("输入保存文件名:")
groups = 50  # int(input("输入族群:"))
GA = Genetic_Algorithm(path, savename, 5, 0.5, groups, 50, 100000000)
GA.fit()
