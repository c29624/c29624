#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.animation as animation
import math
import random
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pylab import mpl
 
# 添加这条可以让图形显示中文，字体显示为黑体
mpl.rcParams['font.sans-serif'] = ['SimHei']


# In[3]:


citys = np.array([])	#城市数组
citys_name = np.array([])
pop_size = 50	#种群大小
c_rate = 0.7	#交叉率
m_rate = 0.05	#突变率
pop = np.array([])	#种群数组
fitness = np.array([])	#适应度数组
city_size = -1		#标记城市数目
ga_num = 200	#最大迭代次数
best_dist = -1	#记录目前最优距离
best_gen = []	#记录目前最优旅行方案


# In[4]:


data=pd.read_excel('107.xlsx')
data.head()


# In[5]:


# 适应度的计算
def calFitness(line, dis_matrix):
    # 贪婪策略得到距离矩阵（解码过程）
    # 计算路径距离（评价函数）
    dis_sum = 0  # 路线距离
    dis = 0
    for i in range(len(line)):
        if i < len(line) - 1:
            # 依次计录一个数以及下一个数的距离，存入城市间的距离矩阵
            dis = dis_matrix.loc[line[i], line[i + 1]]
            dis_sum = dis_sum + dis
        else:
            # 最后一个数，无下一个数的情况
            dis = dis_matrix.loc[line[i], line[0]]
            dis_sum = dis_sum + dis
    # 返回城市间的路线距离矩阵
    return round(dis_sum, 1)
 
 


# In[6]:


# 联赛选择算子
def tournament_select(pops, popsize, fits, tournament_size):
    new_pops, new_fits = [], []
    # 步骤1 从群体中随机选择M个个体，计算每个个体的目标函数值
    while len(new_pops) < len(pops):
        tournament_list = random.sample(range(0, popsize), tournament_size)
        tournament_fit = [fits[i] for i in tournament_list]
        # 转化为df方便索引
        tournament_df = pd.DataFrame             ([tournament_list, tournament_fit]).transpose().sort_values(by=1).reset_index(drop=True)
        # 步骤2 根据每个个体的目标函数值，计算其适应度
        fit = tournament_df.iloc[0, 1]
        pop = pops[int(tournament_df.iloc[0, 0])]
        # 步骤3 选择适应度最大的个体
        new_pops.append(pop)
        new_fits.append(fit)
    return new_pops, new_fits


# In[7]:


# 1. 一点交叉（One-Point Crossover）：随机选择两个个体，然后在某个位置进行切割，将两个个体的基因在该位置进行交换。
# def one_point_crossover(popsize,parent1_pops,parent2_pops, pc):
    
#     child_pops = []
#     for i in range(int(popsize)):
#         # 初始化
#         child = [None] * len(parent1_pops[i])
#         parent1 = parent1_pops[i]
#         parent2 = parent2_pops[i]
#         length = min(len(parent1_pops), len( parent2_pops))
#         point = random.randint(0, length - 1)
#         child1 = parent1_pops[:point] + parent2_pops[point:]
#         # child2 = parent2_pops[:point] + parent1_pops[point:]
#     child_pops.extend(child1)
#     return child_pops
def one_point_crossover(popsize,parent1_pops,parent2_pops, pc):
    
    child_pops = []
    for i in range(int(popsize)):
        # 初始化
        parent1 = parent1_pops[i]
        parent2 = parent2_pops[i]
        child = [None] * len(parent1_pops[i])
        if random.random() >= pc:
            child = parent1.copy()  # 随机生成一个（或者随机保留父代中的一个）
            random.shuffle(child)
        else:
            length = min(len(parent1), len( parent2))
            # 交叉位置
            point = random.randint(0, length - 1)
            # 记录交叉项
            fragment1 = parent1[point:]
            fragment2 = parent2[point:]
            a1_2=[]
            for i in parent1[:point]:
                while i in fragment2:
                    i = fragment1[fragment2.index(i)]
                a1_2.append(i)
            child=a1_2+fragment2
        child_pops.append(child)
    return child_pops


# In[8]:


# 2. 两点交叉（Two-Point Crossover）：随机选择两个位置进行切割，将两个位置之间的基因进行交换。
def Two_Point_Crossover(popsize, parent1_pops, parent2_pops, pc):
    child_pops = []
    for i in range(popsize):
        # 初始化
        child = [None] * len(parent1_pops[i])
        parent1 = parent1_pops[i]
        parent2 = parent2_pops[i]
        if random.random() >= pc:
            child = parent1.copy()  # 随机生成一个（或者随机保留父代中的一个）
            random.shuffle(child)
        else:
            # parent1
            start_pos = random.randint(0, len(parent1) - 1)
            end_pos = random.randint(0, len(parent1) - 1)
            if start_pos > end_pos:
                tem_pop = start_pos
                start_pos = end_pos
                end_pos = tem_pop
            child[start_pos:end_pos + 1] = parent1[start_pos:end_pos + 1].copy()
            # parent2 -> child
            list1 = list(range(end_pos + 1, len(parent2)))
            list2 = list(range(0, start_pos))
            list_index = list1 + list2
            j = -1
            for i in list_index:
                for j in range(j + 1, len(parent2)):
                    if parent2[j] not in child:
                        child[i] = parent2[j]
                        break
        child_pops.append(child)
    return child_pops


# In[9]:


# 3. 均匀交叉（Uniform Crossover）：将两个个体的每个基因按照一定概率进行交换。
def Uniform_crossover(popsize, parent1_pops, parent2_pops):
    child_pops = []
    for i in range(popsize):
        # 初始化
        child = [None] * len(parent1_pops[i])
        parent1 = parent1_pops[i]
        parent2 = parent2_pops[i]
        for j in range(len(child)):
            r=random.random()
            if(r>0.5):
                child[j]=parent1[j]
            else:
                child[j]=parent2[j]
        child_pops.append(child)
    return child_pops

# In[10]:


#  4.使用基于位置的交叉算法对两个父代进行交叉操作。

#     :param parent1: 第一个父代，一个列表。
#     :param parent2: 第二个父代，一个列表。
#     :return: 两个子代，也是列表。
def position_based_crossover(parent1, parent2):
    child_pops=[]

    length = len(parent1)
    # 生成一个随机的交叉点
    crossover_point = random.randint(0, length - 1)

    # 生成两个子代的空列表
    child1 = [-1] * length
    child2 = [-1] * length

    # 将第一个父代的前半段复制到第一个子代中
    for i in range(crossover_point):
        child1[i] = parent1[i]

    # 将第二个父代中在第一个子代中未出现的元素添加到第一个子代的后半段
    j = crossover_point
    for i in range(length):
        if parent2[i] not in child1:
            child1[j] = parent2[i]
            j += 1

    # 将第二个父代的前半段复制到第二个子代中
    for i in range(crossover_point):
        child2[i] = parent2[i]
    # 将第一个父代中在第二个子代中未出现的元素添加到第二个子代的后半段
    j = crossover_point
    for i in range(length):
        if parent1[i] not in child2:
            child2[j] = parent1[i]
            j += 1

    child_pops.extend(child1)
    child_pops.extend(child2)
    return child_pops
# 循环交叉（Cycle Crossover）是遗传算法中的一种交叉因子，它可以保留父代个体的部分次序关系，适用于求解序列型问题。其基本思想是，首先随机选择两个父代个体，然后选择一个交叉起点，在该起点处将两个父代个体进行交叉，得到两个子代个体。在交叉时，子代个体的某些位置可能会重复出现，因此需要进行循环处理，直到子代个体中不再出现重复的位置。
def cycle_crossover(popsize, parent1_pops, parent2_pops, pc):
    child_pops = []
    for i in range(popsize):
        parent1=parent1_pops[i]
        parent2 = parent2_pops[i]
        length = min(len(parent1), len(parent2))
        cycle_start = random.randint(0, length - 1)
        child1 = [-1] * length
        child2 = [-1] * length
        while True:
            value = parent1[cycle_start]
            child1[cycle_start] = value
            index = parent2.index(value)
            if child1[index] != -1:
                break
            cycle_start = index
        for i in range(length):
            if child1[i] == -1:
                child1[i] = parent2[i]
                child2[i] = parent1[i]
            else:
                child2[i] = parent2[i]
        child_pops.append(child1)
    return child_pops


# In[12]:


# 变异操作
def mutate(pops, pm):
    pops_mutate = []
    for i in range(len(pops)):
        pop = pops[i].copy()
        # 随机多次成对变异
        # 随机选出两个位置进行交换
        t = random.randint(1, 5)
        count = 0
        while count < t:
            if random.random() < pm:
                mut_pos1 = random.randint(0, len(pop) - 1)
                mut_pos2 = random.randint(0, len(pop) - 1)
                #如果不相等则进行取反的操作，这里使用交换
                if mut_pos1 != mut_pos2:
                    tem = pop[mut_pos1]
                    pop[mut_pos1] = pop[mut_pos2]
                    pop[mut_pos2] = tem
            pops_mutate.append(pop)
            count += 1
    return pops_mutate

# 画路径图
def draw_path(line, CityCoordinates):
    x, y,c = [], [],[]
    for i in line:
        Coordinate = CityCoordinates.iloc[i]
        x.append(Coordinate[1])
        y.append(Coordinate[2])
        c.append(Coordinate[0])
    x.append(x[0])
    y.append(y[0])
    c.append(c[0])
    for i in range(len(x)):
        plt.text(x[i], y[i], c[i], 
            fontsize=10, color = "black", style = "italic", weight = "light",
            verticalalignment='center', horizontalalignment='left',rotation=0)
    plt.plot(x, y, 'r-', color='blue', alpha=0.8, linewidth=2.2,marker = 'o',markersize = 6,markerfacecolor='#FF3030')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()


# # 运行程序

# ### 1.1初始化参数

# In[14]:


#读取城市经纬度信息
CityCoordinates = pd.read_excel('228.xlsx')
CityNum =len(CityCoordinates)   # 城市数量
MinCoordinate = 0  # 二维坐标最小值
MaxCoordinate = 200  # 二维坐标最大值
# GA参数
generation = 500 # 迭代次数
popsize = 100  # 种群大小
tournament_size = 5  # 锦标赛小组大小
pc = 0.95  # 交叉概率
pm = 0.1  # 变异概率
# CityCoordinates=CityCoordinates.drop('city', axis=1)
# 计算城市之间的距离
dis_matrix =     pd.DataFrame(data=None, columns=range(len(CityCoordinates)), index=range(len(CityCoordinates)))
for i in range(len(CityCoordinates)):
#     xi, yi = CityCoordinates.['longitude'], CityCoordinates['latitude']
    xi,yi= CityCoordinates.iloc[i][1], CityCoordinates.iloc[i][2]
    for j in range(len(CityCoordinates)):
        xj, yj = CityCoordinates.iloc[j][1], CityCoordinates.iloc[j][2]
        dis_matrix.iloc[i, j] = round(math.sqrt((xi - xj) ** 2 + (yi - yj) ** 2), 2)
# print(dis_matrix)


# In[15]:



iteration = 0
# 初始化,随机构造
pops =     [random.sample([i for i in list(range(len(CityCoordinates)))], len(CityCoordinates)) for
     j in range(popsize)]
#画出随机得到的城市连接图
# draw_path(pops[i], CityCoordinates)
# 计算适应度
fits = [None] * popsize
for i in range(popsize):
    fits[i] = calFitness(pops[i], dis_matrix)
# 保留当前最优,最小的fits为最优解
best_fit = min(fits)
best_pop = pops[fits.index(best_fit)]
print('初代最优值 %.1f' % (best_fit))
best_fit_list = []
best_fit_list.append(best_fit)


# In[ ]:



while iteration <= generation:
    # 锦标赛赛选择
    pop1, fits1 = tournament_select(pops, popsize, fits, tournament_size)
    pop2, fits2 = tournament_select(pops, popsize, fits, tournament_size)
#      One_Point_Crossover交叉
#     child_pops = one_point_crossover(popsize, pop1, pop2, pc)

    # Two_Point_Crossover交叉
#     child_pops = Two_Point_Crossover(popsize, pop1, pop2, pc)
    #均匀交叉
    # child_pops = Uniform_crossover(popsize, pop1, pop2)
    #循环交叉
    child_pops = cycle_crossover(popsize, pop1, pop2, pc)

    # child_pops =position_based_crossover(pop1, pop2)
    # 变异
    child_pops = mutate(child_pops, pm)
    # 计算子代适应度
    child_fits = [None] * popsize
    for i in range(popsize):
        child_fits[i] = calFitness(child_pops[i], dis_matrix)
        # 一对一生存者竞争
    for i in range(popsize):
        if fits[i] > child_fits[i]:
            fits[i] = child_fits[i]
            pops[i] = child_pops[i]

    if best_fit > min(fits):
        best_fit = min(fits)
        best_pop = pops[fits.index(best_fit)]

    best_fit_list.append(best_fit)
    if iteration%50==0:
        draw_path(pops[i], CityCoordinates)
    print('第%d代最优值 %.1f' % (iteration, best_fit))
    iteration += 1

# 路径顺序
print(best_pop)

draw_path(best_pop, CityCoordinates)




