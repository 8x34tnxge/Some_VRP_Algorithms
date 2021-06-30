# -*- coding: utf-8 -*-
from collections import namedtuple
from copy import deepcopy
from math import sqrt
from operator import attrgetter
from random import randint, random, seed
from time import sleep, time

import matplotlib.pyplot as plt
from tqdm import tqdm


def main():
    start = time()
    seed(start)
    mainProgram = PSO(filePath=r"./Data/convrp_12_test_4.vrp",  # data file's local path
                      particleNum=25,  # 粒子数
                      iteration=30,  # 迭代次数
                      alpha=0.5,  # partial probability
                      beta=0.7)  # global probability
    mainProgram.run()
    mainProgram.display(startTime=start,
                        endTime=time())


class customerInfomation(object):
    def __init__(self, filePath):
        customer = namedtuple("customer", ('name', 'x', 'y', 'requirement', 'serviceTime'))
        self.customerInfo: list[customer] = []  # 客户信息（依次为名称、X坐标、Y坐标、货物需求、服务时间）
        self.dis: list[list] = []  # 客户间距离
        self.carCapacity: int = 0  # 车辆最大容量
        self.minDistance: int = 0  # 最短距离
        self.population: int = 0  # 节点数量
        
        self.io(filePath, customer)
        self.calcDis()

    def io(self, filePath, infoType):
        """
        * 加载数据
        * 读取数据文件，再根据数据内容
        * 依次读取customerInfo, carCapacity
        """
        customerInfo = []
        with open(filePath) as f:
            # 按行读取内容
            text = f.readlines()
            for i in range(len(text)):
                # 读取结点数
                if "DIMENSION" in text[i]:
                    index = text[i].index(":") + 1
                    self.population = eval(text[i][index:])
                    customerInfo = [[0 for _ in range(2)] for _ in range(self.population)]
                # 读取车辆最大容量
                if "CAPACITY" in text[i]:
                    index = text[i].index(":") + 1
                    self.carCapacity = eval(text[i][index:])
                # 读取最短距离
                if "DISTANCE" in text[i]:
                    index = text[i].index(":") + 1
                    self.minDistance = eval(text[i][index:])

                if text[i] == "NODE_COORD_SECTION\n":
                    # 读取客户位置
                    i = i + 1
                    while text[i] != "DEMAND_SECTION\n":
                        info = text[i].split()
                        info = [eval(info[x]) for x in range(len(info))]
                        customerInfo[info[0]] = info[1:]
                        i = i + 1

                    # 读取客户需求
                    i = i + 1
                    while text[i] != "SVC_TIME_SECTION\n":
                        info = text[i].split()
                        info = [eval(info[x]) for x in range(len(info))]
                        customerInfo[info[0]].append(max(info[1:]) if max(info[1:]) != -1 else 0)
                        i = i + 1

                    # 读取客户服务时间
                    i = i + 1
                    while text[i] != "DEPOT_SECTION\n":
                        info = text[i].split()
                        info = [eval(info[x]) for x in range(len(info))]
                        customerInfo[info[0]].append(max(info[1:]))
                        i = i + 1

                    # 读取仓库坐标
                    i = i + 1
                    info = text[i].split()
                    info = [eval(info[x]) for x in range(len(info))]
                    customerInfo[0] = info
                    i + 1
                    
                    # 初始化仓库需求
                    info = text[i].split()
                    info = [eval(info[x]) for x in range(len(info))]
                    customerInfo[0].append(info[0] if info[0] != -1 else 0)
                    # 初始化仓库服务时间
                    if len(customerInfo[0]) == 3:
                        customerInfo[0].append(0)

                    # 结束初始化参数
                    break
        
        for i, customer in enumerate(customerInfo):
            x, y, requirement, serviceTime = customer
            newCustomer = infoType(i, x, y, requirement, serviceTime)
            self.customerInfo.append(newCustomer)
            
    def calcDis(self):
        self.dis = list([[0 for _ in range(len(self.customerInfo))]
                          for _ in range(len(self.customerInfo))])
        for i in range(len(self.customerInfo)):
            for j in range(len(self.customerInfo)):
                distance = sqrt((self.customerInfo[i].x - self.customerInfo[j].x ) ** 2 +
                                (self.customerInfo[i].y - self.customerInfo[j].y ) ** 2)
                self.dis[i][j] = distance

    def getDis(self, index1, index2):
        return self.dis[index1][index2]

    def getPopulation(self):
        return self.population

    def getPosX(self, index):
        return self.customerInfo[index].x
    
    def getPosY(self, index):
        return self.customerInfo[index].y

    def generateRandomPaths(self, num):
        paths = []
        for _ in range(num):
            isUsed = set()
            path = list()
            while len(path) < self.population:
                customerIndex = randint(0, self.population - 1)
                if customerIndex not in isUsed:
                    isUsed.add(customerIndex)
                    path.append(customerIndex)
            paths.append(path)
        return paths


class particle(object):
    def __init__(self, path, cost):
        self._pBest: list = path
        self._pCost: float = cost
        self._path: list = path
        self._cost: float = cost

    @property
    def pBest(self):
        return deepcopy(self._pBest)

    @pBest.setter
    def pBest(self, pBest):
        self._pBest = pBest

    @property
    def pCost(self):
        return self._pCost

    @pCost.setter
    def pCost(self, pCost):
        self._pCost = pCost

    @property
    def path(self):
        return deepcopy(self._path)

    @path.setter
    def path(self, path):
        self._path = path

    @property
    def cost(self):
        return self._cost

    @cost.setter
    def cost(self, cost):
        self._cost = cost


class PSO(object):
    def __init__(self, filePath, particleNum=30, iteration=100, alpha=0.5, beta=0.7):
        """
        reference: Solving City Routing Issue with Particle Swarm Optimization\n
        using the specific formula: velocity = m * velocity + alpha * partial operation + beta * global operation\n
        m is a momentum ratio between 0 and 1 or [0, 1] and its default is 0\n
        alpha & beta are probabilities between 0 and 1 or [0, 1]
        """
        self.customerInfo: customerInfomation = customerInfomation(filePath)
        self.particleNum: int = particleNum
        self.iteration: int = iteration
        self.particles: list[particle] = []
        self.alpha = alpha
        self.beta = beta
        self.costRecord: list = []
        
        for solution in self.customerInfo.generateRandomPaths(self.particleNum):
            self.particles.append(particle(solution, self.calcPathDis(solution)))
        
        self.gBest: particle = min(self.particles, key=attrgetter('pCost'))

    def calcPathDis(self, path):
        dis = 0
        for i in range(len(path) - 1):
            dis += self.customerInfo.getDis(path[i], path[i+1])
        dis += self.customerInfo.getDis(path[0], path[-1])
        return dis

    def run(self):
        swapOperation = namedtuple('swapOperation', ['index1', 'index2', 'probability'])
        self.gBest = min(self.particles, key=attrgetter('pCost'))
        basicOperation: list[swapOperation] = []
        for it in tqdm(range(self.iteration)):
            self.costRecord.append(self.gBest.pCost)
            for particle in self.particles:
                gBestSolution: list = self.gBest.pBest
                pBestsolution: list = particle.pBest
                currentSolution: list = particle.path
                
                for i in range(len(currentSolution)):
                    if currentSolution[i] != pBestsolution[i]:
                        basicOperation.append(swapOperation(index1=i,
                                                            index2=pBestsolution.index(currentSolution[i]),
                                                            probability=self.alpha))
                        
                for i in range(len(currentSolution)):
                    if currentSolution[i] != gBestSolution[i]:
                        basicOperation.append(swapOperation(index1=i,
                                                            index2=gBestSolution.index(currentSolution[i]),
                                                            probability=self.beta))
                delta = 0
                for operation in basicOperation:
                    if random() < operation.probability:
                        delta += self.calcSwapDelta(currentSolution, operation.index1, operation.index2)
                        currentSolution = self.swap(currentSolution, operation.index1, operation.index2)
                basicOperation.clear()

                particle.path = currentSolution
                particle.cost = particle.cost + delta
                while it >= self.iteration * 2 // 3 and self.intersectionAnalysis(particle):
                    self.refactor(particle)
                
                if particle.cost < particle.pCost:
                    particle.pBest = particle.path
                    particle.pCost = particle.cost
                    
                if particle.cost < self.gBest.pCost:
                    self.gBest.pBest = particle.path
                    self.gBest.pCost = particle.cost

    @staticmethod
    def swap(solution, index1, index2):
        solution = deepcopy(solution)
        solution[index1], solution[index2] = solution[index2], solution[index1]
        return solution

    def calcSwapDelta(self, solution, index1, index2):
        index1, index2 = sorted([index1, index2])
        delta = - self.customerInfo.getDis(solution[index1],
                                           solution[(index1-1+self.customerInfo.population)%self.customerInfo.population]) \
                - self.customerInfo.getDis(solution[index2],
                                           solution[(index2+1)%self.customerInfo.population]) \
                + self.customerInfo.getDis(solution[index1],
                                           solution[(index2+1)%self.customerInfo.population]) \
                + self.customerInfo.getDis(solution[index2],
                                           solution[(index1-1+self.customerInfo.population)%self.customerInfo.population])

        if (index1 == 0 and index2 == self.customerInfo.population - 1):
            delta = - self.customerInfo.getDis(solution[index1], solution[index1+1]) \
                    - self.customerInfo.getDis(solution[index2], solution[index2-1]) \
                    + self.customerInfo.getDis(solution[index1], solution[index2-1]) \
                    + self.customerInfo.getDis(solution[index2], solution[index1+1])
        elif not (abs(index1 - index2) <= 1) :
            delta += - self.customerInfo.getDis(solution[index1], solution[index1+1]) \
                     - self.customerInfo.getDis(solution[index2], solution[index2-1]) \
                     + self.customerInfo.getDis(solution[index1], solution[index2-1]) \
                     + self.customerInfo.getDis(solution[index2], solution[index1+1])

        return delta

    def intersectionAnalysis(self, particle, key='bool'):
        """
        reference: https://blog.csdn.net/rickliuxiao/article/details/6259322?utm_medium=distribute.pc_relevant_t0.none-task-blog-BlogCommendFromMachineLearnPai2-1.compare&depth_1-utm_source=distribute.pc_relevant_t0.none-task-blog-BlogCommendFromMachineLearnPai2-1.compare
        """
        def isIntersected(points):
            func = lambda v1, v2, v3, v4: v1 * v4 - v2 * v3
            delta = func(points[1].x - points[0].x,
                         points[2].x - points[3].x,
                         points[1].y - points[0].y,
                         points[2].y - points[3].y)
            if -(1e-6) <= delta <= 1e-6:
                return False
            alpha = func(points[2].x - points[0].x,
                         points[2].x - points[3].x,
                         points[2].y - points[0].y,
                         points[2].y - points[3].y)
            if alpha / delta > 1 or alpha / delta < 0:
                return False
            beta = func(points[1].x - points[0].x,
                     points[2].x - points[0].x,
                     points[1].y - points[0].y,
                     points[2].y - points[0].y)
            if beta / delta > 1 or beta / delta < 0:
             return False
            return True 
        
        solution =  particle.path
        point = namedtuple('point', ['x', 'y'])
        for i in range(len(solution)):
            for j in range(len(solution)):
                if abs(i-j) <= 1 or (i+j == self.customerInfo.population - 1 and i * j == 0):
                    continue
                points = []
                
                points.append(point(x=self.customerInfo.getPosX(solution[i]),
                                    y=self.customerInfo.getPosY(solution[i])))
                points.append(point(x=self.customerInfo.getPosX(solution[(i+1)%self.customerInfo.population]),
                                    y=self.customerInfo.getPosY(solution[(i+1)%self.customerInfo.population])))
                points.append(point(x=self.customerInfo.getPosX(solution[j]),
                                    y=self.customerInfo.getPosY(solution[j])))
                points.append(point(x=self.customerInfo.getPosX(solution[(j+1)%self.customerInfo.population]),
                                    y=self.customerInfo.getPosY(solution[(j+1)%self.customerInfo.population])))
                
                if isIntersected(points):
                    if key == 'bool':
                        return True
                    elif key == 'position':
                        return sorted([i, i+1, j, j+1])
        if key == 'bool':
            return False
        else:
            return None

    def refactor(self, particle):
        _, index1, index2, _ = self.intersectionAnalysis(particle, key='position')
        index1 %= self.customerInfo.population
        index2 %= self.customerInfo.population
        solution = particle.path
        delta = self.calcRefactDelta(solution, index1, index2)
        solution[index1:index2+1] = reversed(solution[index1:index2+1])
        particle.path = solution
        particle.cost = particle.cost + delta

    def calcRefactDelta(self, solution, index1, index2):
        delta = - self.customerInfo.getDis(solution[index1],
                                           solution[(index1-1+self.customerInfo.population)%self.customerInfo.population]) \
                - self.customerInfo.getDis(solution[index2],
                                           solution[(index2+1)%self.customerInfo.population]) \
                + self.customerInfo.getDis(solution[index1],
                                           solution[(index2+1)%self.customerInfo.population]) \
                + self.customerInfo.getDis(solution[index2],
                                           solution[(index1-1+self.customerInfo.population)%self.customerInfo.population])
        return delta
 
    def display(self, startTime, endTime):
        sleep(0.5)
        duration = endTime - startTime
        hr = duration // 3600
        minutes = (duration - hr * 3600) // 60
        seconds = duration - hr * 3600 - minutes * 60
        print(f'using system time: {hr}hr(s) {minutes}min(s) {seconds:.2f}s, the global best solution is following:')
        for index in self.gBest.pBest:
            print(f'{index}', end='->')
        print(f'{self.gBest.pBest[0]}')
        print(f'the cost of this solution is {self.gBest.pCost:.2f}')
        
        plt.figure()
        plt.subplot(2,1,1)
        routeX, routeY = [[self.customerInfo.getPosX(index) for index in self.gBest.pBest],
                          [self.customerInfo.getPosY(index) for index in self.gBest.pBest]]
        routeX.append(routeX[0])
        routeY.append(routeY[0])
        route, = plt.plot(routeX, routeY)
        
        routeX, routeY = [[self.customerInfo.getPosX(index) for index in range(self.customerInfo.getPopulation())],
                          [self.customerInfo.getPosY(index) for index in range(self.customerInfo.getPopulation())]]
        customer = plt.scatter(routeX, routeY, marker='o', color='g')
        
        for i, _ in enumerate(routeX):
            plt.annotate(i, (routeX[i] + 0.05, routeY[i] + 0.25))
        
        plt.legend([customer, route], ['customers', 'route'])
        plt.xlabel('Pos X')
        plt.ylabel('Pos Y')
        plt.title('PSO for the simple TSP')

        plt.subplot(2,1,2)
        plt.plot([x+1 for x in range(len(self.costRecord))], self.costRecord)
        plt.xlabel('epoch')
        plt.ylabel('cost')
        plt.show()


if __name__ == '__main__':
    main()
