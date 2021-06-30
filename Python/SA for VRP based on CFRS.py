from time import time
from copy import deepcopy
from math import sqrt, exp
from matplotlib import pyplot as plt
from random import seed, randint, random


# 全局参数设置区
filePath: str = r".\Data\convrp_10_test_1.vrp"  # 数据文件路径
inf = 0x2f2f2f2f  # 手动设置无穷大[doge]

T0 = 50000  # 起始温度
T_end = 1  # 截止温度
q = 0.98  # 降温速率
L = 100  # 迭代次数

removeNum = 2  # destroy每次移除顶点的个数
customerInfo: list = []  # 客户信息（依次为X坐标、Y坐标、货物需求、服务时间）
dis: list = []  # 客户间距离
carCapacity: int = 0  # 车辆最大容量
minDistance: int = 0  # 最短距离

currentSolution: list = []  # 当前解
currentValue: float = inf  # 当前解长度

bestSolution: list = []  # 最好解
bestValue: float = inf  # 最好解长度


def main():
    """
    *程序主函数
    *采用的方式为 Cluster First & Route Second (CFRS)
    *框架为 Simulated Annealing algorithm (SA)
    *算子为 Destroy （随机移除一定数量的vertex） & Repair （此处采用插入法）
    """
    # 开始运行算法
    start = time()
    # 设定随机种子
    seed(start)
    # 读取数据
    io()
    # 初始化距离
    calcDis()
    # 初始化解
    initSolution()
    # 分组
    Cluster()
    # 模拟退火算法主程序
    SimulatedAnnealing()
    # 显示结果
    display(start, time())
    # 绘制结果
    draw()


# 模拟退火算法主体
def SimulatedAnnealing():
    global T0, currentSolution, currentValue, bestSolution, bestValue
    while T0 >= T_end:
        for it in range(L):
            tmpSolution = deepcopy(currentSolution)
            tmpValue = deepcopy(currentValue)
            tmpSolution, tmpValue, idleNodes = destroy(tmpSolution, tmpValue)
            tmpSolution, tmpValue = repair(tmpSolution, tmpValue, idleNodes)
            delta = tmpValue - currentValue
            if delta < 0 or (delta > 0 and random() > exp(-delta / T0)):
                currentSolution = tmpSolution
                currentValue = tmpValue

            if bestValue > currentValue:
                bestSolution = currentSolution
                bestValue = currentValue
        print(f'current temperature : {T0:.2f}, currentValue: {currentValue:.2f}, bestValue: {bestValue:.2f}')
        T0 *= q


def destroy(solution, value):
    idleNodes = []
    for i in range(removeNum):
        j = randint(0, len(solution) - 1)
        k = randint(0, len(solution[j]) - 1)
        if k == 0:
            value -= dis[solution[j][k]][0] + \
                     dis[solution[j][k]][solution[j][k + 1]] - \
                     dis[0][solution[j][k + 1]]
        elif k == len(solution[j]) - 1:
            value -= dis[solution[j][k]][solution[j][k - 1]] + \
                     dis[solution[j][k]][0] - \
                     dis[solution[j][k - 1]][0]
        else:
            value -= dis[solution[j][k]][solution[j][k - 1]] + \
                     dis[solution[j][k]][solution[j][k + 1]] - \
                     dis[solution[j][k - 1]][solution[j][k + 1]]
        idleNodes.append(solution[j].pop(k))

    return solution, value, idleNodes


def repair(solution, value, idleNodes):
    for i in range(len(idleNodes)):
        minDelta = inf
        index1, index2 = -1, -1
        for j in range(len(solution)):
            for k in range(len(solution[j]) + 1):
                if not isFeasible(solution[j], idleNodes[i]):
                    continue
                if k == 0:
                    delta = dis[idleNodes[i]][0] + \
                            dis[idleNodes[i]][solution[j][k]] - \
                            dis[0][solution[j][k]]
                elif k == len(solution[j]):
                    delta = dis[idleNodes[i]][solution[j][k - 1]] + \
                            dis[idleNodes[i]][0] - \
                            dis[solution[j][k - 1]][0]
                else:
                    delta = dis[idleNodes[i]][solution[j][k - 1]] + \
                            dis[idleNodes[i]][solution[j][k]] - \
                            dis[solution[j][k - 1]][solution[j][k]]
                if delta < minDelta:
                    minDelta = delta
                    index1, index2 = j, k
        value += minDelta
        solution[index1].insert(index2, idleNodes[i])

    return solution, value


def isFeasible(sequence, vertex):
    weight = 0
    for i in range(len(sequence)):
        weight += customerInfo[sequence[i]][2]

    if weight + customerInfo[vertex][2] > carCapacity:
        return False
    return True


# 初始化客户间距离
def calcDis():
    global dis
    # 初始化距离
    dis = [[0 for _ in range(len(customerInfo))] for _ in range(len(customerInfo))]

    for i in range(len(customerInfo)):
        for j in range(len(customerInfo)):
            dis[i][j] = \
                sqrt((customerInfo[i][0] - customerInfo[j][0]) ** 2 + (customerInfo[i][1] - customerInfo[j][1]) ** 2)


# 计算路径长度
def calcPath(path):
    Sum = 0
    for i in range(len(path) - 1):
        Sum += dis[path[i]][path[i + 1]]

    Sum += dis[0][path[0]] + dis[0][path[-1]]
    return Sum


# 初始化解
def initSolution():
    global currentSolution, currentValue
    solution = []
    while len(solution) < len(customerInfo) - 1:
        element = randint(1, len(customerInfo) - 1)
        if element not in solution:
            solution.append(element)

    currentSolution = solution
    currentValue = calcPath(currentSolution)


# 进行分组
def Cluster():
    global currentSolution, currentValue, bestSolution, bestValue
    finalSolution = [[]]
    weight = 0
    index = 0

    for i in range(len(currentSolution)):
        if weight + customerInfo[currentSolution[i]][2] > carCapacity:
            finalSolution.append([])
            weight = 0
            index += 1
        finalSolution[index].append(currentSolution[i])
        weight += customerInfo[currentSolution[i]][2]

    currentSolution = finalSolution
    currentValue = 0
    for i in range(len(finalSolution)):
        currentValue += calcPath(finalSolution[i])

    for i in range(1, len(customerInfo)):
        currentValue += customerInfo[i][3]

    bestSolution, bestValue = currentSolution, currentValue


# 显示结果
def display(start, end):
    # 最终的路径为
    print("solutions:")
    for i in range(len(bestSolution)):
        print(f"{0}", end="->")
        for k in range(len(bestSolution[i]) - 1):
            print(f"{bestSolution[i][k]}", end="->")
        print(f"{bestSolution[i][-1]}->0")

    print(f"value:{bestValue:.2f}")

    duration = end - start
    hours = duration // 3600
    minutes = (duration - hours * 3600) // 60
    secs = duration - hours * 3600 - minutes * 60

    print(f"程序运行完毕\n所用时间为：{hours}hours\t{minutes}minutes\t{secs:.2f}seconds")


def io():
    """
    * 加载数据
    * 读取数据文件，再根据数据内容
    * 依次读取customerInfo, carCapacity
    """
    global customerInfo, carCapacity, minDistance
    with open(filePath) as f:
        # 按行读取内容
        text = f.readlines()
        for i in range(len(text)):
            # 读取结点数
            if "DIMENSION" in text[i]:
                index = text[i].index(":") + 1
                dimension = eval(text[i][index:])
                customerInfo = [[0 for _ in range(2)] for _ in range(dimension)]
            # 读取车辆最大容量
            if "CAPACITY" in text[i]:
                index = text[i].index(":") + 1
                carCapacity = eval(text[i][index:])
            # 读取最短距离
            if "DISTANCE" in text[i]:
                index = text[i].index(":") + 1
                minDistance = eval(text[i][index:])

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


def draw():
    """
    * 可视化路线图
    * 先绘制Depot & Customers，再绘制Routes
    """
    # 绘制Depot & Customers
    routeX, routeY = [x[0] for x in customerInfo], [x[1] for x in customerInfo]
    plt.scatter(routeX, routeY, marker='o')
    # 绘制Routes
    for i in range(len(bestSolution)):
        routeX, routeY = [customerInfo[0][0]], [customerInfo[0][1]]
        for j in range(len(bestSolution[i])):
            routeX.append(customerInfo[bestSolution[i][j]][0])
            routeY.append(customerInfo[bestSolution[i][j]][1])
        routeX.append(0)
        routeY.append(0)
        plt.plot(routeX, routeY)
    plt.show()


if __name__ == "__main__":
    main()
