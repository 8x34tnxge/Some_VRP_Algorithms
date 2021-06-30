from time import time
from math import sqrt, exp
from matplotlib import pyplot as plt
from random import seed, randint, random

# 全局参数设置区
filePath: str = r".\Data\convrp_10_test_1.vrp"
inf = 0x2f2f2f2f

T0 = 50000  # 起始温度
T_end = 1  # 截止温度
q = 0.98  # 降温速率
L = 100  # 迭代次数

customerInfo: list = []  # 客户信息（依次为X坐标、Y坐标、货物需求、服务时间）
dis: list = []  # 客户间距离
carCapacity: int = 0  # 车辆最大容量
minDistance: int = 0  # 最短距离

currentSolution = []  # 当前解
currentValue = inf  # 当前解长度

bestSolution = []  # 最好解
bestValue = inf  # 最好解长度

finalSolution = []  # 最终解（已进行分组)
finalValue = 0  # 最终解长度


def main():
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
    # 模拟退火算法主程序
    SimulatedAnnealing()
    # 对车队进行分组
    Cluster()
    # 显示结果
    display(start, time())
    draw()


# 加载数据
def io():
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


# 初始化客户间距离
def calcDis():
    global dis
    # 初始化距离
    dis = [[0 for _ in range(len(customerInfo))] for _ in range(len(customerInfo))]

    for i in range(len(customerInfo)):
        for j in range(len(customerInfo)):
            dis[i][j] =\
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
    global currentSolution, currentValue, bestSolution, bestValue
    solution = []
    while len(solution) < len(customerInfo) - 1:
        element = randint(1, len(customerInfo) - 1)
        if element not in solution:
            solution.append(element)

    currentSolution = solution
    currentValue = calcPath(currentSolution)

    bestSolution = currentSolution
    bestValue = currentValue


# 模拟退火算法主体
def SimulatedAnnealing():
    global T0, currentSolution, currentValue, bestSolution, bestValue
    while T0 >= T_end:
        print(f'current temperature : {T0:.2f}, currentValue: {currentValue:.2f}, bestValue: {bestValue:.2f}')
        for it in range(L):
            delta = inf
            index_i = -1
            index_k = -1
            for i in range(len(currentSolution)):
                for k in range(i + 1, len(currentSolution)):
                    tmp = calcTwoOpt(currentSolution, i, k)
                    if tmp < delta:
                        delta = tmp
                        index_i = i
                        index_k = k

            if delta > 0 and random() < exp(-delta / T0):
                currentSolution = twoOpt(currentSolution, index_i, index_k)
                currentValue += delta
            else:
                currentSolution = twoOpt(currentSolution, index_i, index_k)
                currentValue += delta

            if bestValue > currentValue:
                bestSolution = currentSolution
                bestValue = currentValue

        T0 *= q


# 进行分组
def Cluster():
    global finalSolution, finalValue
    finalSolution.append([0])
    weight = 0
    index = 0

    for i in range(len(bestSolution)):
        if weight + customerInfo[bestSolution[i]][2] > carCapacity:
            finalSolution.append([0])
            weight = 0
            index += 1
        finalSolution[index].append(bestSolution[i])
        weight += customerInfo[bestSolution[i]][2]

    for i in range(len(finalSolution)):
        finalValue += calcPath(finalSolution[i])

    for i in range(1, len(customerInfo)):
        finalValue += customerInfo[i][3]


# 2-opt算子
def twoOpt(path, i, k):
    tmp = reversed(path[i:k + 1])
    path[i:k + 1] = tmp
    return path


# 2-opt辅助计算变化距离
def calcTwoOpt(path, i, k):
    delta = 0
    if i + len(path) - 1 == k:
        return delta

    delta += - dis[path[(i + len(path) - 1) % len(path)]][path[i]] - dis[path[(k + len(path) + 1) % len(path)]][path[k]] \
             + dis[path[(i + len(path) - 1) % len(path)]][path[k]] + dis[path[(k + len(path) + 1) % len(path)]][path[i]]

    return delta


# 显示结果
def display(start, end):
    # 最终的路径为
    print("solutions:")
    for i in range(len(finalSolution)):
        for k in range(len(finalSolution[i])-1):
            print(f"{finalSolution[i][k]}", end="->")
        print(f"{finalSolution[i][-1]}->0")

    print(f"value:{finalValue:.2f}")

    duration = end - start
    hours = duration // 3600
    minutes = (duration - hours * 3600) // 60
    secs = duration - hours * 3600 - minutes * 60

    print(f"程序运行完毕\n所用时间为：{hours}hours\t{minutes}minutes\t{secs:.2f}seconds")


def draw():
    routeX, routeY = [x[0] for x in customerInfo], [x[1] for x in customerInfo]
    plt.scatter(routeX, routeY, marker='o')
    for i in range(len(finalSolution)):
        routeX, routeY = [0], [0]
        for j in range(len(finalSolution[i])):
            routeX.append(customerInfo[finalSolution[i][j]][0])
            routeY.append(customerInfo[finalSolution[i][j]][1])
        routeX.append(0)
        routeY.append(0)
        plt.plot(routeX, routeY)
    plt.show()


if __name__ == "__main__":
    main()
