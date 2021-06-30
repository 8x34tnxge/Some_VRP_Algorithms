"""
*模拟退火算法
*用于解决旅行商问题（TSP）
"""

from time import time
from copy import deepcopy
from math import sqrt, exp
from random import seed, randint, random

file_path: str = r'D:\Code\Project2\TSPExample_62128'  # 测试文件路径

T0 = 50000  # 起始温度
T_end = 1e-8  # 截止温度
q = 0.98  # 降温速率
L = 1000  # 迭代次数
N = 52  # 城市数量

city_list = []  # 城市路径

city_pos = []  # 城市坐标


def io():
    global N, city_pos
    with open(file_path, "r") as f:
        N = int(f.readline())
        for i in range(N):
            data = f.readline().split(" ")
            city_pos.append([float(data[1]), float(data[-1])])
        if f.readline() != "EOF\n":
            raise EOFError
    print("数据加载成功")


def distance(city1: tuple, city2: tuple) -> float:
    """
    计算城市间距离
    :param city1: 城市1的坐标
    :param city2: 城市2的坐标
    :return: 城市间距离
    """
    x1 = city1[0]
    x2 = city2[0]
    y1 = city1[1]
    y2 = city2[1]

    dis = sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
    return dis


def path_len(cities: list) -> float:
    """
    计算路径长度
    :param cities: 城市路径
    :return: 路径长度
    """
    path = 0
    for i in range(N - 1):
        index1 = cities[i]
        index2 = cities[i + 1]

        dis = distance(city_pos[index1 - 1], city_pos[index2 - 1])
        path += dis

    first_index = cities[0]
    last_index = cities[N - 1]
    last_dis = distance(city_pos[first_index - 1], city_pos[last_index - 1])
    path += last_dis
    return path


def init():
    """
    初始化城市路径
    """
    global city_list
    for i in range(N):
        city_list.append(i)


def create_new():
    """
    创造新的解
    """
    global city_list
    index1 = randint(0, N - 1)
    index2 = randint(0, N - 1)

    city_list[index1 - 1], city_list[index2 - 1] = city_list[index2 - 1], city_list[index1 - 1]


def main():
    global city_list
    seed(time())

    T = T0
    count = 0

    start = time()  # 开始计时
    io()
    init()
    while T > T_end:
        for i in range(L):
            city_list_copy = deepcopy(city_list)
            create_new()

            f1 = path_len(city_list)
            f2 = path_len(city_list_copy)

            dE = f1 - f2
            if dE > 0:  # 若遭遇上坡
                if exp(-dE / T) < random():  # 若拒绝接受新解
                    city_list = deepcopy(city_list_copy)
        print(f"当前温度温度T0={T:.2f}，最优路径长度为：{path_len(city_list):.2f}")
        T *= q
        count += 1
    end = time()  # 停止计时
    duration = end - start

    print(f"模拟退火算法，初始温度T0={T0:.2f}，降温系数q={q:.2f}")
    print(f"每个温度迭代{L:d}次，共降温{count:d}次")
    print(f"得到的TSP最优路径为:")

    for i in range(N - 1):
        print(f"{city_list[i]}--->", end='')
    print(f"{city_list[N - 1]}")
    print(f"最优路径长度为：{path_len(city_list):.2f}")

    hours = int(duration // 3600)
    minutes = int((duration - 3600 * hours) // 60)
    seconds = duration - 3600 * hours - 60 * minutes
    print(f"程序运行耗时：{hours:d}时 {minutes:d}分 {seconds:.2f}秒")


if __name__ == '__main__':
    main()
