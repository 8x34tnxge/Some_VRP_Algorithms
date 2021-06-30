"""
*禁忌算法
*用于解决旅行商问题（TSP）
"""

from math import sqrt
from time import time
from copy import deepcopy
from random import seed, randint

file_path: str = r'D:\Code\Project2\TSPExample_62128'  # 测试文件路径

INF: int = 0x3f3f3f3f  # 无穷大
Percent: float = 0.75  # 比例
N: int = 0  # 城市数量
cities: dict = {}  # 城市数据
Dis: list = []  # 城市间距离
TabuList: list = []  # 禁忌表
TabuLength: float = 0  # 禁忌步长
MAX_ITERATION = 10000  # 迭代次数


def main():
    seed(time())  # 设定随机种子

    start = time()

    init()  # 初始化
    TabuSearch()  # 禁忌搜索
    best_solution.display()  # 显示最优解

    end = time()
    duration = end - start
    hours = duration // 3600
    minutes = (duration - 3600 * hours) // 60
    seconds = duration - 3600 * hours - 60 * minutes
    print(f"总共用时：{int(hours):d}小时{int(minutes):d}分钟{seconds:.2f}秒")


class solution(object):
    def __init__(self):
        self.path: list = []
        self.value: float = INF
        self.delta: list = []  # 距离差值

    def set_path(self, path: list) -> None:
        self.path = path
        self.path_len()
        self.delta_update()

    def path_len(self) -> None:
        self.value = 0
        for i in range(N - 1):
            self.value += Dis[self.path[i]][self.path[i + 1]]

        self.value += Dis[self.path[0]][self.path[N - 1]]

    def swap(self, i: int, j: int) -> None:
        self.value_update(i, j)

        k = 0
        while i + k < j - k:
            self.path[i + k], self.path[j - k] = self.path[j - k], self.path[i + k]
            k += 1

        self.delta_update()

    def value_update(self, i: int, j: int) -> None:
        self.value += self.delta[i][j]

    def delta_update(self) -> None:
        if not self.delta:
            self.delta = [[0 for _ in range(N)] for _ in range(N)]
        for i in range(N):
            for j in range(i, N):
                if i == j or i + j == N - 1:
                    self.delta[i][j] = self.delta[j][i] = 0
                elif i == 0 and j != N - 1:
                    self.delta[i][j] = self.delta[j][i] = 0 \
                                                          - Dis[self.path[0]][self.path[N - 1]] \
                                                          - Dis[self.path[j]][self.path[j + 1]] \
                                                          + Dis[self.path[0]][self.path[j + 1]] \
                                                          + Dis[self.path[j]][self.path[N - 1]]
                elif i != 0 and j == N - 1:
                    self.delta[i][j] = self.delta[j][i] = 0 \
                                                          - Dis[self.path[i]][self.path[i - 1]] \
                                                          - Dis[self.path[N - 1]][self.path[0]] \
                                                          + Dis[self.path[i]][self.path[0]] \
                                                          + Dis[self.path[N - 1]][self.path[i - 1]]
                else:
                    self.delta[i][j] = self.delta[j][i] = 0 \
                                                          - Dis[self.path[i]][self.path[i - 1]] \
                                                          - Dis[self.path[j]][self.path[j + 1]] \
                                                          + Dis[self.path[i]][self.path[j + 1]] \
                                                          + Dis[self.path[j]][self.path[i - 1]]

    def display(self) -> None:
        print(f"最短路径长度为：{self.value:.2f}")
        print(f"最佳路径为：")
        for i in range(N):
            print(f"{self.path[i]}",end="-->")
            if (i + 1) % 15 == 0:
                print()
        print(f"{self.path[0]}")


best_solution = solution()


def solution_copy(best: solution, current: solution) -> None:
    best.path = deepcopy(current.path)
    best.value = deepcopy(current.value)


def io() -> None:
    global N, cities
    with open(file_path, "r") as f:
        N = int(f.readline())
        for i in range(N):
            data = f.readline().split(" ")
            cities[int(data[0])] = (float(data[1]), float(data[-1]))
        if f.readline() != "EOF\n":
            raise EOFError
    print("数据加载成功")


def distance_2city(city1: int, city2: int) -> float:
    dis = sqrt((cities[city1+1][0] - cities[city2+1][0]) ** 2 + (cities[city1+1][1] - cities[city2+1][1]) ** 2)

    return dis


def get_next(i: int, forbid: list) -> int:
    MIN = Dis[i][0]
    k = 0
    for j in range(1, N):
        if MIN > Dis[i][j] and forbid[j] == 1:
            MIN = Dis[i][j]
            k = j

    return k


def create_path(now) -> list:
    visits: list = [1 for _ in range(N)]
    path = [now]
    visits[now] = 0
    while any(visits):
        if visits[get_next(now, visits)] != 0:
            now = deepcopy(get_next(now, visits))
            path.append(now)
            visits[now] = 0

    return path


def init():
    global Dis, TabuLength, TabuList
    print(f"程序开始初始化")
    io()
    Dis = [[0 for _ in range(N)] for _ in range(N)]
    print(f"开始生成距离")
    for i in range(N):
        for j in range(N):
            Dis[i][j] = distance_2city(i, j) if i != j else INF
    Dis = tuple(Dis)
    print(f"距离生成完成")
    TabuLength = N * Percent
    TabuList = [[-TabuLength for _ in range(N)] for _ in range(N)]
    print(f"开始生成初始解")
    best_solution.set_path(create_path(randint(0, N - 1)))
    print(f"初始解生成完成")
    solution_copy(best_solution, best_solution)
    print(f"程序初始化完成")


def TabuSearch():
    print(f"开始禁忌搜索")
    for it in range(MAX_ITERATION):
        for i in range(N):
            for j in range(i, N):
                if TabuList[i][j] +\
                        TabuLength < it and best_solution.delta[i][j] < 0:
                    best_solution.swap(i, j)
                    TabuList[i][j] += TabuLength
        print(f"当前迭代次数为：{it + 1:d}, 当前最优解为：{best_solution.value:.2f}")
    solution_copy(best_solution, best_solution)
    print(f"\n禁忌算法搜索完成")


if __name__ == "__main__":
    main()
