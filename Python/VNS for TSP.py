"""
*变邻域搜索算法(VNS)
*解决旅行商问题(TSP)
"""

from time import time
from math import sqrt
from copy import deepcopy
from random import seed, shuffle

file_path: str = r'D:\Code\Project2\TSPExample_62128'  # 测试文件路径
N = 52  # 城市个数

city_pos = []


def io() -> None:
    global N, city_pos
    with open(file_path, "r") as f:
        N = int(f.readline())
        for i in range(N):
            data = f.readline().split(" ")
            city_pos.append([float(data[1]), float(data[-1])])
        if f.readline() != "EOF\n":
            raise EOFError
    print("数据加载成功")


io()


def path_len(permutation: list) -> float:
    """
    计算路径长度（时间复杂度为O(n)）
    :param permutation: 访问城市序列
    :return: 路径长度
    """
    length = 0
    for i in range(N):
        c1 = i
        c2 = i + 1 if i != N - 1 else 0
        dis = sqrt((city_pos[permutation[c1]][0] - city_pos[permutation[c2]][0]) ** 2 +
                   (city_pos[permutation[c1]][1] - city_pos[permutation[c2]][1]) ** 2)

        length += dis

    return length


class solution(object):
    def __init__(self):
        self.permutation = [i for i in range(N)]
        shuffle(self.permutation)
        self.cost = path_len(self.permutation)

    def setPermutation(self, permutation: list) -> None:
        """
        设置序列
        :param permutation: 待设置序列
        """
        self.permutation = deepcopy(permutation)

    def getPermutation(self) -> list:
        """
        获取当前序列
        :return: 当前序列
        """
        return self.permutation

    def setCost(self, cost: float) -> None:
        """
        设置路径长度
        :param cost: 待设置路径长度
        """
        self.cost = cost

    def getCost(self) -> float:
        """
        获取当前路径长度
        :return: 当前路径长度
        """
        return self.cost


best_solution = solution()  # 初始化 全局最优解


def copy(current_solution: solution) -> solution:
    """
    复制并返回与当前解相同的solution
    :param current_solution: 当前解
    :return: 复制解
    """
    new_solution = solution()
    new_solution.setPermutation(current_solution.getPermutation())
    new_solution.setCost(current_solution.getCost())

    return new_solution


def distance_2city(city1: int, city2: int) -> float:
    """
    计算两城市间距离
    :param city1: 城市1的序号
    :param city2: 城市2的序号
    :return: 城市距离
    """
    dis = sqrt((city_pos[city1][0] - city_pos[city2][0]) ** 2 +
               (city_pos[city1][1] - city_pos[city2][1]) ** 2)

    return dis


def calc_delta1(i: int, k: int, permutation: list) -> float:
    """
    修正路径长度（时间复杂度O(1)）
    :param i: 前改变点位置
    :param k: 后改变点位置
    :param permutation: 城市序列
    :return: 修正后路径差
    """
    if i == k or i == k - 1 or i > k:
        delta = 0.
    elif i == 0:
        if k == N - 1:
            delta = 0.
        else:
            delta = 0 \
                    - distance_2city(permutation[i], permutation[N - 1]) \
                    - distance_2city(permutation[k], permutation[k + 1]) \
                    + distance_2city(permutation[i], permutation[k + 1]) \
                    + distance_2city(permutation[k], permutation[N - 1])
    else:
        if k == N - 1:
            delta = 0 \
                    - distance_2city(permutation[i], permutation[i - 1]) \
                    - distance_2city(permutation[k], permutation[0]) \
                    + distance_2city(permutation[i], permutation[0]) \
                    + distance_2city(permutation[k], permutation[i - 1])
        else:
            delta = 0 \
                    - distance_2city(permutation[i], permutation[i - 1]) \
                    - distance_2city(permutation[k], permutation[k + 1]) \
                    + distance_2city(permutation[i], permutation[k + 1]) \
                    + distance_2city(permutation[k], permutation[i - 1])

    return delta


def two_opt_swap(i: int, k: int, current_solution: solution) -> None:
    """
    调换 i-k 之间位置的所有元素
    :param i: 前改变点
    :param k: 后改变点
    :param current_solution: 当前解
    """
    permutation = deepcopy(current_solution.getPermutation())
    new_permutation = [None for _ in range(N)]

    for q in range(i):
        new_permutation[q] = (permutation[q])

    for q in range(k - i + 1):
        new_permutation[i + q] = (permutation[k - q])

    for q in range(k + 1, N):
        new_permutation[q] = (permutation[q])

    current_solution.setPermutation(new_permutation)


def update1(permutation: list) -> list:
    """
    更新修正后的solution对应的修正路径差delta
    :param permutation: 城市序列
    :return: 修正路径差
    """
    delta = [[calc_delta1(i, k, permutation) for k in range(N)] for i in range(N)]

    return delta


def neighborhood_one(current_solution: solution) -> solution:
    """
    邻域结构 一
    :param current_solution: 当前解
    :return: 新解
    """
    count = 0
    max_iteration = 60
    new_solution = copy(current_solution)
    delta = [[calc_delta1(i, k, new_solution.getPermutation()) for k in range(N)] for i in range(N)]
    while count < max_iteration:
        count += 1
        for i in range(N):
            for k in range(N):
                if delta[i][k] < 0:
                    two_opt_swap(i, k, new_solution)

                    new_solution.setCost(new_solution.getCost() + delta[i][k])

                    delta = update1(new_solution.getPermutation())

    return new_solution


def calc_delta2(i: int, k: int, permutation: list) -> float:
    """
    修正路径长度（时间复杂度O(1)）
    :param i: 前改变点位置
    :param k: 后改变点位置
    :param permutation: 城市序列
    :return: 修正后路径差
    """
    if i == k or i == k - 1 or i > k:
        delta = 0.
    elif k == N - 1:
        delta = 0. \
                - distance_2city(permutation[i], permutation[i + 1]) \
                - distance_2city(permutation[k], permutation[k - 1]) \
                - distance_2city(permutation[0], permutation[k]) \
                + distance_2city(permutation[0], permutation[k - 1]) \
                + distance_2city(permutation[i], permutation[k]) \
                + distance_2city(permutation[k], permutation[i + 1])
    else:
        delta = 0. \
                - distance_2city(permutation[i], permutation[i + 1]) \
                - distance_2city(permutation[k], permutation[k - 1]) \
                - distance_2city(permutation[k], permutation[k + 1]) \
                + distance_2city(permutation[i], permutation[k]) \
                + distance_2city(permutation[k], permutation[i + 1]) \
                + distance_2city(permutation[k - 1], permutation[k + 1])

    return delta


def two_h_opt_swap(i: int, k: int, current_solution: solution) -> None:
    """
    将k位置的元素插入到i位置的元素后面
    :param i: 前改变点
    :param k: 后改变点
    :param current_solution: 当前解
    """
    permutation = deepcopy(current_solution.getPermutation())
    tmp = permutation[k]
    permutation.remove(tmp)
    permutation.insert(i + 1, tmp)
    current_solution.setPermutation(permutation)


def update2(permutation: list) -> list:
    """
    更新修正后的solution对应的修正路径差delta
    :param permutation: 城市序列
    :return: 修正路径差
    """
    delta = [[calc_delta2(i, k, permutation) for k in range(N)] for i in range(N)]

    return delta


def neighborhood_two(current_solution: solution) -> solution:
    """
    领域结构 二
    :param current_solution: 当前解
    :return: 新解
    """
    count = 0
    max_iteration = 60
    new_solution = copy(current_solution)
    delta = [[calc_delta2(i, k, new_solution.getPermutation()) for k in range(N)] for i in range(N)]
    while count < max_iteration:
        count += 1
        for i in range(N):
            for k in range(N):
                if delta[i][k] < 0:
                    two_h_opt_swap(i, k, new_solution)

                    new_solution.setCost(new_solution.getCost() + delta[i][k])

                    delta = update2(new_solution.getPermutation())

    return new_solution


def shaking(current_solution: solution) -> None:
    """
    摇动产生新解（此为随机生产新解）
    :param current_solution: 当前解
    """
    shuffle(current_solution.getPermutation())

    current_solution.setCost(path_len(current_solution.getPermutation()))


def Variable_Neighborhood_Descent(solution: solution) -> None:
    """
    进行梯度下降操作，寻找全局最优解
    :param solution: 当前解
    """
    global best_solution
    count = 1
    current_solution = copy(solution)
    while count < 3:
        if count == 1:
            new_solution = neighborhood_one(current_solution)
            print(f"Now in neighborhood_one, current_solution = {new_solution.getCost():.2f}"
                  f"\tsolution = {current_solution.getCost():.2f}")
            if current_solution.getCost() > new_solution.getCost():
                current_solution = new_solution
                count = 0

        elif count == 2:
            new_solution = neighborhood_two(current_solution)
            print(f"Now in neighborhood_two, current_solution = {new_solution.getCost():.2f}"
                  f"\tsolution = {current_solution.getCost():.2f}")
            if current_solution.getCost() > new_solution.getCost():
                current_solution = new_solution
                count = 0

        count += 1

    if current_solution.getCost() < best_solution.getCost():
        best_solution = current_solution


def VariableNeighborhoodSearch() -> None:
    """
    变邻域搜索算法
    """
    print(f"初始总路线长度 = {best_solution.getCost():.2f}\n")
    it = 0
    max_iteration = 40
    while it < max_iteration:
        print(f"\t\t\t\tAlgorithm VNS iterated {it + 1} times")
        print(f"-------------------VariableNeighborhoodDescent-------------------")
        current_solution = copy(best_solution)
        shaking(current_solution)

        Variable_Neighborhood_Descent(current_solution)

        print(f"\t\t\t\t全局best_solution = {best_solution.getCost():.2f}\n")

        it += 1

    print(f"\n变邻域搜索算法计算完成！ 最优路线总长度 = {best_solution.getCost():.2f}")
    print(f"最优访问城市序列如下：")
    permutation = best_solution.getPermutation()
    for i in range(len(permutation) - 1):
        print(permutation[i], end='-->')
        if i != 0 and i % 30 == 0:
            print()
    print(permutation[N - 1])
    print()


def main():
    """
    主函数
    """
    seed(time())

    start = time()

    VariableNeighborhoodSearch()

    end = time()
    duration = end - start
    print(f"程序运行时间为：{duration:.2f}s")


if __name__ == '__main__':
    main()
