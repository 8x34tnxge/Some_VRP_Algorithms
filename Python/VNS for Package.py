"""
*变邻域搜索算法(VNS)
*解决0-1背包问题
"""

from time import time
from copy import deepcopy
from random import seed, randint, random


def init_Items() -> None:
    """
    随机生成物品信息
    :return: 无
    """
    global items_information
    for i in range(N):
        items_information[i] = (randint(1, 20), randint(10, 150))


def item_Value(permutation: list) -> int:
    """
    计算当前选择的总价值
    :param permutation: 背包选择
    :return: 总价值
    """
    value = 0
    for i in range(N):
        if permutation[i] == 1:
            value += items_information[i][1]

    return value


def is_Feasible(permutation: list) -> bool:
    """
    计算当前选择是否超出背包容量
    :param permutation: 背包选择
    :return: 是否可行
    """
    weight = 0
    for i in range(N):
        if permutation[i] == 1:
            weight += items_information[i][0]

    return True if weight <= Q else False


class solution(object):  # 解决方法
    def __init__(self):
        self.permutation = [None for i in range(N)]  # 背包选择
        self.value = 0  # 背包价值

    def set_Permutation(self, permutation: list) -> None:
        self.permutation = deepcopy(permutation)
        self.value = item_Value(self.permutation)

    def get_Permutation(self) -> list:
        return self.permutation

    def get_Value(self) -> int:
        return self.value


def solution_Copy(current_solution: solution) -> solution:
    """
    复制并返回当前解
    :param current_solution: 当前解
    :return: 复制解
    """
    new_solution = solution()
    new_solution.set_Permutation(current_solution.get_Permutation())

    return new_solution


def calc_value1(i: int, k: int, current_solution: solution) -> bool:
    """
    计算该操作是否会使得总价值上升
    :param i: 起始点
    :param k: 结束点
    :param current_solution: 当前解
    :return: 总价值是否上升
    """
    new_permutation = deepcopy(current_solution.get_Permutation())
    for j in range(i, k + 1):
        new_permutation[j] = 1 - new_permutation[j]
    if is_Feasible(new_permutation):
        if item_Value(new_permutation) > current_solution.get_Value():
            return True
    return False


def method_one(i: int, k: int, current_solution: solution) -> None:
    """
    将背包选择的的第i个到第k个元素值进行翻转
    :param i: 起始点
    :param k: 结束点
    :param current_solution: 当前解
    :return: 无
    """
    new_permutation = deepcopy(current_solution.get_Permutation())
    for j in range(i, k + 1):
        new_permutation[j] = 1 - new_permutation[j]

    current_solution.set_Permutation(new_permutation)


def Neighborhood_one(current_solution: solution) -> solution:
    """
    邻域结构 1
    :param current_solution: 当前解
    :return: 新解
    """
    new_solution = solution_Copy(current_solution)
    it = 0
    max_iteration = 60
    while it < max_iteration:
        it += 1
        for i in range(N - 1):
            for k in range(i + 1, N):
                if calc_value1(i, k, new_solution):
                    method_one(i, k, new_solution)

    return new_solution


def calc_Value2(i: int, k: int, current_solution: solution) -> bool:
    """
    计算该操作是否会使得总价值上升
    :param i: 起始点
    :param k: 结束点
    :param current_solution: 当前解
    :return: 总价值是否上升
    """
    permutation = deepcopy(current_solution.get_Permutation())
    permutation[i] = 1 - permutation[i]
    permutation[k] = 1 - permutation[k]

    if is_Feasible(permutation):
        if item_Value(permutation) > current_solution.get_Value():
            return True
    return False


def method_two(i: int, k: int, current_solution: solution) -> None:
    """
    将背包选择的的第i个和第k个元素值进行翻转
    :param i: 起始点
    :param k: 结束点
    :param current_solution: 当前解
    :return: 无
    """
    permutation = deepcopy(current_solution.get_Permutation())
    permutation[i] = 1 - permutation[i]
    permutation[k] = 1 - permutation[k]

    current_solution.set_Permutation(permutation)


def Neighborhood_two(current_solution: solution) -> solution:
    """
    邻域结构 2
    :param current_solution: 当前解
    :return: 新解
    """
    new_solution = solution_Copy(current_solution)
    it = 0
    max_iteration = 60
    while it < max_iteration:
        it += 1
        for i in range(N - 1):
            for k in range(i + 1, N):
                if calc_Value2(i, k, new_solution):
                    method_two(i, k, new_solution)

    return new_solution


def calc_Value3(i: int, current_solution: solution) -> bool:
    """
    计算该操作是否会使得总价值上升
    :param i: 改变点
    :param current_solution: 当前解
    :return: 总价值是否上升
    """
    permutation = deepcopy(current_solution.get_Permutation())
    permutation[i] = 1 - permutation[i]

    if is_Feasible(permutation):
        if item_Value(permutation) > current_solution.get_Value():
            return True
    return False


def method_three(i: int, current_solution: solution) -> None:
    """
    将背包选择的的第i个元素值进行翻转
    :param i: 改变点
    :param current_solution: 当前解
    :return: 无
    """
    permutation = deepcopy(current_solution.get_Permutation())
    permutation[i] = 1 - permutation[i]

    current_solution.set_Permutation(permutation)


def Neighborhood_three(current_solution: solution) -> solution:
    """
    邻域结构 3
    :param current_solution: 当前解
    :return: 新解
    """
    new_solution = solution_Copy(current_solution)
    it = 0
    max_iteration = 60
    while it < max_iteration:
        it += 1
        for i in range(N):
            if calc_Value3(i, new_solution):
                method_three(i, new_solution)

    return new_solution


def shaking(current_solution: solution) -> None:
    """
    随机产生一个可行解
    :param current_solution:
    :return: 无
    """
    permutation = current_solution.get_Permutation()
    for i in range(N):
        if random() < 0.5:
            permutation[i] = 1
            if not is_Feasible(permutation):
                permutation[i] = 0
        else:
            permutation[i] = 0

    current_solution.set_Permutation(permutation)


def Neighborhood_descent(current_solution: solution) -> None:
    """
    进行梯度下降
    :param current_solution: 当前解
    :return: 无
    """
    global best_solution
    it = 0
    number_of_method = 3
    new_solution = solution()
    while it <= number_of_method:
        it += 1
        if it == 1:
            new_solution = Neighborhood_one(current_solution)
            print(f"Now in neighborhood_one, current_solution = {new_solution.get_Value()}"
                  f"\tsolution = {current_solution.get_Value()}")
            if new_solution.get_Value() > current_solution.get_Value():
                current_solution.set_Permutation(new_solution.get_Permutation())
                it = 0
        elif it == 2:
            new_solution = Neighborhood_two(current_solution)
            print(f"Now in neighborhood_two, current_solution = {new_solution.get_Value()}"
                  f"\tsolution = {current_solution.get_Value()}")
            if new_solution.get_Value() > current_solution.get_Value():
                current_solution.set_Permutation(new_solution.get_Permutation())
                it = 0
        elif it == 3:
            new_solution = Neighborhood_three(current_solution)
            print(f"Now in neighborhood_three, current_solution = {new_solution.get_Value()}"
                  f"\tsolution = {current_solution.get_Value()}")
            if new_solution.get_Value() > current_solution.get_Value():
                current_solution.set_Permutation(new_solution.get_Permutation())
                it = 0

    if new_solution.get_Value() > best_solution.get_Value():
        best_solution.set_Permutation(new_solution.get_Permutation())


def Neighborhood_search() -> None:
    """
    变邻域搜索算法
    :return: 无
    """
    it = 0
    max_iteration = 20
    current_solution = solution_Copy(best_solution)
    print(f"物品情况如下：")
    for i in range(N):
        print(f"{i}:\t{items_information[i][0]}\t{items_information[i][1]}", end="\t")
        if i != 0 and (i + 1) % 11 == 0:
            print()
    print()
    while it < max_iteration:
        it += 1
        print(f"\t\t\t       Algorithm VNS iterated {it} times")
        print(f"---------------------VariableNeighborhoodDescent---------------------")

        shaking(current_solution)
        Neighborhood_descent(current_solution)

        print(f"\t\t\t       best_solution = {best_solution.get_Value()}")
        print()

    print(f"搜索完成！\t最高价值 = {best_solution.get_Value()}")
    print(f"背包最优选择为：")
    for i in range(N):
        if best_solution.get_Permutation()[i] == 1:
            print(f"{i}", end="\t")
    print()


def main():
    seed(time())

    start = time()

    init_Items()
    Neighborhood_search()

    end = time()
    duration = end - start
    print(f"程序运行耗时：{duration:.2f}s")
    hour = duration // 3600
    minutes = duration // 60 - hour
    seconds = duration - 3600 * hour - 60 * minutes
    print(f"约合{hour:d}小时{minutes:d}分{seconds:.2f}秒")


if __name__ == '__main__':
    N = 100  # 物品总数
    Q = 300  # 背包容量

    items_information = [None for i in range(N)]
    best_solution = solution()

    main()
