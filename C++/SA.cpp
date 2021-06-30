// 模拟退火算法
#include <iostream>
#include <stdio.h>
#include <string.h>
#include <time.h>
#include <math.h>
#define T0 50000 // 初始温度
#define T_end (1e-8)  // 最低温度
#define q 0.98 // 降温速率
#define L 1000 // 链长
#define N 31 // 城市数量
int city_list[N]; // 用于存放一个解

// 中国31个城市坐标
double city_pos[N][2] = {
{1304, 2312}, {3639, 1315}, {4177, 2244}, {3712, 1399},
{3488, 1535}, {3326, 1556}, {3238, 1229}, {4196, 1004},
{4312, 790}, {4386, 570}, {3007, 1970}, {2562, 1756},
{2788, 1491}, {2381, 1676}, {1332, 695},
{3715, 1678}, {3918, 2179}, {4061, 2370},
{3780, 2212}, {3676, 2578}, {4029, 2838},
{4263, 2931}, {3429, 1908}, {3507, 2367},
{3394, 2643}, {3439, 3201}, {2935, 3240},
{3140, 3550}, {2545, 2357}, {2778, 2826},
{2370, 2975}
};

//函数声明
double distance(double*, double*); // 计算两个城市距离
double path_len(int*); // 计算路径长度
void init(); // 初始化函数
void create_new(); // 产生新解

double distance(double* city1, double* city2) {
	double x1 = *city1;
	double x2 = *city2;
	double y1 = *(city1+1);
	double y2 = *(city2+1);
	double dis = sqrt((x1-x2)*(x1-x2)+(y1-y2)*(y1-y2));
	return dis;
}

double path_len(int* arr) {
	double path = 0;
	int index = *arr;
	for (int i = 0; i < N - 1; i++) {
		int index1 = *(arr + i);
		int index2 = *(arr + i + 1);
		double dis = distance(city_pos[index1-1], city_pos[index2-1]);

		path += dis;
	}
	int last_index = *(arr + N - 1);
	int first_index = *arr;
	double last_dis = distance(city_pos[last_index - 1], city_pos[first_index - 1]);

	path += last_dis;
	return path;
}

void init() {
	for (int i = 0; i < N; i++)
		city_list[i] = i + 1;
}

void create_new() {
	double r1 = ((double)rand() / (RAND_MAX + 1.0));
	double r2 = ((double)rand() / (RAND_MAX + 1.0));
	int pos1 = (int)(N * r1);
	int pos2 = (int)(N * r2);
	int temp = city_list[pos1];
	city_list[pos1] = city_list[pos2];
	city_list[pos2] = temp;
}

int main() {
	srand((unsigned)time(NULL));
	time_t start, finish;
	start = clock();
	double T = T0;
	int count = 0;
	init();
	int city_list_copy[N];
	double f1, f2, df;
	double r;

	while (T > T_end) {
		for (int i = 0; i < L; i++) {
			memcpy(city_list_copy, city_list, N * sizeof(int));
			create_new();
			f1 = path_len(city_list_copy);
			f2 = path_len(city_list);
			df = f2 - f1;
			if (df >= 0) {
				r = ((double)rand()) / RAND_MAX;
				if (exp(-df / T) <= r) {
					memcpy(city_list, city_list_copy, N * sizeof(int));
				}
			}
		}
		T *= q;
		count++;
	}
	finish = clock();
	double duration = ((double)(finish - start))/ CLOCKS_PER_SEC;

	printf("模拟退火算法，初始温度T0=%.2f，降温系数q=%.2f，\n每个温度迭代%d次，共降温%d次，\n得到的TSP最优路径为：\n",(double)T0, q, L, count);

	for (int i = 0; i < N - 1; i++) {
		printf("%d--->", city_list[i]);
	}
	printf("%d\n", city_list[N - 1]);
	double len = path_len(city_list);
	printf("最优路径长度为：%lf\n", len);
	printf("\n\n\n程序运行耗时：%lfs\n", duration);
	return 0;
}

