import math
import random
import time
import sys
import os
from typing import List, Tuple, Dict, Optional, Union
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

class Coordinate:
    def __init__(self, longitude: float = 0, latitude: float = 0):
        self.longitude = longitude
        self.latitude = latitude

    def __repr__(self) -> str:
        return f"Coordinate({self.longitude:.6f}, {self.latitude:.6f})"
class DistanceCalculator:
    def __init__(self, enable_cache: bool = True):
        self.distance_cache: Dict[Tuple[int, int], float] = {}
        self.use_cache = enable_cache

    def clear_distance_cache(self) -> None:
        self.distance_cache.clear()

    def calculate(self, a: Coordinate, b: Coordinate) -> float:
        R = 6371000
        lat1 = math.radians(a.latitude)
        lat2 = math.radians(b.latitude)
        delta_lat = math.radians(b.latitude - a.latitude)
        delta_lon = math.radians(b.longitude - a.longitude)
        a_haver = math.sin(delta_lat / 2) ** 2 + \
                 math.cos(lat1) * math.cos(lat2) * \
                 math.sin(delta_lon / 2) ** 2
        c = 2 * math.atan2(math.sqrt(a_haver), math.sqrt(1 - a_haver))
        return R * c
    def get_distance(self, points: List[Coordinate], i: int, j: int) -> float:
        if not self.use_cache:
            return self.calculate(points[i], points[j])
        key = tuple(sorted((i, j)))
        if key in self.distance_cache:
            return self.distance_cache[key]
        distance = self.calculate(points[i], points[j])
        self.distance_cache[key] = distance
        return distance
class AlgorithmConfig:
    def __init__(self, 
                 population_size: int = 100,
                 cooling_rate: float = 0.995,
                 initial_temperature: float = 10000.0,
                 max_iterations: int = 100000,
                 use_cache: bool = True,
                 enable_local_search: bool = True):
        self.population_size = population_size
        self.cooling_rate = cooling_rate
        self.initial_temperature = initial_temperature
        self.max_iterations = max_iterations
        self.use_cache = use_cache
        self.enable_local_search = enable_local_search
class RouteResult:
    def __init__(self):
        self.path: List[Coordinate] = []
        self.total_distance: float = 0.0
        self.execution_time_ms: float = 0.0
        self.start_index: int = 0
        self.end_index: int = 0
        self.starts: List[Coordinate] = []
        self.waypoints: List[Coordinate] = []
        self.ends: List[Coordinate] = []
class RoutePlanner:
    def __init__(self, config: AlgorithmConfig = AlgorithmConfig()):
        self.config = config
        self.dist_calc = DistanceCalculator(config.use_cache)
        self.random = random.SystemRandom()
    def calculate_path_distance(self, path: List[Coordinate]) -> float:
        if len(path) < 2:
            return 0.0
        total = 0.0
        for i in range(len(path) - 1):
            total += self.dist_calc.calculate(path[i], path[i+1])
        return total
    def construct_greedy_path(self, waypoints: List[Coordinate]) -> List[int]:
        n = len(waypoints)
        if n == 0:
            return []       
        path = [0] * n
        visited = [False] * n
        current = self.random.randint(0, n - 1)
        path[0] = current
        visited[current] = True

        for i in range(1, n):
            best_dist = float('inf')
            next_point = current
            for j in range(n):
                if not visited[j]:
                    d = self.dist_calc.calculate(waypoints[current], waypoints[j])
                    if d < best_dist:
                        best_dist = d
                        next_point = j
            if next_point != current:
                path[i] = next_point
                visited[next_point] = True
                current = next_point
        return path
    def two_opt_swap(self, path: List[int], i: int, j: int) -> None:
        while i < j:
            path[i], path[j] = path[j], path[i]
            i += 1
            j -= 1
    def two_opt_optimize(self, waypoints: List[Coordinate], path: List[int]) -> List[int]:
        if not self.config.enable_local_search or len(path) < 4:
            return path.copy()
        improved = True
        best_distance = self.calculate_path_distance(self.extract_path(waypoints, path))
        n = len(path)
        current_path = path.copy()
        while improved:
            improved = False
            for i in range(1, n - 1):
                for j in range(i + 1, n):
                    self.two_opt_swap(current_path, i, j)
                    new_distance = self.calculate_path_distance(self.extract_path(waypoints, current_path))
                    
                    if new_distance < best_distance:
                        best_distance = new_distance
                        improved = True
                    else:
                        self.two_opt_swap(current_path, i, j)
        return current_path
    def extract_path(self, waypoints: List[Coordinate], indices: List[int]) -> List[Coordinate]:
        """从索引列表提取实际路径坐标"""
        return [waypoints[idx] for idx in indices]

    def simulated_annealing(self, waypoints: List[Coordinate]) -> List[int]:
        """使用模拟退火算法寻找近似最优路径"""
        n = len(waypoints)
        if n <= 1:
            return list(range(n))         
        current_path = self.construct_greedy_path(waypoints)
        best_path = current_path.copy()
        current_dist = self.calculate_path_distance(self.extract_path(waypoints, current_path))
        best_dist = current_dist
        temp = self.config.initial_temperature
        cooling_rate = self.config.cooling_rate
        max_iter = min(self.config.max_iterations, n * n * 50)
        for _ in range(max_iter):
            i = self.random.randint(1, n - 1)
            j = self.random.randint(1, n - 1)
            if i == j:
                continue
            if i > j:
                i, j = j, i
            self.two_opt_swap(current_path, i, j)
            new_dist = self.calculate_path_distance(self.extract_path(waypoints, current_path))
            delta = new_dist - current_dist
            if delta < 0 or (temp > 1e-9 and math.exp(-delta / temp) > self.random.random()):
                current_dist = new_dist
                if new_dist < best_dist:
                    best_dist = new_dist
                    best_path = current_path.copy()
            else:
                self.two_opt_swap(current_path, i, j)
            temp *= cooling_rate
            if temp < 1.0:
                break

        return best_path

    def solve_exact_internal(self, waypoints: List[Coordinate]) -> List[int]:
        n = len(waypoints)
        if n > 12:
            return self.simulated_annealing(waypoints)
        from itertools import permutations
        best_perm = None
        best_dist = float('inf')
        for perm in permutations(range(n)):
            dist = self.calculate_path_distance(self.extract_path(waypoints, list(perm)))
            if dist < best_dist:
                best_dist = dist
                best_perm = list(perm)       
        return best_perm if best_perm is not None else list(range(n))
    def plan_route(self, 
                  starts: List[Coordinate],
                  waypoints: List[Coordinate],
                  ends: List[Coordinate]) -> RouteResult:
        start_time = time.time()
        best_full_path: List[Coordinate] = []
        best_total_distance = float('inf')
        best_start_idx = 0
        best_end_idx = 0
        for si in range(len(starts)):
            for ei in range(len(ends)):
                if len(waypoints) <= 12:
                    internal_order = self.solve_exact_internal(waypoints)
                else:
                    internal_order = self.simulated_annealing(waypoints)
                    if self.config.enable_local_search:
                        internal_order = self.two_opt_optimize(waypoints, internal_order)
                full_path = [starts[si]]
                full_path.extend(self.extract_path(waypoints, internal_order))
                full_path.append(ends[ei])
                total_dist = self.calculate_path_distance(full_path)
                if total_dist < best_total_distance:
                    best_total_distance = total_dist
                    best_full_path = full_path
                    best_start_idx = si
                    best_end_idx = ei
        end_time = time.time()
        exec_time = (end_time - start_time) * 1000.0
        result = RouteResult()
        result.path = best_full_path
        result.total_distance = best_total_distance
        result.execution_time_ms = exec_time
        result.start_index = best_start_idx
        result.end_index = best_end_idx
        result.starts = starts
        result.waypoints = waypoints
        result.ends = ends
        return result
    def clear_distance_cache(self) -> None:
        self.dist_calc.clear_distance_cache()

def plot_route(result: RouteResult) -> None:
    plt.figure(figsize=(12, 8))
    start_lons = [c.longitude for c in result.starts]
    start_lats = [c.latitude for c in result.starts]
    plt.scatter(start_lons, start_lats, color='lightgray', s=100, marker='o', label='候选起点')
    end_lons = [c.longitude for c in result.ends]
    end_lats = [c.latitude for c in result.ends]
    plt.scatter(end_lons, end_lats, color='lightgray', s=100, marker='s', label='候选终点')
    waypoint_lons = [c.longitude for c in result.waypoints]
    waypoint_lats = [c.latitude for c in result.waypoints]
    plt.scatter(waypoint_lons, waypoint_lats, color='lightgray', s=100, marker='^', label='候选途经点')
    path_lons = [c.longitude for c in result.path]
    path_lats = [c.latitude for c in result.path]
    plt.scatter(path_lons[0], path_lats[0], color='green', s=150, marker='o', edgecolors='black', linewidths=2, label='选定起点')
    plt.scatter(path_lons[-1], path_lats[-1], color='red', s=150, marker='s', edgecolors='black', linewidths=2, label='选定终点')
    if len(path_lons) > 2:
        plt.scatter(path_lons[1:-1], path_lats[1:-1], color='blue', s=150, marker='^', edgecolors='black', linewidths=2, label='选定途经点')
    plt.plot(path_lons, path_lats, 'b-', linewidth=2, alpha=0.7)
    for i, (lon, lat) in enumerate(zip(path_lons, path_lats)):
        plt.text(lon, lat, f' {i+1}', fontsize=12, fontweight='bold')
    plt.title(f'最优路径规划 (总距离: {result.total_distance:.2f} 米)', fontsize=16)
    plt.xlabel('经度', fontsize=14)
    plt.ylabel('纬度', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(loc='best', fontsize=12)
    plt.tight_layout()
    plt.show()
def read_coordinates_from_file(filename: str) -> Tuple[List[Coordinate], List[Coordinate], List[Coordinate]]:
    starts = []
    waypoints = []
    ends = []
    current_section = None
    try:
        with open(filename, 'r', encoding='utf-8') as file:
            for line_num, line in enumerate(file, 1):
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                if line.startswith('[') and line.endswith(']'):
                    section = line[1:-1].lower()
                    if section in ['starts', 'waypoints', 'ends']:
                        current_section = section
                    else:
                        print(f"警告: 文件{filename}第{line_num}行存在未知标记 {line}，已忽略")
                    continue
                if current_section is None:
                    print(f"警告: 文件{filename}第{line_num}行的坐标没有指定所属部分，已忽略")
                    continue
                try:
                    lon_str, lat_str = line.split(',')
                    longitude = float(lon_str.strip())
                    latitude = float(lat_str.strip())
                    if current_section == 'starts':
                        starts.append(Coordinate(longitude, latitude))
                    elif current_section == 'waypoints':
                        waypoints.append(Coordinate(longitude, latitude))
                    elif current_section == 'ends':
                        ends.append(Coordinate(longitude, latitude))        
                except ValueError:
                    print(f"警告: 文件{filename}第{line_num}行格式错误，已忽略")
                    continue
    
    except FileNotFoundError:
        print(f"错误: 文件 {filename} 不存在")
        print(f"请确保在程序同一目录下创建了points.txt文件，格式符合要求")
        sys.exit(1)
    except Exception as e:
        print(f"读取文件时发生错误: {str(e)}")
        sys.exit(1)
    if not starts:
        print("错误: 文件中未找到任何起点数据")
        print("请检查points.txt文件中是否包含[starts]部分及有效的坐标点")
        sys.exit(1)
    if not ends:
        print("错误: 文件中未找到任何终点数据")
        print("请检查points.txt文件中是否包含[ends]部分及有效的坐标点")
        sys.exit(1)
    
    print(f"成功读取坐标数据:")
    print(f"  起点数量: {len(starts)}")
    print(f"  途经点数量: {len(waypoints)}")
    print(f"  终点数量: {len(ends)}")
    
    return starts, waypoints, ends

def print_coordinates(coords: List[Coordinate], title: str) -> None:
    print(f"===== {title} =====")
    for i, coord in enumerate(coords):
        print(f"  点{i+1}: 经度={coord.longitude:.6f}, 纬度={coord.latitude:.6f}")
def print_result(result: RouteResult) -> None:
    print("\n===== 路线规划结果 =====")
    print(f"总距离: {result.total_distance:.2f} 米")
    print(f"计算耗时: {result.execution_time_ms:.0f} 毫秒")
    print(f"路线顺序 ({len(result.path)}个点):")
    for i, coord in enumerate(result.path):
        print(f"  {i+1}. 经度={coord.longitude:.6f}, 纬度={coord.latitude:.6f}")
    dist_calc = DistanceCalculator()
    verify_dist = 0.0
    print("\n===== 路线验证 =====")
    for i in range(len(result.path) - 1):
        d = dist_calc.calculate(result.path[i], result.path[i+1])
        print(f"  路段 {i+1}: {d:.2f} 米")
        verify_dist += d
    print(f"  算法计算总距离: {result.total_distance:.2f} 米")
    print(f"  验证计算总距离: {verify_dist:.2f} 米")
    print(f"  误差: {abs(result.total_distance - verify_dist):.2f} 米")
    print("所有途经点均已访问（算法保证）")
def main():
    # 设置中文字体，添加错误处理以避免字体报错显示
    try:
        plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC", "sans-serif"]
        plt.rcParams["axes.unicode_minus"] = False
    except Exception:
        # 如果字体设置失败，静默忽略错误
        pass
    config = AlgorithmConfig(
        cooling_rate=0.995,
        initial_temperature=10000.0,
        max_iterations=100000,
        use_cache=True,
        enable_local_search=True
    )
    script_dir = os.path.dirname(os.path.abspath(sys.argv[0]))
    filename = os.path.join(script_dir, 'points.txt') 
    print(f"自动读取坐标文件: {filename}")
    starts, waypoints, ends = read_coordinates_from_file(filename)
    print_coordinates(starts, "待选起点")
    print_coordinates(waypoints, "途经点")
    print_coordinates(ends, "待选终点")
    planner = RoutePlanner(config)
    result = planner.plan_route(starts, waypoints, ends)
    print_result(result)
    plot_route(result)
    planner.clear_distance_cache()

if __name__ == "__main__":
    main()