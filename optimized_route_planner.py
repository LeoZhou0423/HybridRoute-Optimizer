import math
import random
import time
import sys
import os
import concurrent.futures
from typing import List, Tuple, Dict, Optional, Union, NamedTuple
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import numpy as np
import threading
from amap_api import amap_api
from map import AmapMapDisplay
import win32gui

# 根据系统类型选择合适的地图显示方式
USE_PYQT = False
app = None
HWND = None

# 尝试导入PyQt5
try:
    from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QMainWindow
    from PyQt5.QtWebEngineWidgets import QWebEngineView
    from PyQt5.QtCore import QUrl
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    USE_PYQT = True
    # 尝试导入win32gui用于嵌入tkinter
    try:
        import win32gui
        HWND = True
    except ImportError:
        HWND = False
        import webbrowser
except ImportError:
    USE_PYQT = False
    import webbrowser

# --- 地理坐标类 ---
class Coordinate:
    def __init__(self, longitude: float = 0, latitude: float = 0):
        self.longitude = longitude
        self.latitude = latitude

    def __repr__(self) -> str:
        return f"Coordinate({self.longitude:.6f}, {self.latitude:.6f})"

# --- 路径规划结果 ---
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
        self.start_candidate_index: int = 0  # 起点在候选列表中的索引
        self.end_candidate_index: int = 0    # 终点在候选列表中的索引
        self.driving_distance: float = 0.0   # 驾车距离（米）
        self.driving_time: float = 0.0       # 驾车时间（秒）
        self.route_points: List[Tuple[float, float]] = []  # 存储实际驾车路线的坐标点列表

    def __lt__(self, other):
        # 默认按照直线距离升序排列
        return self.total_distance < other.total_distance

# --- 距离计算器（支持缓存可选）---
class DistanceCalculator:
    def __init__(self, use_cache: bool = True):
        self.use_cache = use_cache
        self.cache = {}
        
    def calculate(self, coord1: Coordinate, coord2: Coordinate) -> float:
        # 计算两点间的直线距离（米）
        cache_key = (coord1.longitude, coord1.latitude, coord2.longitude, coord2.latitude)
        
        if self.use_cache and cache_key in self.cache:
            return self.cache[cache_key]
        
        # 使用Haversine公式计算直线距离
        lon1, lat1 = coord1.longitude, coord1.latitude
        lon2, lat2 = coord2.longitude, coord2.latitude
        
        lon1, lat1, lon2, lat2 = map(math.radians, [lon1, lat1, lon2, lat2])
        
        dlon = lon2 - lon1 
        dlat = lat2 - lat1 
        a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
        c = 2 * math.asin(math.sqrt(a)) 
        r = 6371  # 地球半径（公里）
        distance = c * r * 1000  # 转换为米
        
        if self.use_cache:
            self.cache[cache_key] = distance
            # 保存反向查询，因为距离是对称的
            reverse_key = (coord2.longitude, coord2.latitude, coord1.longitude, coord1.latitude)
            self.cache[reverse_key] = distance
        
        return distance
        
    def clear_distance_cache(self) -> None:
        # 清除距离缓存
        self.cache.clear()
        
# --- 算法配置类 ---
class AlgorithmConfig:
    def __init__(self,
                 population_size: int = 100,
                 cooling_rate: float = 0.995,
                 initial_temperature: float = 10000.0,
                 min_temperature: float = 1.0,
                 max_iterations: int = 100000,
                 use_cache: bool = True,
                 enable_local_search: bool = True,
                 enable_pruning: bool = True,
                 pruning_threshold: float = 1.2,
                 parallel_workers: int = 4,
                 top_n_results: int = 3,
                 adaptive_cooling: bool = True,
                 speed_level: int = 3,
                 accuracy_level: int = 3):
        self.population_size = population_size
        self.cooling_rate = cooling_rate
        self.initial_temperature = initial_temperature
        self.min_temperature = min_temperature
        self.max_iterations = max_iterations
        self.use_cache = use_cache
        self.enable_local_search = enable_local_search
        self.enable_pruning = enable_pruning
        self.pruning_threshold = pruning_threshold
        self.parallel_workers = parallel_workers
        self.top_n_results = top_n_results
        self.adaptive_cooling = adaptive_cooling
        self.speed_level = speed_level
        self.accuracy_level = accuracy_level
        
        # 初始化时根据级别设置参数
        self.update_levels(speed_level, accuracy_level)
        
    def update_levels(self, speed_level: int, accuracy_level: int) -> None:
        # 根据速度和准确度级别更新算法参数
        self.speed_level = speed_level
        self.accuracy_level = accuracy_level
        
        # 根据速度级别调整参数
        if speed_level == 1:  # 极速模式
            self.max_iterations = 5000
            self.parallel_workers = 8
            self.enable_pruning = True
            self.pruning_threshold = 1.5
            self.adaptive_cooling = False
        elif speed_level == 2:  # 快速模式
            self.max_iterations = 20000
            self.parallel_workers = 6
            self.enable_pruning = True
            self.pruning_threshold = 1.3
            self.adaptive_cooling = True
        elif speed_level == 3:  # 平衡模式
            self.max_iterations = 50000
            self.parallel_workers = 4
            self.enable_pruning = True
            self.pruning_threshold = 1.2
            self.adaptive_cooling = True
        elif speed_level == 4:  # 精准模式
            self.max_iterations = 80000
            self.parallel_workers = 3
            self.enable_pruning = True
            self.pruning_threshold = 1.1
            self.adaptive_cooling = True
        else:  # 最佳模式 (speed_level == 5)
            self.max_iterations = 150000
            self.parallel_workers = 2
            self.enable_pruning = False
            self.pruning_threshold = 1.0
            self.adaptive_cooling = True
            
        # 根据准确度级别调整参数
        if accuracy_level == 1:  # 基础模式
            self.initial_temperature = 5000
            self.cooling_rate = 0.99
            self.enable_local_search = False
        elif accuracy_level == 2:  # 一般模式
            self.initial_temperature = 8000
            self.cooling_rate = 0.993
            self.enable_local_search = True
        elif accuracy_level == 3:  # 标准模式
            self.initial_temperature = 10000
            self.cooling_rate = 0.995
            self.enable_local_search = True
        elif accuracy_level == 4:  # 高精模式
            self.initial_temperature = 15000
            self.cooling_rate = 0.997
            self.enable_local_search = True
        else:  # 超高精度模式 (accuracy_level == 5)
            self.initial_temperature = 20000
            self.cooling_rate = 0.999
            self.enable_local_search = True
            
    def get_level_description(self) -> str:
        # 获取当前级别的描述文本
        speed_descriptions = {
            1: "极速模式 - 最快计算速度，中等准确度",
            2: "快速模式 - 较快计算速度，良好准确度",
            3: "平衡模式 - 计算速度与准确度的良好平衡",
            4: "精准模式 - 较高准确度，计算速度中等",
            5: "最佳模式 - 最高准确度，计算速度较慢"
        }
        
        accuracy_descriptions = {
            1: "基础准确度 - 适合粗略估算",
            2: "一般准确度 - 适合大多数场景",
            3: "标准准确度 - 平衡的准确度设置",
            4: "高精度 - 适合对结果准确度要求较高的场景",
            5: "超高精度 - 适合专业应用场景"
        }
        
        speed_desc = speed_descriptions.get(self.speed_level, "未知速度级别")
        accuracy_desc = accuracy_descriptions.get(self.accuracy_level, "未知准确度级别")
        
        return f"{speed_desc}\n{accuracy_desc}"

# --- 路径规划器 ---
class RoutePlanner:
    def __init__(self, config: AlgorithmConfig = AlgorithmConfig()):
        self.config = config
        self.dist_calc = DistanceCalculator(config.use_cache)
        
        # 始终使用系统随机源
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
        
        # 随机选择起点
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
        return [waypoints[idx] for idx in indices]

    def simulated_annealing(self, waypoints: List[Coordinate]) -> List[int]:
        n = len(waypoints)
        if n <= 1:
            return list(range(n))
            
        # 构造初始路径 - 使用贪心算法
        current_path = self.construct_greedy_path(waypoints)
        best_path = current_path.copy()
        current_dist = self.calculate_path_distance(self.extract_path(waypoints, current_path))
        best_dist = current_dist

        # 初始化参数
        temp = self.config.initial_temperature
        base_cooling_rate = self.config.cooling_rate
        cooling_rate = base_cooling_rate
        
        # 根据数据量动态调整最大迭代次数
        base_iterations = self.config.max_iterations
        data_size_factor = max(1, n // 50)  # 数据量越大，迭代次数越多
        max_iter = min(base_iterations, n * n * 10 * data_size_factor)
        
        # 自适应冷却参数
        no_improvement_count = 0
        no_improvement_threshold = min(1000, max_iter // 5)  # 无改进阈值
        best_improvement = float('inf')
        
        # 记录温度变化历史，用于自适应调整
        temp_history = []
        dist_history = []

        for iter_count in range(max_iter):
            # 生成邻域解（使用2-opt交换）
            i = self.random.randint(1, n - 1)
            j = self.random.randint(1, n - 1)
            if i == j:
                continue
            if i > j:
                i, j = j, i

            # 执行交换并计算新距离
            self.two_opt_swap(current_path, i, j)
            new_dist = self.calculate_path_distance(self.extract_path(waypoints, current_path))

            # 判断是否接受新解
            delta = new_dist - current_dist
            if delta < 0 or (temp > self.config.min_temperature and math.exp(-delta / temp) > self.random.random()):
                # 接受新解
                current_dist = new_dist
                dist_history.append(current_dist)
                
                # 更新最优解
                if new_dist < best_dist:
                    best_dist = new_dist
                    best_path = current_path.copy()
                    no_improvement_count = 0
                    improvement = best_dist_prev - best_dist if 'best_dist_prev' in locals() else float('inf')
                    best_improvement = min(best_improvement, improvement)
                    best_dist_prev = best_dist
                else:
                    no_improvement_count += 1
            else:
                # 拒绝新解，恢复原路径
                self.two_opt_swap(current_path, i, j)
                no_improvement_count += 1

            # 记录温度历史
            temp_history.append(temp)
            
            # 自适应冷却率调整
            if self.config.adaptive_cooling and iter_count % 100 == 0 and iter_count > 0:
                # 根据近期无改进次数动态调整冷却率
                if no_improvement_count > no_improvement_threshold:
                    # 加快冷却速度
                    cooling_rate = base_cooling_rate * 0.9
                elif 'best_dist_prev' in locals() and best_dist < best_dist_prev:
                    # 发现更好的解，减慢冷却速度
                    cooling_rate = min(base_cooling_rate * 1.05, 0.999)
                else:
                    # 恢复基础冷却率
                    cooling_rate = base_cooling_rate

            # 更新温度
            temp *= cooling_rate
            
            # 提前终止条件
            # 温度低于最小阈值
            if temp < self.config.min_temperature:
                break
            
            # 连续多次迭代无改进且温度很低
            if no_improvement_count > no_improvement_threshold * 3 and temp < 10:
                break
            
            # 变化率非常小（收敛）
            if len(dist_history) > 1000:
                recent_change = abs(dist_history[-1] - dist_history[-1000]) / dist_history[-1000]
                if recent_change < 0.001 and temp < 100:
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

    def _process_route_combination(self, starts: List[Coordinate], waypoints: List[Coordinate], 
                                 ends: List[Coordinate], si: int, ei: int, 
                                 current_best: float = float('inf')) -> List[RouteResult]:
        # 处理单个起点-终点组合的路径规划，返回多个不同的途经点顺序方案
        results = []
        
        # 组合剪枝：如果当前最优解已经存在，且当前组合的近似距离已经超过阈值，则直接返回空列表
        if self.config.enable_pruning and current_best != float('inf'):
            # 计算起点到终点的直接距离作为近似值
            approx_dist = self.dist_calc.calculate(starts[si], ends[ei])
            # 如果近似距离已经超过当前最优解的阈值倍数，则剪枝
            if approx_dist > current_best * self.config.pruning_threshold:
                return results
                
        # 如果有多个途经点，尝试生成多个不同的途经点顺序方案
        if len(waypoints) > 1:
            # 生成最优方案
            if len(waypoints) <= 12:
                optimal_order = self.solve_exact_internal(waypoints)
            else:
                optimal_order = self.simulated_annealing(waypoints)
                if self.config.enable_local_search:
                    optimal_order = self.two_opt_optimize(waypoints, optimal_order)
            
            # 添加最优方案到结果列表
            optimal_result = self._create_route_result(starts, waypoints, ends, si, ei, optimal_order)
            results.append(optimal_result)
            
            # 为了生成多样化的方案，我们可以：
            # 1. 使用不同初始温度的模拟退火
            original_temp = self.config.initial_temperature
            
            # 尝试高初始温度
            self.config.initial_temperature = original_temp * 2
            high_temp_order = self.simulated_annealing(waypoints)
            if self.config.enable_local_search:
                high_temp_order = self.two_opt_optimize(waypoints, high_temp_order)
            high_temp_result = self._create_route_result(starts, waypoints, ends, si, ei, high_temp_order)
            
            # 尝试低初始温度
            self.config.initial_temperature = original_temp / 2
            low_temp_order = self.simulated_annealing(waypoints)
            if self.config.enable_local_search:
                low_temp_order = self.two_opt_optimize(waypoints, low_temp_order)
            low_temp_result = self._create_route_result(starts, waypoints, ends, si, ei, low_temp_order)
            
            # 恢复原始温度设置
            self.config.initial_temperature = original_temp
            
            # 添加多样化的方案（只添加与最优方案有明显不同的方案）
            for result in [high_temp_result, low_temp_result]:
                # 检查结果是否与已有结果明显不同（距离差异超过5%或路径顺序不同）
                is_different = True
                for existing_result in results:
                    # 检查距离差异
                    dist_diff = abs(result.total_distance - existing_result.total_distance) / existing_result.total_distance
                    
                    # 检查路径顺序是否相同
                    same_order = True
                    if len(result.path) == len(existing_result.path) and len(result.path) > 2:
                        # 比较途经点顺序（忽略起点和终点）
                        for i in range(1, len(result.path) - 1):
                            if result.path[i] != existing_result.path[i]:
                                same_order = False
                                break
                    
                    # 如果距离差异小于5%且路径顺序相同，则认为是相同方案
                    if dist_diff < 0.05 and same_order:
                        is_different = False
                        break
                
                # 如果是不同的方案，则添加到结果列表
                if is_different:
                    results.append(result)
        else:
            # 只有一个途经点时，直接生成路径
            single_waypoint_order = [0] if waypoints else []
            result = self._create_route_result(starts, waypoints, ends, si, ei, single_waypoint_order)
            results.append(result)
        
        return results
        
    def _create_route_result(self, starts: List[Coordinate], waypoints: List[Coordinate], 
                            ends: List[Coordinate], si: int, ei: int, 
                            waypoint_order: List[int]) -> RouteResult:
        # 创建路径规划结果对象
        # 构建完整路径
        full_path = [starts[si]]
        full_path.extend(self.extract_path(waypoints, waypoint_order))
        full_path.append(ends[ei])
        
        # 计算总距离
        total_dist = self.calculate_path_distance(full_path)
        
        # 准备结果
        result = RouteResult()
        result.path = full_path
        result.total_distance = total_dist
        result.start_candidate_index = si
        result.end_candidate_index = ei
        result.starts = starts
        result.waypoints = waypoints
        result.ends = ends
        
        # 调用高德地图API计算驾车距离和时间
        driving_distance, driving_time, route_points = self.calculate_driving_info(full_path)
        result.driving_distance = driving_distance
        result.driving_time = driving_time
        
        # 保存路线点数据
        result.route_points = route_points
        
        return result

    def calculate_driving_info(self, path: List[Coordinate]) -> Tuple[float, float, List[Tuple[float, float]]]:
        # 使用高德地图API计算驾车距离和时间及路线点
        if len(path) < 2:
            return 0.0, 0.0, []
        
        try:
            # 提取途经点（如果有）
            waypoints = []
            if len(path) > 2:
                # 将路径中的中间点作为途经点
                waypoints = [(p.longitude, p.latitude) for p in path[1:-1]]
            
            # 调用高德地图API，传入起点、终点和途经点
            result = amap_api.get_driving_info(
                origin=(path[0].longitude, path[0].latitude),
                destination=(path[-1].longitude, path[-1].latitude),
                waypoints=waypoints
            )
            
            if result:
                distance = float(result.get('distance', 0))
                duration = float(result.get('duration', 0))
                
                # 解析polyline数据
                route_points = []
                if 'polyline' in result and result['polyline']:
                    route_points = amap_api.parse_polyline(result['polyline'])
                
                return distance, duration, route_points
        except Exception as e:
            # 静默处理API调用错误
            pass
            
        # 如果API调用失败，返回默认值
        return 0.0, 0.0, []
        
    def plan_route(self, starts: List[Coordinate], waypoints: List[Coordinate], 
                  ends: List[Coordinate]) -> List[RouteResult]:
        # 规划最优路线 - 支持组合剪枝和并行计算
        start_time = time.time()

        # 存储Top N个最优结果
        top_results: List[RouteResult] = []
        
        # 线程安全的共享变量，用于存储当前最优解
        # 使用锁来保护对共享变量的访问
        best_distance_lock = threading.Lock()
        shared_best_distance = [float('inf')]  # 使用列表作为可变对象
        
        # 生成所有需要处理的起点-终点组合
        combinations = [(si, ei) for si in range(len(starts)) for ei in range(len(ends))]
        
        # 预处理：如果有大量组合，先进行一次快速估算，找出可能的好组合
        # 这对于大数据量特别有效
        if self.config.enable_pruning and len(combinations) > 100:
            # 快速估算所有组合的近似距离（起点到终点的直线距离）
            # 直线距离通过Haversine公式在本地计算，不依赖外部API
            estimated_combinations = []
            for si, ei in combinations:
                # 使用本地的DistanceCalculator计算直线距离，不依赖外部API
                approx_dist = self.dist_calc.calculate(starts[si], ends[ei])
                estimated_combinations.append((approx_dist, si, ei))
            
            # 按照近似距离排序，优先处理看起来更优的组合
            estimated_combinations.sort(key=lambda x: x[0])
            
            # 只保留前80%的组合，剪枝掉明显不好的组合
            retention_ratio = 0.8
            retention_count = max(10, int(len(estimated_combinations) * retention_ratio))
            combinations = [(si, ei) for _, si, ei in estimated_combinations[:retention_count]]
        
        # 定义并行处理的包装函数，支持更新共享的最优解
        def process_combination_with_shared_state(si, ei):
            # 获取当前共享的最优解
            with best_distance_lock:
                current_best = shared_best_distance[0]
            
            # 处理单个组合，获取多个可能的方案
            results = self._process_route_combination(starts, waypoints, ends, si, ei, current_best)
            
            # 如果找到更好的解，更新共享状态
            for result in results:
                if result and result.total_distance < shared_best_distance[0]:
                    with best_distance_lock:
                        if result.total_distance < shared_best_distance[0]:  # 双重检查锁定模式
                            shared_best_distance[0] = result.total_distance
            
            return results
        
        # 如果启用了并行计算
        if self.config.parallel_workers > 1 and len(combinations) > self.config.parallel_workers:
            # 使用线程池进行并行计算
            # 根据数据规模动态调整线程数，避免过多线程导致的上下文切换开销
            effective_workers = min(self.config.parallel_workers, max(1, len(combinations) // 10))
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=effective_workers) as executor:
                # 提交所有任务
                future_to_combination = {
                    executor.submit(process_combination_with_shared_state, si, ei): 
                    (si, ei) for si, ei in combinations
                }
                
                # 处理完成的任务
                completed_count = 0
                total_combinations = len(combinations)
                
                for future in concurrent.futures.as_completed(future_to_combination):
                    try:
                            results = future.result()
                            if results:
                                # 添加所有结果到Top N结果中
                                with best_distance_lock:
                                    top_results.extend(results)
                                    # 保持Top N排序
                                    top_results.sort()
                                    # 只保留Top N个结果
                                    if len(top_results) > self.config.top_n_results:
                                        top_results = top_results[:self.config.top_n_results]
                    except Exception as e:
                        # 静默处理异常
                        pass
                    finally:
                        completed_count += 1
                        
                        # 定期更新当前最优解，提高剪枝效率
                        if completed_count % 10 == 0 and top_results:
                            with best_distance_lock:
                                if top_results and top_results[0].total_distance < shared_best_distance[0]:
                                    shared_best_distance[0] = top_results[0].total_distance
        else:
            # 串行计算
            for si, ei in combinations:
                # 检查是否需要剪枝
                if self.config.enable_pruning and shared_best_distance[0] != float('inf'):
                    approx_dist = self.dist_calc.calculate(starts[si], ends[ei])
                    if approx_dist > shared_best_distance[0] * self.config.pruning_threshold:
                        continue
                
                results = self._process_route_combination(starts, waypoints, ends, si, ei, shared_best_distance[0])
                if results:
                    # 更新当前最优解
                    for result in results:
                        if result.total_distance < shared_best_distance[0]:
                            shared_best_distance[0] = result.total_distance
                    
                    # 添加所有结果到Top N结果中
                    top_results.extend(results)
                    # 保持Top N排序
                    top_results.sort()
                    # 只保留Top N个结果
                    if len(top_results) > self.config.top_n_results:
                        top_results = top_results[:self.config.top_n_results]
        
        # 计算执行时间
        end_time = time.time()
        exec_time = (end_time - start_time) * 1000.0  # 转换为毫秒
        
        # 设置每个结果的执行时间
        for result in top_results:
            result.execution_time_ms = exec_time
        
        return top_results

    def clear_distance_cache(self) -> None:
        # 清除距离缓存
        self.dist_calc.clear_distance_cache()

# ===== 可视化配置界面 =====
class InteractiveRoutePlannerApp:
    def __init__(self, root):
        # 初始化根窗口
        self.root = root
        self.root.title("智能路径规划系统")
        self.root.geometry("1200x800")
        
        # 设置窗口关闭事件处理函数
        self.root.protocol("WM_DELETE_WINDOW", lambda: sys.exit(0))
        
        # 初始化优化目标属性
        self.optimization_target = tk.StringVar(value='straight_line')
        
        # 数据存储
        self.starts = []
        self.waypoints = []
        self.ends = []
        self.results = []
        self.selected_result_index = 0
        self.points_file_path = "points.txt"
        
        # 初始化路径规划器
        self.config = AlgorithmConfig()  # 使用默认的速度级别3和准确度级别3
        self.planner = RoutePlanner(self.config)
        
        # 初始化高德地图显示对象
        self.map_display = AmapMapDisplay()
        
        # 创建界面
        self._create_widgets()
        
        # 尝试加载默认数据
        self._load_default_data()
        
    def _create_widgets(self):
        # 创建主框架
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # 创建顶部控制面板
        control_frame = ttk.LabelFrame(main_frame, text="控制面板", padding="10")
        control_frame.pack(fill=tk.X, pady=(0, 10))
        
        # 创建左侧配置面板
        self.left_frame = ttk.LabelFrame(main_frame, text="参数配置", padding="10", width=300)
        self.left_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        self.left_frame.pack_propagate(False)  # 防止子组件改变框架大小
        
        # 创建右侧地图和结果面板
        self.right_frame = ttk.Frame(main_frame)
        self.right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # 创建地图面板
        self.map_frame = ttk.LabelFrame(self.right_frame, text="地图可视化", padding="10")
        self.map_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        # 创建结果面板
        self.result_frame = ttk.LabelFrame(self.right_frame, text="规划结果", padding="10")
        self.result_frame.pack(fill=tk.BOTH, expand=True)
        
        # --- 控制面板 ---
        # 加载文件按钮
        ttk.Button(control_frame, text="加载坐标文件", command=self._load_file).pack(side=tk.LEFT, padx=5)
        
        # 生成测试数据按钮
        ttk.Button(control_frame, text="生成测试数据", command=self._generate_test_data).pack(side=tk.LEFT, padx=5)
        
        # 规划路径按钮
        ttk.Button(control_frame, text="开始路径规划", command=self._start_planning).pack(side=tk.LEFT, padx=5)
        
        # 清除缓存按钮
        ttk.Button(control_frame, text="清除距离缓存", command=self._clear_cache).pack(side=tk.LEFT, padx=5)
        
        # 优化目标选择
        optimization_frame = ttk.Frame(control_frame)
        optimization_frame.pack(side=tk.LEFT, padx=5)
        ttk.Label(optimization_frame, text="优化目标:").pack(side=tk.LEFT)
        ttk.Radiobutton(optimization_frame, text="驾车距离", variable=self.optimization_target, value='driving_distance', command=self._on_optimization_target_change).pack(side=tk.LEFT)
        ttk.Radiobutton(optimization_frame, text="驾车时间", variable=self.optimization_target, value='driving_time', command=self._on_optimization_target_change).pack(side=tk.LEFT)
        ttk.Radiobutton(optimization_frame, text="直线距离", variable=self.optimization_target, value='straight_line', command=self._on_optimization_target_change).pack(side=tk.LEFT)
        
        # --- 参数配置面板 ---
        # 当前配置描述
        self.config_description_var = tk.StringVar(value=self.config.get_level_description())
        ttk.Label(self.left_frame, textvariable=self.config_description_var, wraplength=280, justify=tk.LEFT).pack(anchor=tk.W, pady=5)
        
        # 速度级别滑块
        ttk.Label(self.left_frame, text="速度级别 (1-最快, 5-最佳)").pack(anchor=tk.W, pady=2)
        speed_frame = ttk.Frame(self.left_frame)
        speed_frame.pack(fill=tk.X, pady=2)
        self.speed_level_var = tk.IntVar(value=self.config.speed_level)
        ttk.Scale(speed_frame, from_=1, to=5, orient=tk.HORIZONTAL, variable=self.speed_level_var, 
                 command=lambda v: (self.speed_level_var.set(int(float(v))), self._update_levels())).pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Label(speed_frame, textvariable=self.speed_level_var, width=3).pack(side=tk.RIGHT)
        
        # 速度级别描述
        speed_labels_frame = ttk.Frame(self.left_frame)
        speed_labels_frame.pack(fill=tk.X, pady=1)
        ttk.Label(speed_labels_frame, text="极速", width=4).pack(side=tk.LEFT)
        ttk.Label(speed_labels_frame, text="快速", width=4).pack(side=tk.LEFT)
        ttk.Label(speed_labels_frame, text="平衡", width=4).pack(side=tk.LEFT)
        ttk.Label(speed_labels_frame, text="精准", width=4).pack(side=tk.LEFT)
        ttk.Label(speed_labels_frame, text="最佳", width=4).pack(side=tk.LEFT)
        
        # 准确度级别滑块
        ttk.Label(self.left_frame, text="准确度级别 (1-基础, 5-超高精度)").pack(anchor=tk.W, pady=2)
        accuracy_frame = ttk.Frame(self.left_frame)
        accuracy_frame.pack(fill=tk.X, pady=2)
        self.accuracy_level_var = tk.IntVar(value=self.config.accuracy_level)
        ttk.Scale(accuracy_frame, from_=1, to=5, orient=tk.HORIZONTAL, variable=self.accuracy_level_var, 
                 command=lambda v: (self.accuracy_level_var.set(int(float(v))), self._update_levels())).pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Label(accuracy_frame, textvariable=self.accuracy_level_var, width=3).pack(side=tk.RIGHT)
        
        # 准确度级别描述
        accuracy_labels_frame = ttk.Frame(self.left_frame)
        accuracy_labels_frame.pack(fill=tk.X, pady=1)
        ttk.Label(accuracy_labels_frame, text="基础", width=4).pack(side=tk.LEFT)
        ttk.Label(accuracy_labels_frame, text="一般", width=4).pack(side=tk.LEFT)
        ttk.Label(accuracy_labels_frame, text="标准", width=4).pack(side=tk.LEFT)
        ttk.Label(accuracy_labels_frame, text="高精", width=4).pack(side=tk.LEFT)
        ttk.Label(accuracy_labels_frame, text="超高", width=4).pack(side=tk.LEFT)
        
        # 其他基本参数
        ttk.Separator(self.left_frame).pack(fill=tk.X, pady=10)
        
        # 并行线程数
        ttk.Label(self.left_frame, text="并行线程数:").pack(anchor=tk.W, pady=2)
        self.parallel_workers_var = tk.IntVar(value=self.config.parallel_workers)
        ttk.Scale(self.left_frame, from_=1, to=16, orient=tk.HORIZONTAL, variable=self.parallel_workers_var, 
                 command=lambda v: (self.parallel_workers_var.set(int(float(v))), self._update_param("parallel_workers", int(float(v))))).pack(fill=tk.X, pady=2)
        ttk.Label(self.left_frame, textvariable=self.parallel_workers_var).pack(anchor=tk.W, pady=1)
        
        # 输出Top N结果
        ttk.Label(self.left_frame, text="输出Top N结果:").pack(anchor=tk.W, pady=2)
        self.top_n_results_var = tk.IntVar(value=self.config.top_n_results)
        ttk.Scale(self.left_frame, from_=1, to=10, orient=tk.HORIZONTAL, variable=self.top_n_results_var, 
                 command=lambda v: (self.top_n_results_var.set(int(float(v))), self._update_param("top_n_results", int(float(v))))).pack(fill=tk.X, pady=2)
        ttk.Label(self.left_frame, textvariable=self.top_n_results_var).pack(anchor=tk.W, pady=1)
        
        # --- 地图面板 ---
        # 地图信息标签
        self.map_info_var = tk.StringVar(value="地图正在加载中...")
        ttk.Label(self.map_frame, textvariable=self.map_info_var, wraplength=700, justify=tk.LEFT).pack(fill=tk.X, pady=5)
        
        # 创建地图容器，用于嵌入PyQt地图组件
        self.map_container = ttk.Frame(self.map_frame)
        self.map_container.pack(fill=tk.BOTH, expand=True, padx=0, pady=0)
        
        # 初始化地图视图组件标志
        self.map_web_view = None
        
        # 绑定地图容器大小变化事件
        self.map_container.bind('<Configure>', self._on_map_container_resize)
        
        # 如果没有PyQt5支持，显示提示信息
        if not ('USE_PYQT' in globals() and USE_PYQT):
            ttk.Label(self.map_frame, text="地图将在默认浏览器中打开").pack(fill=tk.X, pady=5)
        
        # --- 结果面板 ---
        # 结果方案选择下拉框
        self.result_combo_frame = ttk.Frame(self.result_frame)
        self.result_combo_frame.pack(fill=tk.X, pady=3)
        ttk.Label(self.result_combo_frame, text="选择方案:").pack(side=tk.LEFT, padx=5)
        self.result_combo = ttk.Combobox(self.result_combo_frame, state="readonly")
        self.result_combo.pack(fill=tk.X, padx=5, expand=True)
        self.result_combo.bind("<<ComboboxSelected>>", self._on_result_selected)
        
        # 结果详情文本框
        self.result_text = tk.Text(self.result_frame, height=10, wrap=tk.WORD)
        self.result_text.pack(fill=tk.BOTH, expand=True, pady=5)  
        # 滚动条
        scrollbar = ttk.Scrollbar(self.result_text, command=self.result_text.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.result_text.config(yscrollcommand=scrollbar.set)
        
    def _update_param(self, param_name, value):
        # 更新算法参数
        setattr(self.config, param_name, value)
        # 重新创建路径规划器以应用新参数
        self.planner = RoutePlanner(self.config)
        
    def _update_levels(self):
        # 更新速度和准确度级别，并重新计算算法参数
        speed_level = self.speed_level_var.get()
        accuracy_level = self.accuracy_level_var.get()
        
        # 更新配置中的级别参数
        self.config.update_levels(speed_level, accuracy_level)
        
        # 更新配置描述
        self.config_description_var.set(self.config.get_level_description())
        
        # 重新创建路径规划器以应用新参数
        self.planner = RoutePlanner(self.config)
    
    def _load_file(self):
        # 加载坐标文件
        file_path = filedialog.askopenfilename(title="选择坐标文件", filetypes=[("文本文件", "*.txt"), ("所有文件", "*.*")])
        if file_path:
            self.points_file_path = file_path
            try:
                starts, waypoints, ends = read_coordinates_from_file(file_path)
                self.starts = starts
                self.waypoints = waypoints
                self.ends = ends
                self._plot_points()
                messagebox.showinfo("成功", f"成功加载文件，读取到 {len(starts)} 个起点，{len(waypoints)} 个途经点，{len(ends)} 个终点")
            except Exception as e:
                messagebox.showerror("错误", f"加载文件失败: {str(e)}")
        
    def _generate_test_data(self):
        # 生成测试数据对话框
        dialog = tk.Toplevel(self.root)
        dialog.title("生成测试数据")
        dialog.geometry("300x200")
        dialog.transient(self.root)
        dialog.grab_set()
        
        # 起点数量
        ttk.Label(dialog, text="起点数量:").grid(row=0, column=0, sticky=tk.W, padx=10, pady=5)
        start_spinbox = ttk.Spinbox(dialog, from_=1, to=100, width=10)
        start_spinbox.grid(row=0, column=1, padx=10, pady=5)
        start_spinbox.delete(0, "end")
        start_spinbox.insert(0, "2")
        
        # 途经点数量
        ttk.Label(dialog, text="途经点数量:").grid(row=1, column=0, sticky=tk.W, padx=10, pady=5)
        waypoint_spinbox = ttk.Spinbox(dialog, from_=0, to=500, width=10)
        waypoint_spinbox.grid(row=1, column=1, padx=10, pady=5)
        waypoint_spinbox.delete(0, "end")
        waypoint_spinbox.insert(0, "5")
        
        # 终点数量
        ttk.Label(dialog, text="终点数量:").grid(row=2, column=0, sticky=tk.W, padx=10, pady=5)
        end_spinbox = ttk.Spinbox(dialog, from_=1, to=100, width=10)
        end_spinbox.grid(row=2, column=1, padx=10, pady=5)
        end_spinbox.delete(0, "end")
        end_spinbox.insert(0, "2")
        
        # 确定按钮
        def on_ok():
            try:
                starts_num = int(start_spinbox.get())
                waypoints_num = int(waypoint_spinbox.get())
                ends_num = int(end_spinbox.get())
                
                # 生成测试数据
                self._generate_sample_data(starts_num, waypoints_num, ends_num)
                dialog.destroy()
            except ValueError:
                messagebox.showerror("错误", "请输入有效的数字")
        
        ttk.Button(dialog, text="确定", command=on_ok).grid(row=3, column=0, columnspan=2, pady=10)
        
    def _is_land_in_zhejiang(self, lon, lat):
        # 判断坐标是否在浙江省陆地区域内（简化版）
        # 浙江范围：经度118.5-122.5，纬度27.5-31.0
        if (lon < 118.5 or lon > 122.5 or lat < 27.5 or lat > 31.0):
            return False
        
        # 避免杭州湾区域
        if (lon > 120.5 and lon < 121.5 and lat > 30.0 and lat < 30.8):
            coastline_lat = -0.05 * lon + 36.3
            if lat < coastline_lat:
                return False
        
        # 避免温州以东海域
        if (lon > 121.1 and lat < 28.1):
            return False
        
        # 避免宁波以东海域
        if (lon > 121.8 and lat < 29.8):
            return False
        
        return True
        
    def _generate_sample_data(self, starts_num, waypoints_num, ends_num):
        # 生成浙江省范围内的随机测试数据
        self.starts = []
        self.waypoints = []
        self.ends = []
        
        # 最大尝试次数，避免无限循环
        max_attempts = 1000
        
        # 生成起点
        attempts = 0
        while len(self.starts) < starts_num and attempts < max_attempts:
            attempts += 1
            lon = round(random.uniform(118.5, 122.5), 6)
            lat = round(random.uniform(27.5, 31.0), 6)
            if self._is_land_in_zhejiang(lon, lat):
                self.starts.append(Coordinate(lon, lat))
        
        # 生成途经点
        attempts = 0
        while len(self.waypoints) < waypoints_num and attempts < max_attempts:
            attempts += 1
            lon = round(random.uniform(118.5, 122.5), 6)
            lat = round(random.uniform(27.5, 31.0), 6)
            if self._is_land_in_zhejiang(lon, lat):
                self.waypoints.append(Coordinate(lon, lat))
        
        # 生成终点
        attempts = 0
        while len(self.ends) < ends_num and attempts < max_attempts:
            attempts += 1
            lon = round(random.uniform(118.5, 122.5), 6)
            lat = round(random.uniform(27.5, 31.0), 6)
            if self._is_land_in_zhejiang(lon, lat):
                self.ends.append(Coordinate(lon, lat))
        
        # 如果无法生成足够的陆地坐标，使用默认范围（作为后备方案）
        if len(self.starts) < starts_num:
            remaining = starts_num - len(self.starts)
            for _ in range(remaining):
                lon = round(random.uniform(118.0, 123.0), 6)
                lat = round(random.uniform(27.0, 31.0), 6)
                self.starts.append(Coordinate(lon, lat))
        
        if len(self.waypoints) < waypoints_num:
            remaining = waypoints_num - len(self.waypoints)
            for _ in range(remaining):
                lon = round(random.uniform(118.0, 123.0), 6)
                lat = round(random.uniform(27.0, 31.0), 6)
                self.waypoints.append(Coordinate(lon, lat))
        
        if len(self.ends) < ends_num:
            remaining = ends_num - len(self.ends)
            for _ in range(remaining):
                lon = round(random.uniform(118.0, 123.0), 6)
                lat = round(random.uniform(27.0, 31.0), 6)
                self.ends.append(Coordinate(lon, lat))
        
        self._plot_points()
        
        # 将生成的坐标保存到points.txt文件
        try:
            with open(self.points_file_path, 'w', encoding='utf-8') as f:
                f.write('# 路径规划系统生成的测试数据\n')
                f.write('# 生成时间: ' + time.strftime('%Y-%m-%d %H:%M:%S') + '\n\n')
                
                # 写入起点
                f.write('[starts]\n')
                for coord in self.starts:
                    f.write(f'{coord.longitude},{coord.latitude}\n')
                f.write('\n')
                
                # 写入途经点
                if self.waypoints:
                    f.write('[waypoints]\n')
                    for coord in self.waypoints:
                        f.write(f'{coord.longitude},{coord.latitude}\n')
                    f.write('\n')
                
                # 写入终点
                f.write('[ends]\n')
                for coord in self.ends:
                    f.write(f'{coord.longitude},{coord.latitude}\n')
                
            messagebox.showinfo("成功", f"成功生成测试数据并保存到points.txt:\n{starts_num}个起点, {waypoints_num}个途经点, {ends_num}个终点")
        except Exception as e:
            messagebox.showerror("错误", f"保存数据到文件失败: {str(e)}")
        
    def _load_default_data(self):
        # 尝试加载默认数据并初始化地图
        # 使用延迟执行，确保UI已经完全初始化
        self.root.after(1000, self._show_map)
        
        # 尝试加载默认数据
        if os.path.exists(self.points_file_path):
            try:
                starts, waypoints, ends = read_coordinates_from_file(self.points_file_path)
                self.starts = starts
                self.waypoints = waypoints
                self.ends = ends
                self._plot_points()
                
                # 更新地图显示
                if hasattr(self, 'map_display'):
                    self.map_display.update_points(
                        [(s.longitude, s.latitude) for s in starts],
                        [(w.longitude, w.latitude) for w in waypoints],
                        [(e.longitude, e.latitude) for e in ends]
                    )
                    
            except Exception as e:
                pass
        
    def _start_planning(self):
        # 开始路径规划
        if not self.starts or not self.ends:
            messagebox.showwarning("警告", "请先加载或生成坐标数据，确保有起点和终点")
            return
        
        # 显示等待对话框
        wait_dialog = tk.Toplevel(self.root)
        wait_dialog.title("路径规划中")
        wait_dialog.geometry("300x100")
        wait_dialog.transient(self.root)
        wait_dialog.grab_set()
        
        ttk.Label(wait_dialog, text="正在进行路径规划，请稍候...").pack(pady=20)
        
        # 更新界面以显示等待状态
        self.root.update()
        
        try:
            # 执行路径规划
            self.results = self.planner.plan_route(self.starts, self.waypoints, self.ends)
            if self.results:
                # 首先根据当前的优化目标排序结果
                self._sort_results_by_target()
                
                # 设置当前选中的结果索引
                self.selected_result_index = 0
                
                # 更新结果下拉框选项
                result_names = self._generate_result_names()
                self.result_combo['values'] = result_names
                if result_names:
                    self.result_combo.current(0)
                
                # 直接显示第一个结果
                self._display_result(0)
            else:
                messagebox.showinfo("信息", "未找到可行的路径方案")
        except Exception as e:
            messagebox.showerror("错误", f"路径规划过程中发生错误: {str(e)}")
        finally:
            wait_dialog.destroy()
        
    def _on_optimization_target_change(self):
        # 当优化目标改变时重新排序并显示结果
        if self.results:
            # 重新排序结果
            self._sort_results_by_target()
            
            # 更新结果下拉框选项以反映新的优化目标
            result_names = self._generate_result_names()
            self.result_combo['values'] = result_names
            if result_names and 0 <= self.selected_result_index < len(result_names):
                self.result_combo.current(self.selected_result_index)
            
            # 如果当前有选中的结果，重新显示
            if 0 <= self.selected_result_index < len(self.results):
                self._display_result(self.selected_result_index)
        
    def _sort_results_by_target(self):
        # 根据当前的优化目标排序结果
        target = self.optimization_target.get()
        if target == 'driving_distance':
            # 按驾车距离升序排序
            self.results.sort(key=lambda x: x.driving_distance if hasattr(x, 'driving_distance') and x.driving_distance > 0 else float('inf'))
        elif target == 'driving_time':
            # 按驾车时间升序排序
            self.results.sort(key=lambda x: x.driving_time if hasattr(x, 'driving_time') and x.driving_time > 0 else float('inf'))
        elif target == 'straight_line':
            # 按直线距离升序排序
            self.results.sort(key=lambda x: x.total_distance)
            
    def _generate_result_names(self):
        # 根据当前的优化目标生成结果下拉框的显示名称
        target = self.optimization_target.get()
        if target == 'driving_distance':
            return [f"方案 {i+1}: 驾车距离 {result.driving_distance/1000:.2f} 公里" if hasattr(result, 'driving_distance') and result.driving_distance > 0 else 
                    f"方案 {i+1}: 无法获取驾车距离" for i, result in enumerate(self.results)]
        elif target == 'driving_time':
            return [f"方案 {i+1}: 驾车时间 {result.driving_time/60:.1f} 分钟" if hasattr(result, 'driving_time') and result.driving_time > 0 else 
                    f"方案 {i+1}: 无法获取驾车时间" for i, result in enumerate(self.results)]
        elif target == 'straight_line':
            return [f"方案 {i+1}: 直线距离 {result.total_distance/1000:.2f} 公里" for i, result in enumerate(self.results)]
        return []
        
    def _on_result_selected(self, event):
        # 选择不同的结果方案
        if self.results:
            selected_index = self.result_combo.current()
            if 0 <= selected_index < len(self.results):
                self.selected_result_index = selected_index
                self._display_result(selected_index)
        
    def _on_map_container_resize(self, event):
        # 处理地图容器大小变化事件，确保地图填满整个容器
        # 获取新的容器尺寸
        new_width = event.width
        new_height = event.height
        
        # 如果地图WebView已创建，调整其大小以填满容器
        if hasattr(self, 'map_web_view') and self.map_web_view is not None:
            try:
                # 调整WebView的大小为新的容器大小
                self.map_web_view.resize(new_width, new_height)
                # 确保地图填满整个容器
                self.map_web_view.setGeometry(0, 0, new_width, new_height)
            except Exception as e:
                    pass  # 忽略调整地图大小时的错误
    
    def _show_map(self):
        # 显示高德地图
        try:
            # 生成地图HTML并显示
            html_path = self.map_display.generate_map_html()
            
            # 检查文件是否存在
            if os.path.exists(html_path):
                # 读取并验证文件内容
                try:
                    with open(html_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                except Exception as e:
                    pass
            else:
                self.map_info_var.set(f"显示地图失败: HTML文件未创建成功")
                return
            
            if 'USE_PYQT' in globals() and USE_PYQT:
                try:
                    # 创建PyQt应用程序实例（如果不存在）
                    if QApplication.instance() is None:
                        app = QApplication([])
                    else:
                        app = QApplication.instance()
                    
                    # 检查地图视图组件是否已创建
                    if self.map_web_view is None:
                        # 创建QWebEngineView
                        self.map_web_view = QWebEngineView()
                        from PyQt5.QtCore import Qt
                        self.map_web_view.setWindowFlags(Qt.FramelessWindowHint)
                        # 如果有HWND支持，将地图嵌入到tkinter容器中
                        if 'HWND' in globals() and HWND:
                            # 获取tkinter容器的窗口句柄
                            container_handle = self.map_container.winfo_id()
                            
                            # 获取PyQt窗口的窗口句柄
                            qt_widget_handle = int(self.map_web_view.winId())    
                            # 使用win32gui将PyQt窗口嵌入到tkinter容器中
                            win32gui.SetParent(qt_widget_handle, container_handle)
                        else:
                            # 如果没有HWND支持，使用独立窗口
                            self.map_web_view.setWindowTitle("地图可视化 - 高德地图")
                            
                            # 设置无边框样式
                            from PyQt5.QtCore import Qt
                            self.map_web_view.setWindowFlags(Qt.FramelessWindowHint)
                            
                        # 确保容器有合理的尺寸
                        min_width = 800
                        min_height = 600
                        
                        # 获取容器的实际尺寸
                        container_width = self.map_container.winfo_width()
                        container_height = self.map_container.winfo_height()
                        
                        # 如果容器大小不够，调整容器大小
                        if container_width < min_width or container_height < min_height:
                            self.map_container.config(width=min_width, height=min_height)
                            self.map_container.update_idletasks()
                            container_width = min_width
                            container_height = min_height
                        
                        # 设置WebView的大小为容器大小
                        self.map_web_view.resize(container_width, container_height)
                        # 确保地图填满整个容器
                        self.map_web_view.setGeometry(0, 0, container_width, container_height)
                        
                        # 确保地图窗口在容器中左上角显示，使其充满整个容器
                        self.map_web_view.move(0, 0)
                    
                    # 加载地图HTML文件到已有的web_view
                    self.map_web_view.load(QUrl.fromLocalFile(html_path))
                    
                    # 显示WebView
                    self.map_web_view.show()
                    
                    # 对于独立窗口，实现窗口居中
                    if not ('HWND' in globals() and HWND):
                        # 获取屏幕可用几何区域
                        screen = app.desktop().availableGeometry()
                        # 获取窗口大小
                        window_size = self.map_web_view.size()
                        # 计算窗口居中位置
                        x = (screen.width() - window_size.width()) // 2
                        y = (screen.height() - window_size.height()) // 2
                        # 设置窗口位置
                        self.map_web_view.move(x, y)
                    
                    # 更新状态信息
                    if 'HWND' in globals() and HWND:
                        self.map_info_var.set(f"地图已在UI中显示")
                    else:
                        self.map_info_var.set(f"已在独立窗口中显示高德地图")
                    
                except Exception as e:
                    # 不再回退到浏览器打开
                    self.map_info_var.set(f"地图显示失败: {str(e)}")
            else:
                # 不再自动打开浏览器
                self.map_info_var.set(f"地图已准备就绪，请手动查看文件: {os.path.abspath(html_path)}")
        except Exception as e:
            self.map_info_var.set(f"显示地图失败: {str(e)}")
            messagebox.showerror("错误", f"显示地图失败: {str(e)}")
        
    def _display_result(self, index):
        # 显示选中的结果详情
        if 0 <= index < len(self.results):
            result = self.results[index]
            
            # 清空文本框
            self.result_text.delete(1.0, tk.END)
            
            # 添加结果详情
            self.result_text.insert(tk.END, f"===== 路径规划结果 (方案 {index+1}) =====\n")
            
            # 获取当前优化目标
            target = self.optimization_target.get()
            
            # 显示对应优化目标的结果
            if target == 'driving_distance' and hasattr(result, 'driving_distance') and result.driving_distance > 0:
                self.result_text.insert(tk.END, f"驾车距离: {result.driving_distance/1000:.2f} 公里\n")
            elif target == 'driving_time' and hasattr(result, 'driving_time') and result.driving_time > 0:
                self.result_text.insert(tk.END, f"驾车时间: {result.driving_time/60:.1f} 分钟\n")
            else:
                self.result_text.insert(tk.END, f"直线距离: {result.total_distance/1000:.2f} 公里\n")
            
            # 显示其他相关信息
            if hasattr(result, 'driving_distance') and result.driving_distance > 0:
                self.result_text.insert(tk.END, f"驾车距离: {result.driving_distance/1000:.2f} 公里\n")
            if hasattr(result, 'driving_time') and result.driving_time > 0:
                self.result_text.insert(tk.END, f"驾车时间: {result.driving_time/60:.1f} 分钟\n")
            
            self.result_text.insert(tk.END, f"直线距离: {result.total_distance/1000:.2f} 公里\n")
            self.result_text.insert(tk.END, f"计算耗时: {result.execution_time_ms:.0f} 毫秒\n")
            self.result_text.insert(tk.END, f"选择起点: 候选点 #{result.start_candidate_index+1}\n")
            self.result_text.insert(tk.END, f"选择终点: 候选点 #{result.end_candidate_index+1}\n")
            self.result_text.insert(tk.END, f"路线顺序 ({len(result.path)}个点):\n")
            
            for i, coord in enumerate(result.path):
                self.result_text.insert(tk.END, f"  {i+1}. 经度={coord.longitude:.6f}, 纬度={coord.latitude:.6f}\n")
            
            # 绘制选中的路径
            self._plot_route(result)

    def _plot_route(self, result: RouteResult):
        # 绘制路径，使用高德地图API
        # 清空地图显示对象中的数据
        self.map_display.clear_points()
        
        # 设置起点（绿色标记）
        start = result.path[0]
        self.map_display.add_start_point(start.longitude, start.latitude)
        
        # 设置终点（红色标记）
        end = result.path[-1]
        self.map_display.add_end_point(end.longitude, end.latitude)
        
        # 设置途经点（蓝色标记）
        if len(result.path) > 2:
            for i, point in enumerate(result.path[1:-1]):
                self.map_display.add_waypoint(point.longitude, point.latitude, f'途经点 {i+1}')
        
        # 设置路线点
        target = self.optimization_target.get()
        if target == 'driving_distance' or target == 'driving_time':
            # 对于驾车距离和驾车时间优化目标，优先显示实际驾车路线
            if hasattr(result, 'route_points') and result.route_points:
                # 使用实际驾车路线
                self.map_display.route_points = result.route_points
            else:
                # 如果没有route_points数据，则使用路径点作为路线
                route_points = [(p.longitude, p.latitude) for p in result.path]
                self.map_display.route_points = route_points
        else:
            # 对于直线距离优化目标，使用路径点作为路线
            route_points = [(p.longitude, p.latitude) for p in result.path]
            self.map_display.route_points = route_points
        
        # 获取起点和终点列表
        starts = [(result.path[0].longitude, result.path[0].latitude)]
        ends = [(result.path[-1].longitude, result.path[-1].latitude)]
        waypoints = [(p.longitude, p.latitude) for p in result.path[1:-1]]
        
        # 调用update_points方法更新地图数据
        self.map_display.update_points(starts, waypoints, ends, self.map_display.route_points)
        
        # 更新地图信息
        if target == 'driving_distance' and hasattr(result, 'driving_distance') and result.driving_distance > 0:
            title = f'最优路径规划 (驾车距离: {result.driving_distance/1000:.2f} 公里)'
        elif target == 'driving_time' and hasattr(result, 'driving_time') and result.driving_time > 0:
            title = f'最优路径规划 (驾车时间: {result.driving_time/60:.1f} 分钟)'
        else:
            title = f'最优路径规划 (直线距离: {result.total_distance/1000:.2f} 公里)'
        
        self.map_info_var.set(f"路径规划完成: {title}\n地图正在自动更新...")
        
        # 自动更新并显示地图
        try:
            # 生成地图HTML
            html_path = self.map_display.generate_map_html()
            
            # 检查是否使用PyQt
            if 'USE_PYQT' in globals() and USE_PYQT:
                        if hasattr(self, 'map_web_view') and self.map_web_view:
                            # 如果地图窗口已创建，直接更新内容
                            self.map_web_view.load(QUrl.fromLocalFile(html_path))
                            self.map_web_view.show()
                            if 'HWND' in globals() and HWND:
                                self.map_info_var.set(f"路径规划完成: {title}\n地图已在UI中自动更新，并已调整视野")
                            else:
                                self.map_info_var.set(f"路径规划完成: {title}\n地图已在独立窗口中自动更新，并已调整视野")
                        else:
                            # 否则调用_show_map方法创建窗口
                            self._show_map()
        except Exception as e:
            self.map_info_var.set(f"路径规划完成: {title}\n自动更新地图失败，请点击'查看地图'按钮")
        
    def _on_map_click(self):
        # 处理地图点击事件
        # 由于现在使用高德地图，此功能已由地图本身处理
        pass
        
    def _plot_points(self):
        # 绘制所有候选点（起点、途经点、终点），使用高德地图API
        # 清空地图显示对象中的数据
        self.map_display.clear_points()
        
        # 设置起点
        if hasattr(self, 'starts') and self.starts:
            # 如果有多个起点，只显示第一个作为示例
            start = self.starts[0]
            self.map_display.add_start_point(start.longitude, start.latitude)
            
            # 其他起点作为途经点显示
            for i, point in enumerate(self.starts[1:], 1):
                self.map_display.add_waypoint(point.longitude, point.latitude, f'起点候选 {i}')
        
        # 设置终点
        if hasattr(self, 'ends') and self.ends:
            # 如果有多个终点，只显示第一个作为示例
            end = self.ends[0]
            self.map_display.add_end_point(end.longitude, end.latitude)
            
            # 其他终点作为途经点显示
            for i, point in enumerate(self.ends[1:], 1):
                self.map_display.add_waypoint(point.longitude, point.latitude, f'终点候选 {i}')
        
        # 设置途经点
        if hasattr(self, 'waypoints') and self.waypoints:
            for i, point in enumerate(self.waypoints):
                self.map_display.add_waypoint(point.longitude, point.latitude, f'途经点 {i+1}')
        
        # 获取起点、途经点和终点列表
        starts_list = [(p.longitude, p.latitude) for p in self.starts]
        waypoints_list = [(p.longitude, p.latitude) for p in self.waypoints]
        ends_list = [(p.longitude, p.latitude) for p in self.ends]
        
        # 调用update_points方法更新地图数据
        self.map_display.update_points(starts_list, waypoints_list, ends_list)
        
        # 更新地图信息
        total_points = len(self.starts) + len(self.waypoints) + len(self.ends)
        self.map_info_var.set(f"已加载 {total_points} 个坐标点 (起点: {len(self.starts)}, 途经点: {len(self.waypoints)}, 终点: {len(self.ends)})\n地图正在自动更新...")
        
        # 自动更新并显示地图
        try:
            # 生成地图HTML
            html_path = self.map_display.generate_map_html()
            
            # 检查是否使用PyQt
            if 'USE_PYQT' in globals() and USE_PYQT:
                if hasattr(self, 'map_web_view') and self.map_web_view:
                    # 如果地图窗口已创建，直接更新内容
                    self.map_web_view.load(QUrl.fromLocalFile(html_path))
                    self.map_web_view.show()
                    if 'HWND' in globals() and HWND:
                        self.map_info_var.set(f"已加载 {total_points} 个坐标点 (起点: {len(self.starts)}, 途经点: {len(self.waypoints)}, 终点: {len(self.ends)})\n地图已在UI中自动更新，并已调整视野")
                    else:
                        self.map_info_var.set(f"已加载 {total_points} 个坐标点 (起点: {len(self.starts)}, 途经点: {len(self.waypoints)}, 终点: {len(self.ends)})\n地图已在独立窗口中自动更新，并已调整视野")
                else:
                    # 否则调用_show_map方法创建窗口
                    self._show_map()
        except Exception as e:
            self.map_info_var.set(f"已加载 {total_points} 个坐标点 (起点: {len(self.starts)}, 途经点: {len(self.waypoints)}, 终点: {len(self.ends)})\n自动更新地图失败，请稍候再试或刷新页面")
    
    def _clear_cache(self):
        # 清除距离缓存
        self.planner.clear_distance_cache()
        messagebox.showinfo("成功", "距离缓存已清除")

# ===== 从文件读取坐标 =====
def read_coordinates_from_file(filename: str) -> Tuple[List[Coordinate], List[Coordinate], List[Coordinate]]:
    # 从文本文件读取坐标数据（以#开头的行为注释，关键字[starts]、[waypoints]、[ends]分别标记不同部分）
    starts = []
    waypoints = []
    ends = []
    current_section = None  # 记录当前正在读取的部分
    
    try:
        with open(filename, 'r', encoding='utf-8') as file:
            for line_num, line in enumerate(file, 1):
                line = line.strip()
                
                # 跳过空行和注释
                if not line or line.startswith('#'):
                    continue
                
                # 检查是否是section标记
                if line.startswith('[') and line.endswith(']'):
                    section = line[1:-1].lower()
                    if section in ['starts', 'waypoints', 'ends']:
                        current_section = section
                    else:
                        pass  # 忽略未知标记
                    continue
                
                # 解析坐标点
                if current_section is None:
                    pass  # 忽略没有指定所属部分的坐标
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
                    pass  # 忽略格式错误的行
                    continue
    
    except FileNotFoundError:
        raise FileNotFoundError(f"文件 {filename} 不存在")
    except Exception as e:
        raise Exception(f"读取文件时发生错误: {str(e)}")
    
    # 验证至少有一个起点和一个终点
    if not starts:
        raise ValueError("文件中未找到任何起点数据")
    if not ends:
        raise ValueError("文件中未找到任何终点数据")
    
    return starts, waypoints, ends

# ===== 主函数 =====
def main():
    # 启动交互式界面
    root = tk.Tk()
    # 设置窗口标题
    root.title("智能路径规划系统")
    
    # 设置窗口图标 - minilogo.ico用于tk窗口图标
    try:
        import os
        # 获取当前脚本所在目录
        base_dir = os.path.dirname(os.path.abspath(__file__))
        # 构建minilogo.ico的绝对路径
        minilogo_path = os.path.join(base_dir, "icon/minilogo.ico")
        # 确保使用Windows兼容的路径格式
        minilogo_path = minilogo_path.replace('/', '\\')
        
        # 设置tk窗口图标为minilogo.ico
        root.iconbitmap(minilogo_path)
    except Exception as e:
        pass  # 忽略图标设置错误
    
    app = InteractiveRoutePlannerApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()