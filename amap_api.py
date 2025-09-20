#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
高德地图API客户端 - 用于获取驾车路径规划的距离、时间和路线
"""
import requests
import json
import time
from typing import Dict, Tuple, Optional, List

class AmapAPI:
    def __init__(self, api_key: str):
        """初始化高德地图API客户端"""
        self.api_key = api_key
        self.base_url = "https://restapi.amap.com/v3/direction/driving"
        # 添加缓存以避免重复请求
        self.cache = {}
        self.cache_timeout = 3600  # 缓存有效期（秒）
        
    def get_driving_info(self, origin: Tuple[float, float], destination: Tuple[float, float], waypoints: List[Tuple[float, float]] = None) -> Optional[Dict[str, any]]:
        """
        获取两点间的驾车距离和时间
        
        参数:
            origin: (longitude, latitude) - 起点坐标
            destination: (longitude, latitude) - 终点坐标
            
        返回:
            Dict - 包含distance（米）和duration（秒）的字典，如果请求失败则返回None
        """
        # 构建缓存键，包含途经点信息
        cache_key = (origin[0], origin[1], destination[0], destination[1]) + tuple((w[0], w[1]) for w in (waypoints or []))
        
        # 检查缓存
        current_time = time.time()
        if cache_key in self.cache:
            cached_data, timestamp = self.cache[cache_key]
            if current_time - timestamp < self.cache_timeout:
                return cached_data.copy()
            else:
                # 缓存已过期，移除
                del self.cache[cache_key]
        
        # 构建请求参数
        params = {
            'key': self.api_key,
            'origin': f"{origin[0]},{origin[1]}",
            'destination': f"{destination[0]},{destination[1]}",
            'output': 'json',
            'extensions': 'all'  # 添加此参数以获取完整的路线信息，包括polyline
        }
        
        # 如果有途经点，添加途经点参数，并添加drag前缀确保经过所有途经点
        if waypoints and len(waypoints) > 0:
            waypoints_str = ';'.join([f"{w[0]},{w[1]}" for w in waypoints])
            params['waypoints'] = f"{waypoints_str}"  # 在途经点前添加drag前缀
            print(f"发送给高德API的途经点=: {params['waypoints']}")  # 添加调试信息
        
        try:
            # 发送请求
            time.sleep(0.5)  # 避免请求频率过快
            response = requests.get(self.base_url, params=params, timeout=10)
            response.raise_for_status()
            
            # 解析响应
            data = response.json()
            
            # 检查响应状态
            if data.get('status') == '1' and data.get('route') and data['route'].get('paths'):
                path = data['route']['paths'][0]
                
                # 调试信息：打印获取到的完整路径数据
                # print(f"获取到的路径数据: distance={path.get('distance')}, duration={path.get('duration')}")
                # print(f"polyline数据长度: {len(path.get('polyline', ''))} 字符")
                # print(f"路径数据完整内容: {path}")  # 打印完整的path对象以检查polyline的位置
                # print(f"API响应完整内容: {data}")  # 打印完整的API响应以了解数据结构
                
                # 构建结果字典
                result = {
                    'distance': float(path.get('distance', 0)),  # 米
                    'duration': float(path.get('duration', 0)),  # 秒
                    'polyline': path.get('polyline', '')  # 添加polyline数据
                }
                
                # 检查是否有其他可能包含路线信息的字段
                if 'steps' in path:
                    print(f"路径包含{len(path['steps'])}个步骤")
                    # 尝试从steps中提取polyline数据
                    if not result['polyline']:
                        combined_polyline = []
                        for step in path['steps']:
                            if 'polyline' in step and step['polyline']:
                                combined_polyline.append(step['polyline'])
                        if combined_polyline:
                            result['polyline'] = ';'.join(combined_polyline)
                            print(f"从steps中合并polyline数据，长度: {len(result['polyline'])} 字符")
                    # 检查路径经过的途经点
                    if 'waypoints' in params:
                        print(f"请求的途经点: {params['waypoints']}")
                
                # 保存到缓存
                self.cache[cache_key] = (result, current_time)
                
                return result
            else:
                print(f"高德地图API返回错误: {data.get('info', '未知错误')}")
                return None
                
        except requests.exceptions.RequestException as e:
            print(f"请求高德地图API时出错: {str(e)}")
            return None
        except json.JSONDecodeError:
            print("解析高德地图API响应时出错")
            return None
    
    def parse_polyline(self, polyline_str: str) -> List[Tuple[float, float]]:
        """
        解析polyline字符串为坐标点列表
        
        参数:
            polyline_str: 高德地图API返回的polyline字符串
            
        返回:
            List[Tuple[float, float]]: 坐标点列表 [(longitude, latitude), ...]
        """
        if not polyline_str:
            return []
        
        coordinates = []
        try:
            # 分割polyline字符串
            points = polyline_str.split(';')
            for point in points:
                if point:
                    lon, lat = map(float, point.split(','))
                    coordinates.append((lon, lat))
        except Exception as e:
            print(f"解析polyline时出错: {e}")
            return []
        
        return coordinates
    
    def clear_cache(self) -> None:
        """清除缓存"""
        self.cache.clear()
        print("高德地图API缓存已清除")

# 创建全局API实例
# 注意：在实际应用中，建议将API Key存储在配置文件中
API_KEY = "*******************"  # 用户提供的API Key
amap_api = AmapAPI(API_KEY)