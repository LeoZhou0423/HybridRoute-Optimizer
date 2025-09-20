# -*- coding: utf-8 -*-
"""
高德地图底图显示模块 - 使用高德地图轻量级地图API显示路线和点
"""
import tkinter as tk
from tkinter import ttk
import os
import webbrowser
import tempfile
from typing import List, Tuple, Dict

class AmapMapDisplay:
    def __init__(self, parent=None, width=800, height=600):
        """初始化高德地图显示组件"""
        self.parent = parent
        self.width = width
        self.height = height
        
        # 用户提供的API密钥
        self.api_key = "********************"
        self.security_key = "********************"
        
        # 创建临时HTML文件用于显示地图
        self.html_file = None
        self.html_path = None
        
        # 初始化地图数据
        self.starts = []  # 起点坐标列表 [(lon, lat), ...]
        self.waypoints = []  # 途经点坐标列表 [(lon, lat), ...]
        self.ends = []  # 终点坐标列表 [(lon, lat), ...]
        self.route_points = []  # 路线坐标点列表 [(lon, lat), ...]
        self.point_labels = {}
        
        # 标记地图是否已初始化
        self._map_initialized = False
    
    def clear_points(self):
        """清空所有地图点"""
        self.starts = []
        self.waypoints = []
        self.ends = []
        self.route_points = []
        self.point_labels = {}
    
    def add_start_point(self, longitude, latitude, label='起点'):
        """添加起点"""
        # 转义标签中的特殊字符，防止破坏JavaScript语法
        safe_label = label.replace('"', '\\"').replace("'", "\\'")
        self.starts.append((longitude, latitude))
        self.point_labels[(longitude, latitude)] = safe_label
    
    def add_end_point(self, longitude, latitude, label='终点'):
        """添加终点"""
        safe_label = label.replace('"', '\\"').replace("'", "\\'")
        self.ends.append((longitude, latitude))
        self.point_labels[(longitude, latitude)] = safe_label
    
    def add_waypoint(self, longitude, latitude, label='途经点'):
        """添加途经点"""
        safe_label = label.replace('"', '\\"').replace("'", "\\'")
        self.waypoints.append((longitude, latitude))
        self.point_labels[(longitude, latitude)] = safe_label
        
    def _create_html_content(self) -> str:
        """生成包含高德地图API的HTML内容"""
        # 创建标签字典的JavaScript表示，确保正确转义
        labels_js = '{' + ',\n'.join([
            f'    "{p[0]},{p[1]}": "{self.point_labels[p]}"' 
            for p in self.point_labels
        ]) + '}'
        
        # 将坐标点转换为JavaScript数组格式
        starts_js = []
        for p in self.starts:
            labels_text = self.point_labels.get(p, '起点')
            starts_js.append(f'{{longitude: {p[0]}, latitude: {p[1]}, label: "{labels_text}"}}')
        starts_js = ', '.join(starts_js)
        
        waypoints_js = []
        for p in self.waypoints:
            labels_text = self.point_labels.get(p, '途经点')
            waypoints_js.append(f'{{longitude: {p[0]}, latitude: {p[1]}, label: "{labels_text}"}}')
        waypoints_js = ', '.join(waypoints_js)
        
        ends_js = []
        for p in self.ends:
            labels_text = self.point_labels.get(p, '终点')
            ends_js.append(f'{{longitude: {p[0]}, latitude: {p[1]}, label: "{labels_text}"}}')
        ends_js = ', '.join(ends_js)
        
        # 路线点转换为JavaScript数组
        route_points_js = ', '.join([
            f'{{longitude: {p[0]}, latitude: {p[1]}}}' 
            for p in self.route_points
        ])
        
        # 确保所有变量都正确定义，避免未定义错误
        len_starts = len(self.starts)
        len_waypoints = len(self.waypoints)
        len_ends = len(self.ends)
        len_route_points = len(self.route_points)
        
        # 构建HTML内容，注意所有括号和引号的闭合
        html_content = f'''
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>高德地图路径规划</title>
    <style>
        body, html {{ margin: 0; padding: 0; width: 100%; height: 100%; overflow: hidden; }}
        #container {{ width: 100%; height: 100%; }}

    </style>
</head>
<body>


    <div id="container"></div>
    <script>
        // 安全地加载高德地图API，包含重试机制
        function loadAMapAPI() {{ 
            const maxRetries = 3;
            let retryCount = 0; 
            
            return new Promise((resolve, reject) => {{ 
                function attemptLoad() {{ 
                    retryCount++; 
                    
                    const script = document.createElement('script'); 
                    script.type = 'text/javascript'; 
                    let apiUrl = 'https://webapi.amap.com/maps?v=2.0&key={self.api_key}&plugin=AMap.MarkerClusterer,AMap.PolylineEditor'; 
                    apiUrl += '&t=' + new Date().getTime(); 
                    
                    script.src = apiUrl; 
                    
                    script.onload = function() {{ 
                        console.log('高德地图API脚本加载成功，尝试次数:', retryCount); 
                        resolve(); 
                    }}; 
                    
                    script.onerror = function() {{ 
                        console.error('高德地图API脚本加载失败，尝试次数:', retryCount); 
                        if (retryCount < maxRetries) {{ 
                            console.log('正在重试...（' + retryCount + '/' + maxRetries + '）'); 
                            document.head.removeChild(script); 
                            setTimeout(attemptLoad, 1000 * retryCount); 
                        }} else {{ 
                            reject(new Error('无法加载高德地图API，请检查网络连接和API密钥')); 
                        }} 
                    }}; 
                    
                    document.head.appendChild(script); 
                }} 
                attemptLoad(); 
            }}); 
        }} 
        
        // 检测地图瓦片加载状态
        function checkTileLoading() {{ 
            let tileLoadChecks = 0; 
            const maxChecks = 10;
            const checkInterval = 1000;
            
            const checkTimer = setInterval(() => {{ 
                tileLoadChecks++; 
                try {{ 
                    const container = document.getElementById('container'); 
                    if (container) {{ 
                        const tileElements = container.querySelectorAll('img[src*="amap.com"]'); 
                        console.log('地图瓦片元素数量:', tileElements.length); 
                         
                        if (tileElements.length > 5) {{ 
                            console.log('地图瓦片加载成功'); 
                            clearInterval(checkTimer); 
                            return; 
                        }} 
                    }} 
                }} catch (e) {{ 
                    console.error('检查地图瓦片加载状态时出错:', e); 
                }} 
                
                if (tileLoadChecks >= maxChecks) {{ 
                    console.warn('地图瓦片加载不完整，尝试刷新地图...'); 
                    clearInterval(checkTimer); 
                }} 
            }}, checkInterval); 
        }} 
        
        // 检查网络连接状态
        function checkNetworkStatus() {{
            return new Promise((resolve, reject) => {{
                const xhr = new XMLHttpRequest(); 
                xhr.open('GET', 'https://webapi.amap.com', true);
                xhr.timeout = 5000;
                
                xhr.onload = function() {{ 
                    console.log('网络连接检查成功'); 
                    resolve(true); 
                }}; 
                
                xhr.onerror = function() {{ 
                    console.error('网络连接检查失败'); 
                    reject(new Error('网络连接问题，请检查您的网络设置')); 
                }}; 
                
                xhr.ontimeout = function() {{ 
                    console.error('网络连接超时'); 
                    reject(new Error('网络连接超时，请检查您的网络设置')); 
                }}; 
                
                xhr.send(); 
            }}); 
        }}

        // 创建标记点函数
        function createMarker(point, color, labelText) {{ 
            try {{ 
                console.log('创建标记点:', point, '颜色:', color, '标签:', labelText);
                var marker = new AMap.Marker({{ 
                    position: [point.longitude, point.latitude],
                    icon: new AMap.Icon({{ 
                        image: 'data:image/svg+xml;charset=utf-8,%3Csvg xmlns="http://www.w3.org/2000/svg" width="36" height="36" viewBox="0 0 36 36"%3E%3Ccircle cx="18" cy="18" r="16" fill="' + encodeURIComponent(color) + '" stroke="%23fff" stroke-width="2" opacity="0.9"%3E%3C/circle%3E%3C/svg%3E',
                        size: new AMap.Size(36, 36),
                        imageSize: new AMap.Size(36, 36)
                    }}),
                    offset: new AMap.Pixel(-18, -18),
                    title: labelText
                }});                
                if (window.map) {{ 
                    marker.setMap(window.map);
                    console.log('标记点已添加到地图');
                }} else {{ 
                    console.error('地图对象不存在，无法添加标记点');
                }}                
                return marker;
            }} catch (e) {{ 
                console.error('创建标记点失败:', e, '坐标点:', point);
                try {{ 
                    console.log('尝试使用备选标记样式...');
                    var simpleMarker = new AMap.Marker({{ 
                        position: [point.longitude, point.latitude],
                        title: labelText
                    }});
                    if (window.map) {{ 
                        simpleMarker.setMap(window.map);
                    }}
                    return simpleMarker;
                }} catch (e2) {{ 
                    console.error('备选标记样式也创建失败:', e2);
                }}
                throw e;
            }}
        }}        
        // 等待DOM和API加载完成
        document.addEventListener('DOMContentLoaded', async function() {{ 
            window.map = null;
            
            try {{ 
                console.log('DOM已加载，开始检查网络连接...'); 
                
                try {{
                    await checkNetworkStatus();
                }} catch (networkError) {{
                    console.warn('网络连接检查失败，但继续尝试加载地图:', networkError);
                }}
                
                if ('{self.api_key}' === '') {{
                    console.error('API密钥为空！这可能导致地图无法加载');
                }} else {{
                    console.log('API密钥已配置');
                }}
                
                console.log('开始加载地图API...');                
                await loadAMapAPI();
                console.log('地图API加载成功');
                
                
                console.log('API密钥: [已配置]');
                console.log('起点数量:', {len_starts});
                console.log('途经点数量:', {len_waypoints});
                console.log('终点数量:', {len_ends});
                console.log('路线点数量:', {len_route_points});                
                
                if (typeof AMap === 'undefined') {{ 
                    throw new Error('高德地图API加载失败，AMap对象未定义');
                }}        
                console.log('创建地图实例...');
                
                try {{
                    window.map = new AMap.Map('container', {{ 
                        resizeEnable: true,
                        zoom: 9,
                        center: [120.0, 29.0],
                        viewMode: '2D'
                    }});                
                    
                    window.addEventListener('resize', function() {{
                        if (window.map && window.map.resize) {{
                            window.map.resize();
                        }}
                    }});
                    
                    setTimeout(function() {{
                        if (window.map && window.map.resize) {{
                            window.map.resize();
                        }}
                    }}, 100);                
                    console.log('地图实例创建成功！');
                }} catch (mapCreateError) {{
                    console.error('地图实例创建失败，尝试使用备用配置:', mapCreateError);
                    
                    try {{
                        window.map = new AMap.Map('container', {{ 
                            resizeEnable: true,
                            zoom: 9,
                            center: [120.0, 29.0]
                        }});
                        console.log('使用备用配置创建地图实例成功');
                    }} catch (fallbackError) {{ 
                        console.error('备用配置创建地图实例也失败:', fallbackError); 
                        window.map = window.map || {{}}; 
                        throw new Error('无法创建地图实例，请检查API密钥和网络连接'); 
                    }}
                }}
                
                setTimeout(checkTileLoading, 2000);
                
                window.safeMapAccess = function(callback, defaultValue) {{
                    try {{
                        if (window.map && typeof window.map === 'object') {{
                            return callback(window.map); 
                        }} 
                    }} catch (e) {{ 
                        console.error('安全访问地图对象时出错:', e); 
                    }} 
                    return defaultValue; 
                }};
                
                var pointLabels = {labels_js};
                var starts = [{starts_js}];
                var ends = [{ends_js}];
                var waypoints = [{waypoints_js}];
                var route_points = [{route_points_js}];
                
                var allPoints = [];
                if (starts && starts.length > 0) allPoints = allPoints.concat(starts);
                if (waypoints && waypoints.length > 0) allPoints = allPoints.concat(waypoints);
                if (ends && ends.length > 0) allPoints = allPoints.concat(ends);
                if (route_points && route_points.length > 0) allPoints = allPoints.concat(route_points);
                
                console.log('地图数据详情:');
                console.log('起点数据:', starts);
                console.log('途经点数据:', waypoints);
                console.log('终点数据:', ends);
                console.log('路线点数量:', route_points.length);
                
                // 添加起点标记
                try {{
                    console.log('添加起点标记...');
                    var startMarkers = [];
                    starts.forEach(function(point, index) {{
                        var labelText = point.label || '起点' + (index + 1);
                        var marker = createMarker(point, 'red', '');
                        if (marker) {{
                            startMarkers.push(marker);
                        }}
                    }});
                    console.log('成功添加起点标记:', startMarkers.length, '个');
                }} catch (e) {{
                    console.error('添加起点标记失败:', e);
                }}
                
                // 添加终点标记
                try {{
                    console.log('添加终点标记...');
                    var endMarkers = [];
                    ends.forEach(function(point, index) {{
                        var labelText = point.label || '终点' + (index + 1);
                        var marker = createMarker(point, 'blue', '');
                        if (marker) {{
                            endMarkers.push(marker);
                        }}
                    }});
                    console.log('成功添加终点标记:', endMarkers.length, '个');
                }} catch (e) {{
                    console.error('添加终点标记失败:', e);
                }}
                
                // 添加途经点标记
                try {{
                    console.log('添加途经点标记...');
                    var waypointMarkers = [];
                    waypoints.forEach(function(point, index) {{
                        var labelText = point.label || '途经点' + (index + 1);
                        var marker = createMarker(point, 'green', '');
                        if (marker) {{
                            waypointMarkers.push(marker);
                        }}
                    }});
                    console.log('成功添加途经点标记:', waypointMarkers.length, '个');
                }} catch (e) {{
                    console.error('添加途经点标记失败:', e);
                }}
                
                // 调整地图视图
                if (allPoints && allPoints.length > 0) {{
                    try {{
                        console.log('尝试自动调整地图视图以显示所有点...');
                        
                        var lnglats = allPoints.map(function(point) {{
                            return [point.longitude, point.latitude];
                        }});
                        
                        if (window.map && window.map.setFitView) {{
                            console.log('使用setFitView自动调整地图视图');
                            window.map.setFitView(lnglats, false, [50, 50, 50, 50]);
                        }}
                    }} catch (e) {{
                        console.error('自动调整地图视图失败:', e);
                    }}
                }} else {{
                    console.log('没有找到有效的坐标点，使用默认地图视图');
                }}
                
                // 绘制路线
                if (route_points && route_points.length > 0) {{
                    try {{
                        console.log('开始绘制路线，点数:', route_points.length);
                        
                        var path = route_points.map(function(point) {{
                            return [point.longitude, point.latitude];
                        }});
                        
                        var polyline = new AMap.Polyline({{
                            path: path,
                            strokeColor: '#3685ff',
                            strokeWeight: 5,
                            strokeStyle: 'solid',
                            lineJoin: 'round',
                            lineCap: 'round'
                        }});
                        
                        polyline.setMap(window.map);
                        console.log('路线绘制成功！');
                    }} catch (e) {{
                        console.error('绘制路线失败:', e);
                    }}
                }} else {{
                    console.log('无路线点数据，跳过路线绘制');
                }}
                
                // 收集所有点用于地图视野调整
                var allPoints = [];
                
                // 添加起点
                starts.forEach(function(point) {{
                    if (point && point.longitude && point.latitude) {{
                        allPoints.push([point.longitude, point.latitude]);
                    }}
                }});
                
                // 添加终点
                ends.forEach(function(point) {{
                    if (point && point.longitude && point.latitude) {{
                        allPoints.push([point.longitude, point.latitude]);
                    }}
                }});
                
                // 添加途经点
                waypoints.forEach(function(point) {{
                    if (point && point.longitude && point.latitude) {{
                        allPoints.push([point.longitude, point.latitude]);
                    }}
                }});
                
                // 添加路线点
                route_points.forEach(function(point) {{
                    if (point && point.longitude && point.latitude) {{
                        allPoints.push([point.longitude, point.latitude]);
                    }}
                }});
                
                // 地图视野调整
                console.log('=== 开始地图视野调整 ===');
                
                var markers = [];
                try {{
                    // 创建临时标记点
                    allPoints.forEach(function(point) {{
                        if (Array.isArray(point) && point.length >= 2 && 
                            typeof point[0] === 'number' && !isNaN(point[0]) && 
                            typeof point[1] === 'number' && !isNaN(point[1]) &&
                            Math.abs(point[0]) <= 180 && Math.abs(point[1]) <= 90) {{
                            try {{
                                var tempMarker = new AMap.Marker({{
                                    position: point,
                                    map: null
                                }});
                                markers.push(tempMarker);
                            }} catch (e) {{
                                console.warn('创建临时标记点失败:', e);
                            }}
                        }}
                    }});
                }} catch (e) {{
                    console.error('创建标记点集合时发生错误:', e);
                }}
                
                console.log('创建的临时标记点数量:', markers.length);
                
                // 使用浙江省中心点作为初始视图
                console.log('使用浙江省中心点作为初始视图...');
                var zhejiangCenter = new AMap.LngLat(120.0, 29.0);
                var defaultZoom = 9;
                
                window.safeMapAccess(function(map) {{
                    if (typeof map.setCenter === 'function') {{
                        map.setCenter(zhejiangCenter);
                    }}
                }});
                
                window.safeMapAccess(function(map) {{
                    if (typeof map.setZoom === 'function') {{
                        map.setZoom(defaultZoom);
                    }}
                }});
                
                // 优化视野
                if (markers.length > 0) {{
                    console.log('检测到标记点，尝试优化视野...');
                    
                    try {{
                        console.log('使用setFitView方法调整视野...');
                        window.safeMapAccess(function(map) {{
                            if (typeof map.setFitView === 'function') {{
                                try {{
                                    map.setFitView(markers, false, [50, 50, 50, 50]);
                                    console.log('地图视野优化成功！');
                                }} catch (e) {{
                                    console.warn('setFitView方法执行失败:', e);
                                }}
                            }} else {{
                                console.warn('setFitView方法不存在');
                            }}
                        }});
                    }} catch (e) {{
                        console.warn('setFitView方法失败:', e);
                        
                        // 降级方案
                        try {{
                            var lngs = allPoints.filter(function(p) {{
                                return Array.isArray(p) && p.length >= 2 && typeof p[0] === 'number' && !isNaN(p[0]);
                            }}).map(function(p) {{
                                return p[0];
                            }});
                            var lats = allPoints.filter(function(p) {{
                                return Array.isArray(p) && p.length >= 2 && typeof p[1] === 'number' && !isNaN(p[1]);
                            }}).map(function(p) {{
                                return p[1];
                            }});
                            
                            if (lngs.length > 0 && lats.length > 0) {{
                                var minLng = Math.min.apply(null, lngs);
                                var maxLng = Math.max.apply(null, lngs);
                                var minLat = Math.min.apply(null, lats);
                                var maxLat = Math.max.apply(null, lats);
                                
                                var centerLng = (minLng + maxLng) / 2;
                                var centerLat = (minLat + maxLat) / 2;
                                
                                var lngDiff = maxLng - minLng;
                                var latDiff = maxLat - minLat;
                                var maxDiff = Math.max(lngDiff, latDiff);
                                
                                var zoomLevel = 12;
                                if (maxDiff < 0.001) zoomLevel = 18;
                                else if (maxDiff < 0.01) zoomLevel = 16;
                                else if (maxDiff < 0.1) zoomLevel = 14;
                                else if (maxDiff < 1) zoomLevel = 12;
                                else if (maxDiff < 10) zoomLevel = 10;
                                
                                if (typeof window.map.setCenter === 'function') {{
                                    window.map.setCenter(new AMap.LngLat(centerLng, centerLat));
                                }}
                                if (typeof window.map.setZoom === 'function') {{
                                    window.map.setZoom(zoomLevel);
                                    console.log('使用计算的中心点和缩放级别成功！');
                                }}
                            }}
                        }} catch (e2) {{
                            console.error('计算中心点和缩放级别也失败:', e2);
                        }}
                    }}
                }}
                
                console.log('=== 地图视野调整完成 ===');
                
                // 隐藏加载遮罩层
                setTimeout(function() {{
                    if (window.map && typeof window.map.getCenter === 'function') {{
    
                        console.log('地图加载完成！');
                        
                        // 显示成功通知
                        try {{
                            const notification = document.createElement('div');
                            notification.style.position = 'absolute';
                            notification.style.top = '10px';
                            notification.style.left = '50%';
                            notification.style.transform = 'translateX(-50%)';
                            notification.style.backgroundColor = 'rgba(76, 175, 80, 0.9)';
                            notification.style.color = 'white';
                            notification.style.padding = '8px 16px';
                            notification.style.borderRadius = '4px';
                            notification.style.zIndex = '1000';
                            notification.style.boxShadow = '0 2px 6px rgba(0,0,0,0.2)';
                            notification.innerHTML = '✓ 地图加载完成';
                            document.body.appendChild(notification);
                            
                            setTimeout(function() {{
                                notification.style.transition = 'opacity 0.5s';
                                notification.style.opacity = '0';
                                setTimeout(function() {{
                                    if (document.body.contains(notification)) {{
                                        document.body.removeChild(notification);
                                    }}
                                }}, 500);
                            }}, 3000);
                        }} catch (e) {{
                            console.error('添加地图加载成功通知失败:', e);
                        }}
                    }} else {{
    
                        console.warn('地图实例可能不完整！');
                        
                        try {{
                            const notification = document.createElement('div');
                            notification.style.position = 'absolute';
                            notification.style.top = '10px';
                            notification.style.left = '50%';
                            notification.style.transform = 'translateX(-50%)';
                            notification.style.backgroundColor = 'rgba(255, 152, 0, 0.9)';
                            notification.style.color = 'white';
                            notification.style.padding = '8px 16px';
                            notification.style.borderRadius = '4px';
                            notification.style.zIndex = '1000';
                            notification.style.boxShadow = '0 2px 6px rgba(0,0,0,0.2)';
                            notification.innerHTML = '⚠️ 地图加载可能不完整，点击刷新';
                            notification.style.cursor = 'pointer';
                            notification.onclick = function() {{
                                window.location.reload();
                            }};
                            document.body.appendChild(notification);
                        }} catch (e) {{
                            console.error('添加地图加载问题通知失败:', e);
                        }}
                    }}
                }}, 1000);
                
                // 添加地图图例
                try {{
                    var legend = document.createElement('div');
                    legend.style.position = 'absolute';
                    legend.style.top = '10px';
                    legend.style.right = '10px';
                    legend.style.backgroundColor = 'white';
                    legend.style.padding = '10px';
                    legend.style.borderRadius = '5px';
                    legend.style.boxShadow = '0 2px 6px rgba(0,0,0,0.1)';
                    legend.style.zIndex = '100';
                    legend.style.fontSize = '14px';
                    legend.innerHTML = `
                        <div style="margin-bottom: 5px;"><strong>图例</strong></div>
                        <div style="display: flex; align-items: center; margin-bottom: 5px;">
                            <div style="width: 16px; height: 16px; background-color: red; border-radius: 50%; margin-right: 5px;"></div>
                            <span>起点</span>
                        </div>
                        <div style="display: flex; align-items: center; margin-bottom: 5px;">
                            <div style="width: 16px; height: 16px; background-color: blue; border-radius: 50%; margin-right: 5px;"></div>
                            <span>终点</span>
                        </div>
                        <div style="display: flex; align-items: center;">
                            <div style="width: 16px; height: 16px; background-color: green; border-radius: 50%; margin-right: 5px;"></div>
                            <span>途经点</span>
                        </div>
                    `;
                    document.getElementById('container').appendChild(legend);
                }} catch (e) {{
                    console.error('添加图例失败:', e);
                }}
            }} catch (e) {{
                console.error('地图加载过程中发生错误:', e);
                
                
                // 地图加载失败时的处理逻辑已移除
                console.error('地图加载失败，错误信息:', e.message);
            }} finally {{
                var loadingAnimation = document.getElementById('loading-animation');
                if (loadingAnimation) {{
                    loadingAnimation.style.display = 'none';
                }}
            }}
        }});
    </script>
</body>
</html>
'''
        
        return html_content
    
    def update_points(self, starts: List[Tuple[float, float]], waypoints: List[Tuple[float, float]], 
                     ends: List[Tuple[float, float]], route_points: List[Tuple[float, float]] = None):
        """更新地图上的点和路线"""
        self.starts = starts
        self.waypoints = waypoints
        self.ends = ends
        self.route_points = route_points or []
        self._refresh_map()
    
    def _refresh_map(self, reopen_browser=False):
        """刷新地图显示"""
        if self.html_file is None or self.html_path is None:
            self.html_file = tempfile.NamedTemporaryFile(suffix='.html', delete=False, mode='w', encoding='utf-8')
            self.html_path = self.html_file.name
        else:
            if not self.html_file.closed:
                self.html_file.close()
            self.html_file = open(self.html_path, 'w', encoding='utf-8')
        
        html_content = self._create_html_content()
        self.html_file.write(html_content)
        self.html_file.close()
        
        if reopen_browser:
            webbrowser.open('file://' + os.path.realpath(self.html_path), new=0)
        
        print(f"地图已更新，HTML文件路径：{os.path.realpath(self.html_path)}")
        
        if not hasattr(self, '_map_initialized') or not self._map_initialized:
            self._map_initialized = True
    
    def show_map(self, reopen_browser=False):
        """显示地图"""
        self._refresh_map(reopen_browser)
    
    def refresh_and_show(self):
        """刷新地图并强制显示更新后的内容"""
        self._refresh_map(reopen_browser=True)
    
    def close(self):
        """关闭地图并清理临时文件"""
        if self.html_file is not None and not self.html_file.closed:
            self.html_file.close()
        if self.html_path is not None and os.path.exists(self.html_path):
            try:
                os.unlink(self.html_path)
            except Exception:
                pass
    
    def generate_map_html(self):
        """生成地图HTML文件并返回文件路径"""
        if self.html_file is None or self.html_path is None:
            self.html_file = tempfile.NamedTemporaryFile(suffix='.html', delete=False, mode='w', encoding='utf-8')
            self.html_path = self.html_file.name
        else:
            if not self.html_file.closed:
                self.html_file.close()
            self.html_file = open(self.html_path, 'w', encoding='utf-8')
        
        html_content = self._create_html_content()
        self.html_file.write(html_content)
        self.html_file.close()
        
        return self.html_path