import pandas as pd
import re
import io
from math import radians, cos, sin, asin, sqrt

# --- 列名映射字典 (新版 Phase 2 -> 旧版 Phase 1) ---
COLUMN_MAPPING = {
    # === Drive Test Data (路测数据) ===
    'Serving PCI': '5G KPI PCell RF Serving PCI',
    'Serving RSRP(dBm)': '5G KPI PCell RF Serving SS-RSRP [dBm]',
    'Serving SINR(dB)': '5G KPI PCell RF Serving SS-SINR [dB]',
    'Throughput(Mbps)': '5G KPI PCell Layer2 MAC DL Throughput [Mbps]',
    'RB/slot': '5G KPI PCell Layer1 DL RB Num (Including 0)', 
    
    # 经纬度与时间 (两表通用)
    'Longitude': 'Longitude',
    'Latitude': 'Latitude',
    'Time': 'Timestamp', # Phase 2 叫 Time, Phase 1 叫 Timestamp
    
    # 邻区 1-3 (Phase 2 只有 Top 3)
    'Neighbor 1 PCI': 'Measurement PCell Neighbor Cell Top Set(Cell Level) Top 1 PCI',
    'Neighbor 1 RSRP(dBm)': 'Measurement PCell Neighbor Cell Top Set(Cell Level) Top 1 Filtered Tx BRSRP [dBm]',
    'Neighbor 2 PCI': 'Measurement PCell Neighbor Cell Top Set(Cell Level) Top 2 PCI',
    'Neighbor 2 RSRP(dBm)': 'Measurement PCell Neighbor Cell Top Set(Cell Level) Top 2 Filtered Tx BRSRP [dBm]',
    'Neighbor 3 PCI': 'Measurement PCell Neighbor Cell Top Set(Cell Level) Top 3 PCI',
    'Neighbor 3 RSRP(dBm)': 'Measurement PCell Neighbor Cell Top Set(Cell Level) Top 3 Filtered Tx BRSRP [dBm]',
    
    # === Engineering Parameters (工参数据) ===
    # [关键修复] 添加同名或直接映射的列，防止被丢弃
    'PCI': 'PCI',  # <--- 之前漏了这一行导致报错
    'gNodeB ID': 'gNodeB ID',
    'Cell ID': 'Cell ID',
    'Max Transmit Power': 'Max Transmit Power', # 如果 Phase 2 有这列的话
    
    # 需要改名的列
    'Mech Tilt(deg)': 'Mechanical Downtilt',
    'Elec Tilt(deg)': 'Digital Tilt',
    'Azimuth(deg)': 'Mechanical Azimuth',
    'Ant Height(m)': 'Height'
}

def haversine_speed(row1, row2):
    """根据两行数据的经纬度和时间计算速度 (km/h)"""
    try:
        # 确保是数值类型
        lon1 = float(row1['Longitude'])
        lat1 = float(row1['Latitude'])
        lon2 = float(row2['Longitude'])
        lat2 = float(row2['Latitude'])
        
        # 计算距离 (km)
        lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
        dlon = lon2 - lon1
        dlat = lat2 - lat1
        a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
        c = 2 * asin(sqrt(a))
        dist_km = c * 6371
        
        # 计算时间差 (秒)
        # Phase 2 时间格式通常是 "2024-09-20 22:30:45.500"
        t1 = pd.to_datetime(str(row1['Time']).strip())
        t2 = pd.to_datetime(str(row2['Time']).strip())
        time_diff_hours = (t2 - t1).total_seconds() / 3600.0
        
        if time_diff_hours <= 0: return 0
        speed = dist_km / time_diff_hours
        return speed
    except Exception:
        return 0

def parse_markdown_table(text_block):
    """
    解析 Markdown 风格的表格文本
    """
    lines = text_block.strip().split('\n')
    # 过滤掉分隔行 (含有 --- 的行)
    data_lines = [line for line in lines if '---' not in line]
    
    if len(data_lines) < 2: return None
    
    # 清洗每一行：去除首尾的 |
    cleaned_lines = []
    for line in data_lines:
        # 以 | 分割
        parts = line.split('|')
        # 去除首尾可能的空字符串 (对应行首尾的 |)
        clean_parts = []
        for p in parts:
            p = p.strip()
            # 只有当它不是空字符串，或者它是中间的空值时才保留
            # 简单的处理：Markdown 表格 split('|') 后，首尾通常是空串
            clean_parts.append(p)
            
        # 修正：通常第一项和最后一项是空的（因为 | 在两端）
        if len(clean_parts) > 0 and clean_parts[0] == '': clean_parts.pop(0)
        if len(clean_parts) > 0 and clean_parts[-1] == '': clean_parts.pop()
        
        cleaned_lines.append(clean_parts)
        
    if not cleaned_lines: return None
    
    header = cleaned_lines[0]
    data = cleaned_lines[1:]
    
    # 如果列数对不上，尝试修复
    valid_data = []
    for row in data:
        if len(row) == len(header):
            valid_data.append(row)
        # 有时候 Markdown 解析会导致少一列或多一列，这里做个简单兼容
        elif len(row) == len(header) + 1 and row[-1] == '': 
            valid_data.append(row[:-1])
            
    if not valid_data: return None
    
    return pd.DataFrame(valid_data, columns=header)

def adapt_phase2_data(drive_test_df, eng_params_df):
    """
    将 Phase 2 格式的 DataFrame 转换为 Phase 1 格式
    """
    # 1. 处理路测数据 (Drive Test Data)
    new_drive_df = pd.DataFrame()
    
    # 自动增加 Time 列的别名处理
    if 'Time' in drive_test_df.columns:
        drive_test_df['Timestamp'] = drive_test_df['Time']

    for new_col, old_col in COLUMN_MAPPING.items():
        if new_col in drive_test_df.columns:
            # 尝试转为数值，无法转换的保留原样（如时间）
            if 'Time' in new_col:
                new_drive_df[old_col] = drive_test_df[new_col]
            else:
                new_drive_df[old_col] = pd.to_numeric(drive_test_df[new_col], errors='coerce')
            
    # 2. 补全 GPS Speed
    if 'GPS Speed (km/h)' not in new_drive_df.columns:
        speeds = [0.0] * len(new_drive_df)
        # 需要 Time 和 Longitude/Latitude
        # 此时 new_drive_df 应该已经有了映射后的 Longitude/Latitude
        if 'Longitude' in new_drive_df.columns and len(new_drive_df) > 1 and 'Time' in drive_test_df.columns:
            # 使用原始 DF 的 Time 来计算
            for i in range(1, len(new_drive_df)):
                s = haversine_speed(drive_test_df.iloc[i-1], drive_test_df.iloc[i])
                if s > 250: s = speeds[i-1] # 简单过滤异常
                speeds[i] = s
            speeds[0] = speeds[1]
            
        new_drive_df['GPS Speed (km/h)'] = speeds
        
    # 3. 处理工参数据 (Engineering Params)
    new_eng_df = pd.DataFrame()
    if eng_params_df is not None:
        for new_col, old_col in COLUMN_MAPPING.items():
            if new_col in eng_params_df.columns:
                new_eng_df[old_col] = pd.to_numeric(eng_params_df[new_col], errors='coerce')
        
        # 补全 Beam Scenario
        if 'Beam Scenario' not in new_eng_df.columns:
            new_eng_df['Beam Scenario'] = 'DEFAULT'
            
    return new_drive_df, new_eng_df