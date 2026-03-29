import pandas as pd
import numpy as np
import re
from math import radians, cos, sin, asin, sqrt

# --- 尝试导入适配器 (用于处理 Phase 2 Markdown 数据) ---
try:
    from data_adapter import parse_markdown_table, adapt_phase2_data
except ImportError:
    print("警告: 找不到 data_adapter.py，Phase 2 数据解析可能会失败。")

# --- Helper Functions ---

def haversine(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance between two points 
    on the earth (specified in decimal degrees)
    """
    try:
        lon1, lat1, lon2, lat2 = map(radians, [float(lon1), float(lat1), float(lon2), float(lat2)])
        dlon = lon2 - lon1 
        dlat = lat2 - lat1 
        a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
        c = 2 * asin(sqrt(a)) 
        r = 6371 
        return c * r
    except:
        return 0

def get_vertical_beamwidth(scenario_str):
    """
    Parses the Beam Scenario string to return vertical beamwidth.
    """
    s = str(scenario_str).upper()
    if pd.isna(scenario_str) or 'DEFAULT' in s:
        return 6
    
    match = re.search(r'SCENARIO_(\d+)', s)
    if match:
        num = int(match.group(1))
        if 1 <= num <= 5: return 6
        if 6 <= num <= 11: return 12
        if num >= 12: return 25
    
    # Fallback default
    return 6 

# --- Parsing Functions (Modified for Dual Phase Support) ---

def parse_drive_test_data(text):
    # 1. 尝试 Phase 1 格式 (Pipe Separated, Standard Header)
    header_pattern_p1 = r"Timestamp\|Longitude\|Latitude\|GPS Speed"
    match_p1 = re.search(header_pattern_p1, text)
    
    if match_p1:
        # --- Phase 1 Logic ---
        header_start = match_p1.start()
        header_line = text[header_start:text.find('\n', header_start)]
        columns = [col.strip() for col in header_line.split('|')]
        
        data_start = text.find('\n', header_start) + 1
        eng_start = text.find("Engeneering parameters data", data_start)
        if eng_start == -1: eng_start = text.find("Engineering parameters data", data_start)
        # 兼容性查找
        if eng_start == -1: eng_start = text.find("Parameter Data", data_start)
        if eng_start == -1: eng_start = len(text)
        
        data_text = text[data_start:eng_start].strip()
        rows = []
        for line in data_text.split('\n'):
            line = line.strip()
            if not line or '|' not in line: continue
            parts = [p.strip() for p in line.split('|')]
            if len(parts) < len(columns): continue
            rows.append(parts[:len(columns)])
        
        if not rows: return None
        df = pd.DataFrame(rows, columns=columns)
        
        # 转换数值
        numeric_cols = [
            'Longitude', 'Latitude', 'GPS Speed (km/h)',
            '5G KPI PCell RF Serving PCI', '5G KPI PCell RF Serving SS-RSRP [dBm]',
            '5G KPI PCell RF Serving SS-SINR [dB]', '5G KPI PCell Layer2 MAC DL Throughput [Mbps]',
            '5G KPI PCell Layer1 DL RB Num (Including 0)'
        ]
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        for i in range(1, 6):
            pci_col = f'Measurement PCell Neighbor Cell Top Set(Cell Level) Top {i} PCI'
            rsrp_col = f'Measurement PCell Neighbor Cell Top Set(Cell Level) Top {i} Filtered Tx BRSRP [dBm]'
            if pci_col in df.columns: df[pci_col] = pd.to_numeric(df[pci_col], errors='coerce')
            if rsrp_col in df.columns: df[rsrp_col] = pd.to_numeric(df[rsrp_col], errors='coerce')
            
        return df

    # 2. 尝试 Phase 2 格式 (Markdown Table)
    if "**Drive Test Data**" in text or "| Time | UE |" in text:
        # 提取表格块
        start_marker = "| Time | UE |"
        start_idx = text.find(start_marker)
        if start_idx != -1:
            end_idx = text.find("**Parameter Data**", start_idx)
            if end_idx == -1: end_idx = len(text)
            
            table_text = text[start_idx:end_idx].strip()
            # 调用适配器中的解析函数
            if 'parse_markdown_table' in globals():
                df = parse_markdown_table(table_text)
                if df is not None:
                    # 标记为需要转换
                    df.attrs['format'] = 'phase2' 
                    return df

    return None

def parse_engineering_params(text):
    # 1. Phase 1
    header_pattern_p1 = r"gNodeB ID\|Cell ID\|Longitude\|Latitude"
    match_p1 = re.search(header_pattern_p1, text)
    
    if match_p1:
        header_start = match_p1.start()
        header_line = text[header_start:text.find('\n', header_start)]
        columns = [col.strip() for col in header_line.split('|')]
        
        data_start = text.find('\n', header_start) + 1
        data_end = text.find('"', data_start)
        if data_end == -1: data_end = len(text)
        
        data_text = text[data_start:data_end].strip()
        rows = []
        for line in data_text.split('\n'):
            line = line.strip()
            if not line or '|' not in line: continue
            parts = [p.strip() for p in line.split('|')]
            if len(parts) < len(columns): continue
            rows.append(parts[:len(columns)])
            
        if not rows: return None
        df = pd.DataFrame(rows, columns=columns)
        
        numeric_cols = ['Cell ID', 'Longitude', 'Latitude', 'Mechanical Azimuth', 
                        'Mechanical Downtilt', 'Digital Tilt', 'Digital Azimuth', 
                        'Height', 'PCI', 'Max Transmit Power']
        for col in numeric_cols:
            if col in df.columns: df[col] = pd.to_numeric(df[col], errors='coerce')
        return df

    # 2. Phase 2
    if "**Parameter Data**" in text or "| gNodeB ID | Cell ID |" in text:
        start_marker = "| gNodeB ID | Cell ID |"
        start_idx = text.find(start_marker)
        if start_idx != -1:
            end_idx = text.find("**Configuration Data**", start_idx)
            if end_idx == -1: end_idx = len(text)
            
            table_text = text[start_idx:end_idx].strip()
            if 'parse_markdown_table' in globals():
                df = parse_markdown_table(table_text)
                return df

    return None

# --- Feature Calculation (Integrated Adapter) ---

def calculate_features(drive_test_df, eng_params_df):
    if drive_test_df is None or len(drive_test_df) == 0:
        return None
    
    # === 适配层接入 ===
    # 检查是否是 Phase 2 格式，如果是，调用 adapt_phase2_data 进行转换
    if hasattr(drive_test_df, 'attrs') and drive_test_df.attrs.get('format') == 'phase2':
        if 'adapt_phase2_data' in globals():
            drive_test_df, eng_params_df = adapt_phase2_data(drive_test_df, eng_params_df)
        else:
            return None # 缺少适配器函数
    
    # 检查转换是否成功（关键列是否存在）
    # 如果适配后还是没有 PCI 列，说明可能是数学题或数据严重损坏
    if '5G KPI PCell RF Serving PCI' not in drive_test_df.columns:
        return None

    # === Phase 1 标准特征提取逻辑 ===
    features_list = []
    
    for idx, row in drive_test_df.iterrows():
        serving_pci = row.get('5G KPI PCell RF Serving PCI')
        serving_rsrp = row.get('5G KPI PCell RF Serving SS-RSRP [dBm]', -140)
        if pd.isna(serving_rsrp): serving_rsrp = -140
        serving_sinr = row.get('5G KPI PCell RF Serving SS-SINR [dB]')
        
        gps_speed = row.get('GPS Speed (km/h)')
        rb_num = row.get('5G KPI PCell Layer1 DL RB Num (Including 0)')
        throughput = row.get('5G KPI PCell Layer2 MAC DL Throughput [Mbps]')
        
        downtilt_angle = 0
        beam_width = 6 # Default narrow beam
        distance_km = 0
        
        if eng_params_df is not None and pd.notna(serving_pci):
            matching = eng_params_df[eng_params_df['PCI'] == serving_pci]
            if not matching.empty:
                serving_cell_info = matching.iloc[0]
                
                # Downtilt Calculation
                mech = serving_cell_info.get('Mechanical Downtilt', 0)
                digi = serving_cell_info.get('Digital Tilt', 0)
                if digi == 255: digi = 6
                downtilt_angle = (mech if pd.notna(mech) else 0) + (digi if pd.notna(digi) else 0)
                
                # Beam Width Parsing
                scenario = serving_cell_info.get('Beam Scenario', 'DEFAULT')
                beam_width = get_vertical_beamwidth(scenario)
                
                # Distance Calculation
                u_lon, u_lat = row.get('Longitude'), row.get('Latitude')
                c_lon, c_lat = serving_cell_info.get('Longitude'), serving_cell_info.get('Latitude')
                if all(pd.notna(x) for x in [u_lon, u_lat, c_lon, c_lat]):
                    distance_km = haversine(u_lon, u_lat, c_lon, c_lat)
        
        neighbors = []
        mod30_risk = False
        same_mod_max = -999
        same_mod_count = 0
        
        for i in range(1, 6):
            n_pci = row.get(f'Measurement PCell Neighbor Cell Top Set(Cell Level) Top {i} PCI')
            n_rsrp = row.get(f'Measurement PCell Neighbor Cell Top Set(Cell Level) Top {i} Filtered Tx BRSRP [dBm]')
            
            if pd.notna(n_pci) and pd.notna(n_rsrp):
                neighbors.append({'pci': n_pci, 'rsrp': n_rsrp})
                # Check Mod30
                if pd.notna(serving_pci) and (int(serving_pci) % 30 == int(n_pci) % 30):
                    if n_rsrp > same_mod_max:
                        same_mod_max = n_rsrp
                    if n_rsrp > -105:
                        same_mod_count += 1
                    # Tighten mod30 trigger to strong collisions only
                    if n_rsrp > -95:
                        mod30_risk = True
        
        neighbors.sort(key=lambda x: x['rsrp'], reverse=True)
        
        top1_rsrp = neighbors[0]['rsrp'] if len(neighbors) > 0 else -140
        top2_rsrp = neighbors[1]['rsrp'] if len(neighbors) > 1 else None
        
        best_delta = top1_rsrp - serving_rsrp
        neighbor_dominance = top1_rsrp - top2_rsrp if top2_rsrp is not None else 0
        
        # Absolute and relative strong neighbor counts
        strong_neighbors = sum(1 for n in neighbors if n['rsrp'] > -105)
        rel_strong_neighbors = sum(1 for n in neighbors if (n['rsrp'] > -105) and (n['rsrp'] > serving_rsrp - 6))
        crowdiness = rel_strong_neighbors
        
        if top2_rsrp is None:
            top2_rsrp = -140
    
        features_list.append({
            'idx': idx,
            'serving_pci': serving_pci,
            'serving_rsrp': serving_rsrp,
            'serving_sinr': serving_sinr,
            'gps_speed': gps_speed,
            'rb_num': rb_num,
            'throughput': throughput,
            'distance_km': distance_km,
            'downtilt_angle': downtilt_angle,
            'mod30_risk': mod30_risk,
            'same_mod_max': same_mod_max,
            'same_mod_count': same_mod_count,
            'best_delta': best_delta,
            'top1_rsrp': top1_rsrp,
            'neighbor_dominance': neighbor_dominance,
            'strong_neighbors': strong_neighbors,
            'rel_strong_neighbors': rel_strong_neighbors,
            'crowdiness': crowdiness,
            'beam_width': beam_width
        })

    features_df = pd.DataFrame(features_list)
    
    features_df['ho_count'] = 0
    if 'idx' in features_df.columns and len(features_df) > 1:
        features_df['pci_changed'] = features_df['serving_pci'].diff().fillna(0) != 0
        features_df['ho_count'] = features_df['pci_changed'].rolling(window=10, min_periods=1).sum()
        
    return features_df

# --- Classification Logic (Full Rule Tree) ---

def classify_root_cause(features_df):
    """
    Phase 11 Strategy: The "Environment Signature" Logic.
    1. C4 (Interference): Defined by CROWDS. If Neighbors >= 2 and no dominant leader, it's C4.
       (Prioritize this to stop C4 from leaking into C1).
    2. C3 (Savior): Defined by QUALITY. If there's a good neighbor (> -105) and good Delta, switch.
    3. C1 (Downtilt): Defined by ISOLATION + TILT. If we aren't crowded, and aren't switching,
       and the tilt is high relative to beam width, it's C1.
    """
    if features_df is None or len(features_df) == 0:
        return 'C4' 
    
    # --- Feature Aggregation ---
    max_speed = features_df['gps_speed'].max()
    min_rb = features_df['rb_num'].min()
    pci_changes = (features_df['serving_pci'].diff().fillna(0) != 0).sum()
    
    max_dist = features_df['distance_km'].max()
    max_downtilt = features_df['downtilt_angle'].max()
    avg_serving_rsrp = features_df['serving_rsrp'].mean()
    # Handle optional columns
    min_serving_sinr = features_df['serving_sinr'].min() if 'serving_sinr' in features_df.columns else np.nan
    
    if 'beam_width' in features_df.columns:
        avg_beam_width = features_df['beam_width'].mean()
    else:
        avg_beam_width = 6 
    
    max_delta = features_df['best_delta'].max()
    if pd.isna(max_delta): max_delta = -999
    
    avg_top1_rsrp = features_df['top1_rsrp'].mean()
    if pd.isna(avg_top1_rsrp): avg_top1_rsrp = -140

    avg_strong_neighbors = features_df['strong_neighbors'].mean()
    avg_rel_strong = features_df['rel_strong_neighbors'].mean() if 'rel_strong_neighbors' in features_df.columns else avg_strong_neighbors
    max_crowdiness = features_df['crowdiness'].max() if 'crowdiness' in features_df.columns else avg_strong_neighbors
    
    avg_dominance = features_df['neighbor_dominance'].mean()
    if pd.isna(avg_dominance): avg_dominance = 0
    
    has_mod30 = features_df['mod30_risk'].any()
    max_same_mod = features_df['same_mod_max'].max() if 'same_mod_max' in features_df.columns else -999
    max_same_mod_count = features_df['same_mod_count'].max() if 'same_mod_count' in features_df.columns else 0

    # =======================================================
    # LAYER 1: SOLVED CLASSES (100% Accuracy)
    # =======================================================
    if max_speed > 40: return 'C7'
    if min_rb < 160: return 'C8'
    if max_dist > 1.2: return 'C2'
    if pci_changes >= 2: return 'C5'
    if has_mod30 and (max_same_mod > -92 or max_same_mod_count >= 2):
        return 'C6'

    # =======================================================
    # LAYER 2: INTERFERENCE (C4 Priority)
    # =======================================================
    # Logic: "The Noisy Market".
    # If we hear 2 or more strong-ish neighbors (>-105), it is an interference zone.
    # We check dominance/SINR to ensure we don't accidentally kill a clear handover (C3).
    if (avg_rel_strong >= 2.0 and avg_dominance < 4.0) or \
       (avg_rel_strong >= 1.5 and avg_dominance < 2.5 and pd.notna(min_serving_sinr) and min_serving_sinr < 2):
        return 'C4'

    # =======================================================
    # LAYER 3: THE SAVIOR (C3 Priority)
    # =======================================================
    # Logic: "The Clear Handover".
    
    # Standard C3: Good Delta and the neighbor is actually usable (>-105).
    if max_delta > 6 and avg_top1_rsrp > -105:
        return 'C3'

    # Lifeboat C3: Serving is dead (-112), Neighbor is decent (-108).
    if avg_serving_rsrp < -112 and avg_top1_rsrp > -108:
        return 'C3'

    # =======================================================
    # LAYER 4: BEAM-AWARE HARDWARE FAULTS (C1)
    # =======================================================
    # Logic: "The Isolated Hole".
    # We have filtered out the crowds (C4) and the handovers (C3).
    # If the tilt is high relative to the beam width, it's a coverage hole.
    
    # Dynamic Threshold Calculation
    # Narrow (6 deg) -> Threshold 14
    # Wide (25 deg) -> Threshold 25
    tilt_threshold = 14 + (avg_beam_width - 6) * 0.6 
    
    # Build a C1 flag so we can rescue some tilted-but-neighbor-rich cases back to C3
    c1_flag = False
    if max_downtilt > tilt_threshold and avg_rel_strong < 2.0:
        c1_flag = True
    if max_dist < 0.2 and avg_serving_rsrp < -100 and avg_rel_strong < 1.8:
        c1_flag = True
    if avg_strong_neighbors < 1.5 and avg_serving_rsrp < -105 and avg_rel_strong < 1.5:
        c1_flag = True

    # =======================================================
    # LAYER 5: C3 RESCUE FOR BORDERLINE C1 CASES
    # =======================================================
    # If tilt is high but a clear neighbor exists, prefer handover root cause.
    if c1_flag:
        if (avg_strong_neighbors >= 1.6 and avg_dominance > 11 and avg_top1_rsrp > -92 and max_delta > -6) or \
           (max_downtilt < 26 and avg_dominance > 9 and avg_top1_rsrp > -93 and avg_strong_neighbors >= 1.4 and max_delta > -4):
            return 'C3'
        return 'C1'

    # =======================================================
    # LAYER 6: RESIDUALS / FALLBACKS
    # =======================================================
    
    # Desperation C3: If we survived C1 check, but performance is bad,
    # and there is a neighbor available (even if delta is small).
    if max_delta > 3 and avg_top1_rsrp > -110:
        return 'C3'

    if max_crowdiness >= 1.8 and avg_dominance < 4.0:
        return 'C4'

    # Default -> C4 (Hidden Interference / Overlap)
    return 'C4'

def process_single_question(question_text):
    drive_test_df = parse_drive_test_data(question_text)
    eng_params_df = parse_engineering_params(question_text)
    
    if drive_test_df is None or len(drive_test_df) == 0:
        return 'Unknown'
    
    features_df = calculate_features(drive_test_df, eng_params_df)
    
    if features_df is None or len(features_df) == 0:
        return 'Unknown'
    
    result = classify_root_cause(features_df)
    return result

def main():
    print("Loading training data...")
    # NOTE: Update the path to your actual file location
    try:
        train_df = pd.read_csv('train.csv')
    except:
        print("Error: train.csv not found.")
        return
    
    print(f"Total samples: {len(train_df)}")
    print("\nProcessing samples...")
    
    predictions = []
    correct_answers = []
    
    for idx, row in train_df.iterrows():
        question = row['question']
        answer = row['answer']
        
        try:
            pred = process_single_question(question)
            predictions.append(pred)
            correct_answers.append(answer)
            
            if (idx + 1) % 500 == 0:
                print(f"Processed {idx + 1}/{len(train_df)} samples...")
        except Exception as e:
            print(f"Error processing sample {idx}: {e}")
            predictions.append('Unknown')
            correct_answers.append(answer)
    
    correct = sum([p == a for p, a in zip(predictions, correct_answers)])
    accuracy = correct / len(predictions) * 100
    
    print(f"\n{'='*60}")
    print(f"Results:")
    print(f"{'='*60}")
    print(f"Total samples: {len(predictions)}")
    print(f"Correct predictions: {correct}")
    print(f"Accuracy: {accuracy:.2f}%")

if __name__ == "__main__":
    main()