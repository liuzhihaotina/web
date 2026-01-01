import json
import math
import os
import glob
from datetime import datetime
from typing import List

import numpy as np
import pandas as pd

# 配置目录
CONFIG_DIR = 'data_config'
HEADERS_DIR = os.path.join(CONFIG_DIR, 'headers')
DATA_DIRS = 'data_dirs'  # 包含多个子文件夹的根目录 data_dirs data_lit
IMPORTANT_DIR = 'data_dir_important'  # 标兵数据目录


CELL_H_PX = 40

MIN_COL_W_PX = 40
MAX_COL_W_PX = 260
CHAR_PX = 8
COL_PAD_PX = 22

# x 轴子指标（底部斜排）参数
FOOTER_ROT_DEG = 35.0          # 使用 +35deg 或 -35deg 都可，CSS里统一用 -35deg
FOOTER_FONT_PX = 11
FOOTER_PAD_TOP = 2             # 文字距热力图（footer行顶部）尽量小
FOOTER_PAD_BOTTOM = 6          # 防裁切安全垫
FOOTER_MIN_H = 56
FOOTER_MAX_H = 130

YAXIS_WIDTH_DEFAULT = 170
CBAR_WIDTH = 88
MIN_LEGEND_PX = 220

def get_important_persons():
    """获取标兵人员列表"""
    important_persons = []
    
    if os.path.exists(IMPORTANT_DIR):
        for item in os.listdir(IMPORTANT_DIR):
            item_path = os.path.join(IMPORTANT_DIR, item)
            if os.path.isdir(item_path) and not item.startswith('.'):
                important_persons.append(item)
    
    return sorted(important_persons)

def load_important_person_data(person_name, table_config):
    """加载标兵个人数据"""
    data_key = table_config['data_key']
    
    # 首先尝试从标兵目录加载
    important_file = os.path.join(IMPORTANT_DIR, person_name, f"{data_key}.json")
    
    if os.path.exists(important_file):
        try:
            with open(important_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
                if data_key in data:
                    data_values = data[data_key]
                    if not isinstance(data_values, list):
                        data_values = [data_values]
                    row_data = [f"⭐ {person_name}"] + data_values
                    return row_data
                elif  data_key == 'test':   
                    data_values = [data]
                    # 添加标兵标记
                    row_data = [f"⭐ {person_name}"] + data_values
                    return row_data
        except Exception as e:
            print(f"加载标兵 {person_name} 的 {data_key} 数据时出错: {e}")
    
    # 如果标兵目录没有，尝试从普通目录加载
    normal_file = os.path.join(DATA_DIRS, person_name, f"{data_key}.json")
    
    if os.path.exists(normal_file):
        try:
            with open(normal_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
                if data_key in data:
                    data_values = data[data_key]
                    if not isinstance(data_values, list):
                        data_values = [data_values]
                    
                    # 添加标兵标记
                    row_data = [f"⭐ {person_name}"] + data_values
                    return row_data
        except Exception as e:
            print(f"加载标兵 {person_name} 的普通数据时出错: {e}")
    
    # 都没有则返回NA
    return create_na_row(f"⭐ {person_name}", table_config)

def build_table_data_with_important(table_config,keyword=None):
    """构建包含标兵数据的表格数据"""
    # 加载表头
    headers = load_headers(table_config['header_file'])
    
    # 获取标兵人员
    important_persons = get_important_persons()
    
    # 获取普通人员（排除标兵）
    all_persons = get_person_names(keyword)
    normal_persons = [p for p in all_persons if p not in important_persons]
    
    # 首先添加标兵数据
    important_rows = []
    for person in important_persons:
        row_data = load_important_person_data(person, table_config)
        # 确保数据长度与表头匹配
        if len(row_data) < len(headers) and 'A' not in row_data:
            row_data = row_data + ["NA"] * (len(headers) - len(row_data))
        elif len(row_data) > len(headers):
            row_data = row_data[:len(headers)]
        important_rows.append(row_data)
    
    # 然后添加普通人员数据
    normal_rows = []
    for person in normal_persons:
        person_dir = {'name': person, 'path': os.path.join(DATA_DIRS, person)}
        row_data = load_person_data(person_dir, table_config)
        # 确保数据长度与表头匹配
        if len(row_data) < len(headers):
            row_data = row_data + ["NA"] * (len(headers) - len(row_data))
        elif len(row_data) > len(headers):
            row_data = row_data[:len(headers)]
        normal_rows.append(row_data)
    
    # 合并数据（标兵在前，普通在后）
    all_rows = important_rows + normal_rows
    
    return {
        'id': table_config['id'],
        'title': table_config['title'],
        'description': table_config.get('description', ''),
        'headers': headers,
        'rows': all_rows,
        'important_count': len(important_rows),  # 标兵数量
        'total_count': len(all_rows)             # 总行数
    }

def get_tables_cfg():
    config = load_config()
    return config

def get_all_tables(keyword=None):
    """获取所有表格的数据（包含标兵）"""
    config = get_tables_cfg()
    tables = []
    
    for table_config in config.get('tables', []):
        table_data = build_table_data_with_important(table_config, keyword)
        tables.append(table_data)
    models_all = get_models_all(tables)
    metrics_all, headers_all, metric_data_dict = get_metrics_headers_all(tables)
    chart = get_charts(models_all, metrics_all, headers_all, metric_data_dict)
    return tables

def build_metric_dataframe(models_all, header, rows):
    df = pd.DataFrame(index=models_all, columns=header, dtype=float)
    for model in models_all:
        row_data = rows.get(model, {})
        for sm, row in zip(header, row_data):
            df.loc[model, sm] = row
    return df

def footer_height_px(x_labels: List[str]) -> int:

    max_len = max(len(str(x)) for x in x_labels)
    text_w = max_len * CHAR_PX
    theta = math.radians(FOOTER_ROT_DEG)

    # 斜排后竖向投影 + 字体高度 + 上下安全垫
    h = int(math.sin(theta) * text_w + (FOOTER_FONT_PX * 1.6) + FOOTER_PAD_TOP + FOOTER_PAD_BOTTOM)
    return max(FOOTER_MIN_H, min(FOOTER_MAX_H, h))

def get_charts(models_all, metrics_all, headers_all, metric_data_dict):
    charts = []
    global_footer_h = FOOTER_MIN_H
    
    # 第一遍：算出每个图自己的 footer_h，并取最大作为全局 footer_h
    tmp = []
    for header, metric in zip(headers_all, metrics_all):
        df = build_metric_dataframe(models_all, header[1:], metric_data_dict[metric])
        if df.empty:
            tmp.append({"metric": metric, "empty": True})
            continue

        y_labels = df.index.tolist()
        x_labels = df.columns.tolist()

        fh = footer_height_px(x_labels)
        global_footer_h = max(global_footer_h, fh)

        tmp.append(
            {
                "metric": metric,
                "empty": False,
                "df": df,          # 暂存
                "x_labels": x_labels,
                "y_labels": y_labels,
            }
        )

    # 第二遍：真正生成 charts，统一使用 global_footer_h（保证每个大指标间距一致）
    for item in tmp:
        if item.get("empty"):
            charts.append({"metric": item["metric"], "empty": True})
            continue

        metric = item["metric"]
        df = item["df"]
        y_labels = item["y_labels"]
        x_labels = item["x_labels"]

        raw = df.values.astype(float)
        z_norm, col_min, col_max = normalize_by_column(raw, robust=robust)

        text = None
        if show_text:
            text = [[fmt_cell(float(v), precision) for v in row] for row in raw]

        col_widths = compute_col_widths([str(x) for x in x_labels], text)
        min_matrix_width = int(sum(col_widths))

        n_rows = len(y_labels)
        footer_h = int(global_footer_h)  # ✅统一
        height = int(n_rows * CELL_H_PX + footer_h)

        legend_target = min(height, max(height, MIN_LEGEND_PX))
        yaxis_width = estimate_yaxis_width_px([str(x) for x in y_labels])

        customdata = []
        for i in range(raw.shape[0]):
            row_cd = []
            for j in range(raw.shape[1]):
                row_cd.append([
                    None if np.isnan(raw[i, j]) else float(raw[i, j]),
                    None if np.isnan(col_min[j]) else float(col_min[j]),
                    None if np.isnan(col_max[j]) else float(col_max[j]),
                ])
            customdata.append(row_cd)

        charts.append(
            {
                "metric": metric,
                "empty": False,
                "x": x_labels,
                "y": y_labels,
                "z": z_norm.tolist(),
                "text": text,
                "customdata": customdata,
                "col_widths": col_widths,
                "min_matrix_width": min_matrix_width,
                "height": height,
                "footer_h": footer_h,
                "yaxis_width": int(yaxis_width),
                "legend_target": int(legend_target),
            }
        )

def get_metrics_headers_all(tables):
    """获取所有表格的指标"""
    metrics_all = []
    headers_all = []
    metric_data_dict = {}
    for table in tables:
        title = table['title']
        metric_data_dict[title] = {}
        metrics_all.append(title)
        headers_all.append(table['headers'])
        for row in table['rows']:
            metric_data_dict[title][row[0]] = row[1:]
    return metrics_all, headers_all, metric_data_dict

def get_models_all(tables):
    """获取所有表格的数据模型"""
    models_all = set()
    for table in tables:
        for row in table['rows']:
            models_all.add(row[0]) 
    return list(models_all)
# ------------------------------========================================
def get_important_person_stats():
    """获取标兵统计信息"""
    important_persons = get_important_persons()
    return {
        'count': len(important_persons),
        'persons': important_persons,
        'has_important': len(important_persons) > 0
    }

# 默认配置
DEFAULT_CONFIG = {
    "tables": [
        {
            "id": "performance-table",
            "title": "⭐ 绩效评估",
            "description": "员工季度绩效评分",
            "header_file": "performance_headers.json",
            "data_key": "performance"
        }
    ],
    "default_headers": ["员工", "Q1", "Q2", "Q3", "Q4", "年度平均"]
}

def load_config():
    """加载表格配置文件"""
    config_path = os.path.join(CONFIG_DIR, 'table_configs.json')
    
    if os.path.exists(config_path):
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"加载配置文件时出错: {e}")
    
    # 如果配置文件不存在，返回默认配置
    return DEFAULT_CONFIG

def load_headers(header_file):
    """加载表头配置"""
    header_path = os.path.join(HEADERS_DIR, header_file)
    
    if os.path.exists(header_path):
        try:
            with open(header_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"加载表头文件 {header_file} 时出错: {e}")
    
    # 如果表头文件不存在，使用默认表头
    config = load_config()
    return config.get("default_headers", ["员工", "数据1", "数据2"])

def get_all_person_dirs(keyword=None):
    """获取所有个人数据文件夹"""
    person_dirs = []
    if keyword and keyword not in ['-1', '0', '0.325', '0.625']:
        keyword = keyword.split('; ')
        for i, kw in enumerate(keyword):
            person_dirs.append({})
            person_dirs[i]['name']=kw
            person_dirs[i]['path'] = f'{DATA_DIRS}\\{kw}'
    return person_dirs

def load_person_data(person_dir, table_config):
    """加载个人的表格数据"""
    person_name = person_dir['name']
    data_key = table_config['data_key']
    
    # 构建数据文件路径
    data_file = os.path.join(person_dir['path'], f"{data_key}.json")
    
    if os.path.exists(data_file):
        try:
            with open(data_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
                # 获取对应数据键的值
                if data_key in data:
                    data_values = data[data_key]
                    # 确保数据是列表
                    if not isinstance(data_values, list):
                        data_values = [data_values]
                    
                    # 将员工名称作为第一列
                    row_data = [person_name] + data_values
                    return row_data
                else:
                    # 如果数据键不存在，返回NA
                    return create_na_row(person_name, table_config)
        except Exception as e:
            print(f"加载 {person_name} 的 {data_key} 数据时出错: {e}")
            return create_na_row(person_name, table_config)
    else:
        # 如果文件不存在，返回NA
        return create_na_row(person_name, table_config)

def create_na_row(person_name, table_config):
    """创建包含NA值的行"""
    # 加载表头来确定列数
    headers = load_headers(table_config['header_file'])
    # 第一列是员工名称，其余列填充"NA"
    row_data = [person_name] + ["NA"] * (len(headers) - 1)
    return row_data

def build_table_data(table_config):
    """构建表格数据"""
    # 加载表头
    headers = load_headers(table_config['header_file'])
    
    # 获取所有个人文件夹
    person_dirs = get_all_person_dirs()
    
    # 加载每个人的数据
    rows = []
    for person_dir in person_dirs:
        row_data = load_person_data(person_dir, table_config)
        # 确保数据长度与表头匹配
        if len(row_data) < len(headers):
            # 填充缺失的值
            row_data = row_data + ["NA"] * (len(headers) - len(row_data))
        elif len(row_data) > len(headers):
            # 截断多余的值
            row_data = row_data[:len(headers)]
        
        rows.append(row_data)
    
    return {
        'id': table_config['id'],
        'title': table_config['title'],
        'description': table_config.get('description', ''),
        'headers': headers,
        'rows': rows
    }


def get_table_ids():
    """获取所有表格的ID列表"""
    config = load_config()
    return [table['id'] for table in config.get('tables', [])]

def save_person_data(person_name, data_key, data_values):
    """保存个人数据到JSON文件"""
    try:
        # 创建个人文件夹（如果不存在）
        person_dir = os.path.join(DATA_DIRS, person_name)
        os.makedirs(person_dir, exist_ok=True)
        
        # 构建数据文件路径
        data_file = os.path.join(person_dir, f"{data_key}.json")
        
        # 准备数据
        data_dict = {data_key: data_values}
        
        # 保存到文件
        with open(data_file, 'w', encoding='utf-8') as f:
            json.dump(data_dict, f, ensure_ascii=False, indent=2)
        
        return True
    except Exception as e:
        print(f"保存 {person_name} 的 {data_key} 数据时出错: {e}")
        return False

def delete_person_data(person_name, data_key):
    """删除个人的特定数据文件"""
    try:
        data_file = os.path.join(DATA_DIRS, person_name, f"{data_key}.json")
        
        if os.path.exists(data_file):
            os.remove(data_file)
            return True
        return False
    except Exception as e:
        print(f"删除 {person_name} 的 {data_key} 数据时出错: {e}")
        return False

def get_person_names(keyword=None):
    """获取所有人员名称列表"""
    person_dirs = get_all_person_dirs(keyword)
    return [person['name'] for person in person_dirs]

def get_table_config(table_id):
    """获取指定表格的配置"""
    config = load_config()
    for table_config in config.get('tables', []):
        if table_config['id'] == table_id:
            return table_config
    return None

def get_available_data_keys():
    """获取所有可用的数据键（表格类型）"""
    config = load_config()
    return [table['data_key'] for table in config.get('tables', [])]

def initialize_directories():
    """初始化必要的目录结构"""
    # 创建配置目录
    os.makedirs(CONFIG_DIR, exist_ok=True)
    os.makedirs(HEADERS_DIR, exist_ok=True)
    os.makedirs(DATA_DIRS, exist_ok=True)
    
    # 如果配置文件不存在，创建默认配置
    config_path = os.path.join(CONFIG_DIR, 'table_configs.json')
    if not os.path.exists(config_path):
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(DEFAULT_CONFIG, f, ensure_ascii=False, indent=2)
        print(f"创建默认配置文件: {config_path}")
    
    # 创建默认表头文件
    default_headers = {
        'performance_headers.json': ["员工", "Q1", "Q2", "Q3", "Q4", "年度平均"],
        'sales_headers.json': ["员工", "Q1", "Q2", "Q3", "Q4", "总计"],
        'projects_headers.json': ["员工", "项目名称", "开始日期", "预计完成", "进度", "状态"],
        'financial_headers.json': ["员工", "月份", "营收(万)", "成本(万)", "利润(万)", "利润率"]
    }
    
    for filename, headers in default_headers.items():
        header_path = os.path.join(HEADERS_DIR, filename)
        if not os.path.exists(header_path):
            with open(header_path, 'w', encoding='utf-8') as f:
                json.dump(headers, f, ensure_ascii=False, indent=2)
            print(f"创建表头文件: {header_path}")
    
    print("目录结构初始化完成")

# 应用启动时初始化
initialize_directories()
get_all_tables()