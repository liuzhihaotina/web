import json
import math
import os
import glob
from datetime import datetime
from typing import List, Optional, Tuple

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

def get_important_persons(base_line):
    """获取标兵人员列表"""
    important_persons = []
    
    if os.path.exists(IMPORTANT_DIR):
        if base_line is None:
            for item in os.listdir(IMPORTANT_DIR):
                item_path = os.path.join(IMPORTANT_DIR, item)
                if os.path.isdir(item_path) and not item.startswith('.'):
                    important_persons.append(item)
        else:
            important_persons = base_line
    
    return sorted(important_persons)

def load_important_person_data(person_name, table_config, height, occ_threshold):
    """加载标兵个人数据"""
    data_key = table_config['data_key']
    title = table_config['title']
    
    # 从基线目录加载
    important_file = os.path.join(IMPORTANT_DIR, person_name, f"{data_key}.json")
    
    if os.path.exists(important_file):
        try:
            with open(important_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if data_key in data:
                    data_values = data[data_key]
                    if title in {'🚀 项目进度跟踪'}:
                        height = '-1' if height is None else height
                        data_values = data_values[height]
                    if title in {'🌸 occ阈值'}:
                        occ_threshold = '0.60' if occ_threshold is None else occ_threshold
                        data_values = data_values[occ_threshold]
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
    
    # 没有则返回空列表
    return []

def build_table_data(table_config, base_line, other, keyword, height, occ_threshold):
    # 加载表头
    headers = load_headers(table_config['header_file'])
    
    # 获取基线
    important_persons = get_important_persons(base_line)
    
    # 获取其他
    all_persons = get_person_names(other, keyword)
    normal_persons = [p for p in all_persons if p not in important_persons]
    
    # 首先添加标兵数据
    important_rows = []
    for person in important_persons:
        row_data = load_important_person_data(person, table_config, height, occ_threshold)
        if len(row_data) == 0:
            continue
        # 确保数据长度与表头匹配
        if len(row_data) < len(headers):
            row_data = row_data + ["NA"] * (len(headers) - len(row_data))
        elif len(row_data) > len(headers):
            row_data = row_data[:len(headers)]
        important_rows.append(row_data)
    
    # 然后添加普通人员数据
    normal_rows = []
    for person in normal_persons:
        person_dir = {'name': person, 'path': os.path.join(DATA_DIRS, person)}
        row_data = load_person_data(person_dir, table_config, height, occ_threshold)
        if len(row_data) == 0:
            continue
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

def get_all_tables(base_line, other, keyword, selected_metrics, height, occ_threshold):
    """获取所有表格的数据"""
    config = get_tables_cfg()
    tables = []
    
    for table_config in config.get('tables', []):
        if selected_metrics and table_config['title'] not in selected_metrics:
            continue
        table_data = build_table_data(table_config, base_line, other, keyword, height, occ_threshold)
        tables.append(table_data)
        
    models_all = get_models_all(tables)
    metrics_all, headers_all, metric_data_dict = get_metrics_headers_all(tables)
    charts, settings, constants = \
    get_charts(models_all, metrics_all, headers_all, metric_data_dict)
    return models_all, metrics_all, charts, settings, constants

def build_metric_dataframe(models_all, header, rows):
    # 只包含有数据的模型
    valid_models = [model for model in models_all if rows.get(model)]
    
    if not valid_models:
        return pd.DataFrame(columns=header)  # 返回空DataFrame
    
    df = pd.DataFrame(index=valid_models, columns=header, dtype=float)
    for model in valid_models:
        row_data = rows.get(model, {})
        if not row_data:  # 双重检查
            continue
        for sm, row in zip(header, row_data):
            df.loc[model, sm] = row
    
    return df

def footer_height_px(x_labels: List[str]) -> int:
    """计算底部斜排文字所需高度"""
    max_len = max(len(x) for x in x_labels)
    text_w = max_len * CHAR_PX
    theta = math.radians(FOOTER_ROT_DEG)

    # 斜排后竖向投影 + 字体高度 + 上下安全垫
    h = int(math.sin(theta) * text_w + (FOOTER_FONT_PX * 1.6) + FOOTER_PAD_TOP + FOOTER_PAD_BOTTOM)
    return max(FOOTER_MIN_H, min(FOOTER_MAX_H, h))

def compute_range_1d(v: np.ndarray, robust: bool = True) -> Tuple[Optional[float], Optional[float]]:
    v = v[~np.isnan(v)]
    if v.size == 0:
        return None, None
    if robust and v.size >= 10:
        return float(np.quantile(v, 0.02)), float(np.quantile(v, 0.98))
    return float(np.min(v)), float(np.max(v))

def normalize_by_column(values: np.ndarray, robust: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if values.ndim != 2:
        raise ValueError("values must be 2D")

    n_rows, n_cols = values.shape
    col_min = np.full((n_cols,), np.nan, dtype=float)
    col_max = np.full((n_cols,), np.nan, dtype=float)

    for j in range(n_cols):
        mn, mx = compute_range_1d(values[:, j].astype(float), robust=robust)
        col_min[j] = np.nan if mn is None else float(mn)
        col_max[j] = np.nan if mx is None else float(mx)

    z_norm = np.full_like(values.astype(float), np.nan, dtype=float)
    for j in range(n_cols):
        mn, mx = col_min[j], col_max[j]
        if np.isnan(mn) or np.isnan(mx):
            continue
        denom = (mx - mn)
        col = values[:, j].astype(float)
        if abs(denom) < 1e-12:
            z_norm[:, j] = np.where(np.isnan(col), np.nan, 0.5)
        else:
            z_norm[:, j] = (col - mn) / denom

    return np.clip(z_norm, 0.0, 1.0), col_min, col_max

def compute_col_widths(x_labels: List[str], text: Optional[List[List[str]]]) -> List[int]:
    n_cols = len(x_labels)
    if not text:
        out = []
        for j in range(n_cols):
            max_len = len(str(x_labels[j]))
            w = COL_PAD_PX + max_len * CHAR_PX
            out.append(int(max(MIN_COL_W_PX, min(MAX_COL_W_PX, w))))
        return out

    out = []
    for j in range(n_cols):
        max_len = len(str(x_labels[j]))
        for i in range(len(text)):
            s = text[i][j] if j < len(text[i]) else ""
            if s:
                max_len = max(max_len, len(str(s)))
        w = COL_PAD_PX + max_len * CHAR_PX
        out.append(int(max(MIN_COL_W_PX, min(MAX_COL_W_PX, w))))
    return out

def estimate_yaxis_width_px(labels: List[str]) -> int:
    if not labels:
        return YAXIS_WIDTH_DEFAULT
    max_chars = max(len(x) for x in labels)
    w = 40 + int(max_chars * 9)
    return max(170, min(320, w))

def get_charts(models_all, metrics_all, headers_all, metric_data_dict):
    charts = []
    global_footer_h = FOOTER_MIN_H
    robust = True
    show_text = True

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
            text = [[str(v)+'ww' for v in row] for row in raw]

        col_widths = compute_col_widths(x_labels, text)
        min_matrix_width = int(sum(col_widths))

        n_rows = len(y_labels)
        footer_h = int(global_footer_h)  # ✅统一
        height = int(n_rows * CELL_H_PX + footer_h)

        legend_target = min(height, max(height, MIN_LEGEND_PX))
        yaxis_width = estimate_yaxis_width_px(y_labels)

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
    settings = dict(show_text=show_text, precision=4, robust=robust)
    constants = dict(
        CELL_H_PX=CELL_H_PX,
        CBAR_WIDTH=CBAR_WIDTH,
        MIN_LEGEND_PX=MIN_LEGEND_PX,
    )
    return charts, settings, constants

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
# ================================================================================================
def load_config():
    """加载表格配置文件"""
    config_path = os.path.join(CONFIG_DIR, 'table_configs.json')
    
    if os.path.exists(config_path):
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"加载配置文件时出错: {e}")

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

def get_all_person_dirs(other, keyword=None):
    """获取数据文件夹"""
    person_dirs = []
    if keyword:
        keyword = keyword.split('; ')
        final = keyword if other is None else other
        for k in final:
                person_dirs.append({})
                person_dirs[-1]['name']=k
                person_dirs[-1]['path'] = f'{DATA_DIRS}\\{k}'
    return person_dirs

def load_person_data(person_dir, table_config, height, occ_threshold):
    """加载个人的表格数据"""
    person_name = person_dir['name']
    data_key = table_config['data_key']
    title = table_config['title']

    # 构建数据文件路径
    data_file = os.path.join(person_dir['path'], f"{data_key}.json")
    
    if os.path.exists(data_file):
        try:
            with open(data_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if title in {'🚀 项目进度跟踪'}:
                    data = data[height]
                if title in {'🌸 occ阈值'}:
                    occ_threshold = '0.60' if occ_threshold is None else occ_threshold
                # 获取对应数据键的值
                if data_key in data:
                    data_values = data[data_key]
                    # 确保数据是列表
                    if not isinstance(data_values, list):
                        data_values = [data_values]
                    
                    # 将员工名称作为第一列
                    row_data = [person_name] + data_values
                    return row_data
        except Exception as e:
            print(f"加载 {person_name} 的 {data_key} 数据时出错: {e}")
    return []

def get_person_names(other, keyword=None):
    """获取所有人员名称列表"""
    person_dirs = get_all_person_dirs(other, keyword)
    return [person['name'] for person in person_dirs]

if __name__ == '__main__':
    # 测试加载数据
    get_all_tables(base_line=None, other=None, keyword='lisi; zhangsan', 
                   selected_metrics=None, height='0.625', occ_threshold=None)