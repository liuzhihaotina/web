import json
import os
import glob
from datetime import datetime

# 配置目录
CONFIG_DIR = 'data_config'
HEADERS_DIR = os.path.join(CONFIG_DIR, 'headers')
DATA_DIRS = 'data_dirs'  # 包含多个子文件夹的根目录
IMPORTANT_DIR = 'data_dir_important'  # 标兵数据目录

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

def build_table_data_with_important(table_config):
    """构建包含标兵数据的表格数据"""
    # 加载表头
    headers = load_headers(table_config['header_file'])
    
    # 获取标兵人员
    important_persons = get_important_persons()
    
    # 获取普通人员（排除标兵）
    all_persons = get_person_names()
    normal_persons = [p for p in all_persons if p not in important_persons]
    
    # 首先添加标兵数据
    important_rows = []
    for person in important_persons:
        row_data = load_important_person_data(person, table_config)
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

def get_all_tables():
    """获取所有表格的数据（包含标兵）"""
    config = load_config()
    tables = []
    
    for table_config in config.get('tables', []):
        table_data = build_table_data_with_important(table_config)
        tables.append(table_data)
    
    return tables

def get_important_person_stats():
    """获取标兵统计信息"""
    important_persons = get_important_persons()
    return {
        'count': len(important_persons),
        'persons': important_persons,
        'has_important': len(important_persons) > 0
    }

def initialize_directories():
    """初始化必要的目录结构"""
    # 原有的代码...
    
    # 创建标兵目录
    os.makedirs(IMPORTANT_DIR, exist_ok=True)
    
    # 检查标兵目录
    print(f"标兵数据目录: {IMPORTANT_DIR}")
    important_count = len(get_important_persons())
    print(f"发现 {important_count} 个标兵")


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

def get_all_person_dirs():
    """获取所有个人数据文件夹"""
    if not os.path.exists(DATA_DIRS):
        print(f"数据目录 {DATA_DIRS} 不存在")
        return []
    
    # 获取所有子文件夹（排除隐藏文件夹）
    person_dirs = []
    for item in os.listdir(DATA_DIRS):
        item_path = os.path.join(DATA_DIRS, item)
        if os.path.isdir(item_path) and not item.startswith('.'):
            person_dirs.append({
                'name': item,  # 文件夹名称作为员工标识
                'path': item_path
            })
    
    # 按名称排序
    person_dirs.sort(key=lambda x: x['name'])
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

def get_person_names():
    """获取所有人员名称列表"""
    person_dirs = get_all_person_dirs()
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