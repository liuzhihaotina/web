from flask import Flask, render_template, request, jsonify, session, send_file, redirect, url_for
import data
import json
import os
import io
import pandas as pd
from datetime import datetime

app = Flask(__name__)
app.secret_key = 'your-secret-key-here'
app.config['SESSION_TYPE'] = 'filesystem'
# 从数据模块获取表格配置
def get_default_table_order():
    """获取默认表格顺序"""
    tables = data.get_tables_cfg()['tables']
    print([table['id'] for table in tables])
    return ['test_table']+[table['id'] for table in tables]

DEFAULT_TABLE_ORDER = get_default_table_order()
# -----------------------------------------------------------------------------------------------
@app.route('/')
def index():
    table_order = session.get('table_order', DEFAULT_TABLE_ORDER)
    tables_data = data.get_all_tables()
    
    # 按照保存的顺序重新排序表格
    ordered_tables = []
    table_dict = {table['id']: table for table in tables_data}
    
    # 首先添加顺序中存在的表格
    for table_id in table_order:
        if table_id in table_dict:
            ordered_tables.append(table_dict[table_id])
    
    # 然后添加其他表格
    for table in tables_data:
        if table['id'] not in table_order:
            ordered_tables.append(table)
            table_order.append(table['id'])
    
    session['table_order'] = table_order
    # 获取当前应用的logger
    app.logger.debug('进入 index 函数')
    # app.logger.info(f'表格列表: {table_order}')
    app.logger.info(f'表格内容: {ordered_tables}')
    
    return render_template('index.html', tables=ordered_tables, table_order=table_order, test_defa='A',
            now=datetime.now())

@app.route('/api/refresh-tables', methods=['GET', 'POST'])
def refresh_tables_api():
    """返回JSON格式的表格数据（保持原有逻辑）"""
    try:
        # 获取搜索关键词（如果有）
        keyword = request.args.get('q', '') if request.method == 'GET' else request.json.get('q', '')
        
        # 使用你原有的逻辑，但返回JSON
        table_order = session.get('table_order', DEFAULT_TABLE_ORDER)
        
        # 根据关键词获取数据
        if keyword:
            # 如果有搜索功能
            tables_data = data.get_all_tables(keyword)
        else:
            tables_data = data.get_all_tables()  # 或你的默认参数
        
        # 按照保存的顺序重新排序表格
        ordered_tables = []
        table_dict = {table['id']: table for table in tables_data}
        
        # 首先添加顺序中存在的表格
        for table_id in table_order:
            if table_id in table_dict:
                ordered_tables.append(table_dict[table_id])
        
        # 然后添加其他表格
        for table in tables_data:
            if table['id'] not in table_order:
                ordered_tables.append(table)
                table_order.append(table['id'])
        
        # 更新session中的顺序
        session['table_order'] = table_order
        
        # 返回JSON响应
        return jsonify({
            'success': True,
            'tables': ordered_tables,
            'count': len(ordered_tables),
            'table_order': table_order,
            'keyword': keyword if keyword else '',
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e),
            'message': '刷新表格失败'
        }), 500
    
# 后端新增API，提供表格配置信息
@app.route('/api/get_tables_config')
def get_tables_config():
    tables_cfg = get_default_table_order()
    
    return jsonify({
        'success': True,
        'tables': tables_cfg
    })
# -----------------------------------------------------------------------------------------------
@app.route('/update_order', methods=['POST'])
def update_order():
    try:
        data_req = request.get_json()
        table_order = data_req.get('order', [])
        
        # 验证表格ID是否存在
        all_table_ids = data.get_table_ids()
        valid_order = [table_id for table_id in table_order if table_id in all_table_ids]
        
        # 保存到session
        session['table_order'] = valid_order
        
        return jsonify({
            'success': True,
            'message': '顺序已更新',
            'order': valid_order
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'message': str(e)
        }), 400

# 数据管理API
@app.route('/api/tables', methods=['GET'])
def get_tables():
    """获取所有表格数据"""
    try:
        tables = data.get_all_tables()
        return jsonify({
            'success': True,
            'tables': tables
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'message': str(e)
        }), 500

@app.route('/api/table/<table_id>', methods=['GET'])
def get_table(table_id):
    """获取单个表格数据"""
    try:
        # 通过构建表格数据来获取
        table_config = data.get_table_config(table_id)
        if not table_config:
            return jsonify({
                'success': False,
                'message': '表格配置不存在'
            }), 404
        
        table_data = data.build_table_data(table_config)
        return jsonify({
            'success': True,
            'table': table_data
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'message': str(e)
        }), 500

@app.route('/api/persons', methods=['GET'])
def get_persons():
    """获取所有人员列表"""
    # 获取当前应用的logger
    # app.logger.debug('进入 get_persons 函数')
    
    # keyword = request.args.get('q', '')  # 获取查询参数
    # app.logger.info(f'搜索关键词: {keyword}')
    try:
        persons = data.get_person_names()
        app.logger.info(f'persons: {persons}')
        print(f'person={persons}')
        return jsonify({
            'success': True,
            'persons': persons
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'message': str(e)
        }), 500

@app.route('/api/person/<person_name>/data/<data_key>', methods=['GET'])
def get_person_data(person_name, data_key):
    """获取个人的特定数据"""
    try:
        # 构建个人文件夹路径
        person_dir = os.path.join(data.DATA_DIRS, person_name)
        data_file = os.path.join(person_dir, f"{data_key}.json")
        
        if os.path.exists(data_file):
            with open(data_file, 'r', encoding='utf-8') as f:
                data_content = json.load(f)
                
                # 确保数据键存在
                if data_key in data_content:
                    return jsonify({
                        'success': True,
                        'person': person_name,
                        'data_key': data_key,
                        'data': data_content[data_key]
                    })
                else:
                    return jsonify({
                        'success': False,
                        'message': f'数据键 {data_key} 不存在'
                    }), 404
        else:
            return jsonify({
                'success': False,
                'message': '数据文件不存在'
            }), 404
    except Exception as e:
        return jsonify({
            'success': False,
            'message': str(e)
        }), 500

@app.route('/api/person/<person_name>/data/<data_key>', methods=['POST'])
def save_person_data(person_name, data_key):
    """保存个人数据"""
    try:
        data_req = request.get_json()
        
        # 验证数据格式
        if 'data' not in data_req:
            return jsonify({
                'success': False,
                'message': '缺少data字段'
            }), 400
        
        data_values = data_req['data']
        
        # 验证数据是否为列表
        if not isinstance(data_values, list):
            return jsonify({
                'success': False,
                'message': 'data必须是列表'
            }), 400
        
        # 保存数据
        success = data.save_person_data(person_name, data_key, data_values)
        
        if success:
            return jsonify({
                'success': True,
                'message': f'{person_name}的{data_key}数据已保存'
            })
        else:
            return jsonify({
                'success': False,
                'message': '保存失败'
            }), 500
    except Exception as e:
        return jsonify({
            'success': False,
            'message': str(e)
        }), 500

@app.route('/api/person/<person_name>/data/<data_key>', methods=['DELETE'])
def delete_person_data(person_name, data_key):
    """删除个人数据"""
    try:
        success = data.delete_person_data(person_name, data_key)
        
        if success:
            return jsonify({
                'success': True,
                'message': f'{person_name}的{data_key}数据已删除'
            })
        else:
            return jsonify({
                'success': False,
                'message': '删除失败（可能文件不存在）'
            })
    except Exception as e:
        return jsonify({
            'success': False,
            'message': str(e)
        }), 500

@app.route('/api/table-configs', methods=['GET'])
def get_table_configs():
    """获取表格配置"""
    try:
        config = data.load_config()
        return jsonify({
            'success': True,
            'config': config
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'message': str(e)
        }), 500

@app.route('/api/table/<table_id>/rows')
def get_table_rows(table_id):
    """获取表格的行数据（支持分页）"""
    try:
        page = request.args.get('page', 1, type=int)
        page_size = request.args.get('page_size', 20, type=int)
        start = (page - 1) * page_size
        end = start + page_size
        
        table_config = data.get_table_config(table_id)
        if not table_config:
            return jsonify({
                'success': False,
                'message': '表格配置不存在'
            }), 404
        
        # 获取完整表格数据
        table_data = data.build_table_data(table_config)
        all_rows = table_data['rows']
        
        # 计算分页
        total_rows = len(all_rows)
        total_pages = (total_rows + page_size - 1) // page_size
        
        # 获取当前页的数据
        page_rows = all_rows[start:end]
        
        return jsonify({
            'success': True,
            'rows': page_rows,
            'total_rows': total_rows,
            'total_pages': total_pages,
            'current_page': page,
            'page_size': page_size,
            'has_next': page < total_pages,
            'has_prev': page > 1
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'message': str(e)
        }), 500

# 导出功能
@app.route('/export_all')
def export_all():
    """导出所有表格到多Sheet Excel"""
    try:
        tables_data = data.get_all_tables()
        output = io.BytesIO()
        
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            for table in tables_data:
                sheet_name = clean_sheet_name(table['title'])
                df = pd.DataFrame(table['rows'], columns=table['headers'])
                df.to_excel(writer, sheet_name=sheet_name, index=False)
        
        output.seek(0)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'员工数据汇总_{timestamp}.xlsx'
        
        return send_file(
            output,
            mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
            as_attachment=True,
            download_name=filename
        )
        
    except Exception as e:
        import traceback
        print(f"导出错误: {str(e)}")
        print(traceback.format_exc())
        return jsonify({
            'success': False,
            'message': f'导出失败: {str(e)}'
        }), 500

@app.route('/export_all_single_sheet_simple')
def export_all_single_sheet_simple():
    """导出所有表格到单Sheet（简化版）"""
    try:
        tables_data = data.get_all_tables()
        output = io.BytesIO()
        
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            all_data = []
            
            for table in tables_data:
                # 添加表格标题
                all_data.append([table['title']])
                
                # 添加表格描述
                if table.get('description'):
                    all_data.append([table['description']])
                else:
                    all_data.append([''])
                
                # 添加空行
                all_data.append([''])
                
                # 添加表头
                all_data.append(table['headers'])
                
                # 添加数据
                all_data.extend(table['rows'])
                
                # 添加表格之间的间隔（3个空行）
                all_data.append([''])
                all_data.append([''])
                all_data.append([''])
            
            # 找到最大列数
            max_cols = 0
            for row in all_data:
                if isinstance(row, list):
                    max_cols = max(max_cols, len(row))
                else:
                    max_cols = max(max_cols, 1)
            
            # 确保所有行都有相同的列数
            formatted_data = []
            for row in all_data:
                if isinstance(row, list):
                    if len(row) < max_cols:
                        row = row + [''] * (max_cols - len(row))
                    formatted_data.append(row)
                else:
                    formatted_data.append([str(row)] + [''] * (max_cols - 1))
            
            df = pd.DataFrame(formatted_data)
            df.to_excel(writer, sheet_name='所有表格', index=False, header=False)
        
        output.seek(0)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'员工数据汇总_{timestamp}.xlsx'
        
        return send_file(
            output,
            mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
            as_attachment=True,
            download_name=filename
        )
        
    except Exception as e:
        import traceback
        print(f"导出错误: {str(e)}")
        print(traceback.format_exc())
        return jsonify({
            'success': False,
            'message': f'导出失败: {str(e)}'
        }), 500

@app.route('/api/important-persons', methods=['GET'])
def get_important_persons():
    """获取标兵人员列表"""
    try:
        important_persons = data.get_important_persons()
        stats = data.get_important_person_stats()
        
        return jsonify({
            'success': True,
            'persons': important_persons,
            'stats': stats
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'message': str(e)
        }), 500

@app.route('/api/move-to-important/<person_name>', methods=['POST'])
def move_to_important(person_name):
    """将人员移动到标兵目录"""
    try:
        # 检查人员是否存在
        all_persons = data.get_person_names()
        if person_name not in all_persons:
            return jsonify({
                'success': False,
                'message': f'人员 {person_name} 不存在'
            }), 404
        
        # 检查是否已经是标兵
        important_persons = data.get_important_persons()
        if person_name in important_persons:
            return jsonify({
                'success': False,
                'message': f'{person_name} 已经是标兵'
            }), 400
        
        # 创建标兵目录
        important_person_dir = os.path.join(data.IMPORTANT_DIR, person_name)
        os.makedirs(important_person_dir, exist_ok=True)
        
        # 复制所有数据文件
        person_dir = os.path.join(data.DATA_DIRS, person_name)
        
        if os.path.exists(person_dir):
            # 获取所有数据键
            config = data.load_config()
            data_keys = [table['data_key'] for table in config.get('tables', [])]
            
            for data_key in data_keys:
                source_file = os.path.join(person_dir, f"{data_key}.json")
                target_file = os.path.join(important_person_dir, f"{data_key}.json")
                
                if os.path.exists(source_file):
                    import shutil
                    shutil.copy2(source_file, target_file)
        
        return jsonify({
            'success': True,
            'message': f'{person_name} 已设为标兵'
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'message': str(e)
        }), 500

@app.route('/api/remove-from-important/<person_name>', methods=['POST'])
def remove_from_important(person_name):
    """将人员从标兵目录移除"""
    try:
        # 检查是否是标兵
        important_persons = data.get_important_persons()
        if person_name not in important_persons:
            return jsonify({
                'success': False,
                'message': f'{person_name} 不是标兵'
            }), 400
        
        # 删除标兵目录
        important_person_dir = os.path.join(data.IMPORTANT_DIR, person_name)
        
        if os.path.exists(important_person_dir):
            import shutil
            shutil.rmtree(important_person_dir)
        
        return jsonify({
            'success': True,
            'message': f'{person_name} 已从标兵移除'
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'message': str(e)
        }), 500

@app.route('/api/toggle-important/<person_name>', methods=['POST'])
def toggle_important(person_name):
    """切换标兵状态"""
    try:
        # 检查人员是否存在
        all_persons = data.get_person_names()
        if person_name not in all_persons:
            return jsonify({
                'success': False,
                'message': f'人员 {person_name} 不存在'
            }), 404
        
        important_persons = data.get_important_persons()
        
        if person_name in important_persons:
            # 从标兵移除
            return remove_from_important(person_name)
        else:
            # 设为标兵
            return move_to_important(person_name)
    except Exception as e:
        return jsonify({
            'success': False,
            'message': str(e)
        }), 500

def clean_sheet_name(name):
    """清理sheet名称"""
    import re
    name = re.sub(r'[^\w\s-]', '', name)
    return name[:31]

if __name__ == '__main__':
    # 初始化目录结构
    data.initialize_directories()
    # 设置日志级别
    import logging
    logging.basicConfig(
        level=logging.DEBUG,  # DEBUG级别会显示所有信息
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            # logging.FileHandler('app.log'),  # 输出到文件
            logging.StreamHandler()           # 输出到控制台
        ]
    )
    
    app.logger.setLevel(logging.DEBUG)
    app.run(debug=True, port=5001)