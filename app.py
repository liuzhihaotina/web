from typing import List
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

# -----------------------------------------------------------------------------------------------
@app.route('/', methods=["GET"])
def index():
    keyword = request.args.get('keyword', None)
    height = request.args.get('height', None)
    occ_threshold = request.args.get('occ_threshold', "0.60")

    # 解析模型选择
    selected_models = request.args.getlist("models")
    if selected_models:
        base_line=[]
        other=[]
        for m in selected_models:
            if '⭐' in m:
                base_line.append(m.replace('⭐', '').strip())
            else:
                other.append(m.strip())
    else:
        base_line = None
        other = None

    # 解析指标选择
    selected_metrics = request.args.getlist("metrics")

    models_all, metrics_all, charts, settings, constants = \
        data.get_all_tables(base_line, other, keyword, selected_metrics, height, occ_threshold)


    return render_template(
        "index.html",
        error=None,
        data_dir='data_dir_important',
        models_all=models_all,
        metrics_all=metrics_all,
        charts=charts,
        settings=settings,
        constants=constants,
        keyword=keyword if keyword else "",
        height=height if height else "-1",
    )


if __name__ == '__main__':
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

    # with app.test_request_context("/?models=⭐ lisi83"):
    #     index()

    app.logger.setLevel(logging.DEBUG)
    app.run(debug=True, port=5001)