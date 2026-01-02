from flask import Flask, render_template, request
import data

app = Flask(__name__)
app.secret_key = 'your-secret-key-here'
app.config['SESSION_TYPE'] = 'filesystem'


@app.route('/', methods=["GET"])
def index():
    keyword = request.args.get('keyword', None)
    height = request.args.get('height', None)
    occ_threshold = request.args.get('occ_threshold', "0.60")

    # 解析模型选择
    selected_models = request.args.getlist("models")
    if selected_models:
        base_line = []
        other = []
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
        occ_threshold=occ_threshold if occ_threshold else "0.60",

        # ✅让下拉/多选刷新不丢
        selected_models=selected_models,
        selected_metrics=selected_metrics,
    )


if __name__ == '__main__':
    import logging
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler()]
    )
    app.logger.setLevel(logging.DEBUG)
    app.run(debug=True, port=5001)
