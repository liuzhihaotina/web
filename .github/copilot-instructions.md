## 项目快速说明与 AI 助手使用准则

该仓库是一个以 Flask 为前端展示的轻量数据浏览与管理工具。下面的指导针对 AI 编码代理（例如 Copilot 类助手），让你快速在此仓库里高效、安全地修改代码或实现新特性。

**快速启动**:
- **创建虚拟环境 & 安装依赖**: ``python -m venv .venv`` 然后在 PowerShell 中:
  ``.\.venv\Scripts\Activate.ps1``
  ``pip install -r requirements.txt``
- **运行应用（开发）**: ``python app.py`` （`app.py` 中使用 `app.run(debug=True)`）

**大局观（文件与职责）**:
- `app.py`: Flask 应用入口，定义路由与 API（例如 `/`、`/api/search`、`/api/add_data`）。前端模板位于 `templates/`，静态资源在 `static/`。
- `data.py`: 核心数据层。包含 `DataManager` 类与全局实例 `data_manager`，负责从 `data.json` 加载/保存数据，提供 `get_exemplary_data()`、`search_data()`、`add_data()` 等方法。
- `multi_dir.py`: 文件/目录批量操作工具（例如复制并自动加数字后缀）。对大量文件夹操作请小心（I/O 密集）。
- `data_config/table_configs.json`: 表格元数据（表 id、标题、数据 key、headers 文件），是前端表格生成的配置来源。
- `data_dirs/` 与 `data_files/`: 仓库中包含大量以 `lisi*` 命名的子目录，通常每个子目录对应个人数据。避免未经必要的全量遍历。

**代码约定与关键模式（对 AI 助手很重要）**:
- 全局单例数据管理器: 在 `data.py` 文件末有 `data_manager = DataManager()`，其他模块通过 `import data` 然后使用 `data.data_manager` 操作数据。修改数据后务必调用 `save_data()` 或让 `DataManager` 自动保存。
- 搜索逻辑：`DataManager.search_data(keyword)` 会默认跳过 `type == '标兵'` 的条目（因为主页默认展示标兵），这是项目特定的业务约束 — 不要无意中改变这一行为。
- API 输入/输出约定：`/api/search` 接受 `POST`，从表单字段 `keyword` 读取关键字；返回 JSON 包含 `success`、`results`、`count`。
- 配置优先于扫描：若要知道表格或列头，应优先读取 `data_config/table_configs.json` 与相关 headers 文件，而不是试图解析 `data_dirs/` 内容。

**性能与安全建议（在代码修改中必须注意）**:
- 避免在请求处理路径中执行大范围磁盘 I/O（例如递归遍历 `data_dirs/`）。如需读取大量文件，应在后台任务或按需分页加载。
- 当新增 API 导致写操作（如 `add_data`）时，注意并发写入到 `data.json` 的竞争，必要时序列化写操作或改用轻量数据库。

**如何改动代码（常见任务示例）**:
- 新增页面或 API：修改 `app.py` 添加路由，数据相关操作通过 `data.data_manager` 调用。
- 改变表格/列定义：在 `data_config/table_configs.json` 中添加或更新表记录，并在 `data_config/headers/` 中添加对应 header 文件名。
- 增加依赖：更新 `requirements.txt` 并在 PR 描述中注明安装顺序与兼容的 Python 版本。

**示例片段（典型查询路径）**:
```
# 前端 -> POST /api/search (form: keyword)
# 后端 app.py 调用 -> data.data_manager.search_data(keyword)
# DataManager.search_data 会跳过 type=='标兵' 的条目
```

**对 AI 助手的具体工作规则**:
- 优先读取并引用本仓库的配置文件（如 `data_config/table_configs.json`）而非猜测结构。
- 当修改会影响磁盘 I/O 或大文件夹（`data_dirs/`）时，提示并征求人工确认。
- 在更改 `data.py` 的持久化逻辑时，务必保持 `data.json` 的 JSON 编码（`ensure_ascii=False, indent=2`）以便兼容现有工具链。
- 在提交 PR 时，在描述中列出具体改动的文件、是否涉及数据迁移、以及如何在本地复现（包含上面的启动命令）。

**常见问题/陷阱**:
- 仓库包含大量 `data_dirs/lisi*` 文件夹，运行任何递归扫描前先确认测试子集。
- `app.py` 的 `debug=True` 仅用于开发；不要在生产场景下保留该设置。

---
如果你希望我把某一节扩展为示例 PR、或添加自动化测试/CI 指南（当前仓库未提供测试框架），告诉我需要补充的内容或你关心的改动点。期待你的反馈以便迭代此文件。
