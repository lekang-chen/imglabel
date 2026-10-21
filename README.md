# 人机协同图像标注系统

室内场景图像标注迭代工具 —— 预标注 → 人工审核修正 → 导出数据集 → 模型训练 → 回标，形成完整迭代闭环。

## 快速开始

```bash
# 1. 创建环境（Windows，Python 3.10+）
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# 2. 放入图片
#    把 150 张室内场景图片（JPG，长边 ≤960）放到 data/images/
#    （仓库已附带 data/data_source.txt，可用 tools/prepare_data.py 重新生成）

# 3. 启动
python app.py
```

浏览器会自动打开 `http://127.0.0.1:5000`。Windows 下也可直接双击 `启动.bat`。

桌面已放置快捷启动器 `人机协同图像标注系统.bat`（位于桌面），双击等价于运行 `start.bat`。
如需重建：`python tools/make_shortcut.py`（自动识别 OneDrive 重定向后的桌面路径）。

## 核心设计

> **以实例掩码为唯一核心标注对象。**
> 检测框 = 掩码外接矩形自动生成；语义掩码 = 实例掩码按类别合并自动派生。
> 人工只需修改掩码一次，三种格式自动同步，不存在检测框与掩码对不上的问题。

底层数据模型见 `core.py`，三种模式的渲染实现见 `canvas/app.js`。

## 使用流程

| 步骤 | 操作 | 产出 |
|---|---|---|
| 1 | 左侧「预标注」 | YOLO-World + YOLO26n-seg 生成草稿标注 |
| 2 | 逐张人工审核 | 改类别 / 修边界 / 补漏标，每次改动自动记入修正记录 |
| 3 | 左侧「导出数据集」 | `dataset/` 下生成 YOLO txt、实例掩码、语义 PNG 三种格式 + 格式自检 |
| 4 | 左侧「训练」 | 自动探测硬件配置，微调产出 `model/best.pt` |
| 5 | 左侧「回标」 | 用自训权重重新预标注，进入第二轮迭代 |
| 6 | 左侧「对比结果」 | 各轮 mAP 与修正率对比 |

## 目录结构

```
├─ app.py              Flask 主程序，HTTP 接口
├─ core.py             数据模型与几何工具（唯一真值的定义）
├─ prelabel.py         预标注封装，异步队列，掩码转多边形
├─ exporter.py         三格式导出器 + 格式自检
├─ train.py            模型微调，自动探测硬件配置
├─ canvas/             前端界面（原生 canvas + JS，无框架依赖）
│  ├─ app.js           交互逻辑（画笔、筛选、聚焦、三模式渲染）
│  └─ geom.js          掩码轮廓提取 + 闭合环 D-P 抽稀（可单测）
├─ tools/
│  ├─ prepare_data.py  从 COCO 筛选压缩出符合要求的数据集
│  ├─ fast_download.py 多线程分片下载器
│  ├─ test_core.py     核心逻辑自测（56 项）
│  ├─ test_geom.js     前端几何算法单测（30 项，node 直跑）
│  ├─ ui_test.js       真实浏览器交互测试（23 项，Edge headless）
│  ├─ eval_draft.py    草稿质量评估（对照 COCO 真值算 P/R/F1）
│  ├─ relabel_r2.py    用 best.pt 回标到独立目录（不覆盖 R1 草稿）
│  ├─ compare_rounds.py R1/R2 修正率逐类对比
│  ├─ make_compare_csv.py 生成 records/对比结果.csv
│  ├─ shoot_demo.py    抓 demo 截图（Edge headless）
│  ├─ exp_lr.py        学习率对照实验
│  └─ package.py       打包提交件（含 .git 提交历史）
├─ data/images/        原始图片
├─ data/annotations/   标注 JSON
├─ dataset/            导出的三格式数据集
├─ model/best.pt       自训权重
├─ records/            修正记录.csv + 对比结果.csv
└─ 启动.bat            一键启动
```

## 自测

```bash
python tools/test_core.py     # 56 项：核心逻辑
node tools/test_geom.js       # 30 项：前端几何算法
node tools/ui_test.js         # 23 项：真实浏览器交互（需先启动服务）
```

共 **109 项**。覆盖：
- 核心层：几何工具、掩码转多边形、三格式一致性、导出自检、边界情况、
  PNG 存盘往返一致性（防调色板模式回归）
- 几何层：缝隙追踪的面积精度、闭合环 D-P 抽稀、鞍点配对、顶点数上限
- 交互层：复选框筛选、聚焦模式、画笔涂抹/擦除、轮廓描边

> `ui_test.js` 在 `?uitest=1` 模式下会禁用落盘，自动化测试**不会污染真实标注数据**。

## 质量评估与迭代对比

```bash
# 1. 评估当前草稿质量（对照 COCO 官方真值）
python tools/eval_draft.py

# 2. 用自训权重回标到 data/annotations_r2（不覆盖 R1 草稿）
python tools/relabel_r2.py

# 3. 对比 R1 / R2 的修正率
python tools/compare_rounds.py

# 4. 生成 records/对比结果.csv
python tools/make_compare_csv.py
```

### 实测结果

| 指标 | R1 官方权重 | R2 自训权重 | 变化 |
|---|---|---|---|
| 草稿对象数 | 900 | 627 | −273 |
| 需人工修正数 | 529 | 320 | −209 |
| **修正率** | **58.8%** | **51.0%** | **↓ 7.7 个百分点** |
| 精确率 | 0.412 | 0.490 | +0.077 |
| 召回率 | 0.600 | 0.497 | −0.103 |
| 检测 mAP@0.5 | — | **0.741** | — |
| 掩码 mAP@0.5 | — | **0.474** | — |

> 召回率下降不是模型变差：R1 是两个模型配合、倾向多标（精确率仅 0.412），
> R2 是单一自训模型、只标有把握的。**要人改的对象从 529 降到 320**，
> 这才是闭环的价值。详见 `ai_log.md`。

## 环境要求

- Python 3.10+，Windows 10 x64 / macOS / Linux
- 最低配置：8GB 内存、20GB 可用磁盘（纯 CPU 可训练，仅耗时较长）
- 推荐：8GB 显存以上 NVIDIA 显卡（本项目实测 RTX 5060 Laptop 8.5GB，
  训练 51 轮耗时 3.1 分钟）
- 依赖：ultralytics、flask、psutil、opencv-python、numpy、pillow、clip-anytorch

## 类别词表

固定十类，不可更改：

`person` `chair` `dining table` `couch` `tv` `laptop` `cup` `bottle` `potted plant` `book`

## 数据准备

推荐方式（**不下载 815MB 整包**，实测 59 秒完成 150 张）：

```bash
# 1. 只取标注文件
curl -L -o data/coco/instances_val2017.json https://hf-mirror.com/datasets/merve/coco/resolve/main/annotations/instances_val2017.json

# 2. 按类别均衡策略选出 150 张，从官方直链逐张抓取 + 压缩
python tools/prepare_data.py --fetch-url --coco-ann data/coco/instances_val2017.json --out data/images
```

传统方式（下载整包，慢）：

```bash
python tools/fast_download.py http://images.cocodataset.org/zips/val2017.zip data/coco/val2017.zip --threads 12
python tools/prepare_data.py --coco-ann data/coco/instances_val2017.json --coco-img data/coco/val2017 --out data/images
```

筛选说明自动写入 `data/data_source.txt`，可据此填写 `提交信息.txt`。
