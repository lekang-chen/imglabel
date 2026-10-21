# 人机协同图像标注系统

室内场景图像标注迭代工具 —— 预标注 → 人工审核修正 → 导出数据集 → 模型训练 → 回标，形成完整迭代闭环。

## 快速开始

```bash
# 1. 创建环境
conda create -n imglabel python=3.10 -y
conda activate imglabel
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# 2. 放入图片
#    把 150 张室内场景图片（JPG，长边 ≤960）放到 data/images/

# 3. 启动
python app.py
```

浏览器会自动打开 `http://127.0.0.1:5000`。Windows 下也可直接双击 `启动.bat`。

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
├─ tools/
│  ├─ prepare_data.py  从 COCO 筛选压缩出符合要求的数据集
│  ├─ fast_download.py 多线程分片下载器
│  └─ test_core.py     核心逻辑自测（49 项）
├─ data/images/        原始图片
├─ data/annotations/   标注 JSON
├─ dataset/            导出的三格式数据集
├─ model/best.pt       自训权重
├─ records/            修正记录.csv + 对比结果.csv
└─ 启动.bat            一键启动
```

## 自测

```bash
python tools/test_core.py
```

覆盖几何工具、掩码转多边形、三格式一致性、导出自检、边界情况。

## 环境要求

- Python 3.10+，Windows 10 x64 / macOS / Linux
- 最低配置：8GB 内存、20GB 可用磁盘（纯 CPU 可训练，仅耗时较长）
- 推荐：8GB 显存以上 NVIDIA 显卡，训练约 15~25 分钟
- 依赖：ultralytics、flask、psutil、opencv-python、numpy、pillow、clip-anytorch

## 类别词表

固定十类，不可更改：

`person` `chair` `dining table` `couch` `tv` `laptop` `cup` `bottle` `potted plant` `book`

## 数据准备

```bash
# 下载 COCO val2017
python tools/fast_download.py http://images.cocodataset.org/zips/val2017.zip data/coco/val2017.zip --threads 12
# 标注文件
curl -L -o data/coco/instances_val2017.json https://hf-mirror.com/datasets/merve/coco/resolve/main/annotations/instances_val2017.json

# 筛选 + 压缩（长边 ≤960，JPG 质量 85）
python tools/prepare_data.py --coco-ann data/coco/instances_val2017.json --coco-img data/coco/val2017 --out data/images
```

筛选说明自动写入 `data/data_source.txt`，可据此填写 `提交信息.txt`。
