"""
tools/exp_lr.py —— 学习率对比实验

【为什么需要这个脚本】
第一轮训练（train.py 默认参数）出现了一个反常现象：
  - 第 1 轮 mAP50 就有 0.60（Box）
  - 之后精度和召回同时下降，训到 51 轮早停
  - 损失一直在降，但验证集指标在掉

诊断结论：这是**灾难性遗忘**。
我们的基线权重 yolo26n-seg.pt 是在 COCO 80 类上训的，
它本来就已经认识 person/chair/cup/... 这十类（只是类别编号和我们对不上，
所以零样本 mAP 只有 0.075）。第 1 轮训练主要在做「类别编号对齐」，
这件事完成后，继续用 120 张噪声伪标签硬训，
反而把预训练权重里那些通用视觉特征冲掉了。

所以要试：**更小的学习率**，看能不能保住预训练知识同时对齐类别。

【怎么用】
  python tools/exp_lr.py 0.0001
  python tools/exp_lr.py 0.0003
"""

import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    lr = float(sys.argv[1]) if len(sys.argv) > 1 else 0.0001
    epochs = int(sys.argv[2]) if len(sys.argv) > 2 else 40

    from ultralytics import YOLO

    name = "exp_lr%s" % str(lr).replace("0.", "").replace(".", "p")
    print("=" * 60)
    print("学习率实验: lr0=%s, epochs=%d, 输出 runs/%s" % (lr, epochs, name))
    print("=" * 60)

    m = YOLO(os.path.join(BASE_DIR, "weights", "yolo26n-seg.pt"))
    m.train(
        data=os.path.join(BASE_DIR, "dataset", "data.yaml"),
        epochs=epochs,
        imgsz=640,
        batch=8,
        device="0",
        workers=2,
        project=os.path.join(BASE_DIR, "runs"),
        name=name,
        exist_ok=True,
        optimizer="AdamW",
        lr0=lr,
        lrf=0.01,
        weight_decay=0.0005,
        patience=epochs,
        cos_lr=True,
        warmup_epochs=3,
        verbose=False,
        plots=False,
        seed=42,
    )
    print("完成，结果在 runs/%s/results.csv" % name)


if __name__ == "__main__":
    main()
