"""
e2e_delete.py —— 端到端验证「删除照片 / 恢复」

为什么单独写这个：
  删除是唯一一个会动 data/images 的操作，而那里躺着 150 张图 + 60 条人工修正。
  单元测试只能验证「点了按钮会不会发请求」，验证不了「文件到底搬没搬走、
  标注是不是跟着走、恢复回来 edited 标记还在不在」——这些必须真跑一遍。

流程（跑完自动清理，初始多少张最后还是多少张）：
  1. 上传一张测试图          -> 图片数 +1
  2. 给它造一份带 edited 的标注
  3. 删除                    -> 图片数 -1，图片与标注成对进 data/_trash/<批次>/
  4. 恢复                    -> 图片数回升，标注内容与 edited 标记完整
  5. 再删一次并清掉这一批    -> 回收站不留垃圾

前置：服务已启动（双击 启动.bat，或 python app.py）
用法：python tools/e2e_delete.py
      python tools/e2e_delete.py --base http://127.0.0.1:5000
"""

import argparse
import io
import json
import os
import shutil
import sys
import time
import urllib.error
import urllib.request

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
IMAGE_DIR = os.path.join(DATA_DIR, "images")
ANNO_DIR = os.path.join(DATA_DIR, "annotations")
TRASH_DIR = os.path.join(DATA_DIR, "_trash")

TEST_NAME = "删除自测.png"           # 上传时用的名字（服务端可能微调，以返回值为准）

passed, failed = 0, 0


def ok(name, cond, extra=""):
    global passed, failed
    if cond:
        passed += 1
        print("  PASS  " + name)
    else:
        failed += 1
        print("  FAIL  " + name + (("  -> " + extra) if extra else ""))


# ---------------------------------------------------------------------------
# 最小 HTTP 客户端（只用标准库，评委机器上不必额外装东西）
# ---------------------------------------------------------------------------

def get(base, path):
    with urllib.request.urlopen(base + path, timeout=30) as r:
        return json.loads(r.read().decode("utf-8"))


def post_json(base, path, payload):
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(base + path, data=data, method="POST",
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            return r.status, json.loads(r.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read().decode("utf-8") or "{}")


def post_file(base, path, filename, content, ctype="image/png"):
    """手写 multipart —— 只为一个字段，不值得引第三方库。"""
    boundary = "----wbe2e%d" % int(time.time() * 1000)
    head = (
        "--%s\r\n"
        'Content-Disposition: form-data; name="file"; filename="%s"\r\n'
        "Content-Type: %s\r\n\r\n" % (boundary, filename, ctype)
    ).encode("utf-8")
    tail = ("\r\n--%s--\r\n" % boundary).encode("utf-8")
    body = head + content + tail
    req = urllib.request.Request(
        base + path, data=body, method="POST",
        headers={"Content-Type": "multipart/form-data; boundary=%s" % boundary})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read().decode("utf-8"))


# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:5000")
    args = ap.parse_args()
    base = args.base.rstrip("/")

    try:
        before = get(base, "/api/images")
    except Exception as exc:  # noqa: BLE001
        print("连不上服务 %s：%s" % (base, exc))
        print("请先启动（双击 启动.bat 或 python app.py）再跑这个脚本。")
        sys.exit(2)

    total0 = before["total"]
    print("初始图片数: %d" % total0)
    print("")

    # 服务端会对文件名做安全处理（比如去掉首尾下划线），
    # 所以清理时不能用 TEST_NAME，必须用上传接口返回的实际名字。
    created = []

    # 这个脚本会让服务端往 state.json 写「上传/删除/恢复」三条运算日志。
    # 那些是测试造出来的假记录，会污染答辩时要展示的操作流水，
    # 所以进来先存一份原始字节，结束时原样写回。
    state_path = os.path.join(DATA_DIR, "state.json")
    state_backup = None
    if os.path.exists(state_path):
        with open(state_path, "rb") as f:
            state_backup = f.read()

    try:
        # ---- 1. 上传 ----
        from PIL import Image
        buf = io.BytesIO()
        Image.new("RGB", (320, 240), (150, 140, 130)).save(buf, "PNG")
        r = post_file(base, "/api/upload", TEST_NAME, buf.getvalue())
        saved = [s["name"] for s in r.get("saved", [])]
        ok("上传成功", len(saved) == 1, json.dumps(r, ensure_ascii=False))
        if not saved:
            return
        name = saved[0]
        created.append(name)
        after_up = get(base, "/api/images")
        ok("上传后图片数 +1", after_up["total"] == total0 + 1,
           "%d -> %d" % (total0, after_up["total"]))

        # ---- 2. 造一份带人工修正标记的标注 ----
        anno_path = os.path.join(ANNO_DIR, os.path.splitext(name)[0] + ".json")
        with open(anno_path, "w", encoding="utf-8") as f:
            json.dump({
                "image": name, "width": 320, "height": 240,
                "objects": [{
                    "id": "e2edel0001", "class": "cup", "edited": True,
                    "polygon": [[0.1, 0.1], [0.4, 0.1], [0.4, 0.4], [0.1, 0.4]],
                }],
            }, f, ensure_ascii=False)
        ok("标注已就位（含 1 条 edited）", os.path.exists(anno_path))

        # ---- 3. 删除 ----
        code, r = post_json(base, "/api/images/delete", {"names": [name]})
        ok("删除接口返回成功", code == 200 and name in r.get("deleted", []),
           json.dumps(r, ensure_ascii=False))
        stamp = r.get("trash")
        ok("返回了回收站批次路径", bool(stamp), str(stamp))
        ok("图片文件已从 data/images 移走",
           not os.path.exists(os.path.join(IMAGE_DIR, name)))
        ok("标注文件也一起移走了", not os.path.exists(anno_path))

        batch_dir = os.path.join(BASE_DIR, stamp.replace("/", os.sep)) if stamp else ""
        if batch_dir:
            files = sorted(os.listdir(batch_dir))
            ok("回收站里图片与标注成对出现",
               name in files and os.path.basename(anno_path) in files,
               "、".join(files))

        after_del = get(base, "/api/images")
        ok("删除后图片数回到初始值", after_del["total"] == total0,
           "%d" % after_del["total"])
        ok("列表里不再出现被删的图",
           all(i["name"] != name for i in after_del["images"]))

        # ---- 4. 恢复 ----
        code, r = post_json(base, "/api/images/restore", {"stamp":
                                                          (stamp or "").split("/")[-1]})
        ok("恢复接口返回成功", code == 200, json.dumps(r, ensure_ascii=False))
        ok("图片文件已搬回", os.path.exists(os.path.join(IMAGE_DIR, name)))
        ok("标注文件已搬回", os.path.exists(anno_path))

        if os.path.exists(anno_path):
            with open(anno_path, encoding="utf-8") as f:
                d = json.load(f)
            objs = d.get("objects") or []
            ok("恢复后标注内容完整（1 个对象）", len(objs) == 1, str(len(objs)))
            ok("恢复后 edited 标记还在",
               bool(objs) and objs[0].get("edited") is True)

        after_res = get(base, "/api/images")
        ok("恢复后图片数回到初始值 +1", after_res["total"] == total0 + 1,
           "%d" % after_res["total"])

        # ---- 5. 再删一次，验证「不留空批次」并清理 ----
        code, r = post_json(base, "/api/images/delete", {"names": [name]})
        ok("二次删除成功", code == 200 and name in r.get("deleted", []))
        after_final = get(base, "/api/images")
        ok("最终图片数 = 初始值", after_final["total"] == total0,
           "%d" % after_final["total"])

    finally:
        # 兜底清理：任何一步异常退出，都不给仓库留垃圾
        for n in created:
            stem = os.path.splitext(n)[0]
            for p in (os.path.join(IMAGE_DIR, n),
                      os.path.join(ANNO_DIR, stem + ".json")):
                if os.path.exists(p):
                    os.remove(p)
            # 回收站里可能还有本次的备份（测试就是删了又恢复、再删一次）
            if os.path.isdir(TRASH_DIR):
                for batch in os.listdir(TRASH_DIR):
                    bd = os.path.join(TRASH_DIR, batch)
                    if not os.path.isdir(bd):
                        continue
                    for f in os.listdir(bd):
                        if os.path.splitext(f)[0] == stem:
                            os.remove(os.path.join(bd, f))
                    if not os.listdir(bd):
                        os.rmdir(bd)
        if os.path.isdir(TRASH_DIR) and not os.listdir(TRASH_DIR):
            shutil.rmtree(TRASH_DIR, ignore_errors=True)
        if state_backup is not None:
            with open(state_path, "wb") as f:
                f.write(state_backup)

    print("")
    print("结果：%d 通过 / %d 失败" % (passed, failed))
    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
