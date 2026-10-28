#!/usr/bin/env python3
"""生成阶段三本地测试图片（全部为程序现场绘制的合成图，绝非用户真实照片）。

阶段三相对阶段二的改动：
  - 裁切版 food_cropped 不再 resize 放大回 800x600，裁切后保持
    768x576 较小分辨率（P1-1：母本靠像素总面积胜出）。
  - 新增 0 秒高速连拍组 fastburst_001..004：四张共用同一个 EXIF
    时间戳（模拟真机 1 秒多张时秒级 EXIF 时间差为 0 秒），序列名、
    同分辨率、彩色，用于验证 P0-1 新连拍规则能检出真连拍。
  - 0 秒修图变体验证用现成的食物组：base_food 与 4 张变体共用
    同一时间戳，但变体文件名含 bright/compressed/cropped/filter
    后缀，按 P0-1 规则绝不能误判为连拍。

用法: python3 gen_test_images.py
产物: demo_images/*.jpg|*.png + faces_manifest.json（测试分组专用 sidecar，
      仅用于在真实人脸检测不可行时演练分组逻辑，绝不冒充真实人脸识别）。

图片清单（共 25 张）:
  带 EXIF: base_park / base_beach / base_food / base_city（4 张，不同日期）
  精确重复: park_copy1 / park_copy2（与 base_park 字节相同）、beach_copy（与 base_beach 字节相同）
  疑似重复: food_compressed（低画质）/ food_bright（调亮）/ food_cropped（轻微裁切，
             768x576，不放大回原尺寸）/ food_filter（加滤镜），均以 base_food 为母本
  连拍组: burst_001..004（同一场景小球微移，EXIF 时间间隔 2 秒）
  0 秒高速连拍组: fastburst_001..004（同一场景小球微移，共用同一 EXIF 时间戳）
  截图: screenshot_chat / screenshot_map（高瘦 UI 样式图，无 EXIF）
  无 EXIF 老照片: old_photo_1（棕褐）/ old_photo_2（黑白）
  人像: portrait_a2（测试人物 A）、portrait_b2（测试人物 B）
人脸测试分组: base_park 与 portrait_a2 属测试人物 A；
              base_beach 与 portrait_b2 属测试人物 B。
"""
import json
import os
import shutil
from PIL import Image, ImageDraw, ImageEnhance, ImageOps

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "demo_images")
W, H = 800, 600


def vgrad(img, top, bottom):
    d = ImageDraw.Draw(img)
    for y in range(H):
        t = y / max(H - 1, 1)
        c = tuple(int(top[i] + (bottom[i] - top[i]) * t) for i in range(3))
        d.line([(0, y), (W, y)], fill=c)


def draw_face(d, cx, cy, r, skin, hair):
    d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=skin)
    d.pieslice([cx - r, cy - r, cx + r, cy + r], start=180, end=360, fill=hair)
    er = max(4, r // 9)
    for dx in (-r // 3, r // 3):
        d.ellipse([cx + dx - er, cy - er, cx + dx + er, cy + er], fill=(30, 30, 30))
    d.arc([cx - r // 2, cy, cx + r // 2, cy + r // 2], start=0, end=180,
          fill=(120, 60, 60), width=3)


def save_jpg(img, name, exif_time=None, quality=92):
    path = os.path.join(OUT, name)
    if exif_time:
        exif = Image.Exif()
        exif[36867] = exif_time  # DateTimeOriginal
        exif[36868] = exif_time  # DateTimeDigitized
        exif[306] = exif_time    # DateTime
        img.save(path, "JPEG", quality=quality, exif=exif)
    else:
        img.save(path, "JPEG", quality=quality)
    return path


def scene_park():
    img = Image.new("RGB", (W, H))
    vgrad(img, (150, 200, 240), (200, 230, 180))
    d = ImageDraw.Draw(img)
    d.ellipse([620, 60, 700, 140], fill=(255, 220, 90))
    d.rectangle([0, 420, W, H], fill=(90, 160, 80))
    d.rectangle([380, 250, 410, 430], fill=(120, 80, 40))
    d.ellipse([300, 150, 490, 330], fill=(60, 140, 70))
    draw_face(d, 180, 300, 70, (255, 205, 170), (40, 40, 40))
    d.text((30, 30), "PARK", fill=(255, 255, 255))
    return img


def scene_beach():
    img = Image.new("RGB", (W, H))
    vgrad(img, (140, 200, 245), (255, 235, 170))
    d = ImageDraw.Draw(img)
    d.ellipse([640, 50, 720, 130], fill=(255, 220, 90))
    d.rectangle([0, 380, W, 470], fill=(60, 140, 200))
    d.rectangle([0, 470, W, H], fill=(240, 210, 140))
    draw_face(d, 200, 250, 70, (255, 220, 190), (110, 70, 40))
    d.text((30, 30), "BEACH", fill=(255, 255, 255))
    return img


def scene_food():
    img = Image.new("RGB", (W, H), (235, 200, 170))
    d = ImageDraw.Draw(img)
    d.rectangle([0, 400, W, H], fill=(170, 110, 60))
    d.ellipse([200, 120, 600, 480], fill=(255, 255, 255))
    d.ellipse([280, 190, 520, 410], fill=(220, 120, 60))
    d.ellipse([330, 230, 470, 370], fill=(255, 180, 90))
    d.text((30, 30), "FOOD", fill=(90, 50, 30))
    return img


def scene_city():
    img = Image.new("RGB", (W, H))
    vgrad(img, (120, 160, 210), (220, 220, 220))
    d = ImageDraw.Draw(img)
    for i, x in enumerate([60, 220, 380, 540]):
        h = 300 + (i % 3) * 60
        d.rectangle([x, 560 - h, x + 120, 560], fill=(90, 100, 120))
        for wy in range(560 - h + 20, 540, 50):
            for wx in range(x + 15, x + 105, 35):
                d.rectangle([wx, wy, wx + 18, wy + 22], fill=(255, 230, 140))
    d.text((30, 30), "CITY", fill=(255, 255, 255))
    return img


def scene_burst(frame):
    img = Image.new("RGB", (W, H))
    vgrad(img, (150, 200, 240), (200, 230, 180))
    d = ImageDraw.Draw(img)
    d.rectangle([0, 420, W, H], fill=(90, 160, 80))
    d.rectangle([380, 250, 410, 430], fill=(120, 80, 40))
    d.ellipse([300, 150, 490, 330], fill=(60, 140, 70))
    x = 200 + frame * 80  # 小球逐帧右移，其余场景不动
    d.ellipse([x, 350, x + 60, 410], fill=(220, 60, 60))
    d.text((30, 30), "BURST %d" % frame, fill=(255, 255, 255))
    return img


def scene_fastburst(frame):
    # 0 秒高速连拍：与 burst 组场景相近但小球步进更大、文案不同，
    # 关键是四帧共用同一 EXIF 时间戳（见 main）。
    img = Image.new("RGB", (W, H))
    vgrad(img, (150, 200, 240), (200, 230, 180))
    d = ImageDraw.Draw(img)
    d.rectangle([0, 420, W, H], fill=(90, 160, 80))
    d.rectangle([380, 250, 410, 430], fill=(120, 80, 40))
    d.ellipse([300, 150, 490, 330], fill=(60, 140, 70))
    x = 180 + frame * 50
    d.ellipse([x, 350, x + 60, 410], fill=(220, 60, 60))
    d.text((30, 30), "FASTBURST %d" % frame, fill=(255, 255, 255))
    return img


def screenshot_chat():
    img = Image.new("RGB", (540, 960), (255, 255, 255))
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, 540, 44], fill=(240, 240, 240))
    d.rectangle([0, 44, 540, 120], fill=(250, 250, 250))
    d.text((20, 70), "CHAT", fill=(30, 30, 30))
    y = 160
    for i in range(6):
        if i % 2 == 0:
            d.rounded_rectangle([30, y, 330, y + 70], radius=16, fill=(230, 235, 240))
        else:
            d.rounded_rectangle([210, y, 510, y + 70], radius=16, fill=(180, 220, 150))
        y += 110
    d.rectangle([0, 880, 540, 960], fill=(245, 245, 245))
    return img


def screenshot_map():
    img = Image.new("RGB", (540, 960), (255, 255, 255))
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, 540, 44], fill=(240, 240, 240))
    d.rectangle([0, 44, 540, 700], fill=(190, 220, 180))
    d.line([(0, 300), (540, 380)], fill=(255, 255, 255), width=14)
    d.line([(270, 44), (270, 700)], fill=(255, 255, 255), width=14)
    d.ellipse([230, 320, 310, 400], fill=(220, 60, 60))
    d.text((20, 730), "MAP", fill=(30, 30, 30))
    d.rounded_rectangle([30, 800, 510, 920], radius=18, fill=(245, 245, 245))
    return img


def portrait(person):
    if person == "A":
        img = Image.new("RGB", (W, H), (200, 220, 240))
        d = ImageDraw.Draw(img)
        for x in range(0, W, 80):
            d.rectangle([x, 0, x + 30, H], fill=(180, 205, 235))
        draw_face(d, 300, 280, 110, (255, 205, 170), (40, 40, 40))
        d.text((30, 30), "PORTRAIT A", fill=(30, 30, 30))
    else:
        img = Image.new("RGB", (W, H), (240, 210, 200))
        d = ImageDraw.Draw(img)
        for cx, cy in [(120, 120), (680, 150), (120, 500), (680, 480)]:
            d.ellipse([cx - 60, cy - 60, cx + 60, cy + 60],
                      outline=(210, 170, 150), width=6)
        draw_face(d, 520, 300, 85, (255, 220, 190), (110, 70, 40))
        d.text((30, 30), "PORTRAIT B", fill=(30, 30, 30))
    return img


def main():
    os.makedirs(OUT, exist_ok=True)
    for f in os.listdir(OUT):
        os.remove(os.path.join(OUT, f))
    # 1) 带 EXIF 的照片
    park = save_jpg(scene_park(), "base_park.jpg", "2024:05:01 10:00:00")
    beach = save_jpg(scene_beach(), "base_beach.jpg", "2024:06:15 14:30:00")
    food = scene_food()
    save_jpg(food, "base_food.jpg", "2025:01:20 12:00:00")
    save_jpg(scene_city(), "base_city.jpg", "2025:03:10 18:45:00")
    # 2) 字节级完全重复（直接复制文件字节）
    shutil.copyfile(park, os.path.join(OUT, "park_copy1.jpg"))
    shutil.copyfile(park, os.path.join(OUT, "park_copy2.jpg"))
    shutil.copyfile(beach, os.path.join(OUT, "beach_copy.jpg"))
    # 3) 疑似重复：同一母本的不同轻微改动（转存画质 90，低于母本 92，
    #    符合“变体是转存副本”的真实场景，也让母本保持体积最大）
    save_jpg(food, "food_compressed.jpg", "2025:01:20 12:00:00", quality=20)
    save_jpg(ImageEnhance.Brightness(food).enhance(1.20),
             "food_bright.jpg", "2025:01:20 12:00:00", quality=90)
    # P1-1：裁切后保持 768x576，不再 resize 放大回 800x600。
    cropped = food.crop((16, 12, 784, 588))
    save_jpg(cropped, "food_cropped.jpg", "2025:01:20 12:00:00", quality=90)
    warm = ImageEnhance.Color(food).enhance(1.6)
    warm = Image.blend(warm, Image.new("RGB", (W, H), (255, 150, 60)), 0.15)
    save_jpg(warm, "food_filter.jpg", "2025:01:20 12:00:00", quality=90)
    # 4) 连拍组：EXIF 时间间隔 2 秒
    for i in range(1, 5):
        save_jpg(scene_burst(i), "burst_%03d.jpg" % i,
                 "2025:04:05 09:00:%02d" % ((i - 1) * 2))
    # 4b) 0 秒高速连拍组：四帧共用同一 EXIF 时间戳
    for i in range(1, 5):
        save_jpg(scene_fastburst(i), "fastburst_%03d.jpg" % i,
                 "2025:04:06 10:00:00")
    # 5) 截图样式图（无 EXIF）
    screenshot_chat().save(os.path.join(OUT, "screenshot_chat.png"))
    screenshot_map().save(os.path.join(OUT, "screenshot_map.png"))
    # 6) 无 EXIF 老照片
    old = scene_park()
    save_jpg(ImageOps.colorize(old.convert("L"), black="#1a1208",
                               white="#ffe8c0"), "old_photo_1.jpg")
    save_jpg(scene_beach().convert("L").convert("RGB"), "old_photo_2.jpg")
    # 7) 人像（用于测试分组）
    save_jpg(portrait("A"), "portrait_a2.jpg", "2025:05:01 11:00:00")
    save_jpg(portrait("B"), "portrait_b2.jpg", "2025:05:02 11:00:00")
    # 测试分组 sidecar（测试专用，绝不冒充真实人脸识别）
    manifest = {
        "note": "测试分组专用：生成时已知的人像身份，仅用于演练分组逻辑，"
                "不是真实人脸识别结果。",
        "groups": {
            "test_person_a": {"label": "测试人物 A",
                              "members": ["base_park.jpg", "portrait_a2.jpg"]},
            "test_person_b": {"label": "测试人物 B",
                              "members": ["base_beach.jpg", "portrait_b2.jpg"]},
        },
    }
    with open(os.path.join(HERE, "faces_manifest.json"), "w",
              encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    print("generated:", sorted(os.listdir(OUT)))


if __name__ == "__main__":
    main()
