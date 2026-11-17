"""Create a local, privacy-redacted real-inference example pending image rights."""
from pathlib import Path
import os
import sys
import json
import hashlib
sys.dont_write_bytecode=True
os.environ['YOLO_AUTOINSTALL']='False'
from PIL import Image,ImageDraw,ImageFont
from ultralytics import YOLO

ROOT=Path(__file__).resolve().parents[2]
WORK=ROOT/'portfolio-export/_work/single-frame-example'
WORK.mkdir(parents=True,exist_ok=True)
SOURCE=ROOT/'data/cctv-person/yolo/images/test/01-08-2022__03-09-56PM_jpg.jpg'
WEIGHTS=ROOT/'weights/best.pt'
assert SOURCE.is_file() and WEIGHTS.is_file()
model=YOLO(str(WEIGHTS))
result=model.predict(str(SOURCE),imgsz=640,conf=0.25,classes=[0],device='cpu',save=False,verbose=False)[0]
boxes=result.boxes.xyxy.cpu().numpy().tolist()
scores=result.boxes.conf.cpu().numpy().tolist()
im=Image.open(SOURCE).convert('RGB')
draw=ImageDraw.Draw(im)
# Fixed opaque rectangles, not generative edits. Coordinates refer to the original frame.
redactions=[
    {'kind':'head','box':[275,80,341,153]},
    {'kind':'head','box':[282,269,338,352]},
    {'kind':'head','box':[382,411,452,511]},
    {'kind':'head','box':[481,398,551,513]},
    {'kind':'camera-timestamp','box':[410,0,640,54]},
    {'kind':'camera-label','box':[28,567,169,629]},
]
for item in redactions:draw.rectangle(item['box'],fill='#171C26')
font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',16)
bold=ImageFont.truetype('C:/Windows/Fonts/arialbd.ttf',20)
blue='#1249EE'
for box,score in zip(boxes,scores):
    x1,y1,x2,y2=[int(round(v)) for v in box]
    draw.rectangle((x1,y1,x2,y2),outline=blue,width=3)
    label=f'person {score:.2f}'
    text_box=draw.textbbox((0,0),label,font=font)
    width=text_box[2]-text_box[0]+10
    lx=min(max(x1,2),im.width-width-2)
    ly=max(2,y1-25)
    draw.rectangle((lx,ly,lx+width,ly+23),fill=blue)
    draw.text((lx+5,ly+2),label,font=font,fill='white')
canvas=Image.new('RGB',(im.width,im.height+74),'#111C2B')
canvas.paste(im,(0,74))
top=ImageDraw.Draw(canvas)
top.text((18,10),'YOLO26s · single-frame inference',font=bold,fill='white')
top.text((18,42),f'Person detections: {len(boxes)}   |   conf ≥ 0.25   |   heads / camera text redacted',font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',14),fill='#D9E4F2')
path=WORK/'07-single-frame-person-detection.webp'
canvas.save(path,'WEBP',lossless=True,method=6,exact=True)
preview=canvas.copy();preview.thumbnail((800,800),Image.Resampling.LANCZOS)
preview_path=WORK/'07-single-frame-person-detection-preview.webp'
preview.save(preview_path,'WEBP',lossless=True,method=6,exact=True)
record={
 'status':'local-preview-awaiting-publication-rights',
 'model':'YOLO26s default fine-tune',
 'modelWeightsSha256':hashlib.sha256(WEIGHTS.read_bytes()).hexdigest(),
 'sourceFrameSha256':hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
 'sourceSplit':'test','imgsz':640,'confidenceThreshold':0.25,'device':'cpu',
 'newInferenceRun':True,'newTrainingRun':False,'detectedPeople':len(boxes),
 'boxesXyxy':boxes,'confidences':scores,
 'rendering':'Blue boxes/confidence labels from actual model predictions; fixed opaque privacy rectangles applied after inference.',
 'scope':'One selected dataset frame; not a tracking, unique-visitor, entry/exit, or general accuracy measurement.',
 'redactions':redactions,
 'width':canvas.width,'height':canvas.height,
 'previewWidth':preview.width,'previewHeight':preview.height,
 'datasetLicenseObservation':'README.dataset.txt says License: undefined; no permission has yet been established for the public ZIP.'
}
(WORK/'example-evidence.local.json').write_text(json.dumps(record,indent=2)+'\n',encoding='utf8')
(WORK/'example.en.md').write_text(f'''# Single-frame person detection example

![YOLO26s person detections with privacy redactions](07-single-frame-person-detection.webp)

Actual local inference with the default fine-tuned YOLO26s model detected **{len(boxes)} people** in one selected dataset frame, at image size 640 and confidence threshold 0.25 on CPU. Blue labels show the returned confidence values. Opaque rectangles hide heads and camera text after inference. The frame was not enlarged.

This is a detector-use example. One still image does not demonstrate identity tracking, unique visitors, or entry/exit counting. It is not an accuracy benchmark or a production-camera measurement.

Publication status: local preview awaiting permission for this dataset frame. The existing public ZIP has not been changed to include it.
''',encoding='utf8')
(WORK/'example.id.md').write_text(f'''# Contoh deteksi orang pada satu frame

![Deteksi orang YOLO26s dengan redaksi privasi](07-single-frame-person-detection.webp)

Inferensi lokal sebenarnya dengan model YOLO26s fine-tuned default mendeteksi **{len(boxes)} orang** pada satu frame dataset terpilih, dengan ukuran gambar 640 dan confidence threshold 0,25 pada CPU. Label biru menunjukkan confidence yang dikembalikan model. Kotak polos menutup kepala dan teks kamera setelah inferensi. Frame tidak diperbesar.

Ini merupakan contoh penggunaan detektor. Satu gambar diam tidak membuktikan tracking identitas, pengunjung unik, atau hitungan masuk/keluar. Contoh ini bukan benchmark akurasi atau pengukuran kamera produksi.

Status publikasi: preview lokal menunggu izin untuk frame dataset ini. ZIP publik yang tersedia belum diubah untuk memasukkannya.
''',encoding='utf8')
print(json.dumps({'preview':str(path),'dimensions':canvas.size,'detections':len(boxes),'confidenceThreshold':0.25,'newInferenceRun':True,'publication':'awaiting-rights-confirmation'},ensure_ascii=True))
