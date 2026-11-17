"""Include one specifically user-directed, face-redacted real-model example."""
from pathlib import Path
import sys
import json
import hashlib
import shutil
sys.dont_write_bytecode=True
from PIL import Image,ImageDraw,ImageFont
from build_export import ROOT, DEST, WORK, DATE, pair, sha, write_json, write_text

cache=json.loads((WORK/'single-frame-example/example-evidence.local.json').read_text(encoding='utf8'))
source=ROOT/'data/cctv-person/yolo/images/test/01-08-2022__03-09-56PM_jpg.jpg'
assert sha(source.read_bytes())==cache['sourceFrameSha256']
assert sha((ROOT/'weights/best.pt').read_bytes())==cache['modelWeightsSha256']
boxes,scores=cache['boxesXyxy'],cache['confidences']
assert len(boxes)==len(scores)==4
im=Image.open(source).convert('RGB')
draw=ImageDraw.Draw(im)
redactions=[{'kind':'visible-facial-profile','box':[302,105,342,153]},
            {'kind':'camera-timestamp','box':[410,0,640,54]},
            {'kind':'camera-label','box':[28,567,169,629]}]
for item in redactions:draw.rectangle(item['box'],fill='#171C26')
font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',16)
for box,score in zip(boxes,scores):
    x1,y1,x2,y2=[int(round(v)) for v in box]
    draw.rectangle((x1,y1,x2,y2),outline='#1249EE',width=3)
    label=f'person {score:.2f}';bbox=draw.textbbox((0,0),label,font=font)
    width=bbox[2]-bbox[0]+10;lx=min(max(x1,2),im.width-width-2);ly=max(2,y1-25)
    draw.rectangle((lx,ly,lx+width,ly+23),fill='#1249EE')
    draw.text((lx+5,ly+2),label,font=font,fill='white')
canvas=Image.new('RGB',(640,714),'#111C2B');canvas.paste(im,(0,74))
draw=ImageDraw.Draw(canvas)
draw.text((18,10),'YOLO26s · single-frame inference',font=ImageFont.truetype('C:/Windows/Fonts/arialbd.ttf',20),fill='white')
draw.text((18,42),'Person detections: 4   |   conf ≥ 0.25   |   visible face / camera text redacted',
          font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',14),fill='#D9E4F2')
main='assets/07-single-frame-person-detection.webp';preview='assets/07-single-frame-person-detection-preview.webp'
canvas.save(DEST/main,'WEBP',lossless=True,method=6,exact=True)
thumb=canvas.copy();thumb.thumbnail((480,600),Image.Resampling.LANCZOS)
thumb.save(DEST/preview,'WEBP',lossless=True,method=6,exact=True)
def file_info(rel):
    path=DEST/rel
    with Image.open(path) as img: dimensions={'width':img.width,'height':img.height}
    data=path.read_bytes()
    return {'path':rel,**dimensions,'bytes':len(data),'sha256':sha(data)}
caption=pair('Actual local YOLO26s inference returned 4 person detections in one selected dataset frame at image size 640 and confidence threshold 0.25 on CPU. Blue labels are returned confidence scores. One visible facial profile and camera text are redacted; back-of-head views are retained. This still image does not demonstrate tracking, unique visitors, or entry/exit accuracy.',
             'Inferensi YOLO26s lokal sebenarnya menghasilkan 4 deteksi orang pada satu frame dataset terpilih dengan ukuran gambar 640 dan confidence threshold 0,25 di CPU. Label biru adalah skor confidence yang dikembalikan model. Satu profil wajah yang terlihat dan teks kamera ditutup; tampilan belakang kepala dipertahankan. Gambar diam ini tidak membuktikan tracking, pengunjung unik, atau akurasi masuk/keluar.')
alt=pair('One CCTV frame with four blue person detection boxes and confidence labels; a visible facial profile and camera text are covered while three back-of-head views remain visible.',
         'Satu frame CCTV dengan empat kotak deteksi orang biru dan label confidence; satu profil wajah yang terlihat dan teks kamera ditutup sementara tiga tampilan belakang kepala tetap terlihat.')
asset={'id':'asset-07',**file_info(main),'preview':file_info(preview),'visualType':'observed-model-inference-frame',
       'order':7,'fit':'contain','detailFit':'contain','cardAspectRatio':'8:5','imageTextLanguage':'en',
       'alt':alt,'caption':caption,'recommendedForInitialWebsite':True,'coverReason':None,
       'provenance':{'kind':'observed-single-frame-inference','modelWeightsSha256':cache['modelWeightsSha256'],
                     'sourceFrameSha256':cache['sourceFrameSha256'],'sourceSplit':'test',
                     'evidencePath':'technical/person-counter-example-evidence.json','newTimingBenchmark':False},
       'publication':{'userExplicitlyRequestedInPublicZip':True,'instructionDate':DATE,'upstreamDatasetLicense':'undefined',
                      'upstreamRightsIndependentlyVerified':False,
                      'scope':pair('Only this privacy-redacted example frame is included at the user\'s explicit direction; no general dataset or model redistribution clearance is inferred.',
                                   'Hanya frame contoh dengan redaksi privasi ini yang disertakan atas instruksi eksplisit pengguna; tidak disimpulkan adanya izin distribusi umum untuk dataset atau model.')},
       'review':{'status':'pending-final-visual-review','metadataRemoved':True,'containsPeople':True,
                 'containsQR':False,'containsPersonalData':False,'visibleFacialProfilesRedacted':1,
                 'backOfHeadViewsRetained':3,'aiGenerated':False,'distributionBasis':'specific-user-directed-frame-inclusion-upstream-license-undefined'}}
manifest=json.loads((DEST/'assets-manifest.json').read_text(encoding='utf8'))
assert len(manifest['assets'])==6
manifest['assets'].append(asset)
manifest['recommendedWebsiteAssets']=['asset-01','asset-07','asset-02','asset-05']
manifest['curationNote']=pair('Use the pipeline cover, single-frame model example, domain/density results, and backend benchmark initially. Keep all captions and use contain for the frame and charts.',
                             'Gunakan sampul pipeline, contoh model satu frame, hasil domain/kepadatan, dan benchmark backend terlebih dahulu. Pertahankan semua caption dan gunakan contain untuk frame serta grafik.')
write_json('assets-manifest.json',manifest)
evidence={'model':cache['model'],'modelWeightsSha256':cache['modelWeightsSha256'],
          'sourceFrameSha256':cache['sourceFrameSha256'],'sourceSplit':'test','imgsz':640,'confidenceThreshold':0.25,
          'device':'cpu','detectedPeople':4,'boxesXyxy':boxes,'confidences':scores,'newInferenceRunPerformed':True,
          'trainingRunPerformed':False,'completeVideoRunPerformed':False,
          'renderingReusesObservedPredictions':True,'privacyRectanglesAppliedAfterInference':True,'redactions':redactions,
          'frameWidth':640,'frameHeight':640,'renderedWidth':640,'renderedHeight':714,
          'imageUpscaled':False,'visibleFacialProfilesRedacted':1,'backOfHeadViewsRetained':3,
          'scope':pair('One selected detector-use example; not an accuracy benchmark, live-camera measurement, or tracking/footfall evaluation.',
                       'Satu contoh penggunaan detektor terpilih; bukan benchmark akurasi, pengukuran kamera langsung, atau evaluasi tracking/footfall.'),
          'publication':asset['publication']}
write_json('technical/person-counter-example-evidence.json',evidence)
code='''import os
os.environ["YOLO_AUTOINSTALL"] = "False"
from ultralytics import YOLO

model = YOLO("weights/best.pt")
result = model.predict(
    source="<LOCAL_IMAGE_FILE>",
    imgsz=640,
    conf=0.25,
    classes=[0],
    device="cpu",
    save=False,
    verbose=False,
)[0]
print(f"Person detections: {len(result.boxes)}")'''
write_text('technical/person-counter-example.md',f'''# Single-frame person counter example / Contoh person counter satu frame

![{alt['en']} / {alt['id']}](../{main})

## English

{caption['en']}

The example uses the default fine-tuned person detector. Its four displayed scores round to 0.89, 0.86, 0.82, and 0.81. The original 640 × 640 frame is not enlarged; a 74-pixel documentation header is added above it. Opaque privacy rectangles are applied after inference, so the boxes and scores remain the actual returned predictions. Heads viewed from behind are not covered.

Run the following from a working copy with authorized local weights and an image you are entitled to process. Replace the explicit placeholder. The ZIP does not bundle model weights. A different image may produce a different count.

```python
{code}
```

This is detection on one still frame. `count_people.py` provides the separate video-tracking and line-crossing workflow; this illustration does not prove its accuracy.

Publication provenance: the user explicitly requested this redacted frame in the public ZIP. The upstream export still states `License: undefined`; that observation is recorded without claiming a verified upstream license or granting rights to the rest of the dataset.

## Bahasa Indonesia

{caption['id']}

Contoh memakai detektor person fine-tuned default. Empat skor yang ditampilkan dibulatkan menjadi 0,89, 0,86, 0,82, dan 0,81. Frame asli 640 × 640 tidak diperbesar; header dokumentasi 74 piksel ditambahkan di atasnya. Kotak privasi polos diterapkan setelah inferensi, sehingga kotak dan skor tetap merupakan prediksi yang sebenarnya. Kepala yang terlihat dari belakang tidak ditutup.

Jalankan kode di atas dari salinan kerja dengan bobot lokal berizin dan gambar yang berhak Anda proses. Ganti placeholder eksplisit. ZIP tidak membundel bobot model. Gambar berbeda dapat menghasilkan hitungan berbeda.

Ini merupakan deteksi pada satu frame diam. `count_people.py` menyediakan alur terpisah untuk tracking video dan lintasan garis; ilustrasi ini tidak membuktikan akurasinya.

Asal instruksi publikasi: pengguna secara eksplisit meminta frame dengan redaksi ini di ZIP publik. Ekspor upstream masih menyatakan `License: undefined`; pengamatan tersebut dicatat tanpa mengklaim lisensi upstream yang terverifikasi atau memberikan hak atas seluruh dataset.

Evidence / Bukti: [person-counter-example-evidence.json](person-counter-example-evidence.json).
''')
project=json.loads((DEST/'project.json').read_text(encoding='utf8'))
project['recommendedWebsiteAssets']=manifest['recommendedWebsiteAssets']
project['proof'].insert(0,{'type':'observed','title':pair('Single-frame model-use example','Contoh penggunaan model satu frame'),
                         'text':caption,'evidenceIds':['E20'],'assetIds':['asset-07']})
project['sources'].append({'id':'single-frame-example','label':pair('Single-frame inference example and usage','Contoh inferensi satu frame dan penggunaan'),
                          'packagePath':'technical/person-counter-example.md','type':'evidence'})
for item in project['limitations']:
    if 'Dataset redistribution rights are unverified.' in item['text']['en']:
        item['text']=pair('Full dataset and model redistribution rights are unverified. Only one privacy-redacted example frame is included at the user\'s explicit direction. Training data, annotations, prediction caches, raw video, and model binaries remain excluded, so the ZIP alone cannot reproduce training or inference.',
                          'Hak distribusi dataset lengkap dan model belum terverifikasi. Hanya satu frame contoh dengan redaksi privasi yang disertakan atas instruksi eksplisit pengguna. Data training, anotasi, cache prediksi, video mentah, dan binary model tetap dikeluarkan, sehingga ZIP saja tidak dapat mereproduksi training atau inferensi.')
project['licenseStatus']['datasetAndModelRedistribution']='unverified-full-dataset-and-models-excluded-one-user-directed-redacted-frame'
project['licenseStatus']['singleFrameInclusion']=asset['publication']
project['snapshot']['exportRevision']=pair('The untracked portfolio output now also contains one explicitly user-directed, redacted model-use frame; original tracked repository files remain unchanged.',
                                         'Keluaran portfolio yang tidak terlacak kini juga memuat satu frame penggunaan model dengan redaksi atas instruksi eksplisit pengguna; berkas repository asli yang terlacak tetap tidak berubah.')
write_json('project.json',project)
aggregate=json.loads((DEST/'technical/aggregate-evidence.json').read_text(encoding='utf8'))
aggregate['verification']['newSingleFrameInferenceRun']=True
aggregate['verification']['newModelInferenceRun']=True
aggregate['singleFrameExample']={'evidencePath':'technical/person-counter-example-evidence.json','detectedPeople':4,
                                 'imgsz':640,'confidenceThreshold':0.25,'scope':'selected-single-image-not-a-benchmark'}
write_json('technical/aggregate-evidence.json',aggregate)

def change(rel,old,new,required=True):
    path=DEST/rel;text=path.read_text(encoding='utf8')
    if required:assert old in text,(rel,old[:85])
    path.write_text(text.replace(old,new),encoding='utf8')

for lang in ['en','id']:
    rel=f'case-study.{lang}.md'
    marker='## 9. Evidence and results\n' if lang=='en' else '## 9. Bukti dan hasil\n'
    block=f'\n![{alt[lang]}]({main})\n\n{caption[lang]}\n\n'+('[Usage and inference evidence](technical/person-counter-example.md).\n' if lang=='en' else '[Cara penggunaan dan bukti inferensi](technical/person-counter-example.md).\n')
    change(rel,marker,marker+block)
    old='No new training, model inference, or complete video-counting run was performed for this portfolio export. Source checks and cached-output verification are recorded separately in the evidence ledger.' if lang=='en' else 'Tidak ada training, inferensi model, atau run counting video lengkap yang baru untuk ekspor portfolio ini. Pemeriksaan source dan verifikasi keluaran tersimpan dicatat terpisah dalam daftar bukti.'
    new='One additional local detector inference was performed for the selected single-frame illustration. No new training or complete video-counting run was performed; the historical mAP and benchmark figures were not regenerated.' if lang=='en' else 'Satu inferensi detektor lokal tambahan dijalankan untuk ilustrasi satu frame terpilih. Tidak ada training baru atau run counting video lengkap; angka mAP dan benchmark historis tidak dihasilkan ulang.'
    change(rel,old,new)
    old='Dataset redistribution rights are unverified. This handoff excludes source imagery, video, annotations, predictions, and all model binaries, so training and inference cannot be reproduced from the ZIP alone.' if lang=='en' else 'Hak distribusi dataset belum terverifikasi. Handoff ini mengeluarkan gambar sumber, video, anotasi, prediksi, dan seluruh binary model, sehingga training dan inferensi tidak dapat direproduksi hanya dari ZIP.'
    new=next(x['text'][lang] for x in project['limitations'] if 'E19' in x['evidenceIds'])
    change(rel,old,new)

change('README.md','Start with **asset-01** (pipeline cover), **asset-02** (domain and density results), and **asset-05** (local runtime comparison).',
       'Start with **asset-01** (pipeline cover), **asset-07** (one actual model-use frame), **asset-02** (domain and density results), and **asset-05** (local runtime comparison).')
change('README.md','Mulai dari **asset-01** (sampul pipeline), **asset-02** (hasil menurut domain dan kepadatan), serta **asset-05** (perbandingan runtime lokal).',
       'Mulai dari **asset-01** (sampul pipeline), **asset-07** (satu frame penggunaan model sebenarnya), **asset-02** (hasil menurut domain dan kepadatan), serta **asset-05** (perbandingan runtime lokal).')
change('README.md','No source imagery, identifiable people, QR codes, or recording frames are included.',
       'One selected model-use frame is included at the user\'s explicit direction, with its visible facial profile and camera text redacted. Other source imagery and recordings are excluded. The upstream dataset license remains undefined; no broader redistribution clearance is inferred.')
change('README.md','Tidak ada gambar sumber, orang yang dapat dikenali, QR, atau frame rekaman di dalam paket.',
       'Satu frame penggunaan model terpilih disertakan atas instruksi eksplisit pengguna, dengan profil wajah yang terlihat dan teks kamera ditutup. Gambar sumber lain dan rekaman dikeluarkan. Lisensi dataset upstream tetap belum ditentukan; tidak disimpulkan adanya izin distribusi yang lebih luas.')
readme=DEST/'README.md'
readme.write_text(readme.read_text(encoding='utf8')+'\n## Model-use example / Contoh penggunaan model\n\nThe seven-visual gallery includes one actual single-frame inference result; use `contain` to preserve the complete frame. [View the example and code](technical/person-counter-example.md).\n\nGaleri tujuh visual mencakup satu hasil inferensi satu frame yang sebenarnya; gunakan `contain` agar frame tetap utuh. [Lihat contoh dan kode](technical/person-counter-example.md).\n',encoding='utf8')

change('technical/third-party-notices.md','Accordingly all pretrained/trained weights, compiled graphs, annotations, imagery, and recordings are excluded, even where some are already committed publicly.',
       'Pretrained/trained weights, compiled graphs, annotations, raw imagery, and recordings remain excluded. One redacted derivative frame is now included at the user\'s explicit direction; this instruction is recorded separately from the still-undefined upstream license.')
change('technical/third-party-notices.md','no source photo, identifiable person, camera frame, or per-image label file is included.',
       'the full dataset, raw recordings, and per-image label files are not included. A separate single-frame example shows actual model output; its visible facial profile and camera text are redacted, while back-of-head views remain visible.')
change('technical/third-party-notices.md','Karena itu, seluruh bobot pretrained/terlatih, graph kompilasi, anotasi, gambar, dan rekaman dikeluarkan, termasuk yang sudah tersimpan publik.',
       'Bobot pretrained/terlatih, graph kompilasi, anotasi, gambar mentah, dan rekaman tetap dikeluarkan. Satu frame turunan dengan redaksi kini disertakan atas instruksi eksplisit pengguna; instruksi ini dicatat terpisah dari lisensi upstream yang tetap belum ditentukan.')
change('technical/third-party-notices.md','tidak ada foto sumber, orang yang dapat dikenali, frame kamera, atau berkas label per gambar.',
       'dataset lengkap, rekaman mentah, dan berkas label per gambar tidak disertakan. Contoh satu frame terpisah menunjukkan keluaran model sebenarnya; profil wajah yang terlihat dan teks kamera ditutup, sementara tampilan belakang kepala tetap terlihat.')

for rel in ['technical/development.md','technical/api-and-data.md']:
    path=DEST/rel
    path.write_text(path.read_text(encoding='utf8')+'\n## Additional observed example / Contoh tambahan yang diamati\n\nOne local single-image detector inference is documented in [the model-use example](person-counter-example.md). It returned four detections at confidence 0.25 and image size 640. This does not rerun the historical evaluation or measure video tracking. Only its redacted illustration is included; the full data/model bundle remains absent.\n\nSatu inferensi detektor lokal pada gambar tunggal didokumentasikan pada [contoh penggunaan model](person-counter-example.md). Hasilnya empat deteksi pada confidence 0,25 dan ukuran gambar 640. Ini tidak menjalankan ulang evaluasi historis atau mengukur tracking video. Hanya ilustrasi dengan redaksinya yang disertakan; bundel data/model lengkap tetap tidak tersedia.\n',encoding='utf8')

change('internal/missing-information.md','No desktop/mobile application screenshot is supplied; the gallery uses project charts and a clearly labeled code-derived diagram.',
       'No desktop/mobile application UI screenshot is supplied; the gallery now includes one actual single-frame detector result, project charts, and a clearly labeled code-derived diagram.')
change('internal/missing-information.md','Screenshot aplikasi desktop/mobile tidak disediakan; galeri menggunakan grafik proyek dan diagram berdasarkan kode yang diberi label jelas.',
       'Screenshot UI aplikasi desktop/mobile tidak disediakan; galeri kini mencakup satu hasil detektor satu frame sebenarnya, grafik proyek, dan diagram berdasarkan kode yang diberi label jelas.')
path=DEST/'internal/missing-information.md'
path.write_text(path.read_text(encoding='utf8')+'\n## Single-frame publication instruction / Instruksi publikasi satu frame\n\nThe latest user instruction explicitly requests the one redacted example frame in the public ZIP and asks that back-of-head views remain uncovered. This supersedes the earlier local-only choice for this one derivative. The upstream export still states `License: undefined`; no independent upstream permission or general dataset/model license has been established.\n\nInstruksi pengguna terbaru secara eksplisit meminta satu frame contoh dengan redaksi di ZIP publik dan meminta tampilan belakang kepala tetap tidak ditutup. Instruksi ini menggantikan pilihan lokal sebelumnya untuk satu turunan ini. Ekspor upstream tetap menyatakan `License: undefined`; belum ditetapkan izin upstream independen atau lisensi umum dataset/model.\n',encoding='utf8')
change('internal/evidence.md','No new training, detector inference, complete video processing, browser demo, production deployment, automated repository test suite, or code-coverage measurement is claimed.',
       'One observed single-frame detector inference is documented as E20. No new training, complete video processing, browser demo, production deployment, automated repository test suite, or code-coverage measurement is claimed.')
change('internal/evidence.md','Tidak ada klaim training baru, inferensi detektor, pemrosesan video lengkap, demo browser, deployment produksi, suite tes repository otomatis, atau pengukuran code coverage.',
       'Satu inferensi detektor satu frame yang diamati didokumentasikan sebagai E20. Tidak ada klaim training baru, pemrosesan video lengkap, demo browser, deployment produksi, suite tes repository otomatis, atau pengukuran code coverage.')
path=DEST/'internal/evidence.md'
text=path.read_text(encoding='utf8')
marker='## Executed checks and limits / Pemeriksaan yang dijalankan dan batasnya'
row='| E20 | observed + explicit user direction | One actual detector-use frame with selective face redaction / Satu frame penggunaan detektor sebenarnya dengan redaksi wajah selektif | technical/person-counter-example-evidence.json; actual local inference returned 4 boxes at 640 pixels and confidence 0.25. A visible facial profile and camera text are covered; three back-of-head views retained. The latest user instruction requests this derivative in the public ZIP; upstream license remains undefined / Inferensi lokal sebenarnya menghasilkan 4 kotak pada 640 piksel dan confidence 0,25. Profil wajah yang terlihat serta teks kamera ditutup; tiga tampilan belakang kepala dipertahankan. Instruksi pengguna terbaru meminta turunan ini pada ZIP publik; lisensi upstream tetap belum ditentukan |\n\n'
assert marker in text
text=text.replace(marker,row+marker)
path.write_text(text,encoding='utf8')

change('internal/publication-review.md','six main WebP visuals and six previews','seven main WebP visuals and seven previews')
change('internal/publication-review.md','enam visual WebP utama, dan enam preview','tujuh visual WebP utama, dan tujuh preview')
change('internal/publication-review.md','No faces or documents were edited with AI; no person images were used.',
       'No faces or documents were edited with generative AI. One actual person-detector frame is included at the latest user\'s explicit direction; only a visible facial profile and camera text are covered with permanent plain rectangles. Three back-of-head views remain uncovered.')
change('internal/publication-review.md','Tidak ada wajah atau dokumen yang diedit dengan AI; tidak ada gambar orang yang dipakai.',
       'Tidak ada wajah atau dokumen yang diedit dengan AI generatif. Satu frame detektor orang sebenarnya disertakan atas instruksi eksplisit pengguna terbaru; hanya profil wajah yang terlihat dan teks kamera ditutup dengan kotak polos permanen. Tiga tampilan belakang kepala tetap tidak ditutup.')
path=DEST/'internal/publication-review.md'
path.write_text(path.read_text(encoding='utf8')+'\n## Narrow user-directed inclusion / Penyertaan terbatas atas instruksi pengguna\n\nThe broad imagery exclusions above apply to raw data and galleries. Exactly one privacy-redacted example frame is included as asset-07 at the user\'s explicit request. The underlying export reports `License: undefined`; the user\'s publication instruction is recorded without asserting an independently verified upstream license. No permission for additional dataset frames, raw media, or model binaries is inferred.\n\nPengecualian gambar secara umum di atas berlaku untuk data mentah dan galeri. Tepat satu frame contoh dengan redaksi privasi disertakan sebagai asset-07 atas permintaan eksplisit pengguna. Ekspor asal melaporkan `License: undefined`; instruksi publikasi pengguna dicatat tanpa menyatakan lisensi upstream yang terverifikasi independen. Tidak disimpulkan adanya izin untuk frame dataset lain, media mentah, atau binary model.\n',encoding='utf8')

source_manifest=json.loads((DEST/'source-manifest.json').read_text(encoding='utf8'))
for item in source_manifest['exclusions']:
    if item['category']=='identifiable-images-video-and-media-galleries':
        item['impact']=pair('Raw footage and dataset galleries are omitted; only one explicitly user-directed, face-redacted derivative frame appears in the separate assets gallery, not in the source snapshot.',
                            'Rekaman mentah dan galeri dataset dikeluarkan; hanya satu frame turunan dengan redaksi wajah atas instruksi eksplisit pengguna yang berada pada galeri assets terpisah, bukan snapshot source.')
write_json('source-manifest.json',source_manifest)

# Refresh the local working example so it shows the same selective redaction.
local=ROOT/'portfolio-export/local-examples/person-counter'
shutil.copyfile(DEST/main,local/'single-frame-person-detection.webp')
shutil.copyfile(DEST/preview,local/'single-frame-person-detection-preview.webp')
(local/'example-evidence.local.json').write_text(json.dumps(evidence,indent=2)+'\n',encoding='utf8')
for name in ['README.md','example.en.md','example.id.md']:
    path=local/name;text=path.read_text(encoding='utf8')
    text=text.replace('**LOCAL ONLY / HANYA LOKAL** — not cleared for portfolio publication / belum diizinkan untuk publikasi portfolio.',
                      '**Public derivative requested / Turunan publik diminta** — the latest user instruction includes this redacted example in the public ZIP / instruksi pengguna terbaru menyertakan contoh dengan redaksi ini dalam ZIP publik.')
    text=text.replace('Opaque rectangles hide heads and camera text after inference.','Opaque rectangles hide a visible facial profile and camera text after inference; back-of-head views remain visible.')
    text=text.replace('Kotak polos menutup kepala dan teks kamera setelah inferensi.','Kotak polos menutup profil wajah yang terlihat dan teks kamera setelah inferensi; tampilan belakang kepala tetap terlihat.')
    text=text.replace('The owner chose local preparation only. These files are outside the curated public-export folder and are not included in the public ZIP.',
                      'The latest user instruction requests one redacted frame in the public ZIP; the underlying license remains undefined. This local working copy is still ignored by Git.')
    text=text.replace('Pemilik memilih penyiapan lokal saja. Berkas berada di luar folder ekspor publik terkurasi dan tidak dimasukkan ke ZIP publik.',
                      'Instruksi pengguna terbaru meminta satu frame dengan redaksi di ZIP publik; lisensi asal tetap belum ditentukan. Salinan kerja lokal ini tetap diabaikan Git.')
    text=text.replace('Publication status: LOCAL ONLY. The owner requested local preparation and has not authorized public use of this dataset frame. This image is excluded from the public ZIP.',
                      'Publication status: the latest user instruction includes this one redacted frame in the public ZIP. The underlying dataset license remains undefined.')
    text=text.replace('Status publikasi: HANYA LOKAL. Pemilik meminta penyiapan lokal dan belum mengizinkan penggunaan publik frame dataset ini. Gambar dikeluarkan dari ZIP publik.',
                      'Status publikasi: instruksi pengguna terbaru menyertakan satu frame dengan redaksi ini dalam ZIP publik. Lisensi dataset asal tetap belum ditentukan.')
    path.write_text(text,encoding='utf8')
print(json.dumps({'asset':'asset-07','detections':4,'mainDimensions':canvas.size,'previewDimensions':thumb.size,'visibleFacesRedacted':1,'backOfHeadViewsRetained':3,'mainVisuals':7,'originalSourceChanged':False},ensure_ascii=True))
