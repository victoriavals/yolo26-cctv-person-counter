"""Improve a crowded historical chart without changing its measurements."""
import json
from pathlib import Path
import sys
sys.dont_write_bytecode=True
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
from PIL import Image
from build_export import DEST, ROOT, WORK, sha, pair, write_json

e=json.loads((DEST/'technical/aggregate-evidence.json').read_text(encoding='utf8'))
m=json.loads((DEST/'assets-manifest.json').read_text(encoding='utf8'))
data=e['historicalBackendBenchmark']['backends']
keys=['pytorch_cpu','onnx_cpu','openvino_cpu','pytorch_gpu','openvino_gpu','tensorrt_gpu']
labels=['PyTorch · CPU','ONNX Runtime · CPU','OpenVINO · CPU','PyTorch · GPU','OpenVINO · GPU','TensorRT · GPU']
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':12,'axes.spines.top':False,'axes.spines.right':False,
                     'axes.spines.left':False,'axes.grid':True,'grid.alpha':.2,'axes.axisbelow':True})
fig,axs=plt.subplots(1,2,figsize=(18,10),dpi=100,sharey=True)
fig.subplots_adjust(left=.17,right=.96,bottom=.23,top=.78,wspace=.2)
fig.suptitle('Local inference backends',x=.055,y=.965,ha='left',fontsize=30,fontweight='bold',color='#18364B')
fig.text(.055,.902,'Historical YOLO26s prediction benchmark · 40 test images · confidence 0.25',fontsize=16,color='#4D6274')
fig.text(.055,.865,'Intel i5-13400F CPU / NVIDIA RTX 4060 Ti GPU · one image per prediction call',fontsize=14,color='#4D6274')
y=np.arange(len(keys))
for size,offset,color in [('640',-.17,'#087D8B'),('416',.17,'#A8CED3')]:
    latency=[data[k][size] for k in keys]
    throughput=[1000/v for v in latency]
    for ax,values in zip(axs,[latency,throughput]):
        ax.barh(y+offset,values,height=.28,color=color,label=f'{size} pixels')
        for pos,v in zip(y+offset,values):
            xp=v*1.07 if ax==axs[0] else v+1.1
            ax.text(xp,pos,f'{v:.1f}',va='center',fontsize=11,color='#18364B')
axs[0].set_yticks(y,labels)
axs[0].invert_yaxis()
axs[0].set_xscale('log'); axs[0].set_xlim(8,1300)
axs[0].set_xticks([10,100,1000]); axs[0].xaxis.set_major_formatter(ScalarFormatter())
axs[0].set_xlabel('Milliseconds per image · logarithmic scale',labelpad=15)
axs[0].set_title('Latency · lower is faster',loc='left',fontweight='bold',pad=18)
axs[1].set_xlim(0,108); axs[1].set_xlabel('Frames per second',labelpad=15)
axs[1].set_title('Prediction throughput',loc='left',fontweight='bold',pad=18)
axs[1].axvline(15,color='#6C7F8D',lw=1,ls='--')
axs[1].text(15,.02,' 15 FPS reference',transform=axs[1].get_xaxis_transform(),fontsize=10,color='#4D6274',ha='left')
axs[0].legend(loc='lower right',frameon=False)
fig.text(.055,.125,'Scope: prediction calls only; tracking, video encoding, and production operation are not measured.',fontsize=13,color='#18364B')
fig.text(.055,.083,'Sample check at 640 pixels: 155/155 reference boxes matched across 40/40 images (IoU ≥ 0.95).',fontsize=13,color='#18364B')
fig.text(.055,.041,'This sample match is not universal: crowded-video caches contain different detection totals across runtimes.',fontsize=13,color='#4D6274')
tmp=WORK/'benchmark-rerender.png'
fig.savefig(tmp,facecolor='white',dpi=100)
plt.close(fig)
with Image.open(tmp) as original:
    im=original.convert('RGB')
asset=next(a for a in m['assets'] if a['id']=='asset-05')
im.save(DEST/asset['path'],'WEBP',lossless=True,method=6,exact=True)
preview=im.copy();preview.thumbnail((800,800),Image.Resampling.LANCZOS)
preview.save(DEST/asset['preview']['path'],'WEBP',lossless=True,method=6,exact=True)
for record in [asset,asset['preview']]:
    path=DEST/record['path']
    with Image.open(path) as obj: record.update(width=obj.width,height=obj.height)
    record.update(bytes=path.stat().st_size,sha256=sha(path.read_bytes()))
asset['imageTextLanguage']='en'
asset['alt']=pair('Two horizontal-bar panels compare local prediction latency and throughput across six CPU and GPU backends at 640 and 416 pixels.',
                  'Dua panel batang horizontal membandingkan latensi prediksi lokal dan throughput pada enam backend CPU dan GPU pada 640 serta 416 piksel.')
asset['caption']=pair('Re-rendered from the historical benchmark for legibility; no new timing was measured. The i5-13400F / RTX 4060 Ti results cover prediction calls on 40 images. The 155/155 sampled box match at 640 pixels is not universal; crowded-video caches differ. Tracking and video encoding are excluded from FPS.',
                      'Dirender ulang dari benchmark historis agar terbaca; tidak ada pengukuran waktu baru. Hasil i5-13400F / RTX 4060 Ti mencakup pemanggilan prediksi pada 40 gambar. Kecocokan kotak sampel 155/155 pada 640 piksel tidak bersifat universal; cache video ramai berbeda. Tracking dan encoding video tidak tercakup dalam FPS.')
asset['provenance']={'kind':'re-rendered-historical-project-benchmark','repositoryPath':'reports/eval/backends_default.json',
                     'originalSha256':sha((ROOT/'reports/eval/backends_default.json').read_bytes()),
                     'reason':pair('The original adjacent bar labels overlapped and sample limits were only supplied outside the image. Numerical measurements are unchanged.',
                                   'Label batang berdekatan pada gambar asli bertumpuk dan batas sampel hanya dijelaskan di luar gambar. Angka pengukuran tidak diubah.')}
write_json('assets-manifest.json',m)
print(json.dumps({'updatedAsset':'asset-05','size':im.size,'newMeasurements':False}))
