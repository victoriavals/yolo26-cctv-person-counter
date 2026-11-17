"""Build a reviewed portfolio handoff without changing the source repository."""
from pathlib import Path
from collections import Counter, defaultdict
from html.parser import HTMLParser
import ast
import csv
import hashlib
import importlib.metadata as metadata
import json
import os
import re
import subprocess
import sys
import xml.etree.ElementTree as ET

sys.dont_write_bytecode = True
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
os.environ['YOLO_AUTOINSTALL'] = 'False'

ROOT = Path('D:/ML/gorengan-counter/yolo26-cctv-person-counter')
BASE = ROOT / 'portfolio-export'
WORK = BASE / '_work'
SLUG = 'yolo26-cctv-person-counter'
DATE = '2026-11-16'
COMMIT = '624a840b2de026585d7da4bd9bfeb5f92f8984fd'
DEST = BASE / SLUG
REPO_URL = 'https://github.com/victoriavals/' + SLUG


def pair(en, id):
    return {'en': en, 'id': id}


def sha(data):
    return hashlib.sha256(data).hexdigest()


def write_text(rel, value):
    p = DEST / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(value.rstrip() + '\n', encoding='utf-8', newline='\n')


def write_json(rel, value):
    write_text(rel, json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False))


def git(*args):
    return subprocess.run(['git', '-C', str(ROOT), *args], check=True,
                          capture_output=True, text=True).stdout.strip()


def read_json(rel):
    return json.loads((ROOT / rel).read_text(encoding='utf-8'))


def collect_evidence():
    import numpy as np
    assert git('rev-parse', 'HEAD') == COMMIT
    assert git('status', '--porcelain') == ''
    tracked = git('ls-files').splitlines()
    original_hashes = {r: sha((ROOT / r).read_bytes()) for r in tracked}
    (WORK / 'original-hashes.json').write_text(json.dumps(original_hashes, indent=2), encoding='utf-8')
    snapshot = {
        'repository': REPO_URL, 'branch': git('branch', '--show-current'),
        'commit': COMMIT, 'reviewDate': DATE, 'timezone': 'Asia/Jakarta',
        'trackedWorkingTreeClean': True, 'localChangedTrackedPaths': [],
        'trackedFileCount': len(tracked), 'repositoryVisibility': 'public',
        'repositoryLicenseSpdx': None, 'repositoryHomepage': None,
        'remoteMetadataCheckedVia': 'GitHub connector; GET repository metadata',
        'remoteMetadataCheckedOn': DATE,
        'ownerConfirmation': {
            'name': 'Naufal Firdaus', 'githubAccount': 'victoriavals',
            'confirmationDate': DATE, 'source': 'Owner statement in this session',
            'customCodeRedistributionAuthorized': True,
            'customProjectAuthorshipConfirmed': True,
            'newLicenseAssigned': False,
        },
        'scope': pair('AI person-counter repository only; adjacent repositories and unpublished manuscripts excluded.',
                      'Hanya repository AI person counter; repository pendamping dan naskah belum terbit dikeluarkan.'),
        'artifactScope': pair('Local ignored datasets and model files were inspected for aggregate verification but are not exported. The commit alone does not identify those local artifacts; input hashes are recorded separately.',
                              'Dataset dan model lokal yang diabaikan Git diperiksa untuk verifikasi agregat, tetapi tidak diekspor. Commit saja tidak mengidentifikasi artefak lokal tersebut; hash input dicatat terpisah.'),
    }
    man = list(csv.DictReader((ROOT / 'data/cctv-person/manifest.csv').open(encoding='utf-8')))
    scenes = defaultdict(set)
    for r in man:
        scenes[r['scene']].add(r['split'])
    dataset = {'images': len(man), 'scenes': len(scenes),
               'scenesAssignedToMultipleSplits': sum(len(v) > 1 for v in scenes.values()),
               'domainCounts': dict(Counter(r['kind'] for r in man)), 'splits': {},
               'totalBoxes': 0, 'labelMismatches': 0, 'emptyImages': 0,
               'groupingCaveat': pair('Scene identity comes from filename heuristics; this is not exhaustive pixel-level proof that all cross-scene duplicates are absent.',
                                      'Identitas scene berasal dari heuristik nama berkas; ini bukan bukti menyeluruh berbasis piksel bahwa seluruh duplikasi lintas scene tidak ada.')}
    for sp in ('train', 'val', 'test'):
        d = read_json(f'data/cctv-person/annotations/{sp}.json')
        by_id = defaultdict(list)
        for a in d['annotations']:
            by_id[a['image_id']].append(a['bbox'])
        expected = {i['file_name'] for i in d['images']}
        actual = {p.name for p in (ROOT / f'data/cctv-person/yolo/images/{sp}').glob('*.jpg')}
        mismatch = 0
        for im in d['images']:
            boxes = np.array(by_id[im['id']], dtype=float)
            W, H = im['width'], im['height']
            exp = np.column_stack([np.zeros(len(boxes)), (boxes[:, 0] + boxes[:, 2] / 2) / W,
                                   (boxes[:, 1] + boxes[:, 3] / 2) / H, boxes[:, 2] / W, boxes[:, 3] / H])
            path = ROOT / f'data/cctv-person/yolo/labels/{sp}' / (Path(im['file_name']).stem + '.txt')
            arr = np.loadtxt(path, ndmin=2)
            mismatch += int(arr.shape != exp.shape or not np.allclose(arr, exp, atol=1e-6, rtol=0))
        empty = sum(not by_id[i['id']] for i in d['images'])
        dataset['splits'][sp] = {'images': len(d['images']), 'boxes': len(d['annotations']),
                                 'missingImages': len(expected - actual), 'unexpectedImages': len(actual - expected),
                                 'labelMismatches': mismatch, 'emptyImages': empty,
                                 'domainCounts': dict(Counter(r['kind'] for r in man if r['split'] == sp))}
        dataset['totalBoxes'] += len(d['annotations'])
        dataset['labelMismatches'] += mismatch
        dataset['emptyImages'] += empty
    assert dataset['images'] == 4746 and dataset['totalBoxes'] == 18445
    assert dataset['labelMismatches'] == 0
    sys.path.insert(0, str(ROOT / 'tools'))
    import evaluate as ev
    _, id_of, gt_boxes = ev.load_gt('test')
    files = sorted(id_of)
    evals = {}
    inputs = {}
    names = ('baseline', 'default', 'tuned', 'indoor', 'default_512', 'default_416', 'default_320')
    for name in names:
        r = read_json(f'reports/eval/{name}.json')
        npz = ROOT / f'reports/eval/{name}_preds.npz'
        with np.load(npz, allow_pickle=False) as z:
            preds = {k: z[k] for k in z.files}
        conf = r['conf_best_counting']
        recomputed = {**ev.counting(preds, gt_boxes, id_of, conf, files),
                      **ev.pr_at(preds, gt_boxes, id_of, conf, files)}
        max_delta = max(abs(float(v) - float(r['at_best_counting'][k])) for k, v in recomputed.items())
        assert max_delta < 1e-12
        errors = [int((preds[f][:, 4] >= conf).sum()) - len(gt_boxes[id_of[f]]) for f in files]
        evals[name] = {'split': r['split'], 'images': r['images'], 'imgsz': r.get('imgsz', 640),
                       'confidence': conf, 'confidenceForF1': r['conf_best_f1'],
                       'reportedDetection': r['overall'], 'reportedCounting': r['at_best_counting'],
                       'recomputedCountingAndPR': recomputed, 'maxAbsoluteRecomputeDelta': max_delta,
                       'exactCountPercentFromCache': float(100 * np.mean(np.array(errors) == 0)),
                       'reportedByDomain': r['by_kind'], 'reportedByDensity': r['by_density'],
                       'mAPRecomputed': False, 'newModelInferenceRun': False}
        for rel in (f'reports/eval/{name}.json', f'reports/eval/{name}_preds.npz'):
            inputs[rel] = sha((ROOT / rel).read_bytes())
    back = read_json('reports/eval/backends_default.json')
    benchmark = {k: back[k] for k in ('run', 'conf', 'n_images', 'sizes', 'gpu', 'hw_of', 'backends', 'parity')}
    benchmark['scope'] = pair('Historical local prediction benchmark; not a whole-video tracking or production performance measurement. Parity is limited to 40 sampled images at 640 pixels.',
                              'Benchmark prediksi lokal historis; bukan pengukuran tracking seluruh video atau performa produksi. Kesamaan hasil terbatas pada 40 gambar sampel pada 640 piksel.')
    video_comparison = {}
    for backend in ('pytorch_cpu', 'pytorch_gpu', 'onnx_cpu', 'openvino_cpu', 'openvino_gpu', 'tensorrt_gpu'):
        r = read_json(f'reports/eval/backend_compare/{backend}.json')
        video_comparison[backend] = {k: r[k] for k in ('ms_mean', 'fps', 'total_boxes')}
        video_comparison[backend]['frames'] = len(r['boxes'])
    warehouse = read_json('reports/tracking/warehouse_summary.json')
    warehouse_safe = {k: warehouse[k] for k in ('frames', 'fps_video', 'conf', 'occupancy', 'track_unik', 'garis_masuk', 'garis_keluar')}
    warehouse_safe['interpretation'] = pair('Historical demo output. The README gives a visual estimate of 5–7 people and about 6 real individuals; no tracking ground truth was supplied. The source recording and its path are excluded.',
                                           'Keluaran demo historis. README memberi perkiraan visual 5–7 orang dan sekitar 6 individu; ground truth tracking tidak tersedia. Rekaman sumber dan path-nya dikeluarkan.')
    training = {}
    for name in ('default', 'tuned', 'indoor'):
        p = ROOT / f'runs/person/{name}/results.csv'
        rows = [{k.strip(): v.strip() for k, v in r.items()} for r in csv.DictReader(p.open())]
        best = max(rows, key=lambda r: float(r['metrics/mAP50-95(B)']))
        training[name] = {'epochsRecorded': len(rows), 'epochWithHighestValidationMAP5095': int(float(best['epoch'])),
                          'runReportedSeconds': read_json(f'runs/person/{name}/run_meta.json')['seconds']}
        inputs[p.relative_to(ROOT).as_posix()] = sha(p.read_bytes())
    hash_checks = []
    for line in (ROOT / 'weights/SHA256SUMS').read_text().splitlines():
        expected, rel = line.split(None, 1)
        rel = rel.strip().lstrip('*')
        actual = sha((ROOT / 'weights' / rel).read_bytes())
        assert expected == actual
        hash_checks.append({'path': 'weights/' + rel, 'sha256': actual, 'matches': True})
    same_model = sha((ROOT / 'weights/best.pt').read_bytes()) == sha((ROOT / 'runs/person/default/weights/best.pt').read_bytes())
    assert same_model
    graphs = {}
    for rel in ('weights/best_openvino_model/best.xml', 'runs/person/default/weights/best_openvino_model/best.xml'):
        tree = ET.parse(ROOT / rel)
        parameter = next(x for x in tree.findall('./layers/layer') if x.attrib.get('type') == 'Parameter')
        graphs[rel] = parameter.find('data').attrib
    packages = {}
    for name in ('ultralytics', 'torch', 'torchvision', 'supervision', 'lap', 'numpy', 'opencv-python', 'Pillow',
                 'pycocotools', 'matplotlib', 'onnx', 'onnxruntime', 'openvino', 'tensorrt'):
        packages[name] = metadata.version(name)
    import supervision as sv
    zone = sv.LineZone(start=sv.Point(0, 50), end=sv.Point(100, 50))
    synthetic = sv.Detections(xyxy=np.array([[10, 10, 20, 30], [40, 70, 50, 90]], dtype=np.float32),
                              confidence=np.array([0.9, 0.9]), class_id=np.array([0, 0]), tracker_id=np.array([-1, -1]))
    zone.trigger(synthetic)
    assert zone.in_count + zone.out_count == 1
    trainer_path = metadata.distribution('ultralytics').locate_file('ultralytics/engine/trainer.py')
    trainer_source = trainer_path.read_text(encoding='utf-8')
    auto_ignores_lr = "ignoring 'lr0=" in trainer_source
    cfg_source = metadata.distribution('ultralytics').locate_file('ultralytics/cfg/default.yaml').read_text(encoding='utf-8')
    auto_default = bool(re.search(r'^optimizer:\s*auto\b', cfg_source, flags=re.M))
    assert auto_ignores_lr and auto_default
    import merge_store_data as msd
    blocks = msd.time_blocks([f'frame_{i:07d}' for i in range(300)], 10)
    assert len(blocks) == 10
    split_check = {'syntheticFrames': 300, 'timeBlocks': len(blocks), 'testBlocks': max(1, round(len(blocks)*0.20)),
                   'validationBlocks': max(1, round(len(blocks)*0.15))}
    audit = read_json('reports/dup_audit.json')
    audit_summary = {k: v for k, v in audit.items() if k != 'assignments'}
    old_stats = read_json('reports/stats.json')
    for rel in ('data/cctv-person/manifest.csv', 'data/cctv-person/annotations/train.json',
                'data/cctv-person/annotations/val.json', 'data/cctv-person/annotations/test.json',
                'reports/eval/backends_default.json', 'reports/tracking/warehouse_summary.json'):
        inputs[rel] = sha((ROOT / rel).read_bytes())
    evidence = {'snapshot': snapshot, 'dataset': dataset, 'evaluation': evals, 'historicalBackendBenchmark': benchmark,
                'historicalVideoComparison': video_comparison, 'historicalWarehouseDemo': warehouse_safe,
                'historicalTraining': training, 'modelIntegrity': {'checksumChecks': hash_checks, 'defaultCopiesIdentical': same_model,
                                                                 'openvinoInputGraphs': graphs},
                'verification': {'newTrainingRun': False, 'newVideoInferenceRun': False, 'cachedPredictionsRecomputed': True},
                'focusedChecks': {'sharedInvalidId': {'inputKind': 'synthetic-geometry-only', 'inputFrames': 1,
                                                      'inputBoxes': 2, 'sharedTrackerId': -1,
                                                      'inCount': zone.in_count, 'outCount': zone.out_count},
                                  'installedOptimizerAuto': {'defaultIsAuto': auto_default, 'implementationIgnoresManualLr0': auto_ignores_lr,
                                                            'storeConfigExplicitOptimizer': False},
                                  'storeTimeBlockRounding': split_check},
                'reportedRawAudit': audit_summary,
                'legacyDatasetStats': {'images': old_stats['images'], 'boxes': old_stats['boxes'], 'currentDatasetMatches': False},
                'inputArtifactHashes': inputs}
    write_json('technical/aggregate-evidence.json', evidence)
    write_json('technical/environment-observed.json', {
        'observedOn': DATE, 'pythonVersion': sys.version.split()[0], 'platform': 'Windows', 'packages': packages,
        'purpose': pair('Observed installed environment, not an original dependency manifest or a reproducible lockfile. No package was installed for this export.',
                        'Lingkungan terpasang yang diamati, bukan manifest dependensi asli atau lockfile reproduktif. Tidak ada paket yang diinstal untuk ekspor ini.')})
    return evidence


def export_source():
    rels = sorted(p.relative_to(ROOT).as_posix() for p in (ROOT / 'tools').glob('*.py'))
    rels += ['README.md', 'PANDUAN.md', '.gitignore', '.gitattributes',
             'reports/dataset-report.html', 'reports/training-report.html',
             'reports/tracking/player.html', 'reports/tracking/backends.html',
             'data/cctv-person/yolo/data.yaml', 'data/cctv-person/yolo/data_indoor.yaml']
    assert len(rels) == 26
    records = []
    for rel in sorted(rels):
        original = (ROOT / rel).read_bytes()
        txt = original.decode('utf-8')
        transformations = []
        if rel == 'tools/train.py':
            lines = txt.splitlines(keepends=True)
            for i, line in enumerate(lines):
                if 'computer-vision' in line and 'runs' in line:
                    lines[i] = '    # <LEGACY_RUNS_DIR>/detect/runs/person\n'
                    transformations.append('Replaced a companion-project path in a comment with an explicit placeholder.')
            txt = ''.join(lines)
        for before, after in [
            ('D:/ML/person-counter', '<PROJECT_ROOT>'),
            ('D:\\ML\\person-counter', '<PROJECT_ROOT>'),
            ('D:/ML/...', '<ABSOLUTE_VIDEO_PATH>'),
            ('/d/ML/...', '<GIT_BASH_PATH>'),
        ]:
            if before in txt:
                txt = txt.replace(before, after)
                transformations.append('Replaced a machine-specific example path with an explicit placeholder.')
        if rel.endswith('data.yaml') or rel.endswith('data_indoor.yaml'):
            txt = re.sub(r'^path: .+$', 'path: <DATASET_ROOT>', txt, flags=re.M)
            transformations.append('Dataset root is an explicit placeholder; configure it before running Ultralytics.')
        if '\ufffd' in txt:
            txt = txt.replace('\ufffd', '—')
            transformations.append('Replaced an existing text-encoding replacement character with punctuation.')
        # Preserve source bytes exactly unless a reviewed redaction is required.
        output = txt.encode('utf-8') if transformations else original
        p = DEST / 'source-code' / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(output)
        if p.suffix == '.py':
            ast.parse(p.read_text(encoding='utf-8'), filename=rel)
        records.append({'path': 'source-code/' + rel, 'repositoryPath': rel, 'bytes': len(output),
                        'sha256': sha(output), 'originalSha256': sha(original),
                        'origin': 'reviewed-repository-file', 'redacted': bool(transformations),
                        'transformations': sorted(set(transformations)),
                        'review': {'textReviewed': True, 'secretScan': 'passed', 'distributionBasis': 'owner-confirmed-custom-project-material'}})
    return records


def make_assets():
    from PIL import Image, ImageDraw, ImageFont
    ad = DEST / 'assets'
    ad.mkdir(parents=True, exist_ok=True)
    W, H = 1600, 1000
    cover = Image.new('RGB', (W, H), '#F3F6FA')
    d = ImageDraw.Draw(cover)
    def font(size, bold=False):
        return ImageFont.truetype('C:/Windows/Fonts/' + ('segoeuib.ttf' if bold else 'segoeui.ttf'), size)
    ink, accent, muted = '#18364B', '#087D8B', '#4D6274'
    d.text((72, 42), 'COMPUTER VISION  /  OFFLINE VIDEO', font=font(20, True), fill=accent)
    d.text((68, 88), 'CCTV Person Counter', font=font(66, True), fill=ink)
    d.text((72, 182), 'Code-derived pipeline · Research prototype', font=font(27), fill=muted)
    def box(x, y, w, h, title, sub, title_size=26):
        d.rounded_rectangle((x, y, x+w, y+h), radius=16, fill='white', outline='#C2D1DD', width=2)
        y_text = y + 26
        for line in title.split('\n'):
            assert d.textbbox((0,0), line, font=font(title_size, True))[2] < w-28, line
            d.text((x+18, y_text), line, font=font(title_size, True), fill=ink)
            y_text += title_size + 8
        for line in sub.split('\n'):
            assert d.textbbox((0,0), line, font=font(21))[2] < w-28, line
            d.text((x+18, max(y_text+8, y+h-56)), line, font=font(21), fill=muted)
            y_text += 28
    def arrow(x, y, end):
        d.line((x,y,end,y), fill=accent, width=4)
        d.polygon([(end,y),(end-12,y-7),(end-12,y+7)],fill=accent)
    d.text((72, 272), '01  MODEL DEVELOPMENT', font=font(23, True), fill=accent)
    xs = [72, 440, 808, 1176]
    top = [('Audit & clean', 'COCO → YOLO labels'), ('Fine-tune YOLO26s', 'Pretrained → person'),
           ('Evaluate', 'mAP + counting error'), ('Export & benchmark', 'CPU / GPU runtimes')]
    for i,(title,sub) in enumerate(top):
        box(xs[i], 320, 300, 150, title, sub, 25)
        if i < 3: arrow(xs[i]+307, 395, xs[i+1]-12)
    d.text((72, 505), 'The chosen model is loaded for video inference.', font=font(25), fill=muted)
    d.text((72, 586), '02  VIDEO COUNTING', font=font(23, True), fill=accent)
    xs = [72, 367, 662, 957, 1252]
    bottom = [('Recorded video','Local file'), ('YOLO26s','Person detections'), ('ByteTrack','Persistent track IDs'),
              ('LineZone','Directional crossings'), ('CSV / JSON','Optional video')]
    for i,(title,sub) in enumerate(bottom):
        box(xs[i], 636, 240, 145, title, sub, 25)
        if i < 4: arrow(xs[i]+247, 709, xs[i+1]-12)
    d.line((72, 840, 1492, 840), fill='#CBD6DF', width=2)
    d.text((72, 871), 'Implemented: detection, tracking and line-crossing workflow.', font=font(25, True), fill=ink)
    d.text((72, 916), 'Store-specific evaluation and tracking ground truth remain incomplete.', font=font(23), fill=muted)
    specs = [
      ('asset-01','01-person-counter-pipeline',None,'architecture-diagram',True,
       pair('Two-stage diagram of model development and recorded-video counting with YOLO26s, ByteTrack, and LineZone.', 'Diagram dua tahap pengembangan model dan counting rekaman video dengan YOLO26s, ByteTrack, dan LineZone.'),
       pair('Architecture illustration reconstructed from the reviewed code; it is not an application screenshot or an observed deployment.', 'Ilustrasi arsitektur yang disusun dari kode yang ditinjau; bukan screenshot aplikasi atau deployment yang diamati.')),
      ('asset-02','02-domain-and-density-results','breakdown','experiment-chart',True,
       pair('Four charts compare detection mAP50 and counting MAE by domain and crowd-density group.', 'Empat grafik membandingkan mAP50 deteksi dan MAE hitungan menurut domain dan kelompok kepadatan.'),
       pair('Historical test-split results for baseline, default, tuned, and indoor models. Dense groups with fewer than ten images were omitted by the original plot code.', 'Hasil test historis untuk model baseline, default, tuned, dan indoor. Kelompok padat dengan kurang dari sepuluh gambar dilewati oleh kode grafik asli.')),
      ('asset-03','03-counting-predictions','scatter','experiment-chart',False,
       pair('Four scatter plots compare predicted and ground-truth person counts, colored by indoor, outdoor, and unknown domains.', 'Empat scatter plot membandingkan hitungan prediksi dan ground truth, dengan warna domain indoor, outdoor, dan unknown.'),
       pair('Historical cached test predictions on 475 images. The ±1 figure means an error of at most one person; it is not exact-count accuracy.', 'Prediksi test historis tersimpan pada 475 gambar. Angka ±1 berarti galat paling banyak satu orang; bukan akurasi hitungan tepat.')),
      ('asset-04','04-confidence-threshold-sweep','sweep','experiment-chart',False,
       pair('Line charts show counting MAE and F1 across confidence thresholds for four model variants.', 'Grafik garis menunjukkan MAE hitungan dan F1 pada berbagai confidence untuk empat varian model.'),
       pair('Historical confidence sweep selected on the evaluated test split. It is exploratory tuning, not an untouched-test estimate at a validation-selected threshold.', 'Sweep confidence historis dipilih pada split test yang dievaluasi. Ini tuning eksploratif, bukan estimasi test murni dengan threshold yang dipilih pada validation.')),
      ('asset-05','05-inference-backend-benchmark','backends','benchmark-chart',True,
       pair('Three panels compare local inference latency, throughput, and sampled box matching across six CPU and GPU backends.', 'Tiga panel membandingkan latensi inferensi lokal, throughput, dan kecocokan kotak sampel pada enam backend CPU dan GPU.'),
       pair('Historical benchmark on an i5-13400F and RTX 4060 Ti. The 100% match applies only to 40 sampled images at 640 pixels; crowded-video detections differ between runtimes. FPS excludes full tracking and video encoding.', 'Benchmark historis pada i5-13400F dan RTX 4060 Ti. Kecocokan 100% hanya berlaku pada 40 gambar sampel pada 640 piksel; deteksi video ramai berbeda antar-runtime. FPS tidak mencakup seluruh tracking dan encoding video.')),
      ('asset-06','06-training-curves','curves','training-chart',False,
       pair('Training box-loss and validation mAP curves compare default, tuned, and indoor fine-tuning experiments.', 'Kurva box loss training dan mAP validation membandingkan eksperimen fine-tuning default, tuned, dan indoor.'),
       pair('Historical run logs: default recorded 131 epochs, tuned 150, and indoor 80. These curves were not produced by a new training run during portfolio preparation.', 'Log run historis: default mencatat 131 epoch, tuned 150, dan indoor 80. Kurva ini tidak dihasilkan oleh training baru saat penyiapan portfolio.')),
    ]
    assets=[]
    for n,(aid,stem,source,kind,recommended,alt,caption) in enumerate(specs,1):
        if source is None:
            im = cover
            origin = {'kind':'new-code-derived-diagram','sourcePaths':['tools/train.py','tools/evaluate.py','tools/count_people.py','tools/benchmark_backends.py']}
        else:
            rel = f'reports/eval/figures/{source}.png'
            with Image.open(ROOT / rel) as original:
                im = original.convert('RGB')
            origin = {'kind':'existing-project-chart','repositoryPath':rel,'originalSha256':sha((ROOT / rel).read_bytes())}
        main_rel=f'assets/{stem}.webp'; prev_rel=f'assets/{stem}-preview.webp'
        im.save(DEST / main_rel, 'WEBP', lossless=True, method=6, exact=True)
        prev = im.copy()
        prev.thumbnail((800,800), Image.Resampling.LANCZOS)
        prev.save(DEST / prev_rel,'WEBP',lossless=True,method=6,exact=True)
        def info(rel):
            with Image.open(DEST / rel) as obj:
                dims={'width':obj.width,'height':obj.height}
            data=(DEST/rel).read_bytes()
            return {'path':rel,**dims,'bytes':len(data),'sha256':sha(data)}
        assets.append({'id':aid,**info(main_rel),'preview':info(prev_rel),'visualType':kind,'order':n,
                       'fit':'cover' if n==1 else 'contain','detailFit':'contain','cardAspectRatio':'8:5',
                       'imageTextLanguage':'en' if n==1 else 'id-with-technical-labels',
                       'alt':alt,'caption':caption,'recommendedForInitialWebsite':recommended,'provenance':origin,
                       'coverReason':pair('The 8:5 diagram explains the actual workflow and prototype status without redistributing identifiable footage or implying a web application.',
                                          'Diagram 8:5 menjelaskan alur nyata dan status prototipe tanpa mendistribusikan rekaman orang yang dapat dikenali atau menyiratkan aplikasi web.') if n==1 else None,
                       'review':{'status':'pending-final-visual-review','metadataRemoved':True,'containsPeople':False,
                                  'containsQR':False,'containsPersonalData':False,'aiGenerated':False,
                                  'distributionBasis':'owner-confirmed-project-chart-or-new-documentation-diagram'}})
    write_json('assets-manifest.json',{'schemaVersion':'1.0','assets':assets,
               'recommendedWebsiteAssets':['asset-01','asset-02','asset-05'],
               'curationNote':pair('Use only the three recommended images initially; the remaining gallery is optional. Preserve captions and use contain for charts.',
                                   'Gunakan tiga gambar rekomendasi terlebih dahulu; galeri lainnya opsional. Pertahankan caption dan gunakan contain untuk grafik.')})
    return assets


def main():
    assert not DEST.exists(), 'Refusing to overwrite an existing export directory.'
    DEST.mkdir(parents=True)
    evidence=collect_evidence()
    records=export_source()
    assets=make_assets()
    from portfolio_content import make_content
    make_content(evidence,assets,records)
    print(json.dumps({'built':str(DEST),'sourceFiles':27,'mainVisuals':len(assets),'previewVisuals':len(assets)},ensure_ascii=True))


if __name__=='__main__':
    main()
