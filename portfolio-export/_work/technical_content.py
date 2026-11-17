"""Technical documentation for the public portfolio handoff."""
import json
from build_export import write_text, write_json, pair, DATE, COMMIT


def write_technical(p,e):
    diagram='''flowchart LR
  subgraph DEVELOPMENT["Model development: implemented"]
    RAW["Authorized raw COCO export - excluded"] --> AUDIT["audit_duplicates.py"]
    RAW --> CLEAN["clean_cctv.py + families.py"]
    AUDIT -->|"cluster assignments"| CLEAN
    CLEAN --> DATA["COCO + YOLO labels + manifest"]
    PRE["Pretrained YOLO26s - external"] --> TRAIN["train.py"]
    DATA --> TRAIN
    TRAIN --> WEIGHTS["best.pt - excluded"]
    WEIGHTS --> EVAL["evaluate.py: detection + counting"]
    DATA --> EVAL
    EVAL --> METRICS["JSON + NPZ + figures"]
    WEIGHTS --> EXPORT["export.py / benchmark_backends.py"]
  end
  subgraph COUNTING["Recorded-video counting: implemented"]
    VIDEO["Local video file - excluded"] --> DETECTOR["YOLO backend"]
    DETECTOR --> TRACK["ByteTrack / BoT-SORT"]
    TRACK --> LINE["Supervision LineZone"]
    LINE --> OUT["CSV + summary JSON + optional MP4"]
  end
  WEIGHTS --> DETECTOR
  EXPORT --> DETECTOR
  subgraph ADAPTATION["Store adaptation: partial"]
    OWN["Authorized store footage - not supplied"] --> SAMPLE["prepare_labeling.py"]
    SAMPLE --> HUMAN["Manual label correction"]
    HUMAN --> MERGE["merge_store_data.py"]
    MERGE --> STORE["train.py store - configuration exists"]
    STORE -.-> MISSING["Store-only evaluation - not implemented"]
  end'''
    write_text('technical/architecture.md',f'''# Architecture / Arsitektur

## English

The repository contains 15 standalone Python command-line programs and one shared scene-family module. The programs exchange files; no application server, inference API, message queue, or database is implemented. `serve_reports.py` is an optional static-file viewer with HTTP Range support.

The following diagram describes relationships visible in the inspected code. It is a **code-derived architecture diagram**, not an observed production topology. The store branch is explicitly partial. External inputs and model binaries shown in the diagram are not bundled.

```mermaid
{diagram}
```

### AI lifecycle

1. **Input and provenance.** The training entry point initializes `models/yolo26s.pt`, a pretrained detector. Raw data comes from an externally obtained Roboflow COCO export. Actual inference consumes local recorded video; the current counter rejects non-file paths.
2. **Dataset processing.** The audit uses HSV and gray histograms after suppressing near-black fill pixels. It clusters within filename-stem groups with similarity threshold 0.97. Cleaning selects one representative, maps category IDs 0–3 to person, drops shoe IDs 4–5 and the non-CCTV family, filters degenerate boxes, and creates scene-grouped approximately 80/10/10 splits stratified by domain and density. These are heuristic groups, not universally verified camera identities.
3. **Model and fine-tuning.** Ultralytics handles image resizing/letterboxing and model input preparation. The trained detector has one class, `person`, with end-to-end detection and `reg_max=1` in the inspected metadata/model. The project fine-tunes pretrained weights; it does not train the detector from random initialization. Shared settings request 640-pixel images, batch 16, up to 150 epochs, patience 30, two workers, mixed precision, and seed 0. Default recorded 131 epochs, tuned 150, and indoor 80.
4. **Evaluation.** `predict()` caches low-threshold predictions in chunks of 16. COCOeval supplies mAP; confidence-ordered IoU matching supplies precision/recall and count differences supply MAE/RMSE/bias/within-one. IoU 0.5 is an evaluation matching threshold, not an NMS setting. Confidence is swept from 0.05 through 0.95 in 0.05 steps. The selected threshold currently comes from the split being evaluated, including test.
5. **Runtime and tracking.** `model.track(..., persist=True, stream=True)` requests 640 pixels and default confidence 0.25. ByteTrack is default; BoT-SORT is selectable. The resulting tracked boxes are converted to Supervision detections, passed to LineZone, and counted. Occupancy is `len(det)` after tracking, so it is affected by tracking postprocessing. A normalized line specifies direction; entering/exiting meaning needs camera-specific calibration.
6. **Outputs.** Per-frame counts and cumulative directional crossings go to CSV, aggregate statistics to JSON, and optional boxes/traces/line overlays to MP4. Cumulative IDs are tracker identities, not a validated set of unique visitors. Retrieval, language models, embeddings, and vector stores are absent.

### Runtime artifacts and boundaries

| Runtime | Selected by | Important constraint |
|---|---|---|
| PyTorch | `.pt` | Requires compatible local model and Torch installation |
| ONNX Runtime | `.onnx` | Fixed exports here use batch one and a baked input size |
| OpenVINO | directory ending `_openvino_model` | Inspect graph dimensions; explicit `intel:cpu` is used in benchmark code |
| TensorRT | `.engine` | Compiled artifacts depend on the compatible GPU/runtime environment |

The committed deployment OpenVINO graph inspected locally has shape `1,3,640,640`; a different export beside the run weights has `1,3,416,416`. Both are excluded from the handoff. A requested image size cannot replace a fixed graph's actual shape. The counter CLI does not expose a device flag; do not invent one in deployment instructions.

## Bahasa Indonesia

Repository berisi 15 program command-line Python mandiri dan satu modul bersama untuk keluarga scene. Program bertukar berkas; tidak ada server aplikasi, API inferensi, message queue, atau database yang diimplementasikan. `serve_reports.py` merupakan penampil berkas statis opsional dengan dukungan HTTP Range.

Diagram di atas menjelaskan hubungan yang terlihat pada kode yang diperiksa. Diagram tersebut **disusun dari kode**, bukan topologi produksi yang diamati. Cabang toko ditandai parsial secara eksplisit. Input eksternal dan binary model pada diagram tidak dibundel.

### Siklus AI

1. **Input dan asal.** Entry point training menginisialisasi `models/yolo26s.pt`, yaitu detektor pretrained. Data mentah berasal dari ekspor COCO Roboflow yang diperoleh secara eksternal. Inferensi sebenarnya menerima rekaman video lokal; counter saat ini menolak path selain berkas.
2. **Pemrosesan dataset.** Audit memakai histogram HSV dan abu-abu setelah menekan piksel fill hampir hitam. Pengelompokan berjalan dalam kelompok stem nama berkas dengan threshold kemiripan 0,97. Pembersihan memilih satu representasi, memetakan ID kategori 0–3 menjadi person, membuang ID sepatu 4–5 dan keluarga non-CCTV, menyaring kotak degeneratif, serta membuat split berbasis scene sekitar 80/10/10 yang distratifikasi menurut domain dan kepadatan. Kelompok ini bersifat heuristik, bukan identitas kamera yang terverifikasi universal.
3. **Model dan fine-tuning.** Ultralytics menangani resizing/letterboxing serta persiapan input model. Detektor terlatih memiliki satu kelas, `person`, dengan deteksi end-to-end dan `reg_max=1` pada metadata/model yang diperiksa. Proyek melakukan fine-tuning bobot pretrained; tidak melatih detektor dari inisialisasi acak. Setelan bersama meminta gambar 640 piksel, batch 16, maksimal 150 epoch, patience 30, dua worker, mixed precision, dan seed 0. Default mencatat 131 epoch, tuned 150, dan indoor 80.
4. **Evaluasi.** `predict()` menyimpan prediksi ber-threshold rendah dalam chunk 16. COCOeval menyediakan mAP; pencocokan IoU berdasarkan urutan confidence menyediakan precision/recall, sedangkan selisih jumlah menyediakan MAE/RMSE/bias/within-one. IoU 0,5 merupakan threshold pencocokan evaluasi, bukan setelan NMS. Confidence di-sweep dari 0,05 hingga 0,95 dengan langkah 0,05. Threshold terpilih saat ini berasal dari split yang sedang dievaluasi, termasuk test.
5. **Runtime dan tracking.** `model.track(..., persist=True, stream=True)` meminta 640 piksel dan confidence default 0,25. ByteTrack menjadi default; BoT-SORT dapat dipilih. Kotak hasil tracking dikonversi menjadi deteksi Supervision, diteruskan ke LineZone, dan dihitung. Occupancy merupakan `len(det)` setelah tracking, sehingga dipengaruhi postprocessing tracking. Garis ternormalisasi menentukan arah; arti masuk/keluar memerlukan kalibrasi khusus kamera.
6. **Keluaran.** Hitungan per frame dan lintasan kumulatif berdasarkan arah masuk ke CSV, statistik agregat ke JSON, serta overlay kotak/jejak/garis opsional ke MP4. ID kumulatif merupakan identitas tracker, bukan kumpulan pengunjung unik yang tervalidasi. Retrieval, model bahasa, embedding, dan vector store tidak tersedia.

### Artefak runtime dan batasnya

| Runtime | Dipilih melalui | Batas penting |
|---|---|---|
| PyTorch | `.pt` | Memerlukan model lokal dan instalasi Torch yang kompatibel |
| ONNX Runtime | `.onnx` | Ekspor tetap di sini memakai batch satu dan ukuran input tertanam |
| OpenVINO | direktori berakhiran `_openvino_model` | Periksa dimensi graph; kode benchmark memakai `intel:cpu` eksplisit |
| TensorRT | `.engine` | Artefak kompilasi bergantung pada lingkungan GPU/runtime kompatibel |

Graph deployment OpenVINO tersimpan yang diperiksa lokal memiliki shape `1,3,640,640`; ekspor berbeda di samping bobot run memiliki `1,3,416,416`. Keduanya dikeluarkan dari handoff. Ukuran gambar yang diminta tidak dapat mengganti shape aktual graph tetap. CLI counter tidak menyediakan flag device; jangan mengarang flag tersebut pada instruksi deployment.

Sources / Sumber: [training](../source-code/tools/train.py), [evaluation](../source-code/tools/evaluate.py), [counter](../source-code/tools/count_people.py), [benchmark](../source-code/tools/benchmark_backends.py), [aggregate evidence](aggregate-evidence.json).
''')
    versions='\n'.join(f'| {name} | {version} |' for name,version in json.loads((__import__('build_export').DEST/'technical/environment-observed.json').read_text())['packages'].items())
    write_text('technical/development.md',f'''# Development and reproduction / Pengembangan dan reproduksi

## English

### Folder and entry-point map

```text
source-code/
  tools/                     16 Python modules; CLI steps and one shared module
  data/cctv-person/yolo/     two YAML configurations with explicit root placeholders
  reports/                   four historical HTML sources; media deliberately absent
  README.md, PANDUAN.md      historical project and operator documentation
  EXPORT-NOTES.md            export changes and known historical inconsistencies
```

| Entry point | Responsibility / relevant arguments |
|---|---|
| `audit_duplicates.py` | Read raw COCO export, calculate duplicate clusters; no CLI options |
| `clean_cctv.py` | Consume audit, export cleaned COCO/YOLO dataset; no CLI options |
| `families.py` | Shared filename-family rules, imported by cleaning, training, and dataset figures |
| `make_figures.py` | Dataset figures/statistics; reads raw and cleaned inputs |
| `train.py` | `--list`; runs `smoke`, `default`, `tuned`, `indoor`, `store`; `all` excludes smoke/store |
| `status.py` | Read training CSV logs; optional `--watch`; only default/tuned/indoor are listed |
| `evaluate.py` | Targets baseline/default/tuned/indoor, `--all`, `--split`, `--imgsz` |
| `make_eval_figures.py` | Render figures from stored evaluation JSON/NPZ and run CSVs |
| `export.py` | ONNX export, sampled parity and timing; `--run`, `--n` |
| `benchmark_backends.py` | Stage exports and compare runtimes; `--run`, `--n`, `--sizes`, `--skip` |
| `make_backend_figure.py` | Plot the hard-coded default backend report |
| `count_people.py` | `--source`, `--weights`, `--conf`, `--line`, `--max-frames`, `--tracker`, `--no-video`, `--name` |
| `prepare_labeling.py` | `--source`, `--n`, `--min-gap-sec`, `--weights`, `--no-predraw`, `--min-change`, `--name` |
| `merge_store_data.py` | `--store`, `--store-val`, `--store-test`; requires manually reviewed YOLO labels |
| `compare_backends_video.py` | Fixed source/artifact conventions; `--start`, `--frames`, `--infer`, `--render` |
| `serve_reports.py` | Static viewer only; `--dir`, `--port`, `--bind` (default loopback) |

### Dependency roles

Runtime/model: Python, PyTorch, Ultralytics, OpenCV, NumPy, Supervision, and LAP. Optional inference runtimes: ONNX Runtime, OpenVINO, TensorRT. Evaluation/reporting: pycocotools, Pillow, Matplotlib, and ONNX export tools. Development tools include Git and ordinary Python CLI tooling; there is no package build pipeline or automated test framework configured in the repository.

Roboflow is the historical dataset source, and pretrained YOLO26s is an external model input. They are not network services called for each prediction. Manual annotation tools are operator choices, not bundled services. No companion repository is required by an implemented import in this snapshot.

### Installation examples, not actions performed during export

There is no original `requirements.txt`, lockfile, or `pyproject.toml`. The table below records installed versions observed on {DATE}; it is not proof that every wheel will be available on another machine. This export did not install dependencies or modify global Python.

| Installed distribution | Observed version |
|---|---|
{versions}

Create an isolated environment for review using PowerShell, from a separate working copy of `source-code`:

```powershell
python -m venv .venv
.venv/Scripts/python.exe -m pip install numpy==2.3.5 Pillow==11.3.0 PyYAML
```

For actual inference/training, install compatible PyTorch/Torchvision wheels for your platform first. The observed workstation used Torch `2.14.0+cu126` and Torchvision `0.29.0+cu126`; the original README identifies the official cu126 package index. Then install the required application/evaluation packages, for example:

```powershell
.venv/Scripts/python.exe -m pip install ultralytics==8.4.60 supervision==0.27.0 opencv-python==4.10.0.84 lap==0.5.13 pycocotools==2.0.11 matplotlib==3.10.8
$env:YOLO_AUTOINSTALL = 'False'
$env:PYTHONDONTWRITEBYTECODE = '1'
```

Optional exports require their compatible ONNX/OpenVINO/TensorRT tooling and its terms. Installing those runtimes is not necessary merely to read the snapshot. Do not treat the observed CUDA wheel as a universal CPU installation command.

### Configuration and conditional execution

Replace `<DATASET_ROOT>` in the copied YAML files with the absolute path to an authorized YOLO-format dataset. Other placeholders such as `<PROJECT_ROOT>`, `<ABSOLUTE_VIDEO_PATH>`, and `<LEGACY_RUNS_DIR>` are intentional redactions; substitute actual local values before using those examples. Do not put credentials into these files.

For the original pipeline, separately obtain authorized matching raw data and pretrained weights, place them in the expected local folders, and run from the source root:

```powershell
python tools/audit_duplicates.py
python tools/clean_cctv.py
python tools/train.py --list
python tools/train.py smoke
python tools/train.py default
python tools/evaluate.py default
```

These commands are documented reproduction steps, **not checks executed for this export**. The first two expect the full original raw export. Cleaning rewrites dataset outputs and does not remove old split files; use a fresh working copy to avoid retaining stale files. Training requires model/data artifacts not present in the ZIP. The smoke run trains for two epochs on a fraction of the dataset; it is not a unit test.

With an authorized local video and a compatible single-person PyTorch model, an example is:

```powershell
python tools/count_people.py --source "<AUTHORIZED_VIDEO_FILE>" --weights "<AUTHORIZED_PERSON_MODEL_PT>" --line 0,0.6,1,0.6 --max-frames 900 --no-video --name local-review
```

The line uses normalized `x1,y1,x2,y2`. Direction is determined by endpoint order and must be checked for the camera. Defaults start fresh per run. The model must be a single-person detector; the counter does not add a class filter for arbitrary multiclass weights.

The default model path points into a training run. A separately obtained deployment copy can be supplied using `--weights`; it is not included in this handoff. For direct OpenVINO prediction, benchmark code uses `device="intel:cpu"`, but this device option is not exposed by the counter CLI. Confirm the export's actual input shape before use.

Store adaptation is conditional on owner-reviewed footage and labels. `prepare_labeling.py` creates draft boxes that must be corrected manually, especially missed seated/occluded people. `merge_store_data.py` uses contiguous time blocks. Its default ten-block calculation yields 20% test, 20% validation, and 60% train because `round(1.5)` is two. `train.py store` is configured, but the store-only evaluation and gallery described in the legacy guide are absent. Do not report them as completed.

### Verification and build boundaries

For source review, parse Python using `ast.parse`; the package validator additionally checks hashes, JSON, assets, and privacy. No automated repository test suite or code-coverage measurement exists. Cached counting checks confirm the stored predictions; they do not rerun the detector or validate new footage.

There is no frontend build, container image, CI pipeline, or deployment configuration in this repository. Model export is a separate optional build step that needs authorized weights. Generated reports and caches are outputs, not source dependencies supplied here. Some legacy report HTML links were already inconsistent and media is excluded; the HTML is not a standalone demo.

## Bahasa Indonesia

### Peta folder dan entry point

Peta dan tabel entry point di atas mencakup seluruh 16 modul: audit/pembersihan dataset, aturan keluarga scene, training/pemantauan, evaluasi/grafik, ekspor/benchmark, counting, persiapan/penggabungan data toko, perbandingan video, dan penampil laporan statis. Struktur relatif dipertahankan di `source-code/`; media laporan sengaja tidak disertakan.

### Peran dependensi

Runtime/model memakai Python, PyTorch, Ultralytics, OpenCV, NumPy, Supervision, dan LAP. Runtime inferensi opsional mencakup ONNX Runtime, OpenVINO, serta TensorRT. Evaluasi/pelaporan memakai pycocotools, Pillow, Matplotlib, dan alat ekspor ONNX. Alat pengembangan mencakup Git dan CLI Python biasa; repository tidak mengonfigurasi pipeline build paket atau framework tes otomatis.

Roboflow merupakan sumber dataset historis, sedangkan YOLO26s pretrained merupakan input model eksternal. Keduanya bukan layanan jaringan yang dipanggil pada setiap prediksi. Alat anotasi manual merupakan pilihan operator, bukan layanan yang dibundel. Tidak ada repository pendamping yang dibutuhkan oleh import yang diimplementasikan pada snapshot ini.

### Contoh instalasi, bukan tindakan saat ekspor

Tidak ada `requirements.txt`, lockfile, atau `pyproject.toml` asli. Tabel versi di atas mencatat distribusi terpasang yang diamati pada {DATE}; tabel tersebut bukan bukti bahwa setiap wheel tersedia pada mesin lain. Ekspor ini tidak menginstal dependensi atau mengubah Python global.

Gunakan contoh PowerShell di atas dari salinan kerja `source-code` terpisah untuk membuat `.venv`; pemanggilan interpreter secara langsung tidak memerlukan perubahan kebijakan keamanan shell. Untuk inferensi/training, instal dahulu wheel PyTorch/Torchvision yang kompatibel dengan platform. Workstation yang diamati memakai Torch `2.14.0+cu126` dan Torchvision `0.29.0+cu126`; README asli mencantumkan index paket cu126 resmi. Kemudian instal paket aplikasi/evaluasi yang dibutuhkan dan set `YOLO_AUTOINSTALL=False` sebelum mengimpor Ultralytics.

Ekspor opsional memerlukan alat ONNX/OpenVINO/TensorRT kompatibel beserta ketentuannya. Instalasi runtime tersebut tidak diperlukan hanya untuk membaca snapshot. Wheel CUDA yang diamati bukan perintah instalasi CPU universal.

### Konfigurasi dan eksekusi bersyarat

Ganti `<DATASET_ROOT>` pada YAML salinan dengan path absolut dataset format YOLO yang berhak digunakan. Placeholder lain seperti `<PROJECT_ROOT>`, `<ABSOLUTE_VIDEO_PATH>`, dan `<LEGACY_RUNS_DIR>` merupakan redaksi yang disengaja; ganti dengan nilai lokal sebelum memakai contohnya. Jangan menaruh kredensial di berkas tersebut.

Untuk pipeline asli, peroleh terpisah data mentah dan bobot pretrained yang sesuai serta berhak digunakan, tempatkan pada folder lokal yang diharapkan, lalu ikuti urutan perintah di atas dari root source. Perintah tersebut merupakan langkah reproduksi terdokumentasi, **bukan pemeriksaan yang dijalankan saat ekspor ini**. Dua perintah pertama memerlukan ekspor mentah asli lengkap. Pembersihan menulis ulang keluaran dataset dan tidak menghapus berkas split lama; gunakan salinan kerja baru agar berkas tertinggal tidak tercampur. Training memerlukan model/data yang tidak ada dalam ZIP. Run smoke melatih dua epoch pada sebagian dataset; ini bukan unit test.

Contoh counter di atas memerlukan video lokal berizin dan model PyTorch satu kelas person yang kompatibel. Garis memakai koordinat `x1,y1,x2,y2` ternormalisasi. Urutan titik menentukan arah dan harus diperiksa untuk kamera. Hitungan dimulai baru pada setiap run. Model harus merupakan detektor satu kelas person; counter tidak menambahkan filter kelas untuk bobot multikelas sembarang.

Path model default mengarah ke dalam run training. Salinan deployment yang diperoleh terpisah dapat diberikan melalui `--weights`; salinan tersebut tidak disertakan dalam handoff. Prediksi OpenVINO langsung pada kode benchmark memakai `device="intel:cpu"`, tetapi opsi device ini tidak tersedia pada CLI counter. Pastikan shape input aktual ekspor sebelum digunakan.

Adaptasi toko bergantung pada rekaman serta label yang ditinjau pemilik. `prepare_labeling.py` menghasilkan kotak draft yang harus dikoreksi manual, terutama orang duduk/tertutup yang terlewat. `merge_store_data.py` memakai blok waktu berurutan. Perhitungan default sepuluh blok menghasilkan test 20%, validation 20%, dan train 60% karena `round(1.5)` menjadi dua. `train.py store` sudah dikonfigurasi, tetapi evaluasi khusus toko dan galeri yang dijelaskan panduan lama belum ada. Jangan melaporkannya selesai.

### Batas verifikasi dan build

Untuk tinjauan source, parsing Python menggunakan `ast.parse`; validator paket juga memeriksa hash, JSON, aset, dan privasi. Tidak ada suite tes repository otomatis atau pengukuran code coverage. Pemeriksaan counting dari cache mengonfirmasi prediksi tersimpan; pemeriksaan tersebut tidak menjalankan ulang detektor atau memvalidasi rekaman baru.

Repository ini tidak memiliki build frontend, image container, pipeline CI, atau konfigurasi deployment. Ekspor model merupakan langkah build opsional terpisah yang memerlukan bobot berizin. Laporan dan cache yang dihasilkan adalah keluaran, bukan dependensi source yang disediakan di sini. Sebagian tautan HTML laporan lama sudah tidak sesuai dan media dikeluarkan; HTML tersebut bukan demo mandiri.

See / Lihat: [architecture](architecture.md), [API and data](api-and-data.md), [observed environment](environment-observed.json), [export notes](../source-code/EXPORT-NOTES.md).
''')
    write_text('technical/api-and-data.md','''# API and data contracts / Kontrak API dan data

## English

### Interfaces

There is no REST/GraphQL inference API, application database, ORM, schema migration, or backend service in the inspected repository. The public interfaces are Python CLI arguments and files. `serve_reports.py` serves static files and HTTP byte ranges; it is not a prediction endpoint or authenticated application backend. No endpoint or database schema is invented for this handoff.

### Data and coordinate contracts

| Artifact | Contract | Producer → consumer |
|---|---|---|
| Raw COCO export | Original image metadata, category IDs, bounding boxes | External dataset → audit/cleaning |
| Duplicate audit JSON | Filename-to-cluster assignments plus aggregate audit counts | `audit_duplicates.py` → `clean_cctv.py` |
| Clean COCO | Images plus annotations with category ID 1 (`person`), `bbox=[x,y,width,height]`, area, and `iscrowd=0` | Cleaning → evaluation |
| YOLO labels | Class index 0, then normalized center-x, center-y, width, height | Cleaning/manual labeling → training |
| Split manifest CSV | `file,scene,split,kind,density_bucket,boxes,copies_in_export,split_in_export` | Cleaning → training/evaluation/figures |
| Evaluation JSON | Aggregate detection/counting, confidence sweep, domain/density summaries, and per-image records | Evaluation → figures |
| Prediction NPZ | Per-image arrays of `x,y,width,height,confidence` | Evaluation → cached plotting/checks |
| Tracking CSV | Per-frame counts and cumulative line events, described below | Counter → operator/report analysis |
| Tracking JSON | Video/model references, settings, processing totals, occupancy aggregates, ID/crossing totals | Counter → report analysis |

For an image of width W and height H, the conversion implemented in cleaning is:

```text
center_x = (x + width / 2) / W
center_y = (y + height / 2) / H
box_width = width / W
box_height = height / H
```

COCO class ID 1 and YOLO class index 0 both represent person; they are different format conventions. The counting evaluator converts detector corners to COCO-style XYWH before matching.

### Tracking CSV fields

| Field | Meaning |
|---|---|
| `frame` | Zero-based frame index processed by the loop |
| `t_detik` | Rounded frame index divided by reported video FPS |
| `occupancy` | Number of boxes in the post-tracking result |
| `masuk` | Cumulative LineZone in-count for the configured direction |
| `keluar` | Cumulative LineZone out-count for the configured direction |
| `id_kumulatif` | Number of nonnegative tracker IDs seen in the run |

Counts reset for each invocation; the repository does not persist store occupancy across sessions. `masuk - keluar` is not implemented as a calibrated total of everyone inside the store. The initial visible population, people outside the camera field, line geometry, repeat crossings, missed detections, and identity fragmentation must be accounted for before interpreting footfall.

### Counting and matching formulas

For N evaluated frames, let e_i = predicted count minus annotated count. The code calculates MAE = sum(abs(e_i))/N, RMSE = sqrt(sum(e_i²)/N), bias = sum(e_i)/N, and within-one percent = 100 × count(abs(e_i) ≤ 1)/N. Precision and recall use confidence-ordered greedy matching with IoU ≥ 0.5. COCOeval supplies mAP independently across its IoU thresholds; it is not exact-count accuracy.

### What is actually bundled

Only the two dataset configuration YAML files are in the source snapshot; they contain `<DATASET_ROOT>`, not real dataset paths. Dataset manifests, raw annotations, image lists, prediction arrays, and recordings are excluded. `aggregate-evidence.json` contains reviewed aggregate values and SHA-256 fingerprints of unbundled inputs, without per-frame filenames or original recording paths. The installed-environment record is descriptive and contains no user profile path or device serial number.

The original reports and many ML scripts assume specific local artifacts. Their presence in source proves implementation, not that the ZIP can run those features without the omitted data and model. Store-test YAML generation exists, but no implemented evaluator consumes it to provide store-only counting metrics.

## Bahasa Indonesia

### Antarmuka

Repository yang diperiksa tidak memiliki API inferensi REST/GraphQL, database aplikasi, ORM, migrasi skema, atau service backend. Antarmukanya adalah argumen CLI Python dan berkas. `serve_reports.py` melayani berkas statis serta byte range HTTP; bukan endpoint prediksi atau backend aplikasi terautentikasi. Handoff ini tidak mengarang endpoint atau skema database.

### Kontrak data dan koordinat

Ekspor COCO mentah memasok metadata gambar, ID kategori, dan bounding box untuk audit/pembersihan. JSON audit berisi pemetaan nama berkas ke cluster dan jumlah agregat. COCO bersih memakai kategori ID 1 (`person`) serta `bbox=[x,y,width,height]`, area, dan `iscrowd=0`. Label YOLO memakai indeks kelas 0, diikuti pusat-x, pusat-y, lebar, dan tinggi ternormalisasi.

Manifest split memuat `file,scene,split,kind,density_bucket,boxes,copies_in_export,split_in_export`. JSON evaluasi memuat metrik deteksi/counting, sweep confidence, ringkasan domain/kepadatan, dan catatan per gambar. NPZ prediksi menyimpan array per gambar berupa `x,y,width,height,confidence`. CSV tracking menyimpan hitungan per frame serta lintasan garis kumulatif; JSON tracking menyimpan referensi video/model, setelan, total pemrosesan, agregat occupancy, dan jumlah ID/lintasan.

Untuk gambar berlebar W dan tinggi H, rumus konversi di atas sesuai implementasi pembersihan: `(x + width/2)/W`, `(y + height/2)/H`, `width/W`, dan `height/H`. ID kelas COCO 1 dan indeks YOLO 0 sama-sama mewakili person dengan konvensi format berbeda. Evaluator mengubah sudut kotak detektor menjadi XYWH COCO sebelum pencocokan.

### Kolom CSV tracking

- `frame`: indeks frame berbasis nol yang diproses loop.
- `t_detik`: pembulatan indeks frame dibagi FPS video yang dilaporkan.
- `occupancy`: jumlah kotak dalam hasil setelah tracking.
- `masuk`: in-count LineZone kumulatif menurut arah yang dikonfigurasi.
- `keluar`: out-count LineZone kumulatif menurut arah yang dikonfigurasi.
- `id_kumulatif`: jumlah ID tracker nonnegatif yang terlihat selama run.

Hitungan direset pada setiap pemanggilan; repository tidak menyimpan occupancy toko lintas sesi. `masuk - keluar` tidak diimplementasikan sebagai total terkalibrasi semua orang di dalam toko. Populasi awal yang terlihat, orang di luar bidang kamera, geometri garis, lintasan berulang, deteksi terlewat, dan fragmentasi identitas perlu diperhitungkan sebelum menafsirkan footfall.

### Rumus counting dan pencocokan

Untuk N frame evaluasi, e_i adalah hitungan prediksi dikurangi anotasi. Kode menghitung MAE = jumlah(abs(e_i))/N, RMSE = akar(jumlah(e_i²)/N), bias = jumlah(e_i)/N, serta within-one persen = 100 × jumlah(abs(e_i) ≤ 1)/N. Precision dan recall memakai greedy matching menurut urutan confidence dengan IoU ≥ 0,5. COCOeval menyediakan mAP secara terpisah pada threshold IoU-nya; mAP bukan akurasi hitungan tepat.

### Isi yang benar-benar dibundel

Hanya dua YAML konfigurasi dataset yang ada dalam snapshot source; isinya `<DATASET_ROOT>`, bukan path dataset nyata. Manifest dataset, anotasi mentah, daftar gambar, array prediksi, dan rekaman dikeluarkan. `aggregate-evidence.json` berisi nilai agregat yang ditinjau dan fingerprint SHA-256 input yang tidak dibundel, tanpa nama berkas per frame atau path rekaman asli. Catatan lingkungan terpasang bersifat deskriptif serta tidak memuat path profil pengguna atau nomor seri perangkat.

Laporan asli dan banyak skrip ML mengasumsikan artefak lokal tertentu. Keberadaannya pada source membuktikan implementasi, bukan bahwa ZIP dapat menjalankan fitur tanpa data dan model yang dikeluarkan. Pembuatan YAML test toko tersedia, tetapi belum ada evaluator yang memakainya untuk menyediakan metrik counting khusus toko.
''')
    write_text('technical/third-party-notices.md','''# Provenance and third-party boundaries / Asal dan batas pihak ketiga

## English

Naufal Firdaus confirmed that he owns the victoriavals account, created the custom project, and holds the right to share its custom source and project charts for this portfolio handoff. This is an owner statement, not a conclusion inferred from commit names. No new LICENSE file or SPDX license is assigned to the custom project. GitHub metadata and the local tree show no declared repository license.

The implementation imports external packages; their implementations, installed wheels, font files, and binaries are not bundled. Selected installed metadata identifies Ultralytics as AGPL-3.0, Supervision as MIT, OpenCV as Apache 2.0, ONNX as Apache-2.0, ONNX Runtime as MIT, OpenVINO as Apache-2.0, and TensorRT as Proprietary. Torch metadata contains a composite SPDX expression for bundled components. These are observations of installed metadata, not a legal assessment, compatibility determination, or permission to relicense dependencies.

The model workflow initializes pretrained YOLO26s weights and fine-tunes on an external Roboflow dataset. The deployment model metadata records end-to-end person detection and a 640-pixel OpenVINO export. The dataset's redistribution terms were not established in the repository. Accordingly all pretrained/trained weights, compiled graphs, annotations, imagery, and recordings are excluded, even where some are already committed publicly. Public availability is not treated as rights clearance.

Included experiment charts contain aggregate scientific/engineering results from the owner's project; no source photo, identifiable person, camera frame, or per-image label file is included. The new cover is a code-derived diagram. The original report HTML sources remain historical source examples and are not cleared footage galleries or ready-to-publish demos. Public source links target the reviewed code/README; no direct media, dataset-download, or warehouse-recording link is supplied.

## Bahasa Indonesia

Naufal Firdaus mengonfirmasi bahwa ia pemilik akun victoriavals, pembuat proyek custom, dan pemegang hak untuk membagikan source custom serta grafik proyek untuk handoff portfolio ini. Ini pernyataan pemilik, bukan kesimpulan dari nama commit. Tidak ada berkas LICENSE baru atau lisensi SPDX yang ditetapkan untuk proyek custom. Metadata GitHub dan tree lokal menunjukkan tidak adanya lisensi repository yang dideklarasikan.

Implementasi mengimpor paket eksternal; implementasi paket, wheel terpasang, berkas font, dan binary-nya tidak dibundel. Metadata terpasang terpilih mengidentifikasi Ultralytics sebagai AGPL-3.0, Supervision sebagai MIT, OpenCV sebagai Apache 2.0, ONNX sebagai Apache-2.0, ONNX Runtime sebagai MIT, OpenVINO sebagai Apache-2.0, dan TensorRT sebagai Proprietary. Metadata Torch memuat ekspresi SPDX gabungan untuk komponen yang dibundel. Ini pengamatan metadata terpasang, bukan penilaian hukum, penetapan kompatibilitas, atau izin mengganti lisensi dependensi.

Alur model menginisialisasi bobot YOLO26s pretrained dan melakukan fine-tuning pada dataset Roboflow eksternal. Metadata model deployment mencatat deteksi person end-to-end dan ekspor OpenVINO 640 piksel. Ketentuan distribusi ulang dataset belum ditetapkan di repository. Karena itu, seluruh bobot pretrained/terlatih, graph kompilasi, anotasi, gambar, dan rekaman dikeluarkan, termasuk yang sudah tersimpan publik. Ketersediaan publik tidak dianggap sebagai kepastian hak distribusi.

Grafik eksperimen yang disertakan memuat hasil ilmiah/engineering agregat dari proyek pemilik; tidak ada foto sumber, orang yang dapat dikenali, frame kamera, atau berkas label per gambar. Sampul baru merupakan diagram berdasarkan kode. Source HTML laporan asli tetap berupa contoh source historis, bukan galeri rekaman berizin atau demo siap publikasi. Tautan source publik menargetkan kode/README yang ditinjau; tidak ada tautan media, unduhan dataset, atau rekaman warehouse langsung.
''')
