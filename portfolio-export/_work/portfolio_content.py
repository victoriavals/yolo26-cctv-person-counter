"""Bilingual portfolio content derived from the reviewed repository."""
from pathlib import Path
import json
from build_export import DEST, ROOT, DATE, COMMIT, REPO_URL, pair, sha, write_text, write_json


def entry(en, id, ids):
    return {'text': pair(en, id), 'evidenceIds': ids}


def make_content(evidence, assets, source_records):
    features = [
        entry('Audit augmented duplicates and rebuild a single-person dataset with scene-grouped splits.',
              'Mengaudit duplikasi augmentasi dan membangun ulang dataset satu kelas person dengan split berbasis scene.', ['E02','E03']),
        entry('Fine-tune pretrained YOLO26s and compare default, reduced-augmentation, and indoor-only experiments.',
              'Melakukan fine-tuning YOLO26s pretrained dan membandingkan eksperimen default, augmentasi dikurangi, serta indoor saja.', ['E04','E05']),
        entry('Evaluate detection and per-frame counting using mAP, precision, recall, MAE, RMSE, bias, and within-one-person results.',
              'Mengevaluasi deteksi dan hitungan per frame menggunakan mAP, precision, recall, MAE, RMSE, bias, dan hasil dengan selisih maksimal satu orang.', ['E06','E07']),
        entry('Track detections with ByteTrack or BoT-SORT and count directional line crossings in recorded video.',
              'Melacak deteksi dengan ByteTrack atau BoT-SORT dan menghitung lintasan garis berdasarkan arah pada rekaman video.', ['E08']),
        entry('Export models and compare PyTorch, ONNX Runtime, OpenVINO, and TensorRT on local CPU and GPU hardware.',
              'Mengekspor model dan membandingkan PyTorch, ONNX Runtime, OpenVINO, serta TensorRT pada CPU dan GPU lokal.', ['E09','E10']),
        entry('Sample store-camera frames, generate draft labels, and merge manually corrected labels for a later fine-tuning run.',
              'Mengambil sampel frame kamera toko, membuat label draft, dan menggabungkan label yang dikoreksi manual untuk fine-tuning lanjutan.', ['E11']),
    ]
    focus = [
        entry('Treat data leakage and mixed annotations as engineering problems before comparing model scores.',
              'Menangani kebocoran data dan anotasi campuran sebagai persoalan engineering sebelum membandingkan skor model.', ['E02','E03']),
        entry('Measure counting error separately from object-detection quality and tracking identity counts.',
              'Mengukur galat hitungan secara terpisah dari kualitas deteksi objek dan jumlah identitas tracking.', ['E06','E08']),
        entry('Compare inference runtimes using exported graph sizes, warm-up, sampled detection matching, and local timing.',
              'Membandingkan runtime inferensi dengan ukuran graph ekspor, warm-up, kecocokan deteksi sampel, dan pengukuran waktu lokal.', ['E09','E10','E12']),
        entry('Document the gap between public-dataset results and the behavior of seated or partially occluded people on a store-like camera.',
              'Mendokumentasikan kesenjangan antara hasil dataset publik dan perilaku orang duduk atau tertutup sebagian pada kamera yang menyerupai toko.', ['E13']),
    ]
    implementations = [
        {'feature':pair('Dataset reconstruction','Rekonstruksi dataset'),
         'paths':['tools/audit_duplicates.py','tools/clean_cctv.py','tools/families.py'],
         'explanation':pair('A histogram-based audit clusters copies within filename stems; cleaning keeps one representative, maps person categories, drops shoe annotations and non-CCTV families, and exports COCO and normalized YOLO labels.',
                            'Audit berbasis histogram mengelompokkan salinan dalam stem nama berkas; pembersihan mempertahankan satu representasi, memetakan kategori orang, membuang anotasi sepatu dan keluarga non-CCTV, lalu mengekspor COCO dan label YOLO ternormalisasi.'),
         'evidenceIds':['E02','E03']},
        {'feature':pair('Experiment configuration','Konfigurasi eksperimen'), 'paths':['tools/train.py','tools/status.py'],
         'explanation':pair('COMMON and RUNS define shared training settings and explicit overrides. The progress monitor reads existing CSV logs for the three public-data experiments.',
                            'COMMON dan RUNS mendefinisikan setelan training bersama serta override eksplisit. Pemantau progres membaca log CSV yang tersedia untuk tiga eksperimen data publik.'), 'evidenceIds':['E04','E05']},
        {'feature':pair('Evaluation and reporting','Evaluasi dan pelaporan'), 'paths':['tools/evaluate.py','tools/make_eval_figures.py','tools/make_figures.py'],
         'explanation':pair('COCO evaluation measures detection; confidence-ordered matching and count differences measure counting. Predictions are cached as NPZ for report figures without repeated inference.',
                            'Evaluasi COCO mengukur deteksi; pencocokan berdasarkan confidence dan selisih jumlah mengukur counting. Prediksi disimpan sebagai NPZ untuk gambar laporan tanpa inferensi berulang.'), 'evidenceIds':['E06','E07']},
        {'feature':pair('Video counting','Counting video'), 'paths':['tools/count_people.py'],
         'explanation':pair('model.track uses persistent streaming results. Supervision LineZone maintains in/out crossing totals; the script writes per-frame CSV, summary JSON, and optional annotated MP4.',
                            'model.track menggunakan hasil streaming yang persisten. Supervision LineZone menyimpan total lintasan masuk/keluar; skrip menulis CSV per frame, JSON ringkasan, dan MP4 anotasi opsional.'), 'evidenceIds':['E08']},
        {'feature':pair('Runtime experiments','Eksperimen runtime'), 'paths':['tools/export.py','tools/benchmark_backends.py','tools/compare_backends_video.py','tools/make_backend_figure.py'],
         'explanation':pair('Exports are staged by format and input size. The comparison tools collect prediction timings and cached video detections; the video renderer does not provide tracking ground truth.',
                            'Ekspor dipisahkan menurut format dan ukuran input. Alat perbandingan mengumpulkan waktu prediksi dan deteksi video tersimpan; perender video tidak menyediakan ground truth tracking.'), 'evidenceIds':['E09','E10']},
        {'feature':pair('Store adaptation and static reports','Adaptasi toko dan laporan statis'),
         'paths':['tools/prepare_labeling.py','tools/merge_store_data.py','tools/serve_reports.py','reports/'],
         'explanation':pair('Frame sampling, draft labeling, time-block merging, and a store training configuration exist. Store-specific evaluation remains missing. The HTTP helper only serves static report files.',
                            'Pengambilan sampel frame, label draft, penggabungan blok waktu, dan konfigurasi training toko tersedia. Evaluasi khusus toko belum tersedia. Helper HTTP hanya melayani berkas laporan statis.'), 'evidenceIds':['E11','E14','E15']},
    ]
    decisions = [
        {'decision':pair('Deduplicate before splitting','Dedup sebelum split'),
         'rationale':pair('The source audit reports 10,062 files representing 4,886 photo clusters and copies shared across the original splits.',
                          'Audit sumber melaporkan 10.062 berkas yang mewakili 4.886 cluster foto serta salinan yang tersebar pada split awal.'),
         'tradeoff':pair('The 0.97 histogram-similarity rule is heuristic and is applied within filename-stem groups; it does not prove universal deduplication.',
                         'Aturan kemiripan histogram 0,97 bersifat heuristik dan diterapkan dalam kelompok stem nama berkas; ini tidak membuktikan deduplikasi universal.'), 'evidenceIds':['E02']},
        {'decision':pair('Group splits by scene and stratify by domain and density','Mengelompokkan split menurut scene serta stratifikasi domain dan kepadatan'),
         'rationale':pair('The intended 80/10/10 allocation keeps frames from a named scene together while balancing the target indoor share.',
                          'Alokasi 80/10/10 yang dituju menjaga frame dari scene bernama tetap bersama sambil menyeimbangkan proporsi indoor sasaran.'),
         'tradeoff':pair('The 142 semi/entrance images are all in train; entrance-specific performance is not measured on validation or test.',
                         'Seluruh 142 gambar semi/pintu masuk berada pada train; performa khusus pintu masuk tidak diukur pada validation atau test.'), 'evidenceIds':['E03']},
        {'decision':pair('Retain the default experiment','Mempertahankan eksperimen default'),
         'rationale':pair('The recorded default result has lower aggregate counting MAE than the reduced-augmentation and indoor-only runs.',
                          'Hasil default yang tercatat memiliki MAE counting agregat lebih rendah daripada run augmentasi dikurangi dan indoor saja.'),
         'tradeoff':pair('These are recorded experiments with a single configured seed, not a repeated-run statistical comparison or a universal rule about augmentation.',
                         'Ini adalah eksperimen tercatat dengan satu seed konfigurasi, bukan perbandingan statistik run berulang atau aturan universal mengenai augmentasi.'), 'evidenceIds':['E04','E07']},
        {'decision':pair('Choose confidence using counting error','Memilih confidence berdasarkan galat hitungan'),
         'rationale':pair('The product goal is counting; the evaluator compares MAE and F1 over confidence thresholds.',
                          'Tujuan produk adalah counting; evaluator membandingkan MAE dan F1 pada berbagai threshold confidence.'),
         'tradeoff':pair('The stored operating point was selected on the same test split. A future evaluation should choose it on validation before testing.',
                         'Titik operasi tersimpan dipilih pada split test yang sama. Evaluasi berikutnya sebaiknya memilihnya pada validation sebelum test.'), 'evidenceIds':['E06','E07']},
        {'decision':pair('Stage exports and inspect fixed graph dimensions','Memisahkan ekspor dan memeriksa dimensi graph tetap'),
         'rationale':pair('Ultralytics exports beside the weights; later resolutions can overwrite earlier files. The inspected copies include both 640 and 416 pixel OpenVINO graphs.',
                          'Ultralytics mengekspor di samping bobot; resolusi berikutnya dapat menimpa berkas sebelumnya. Salinan yang diperiksa mencakup graph OpenVINO 640 dan 416 piksel.'),
         'tradeoff':pair('An imgsz argument does not replace the dimensions baked into a fixed export. Device and graph selection still need explicit checking.',
                         'Argumen imgsz tidak mengganti dimensi yang sudah tertanam pada ekspor tetap. Pemilihan device dan graph tetap perlu diperiksa secara eksplisit.'), 'evidenceIds':['E09','E12']},
        {'decision':pair('Use local runtime measurements to guide backend choice','Menggunakan pengukuran runtime lokal untuk memandu pilihan backend'),
         'rationale':pair('The historical sample favors OpenVINO for this CPU and TensorRT for this GPU; ONNX Runtime CPU was substantially slower on this workstation.',
                          'Sampel historis lebih menguntungkan OpenVINO untuk CPU ini dan TensorRT untuk GPU ini; ONNX Runtime CPU jauh lebih lambat pada workstation ini.'),
         'tradeoff':pair('The 40-image parity result is sample-specific; crowded-video caches show different detection totals. Prediction throughput is not whole-pipeline throughput.',
                         'Hasil kesamaan pada 40 gambar bersifat khusus sampel; cache video ramai menunjukkan total deteksi berbeda. Throughput prediksi bukan throughput seluruh pipeline.'), 'evidenceIds':['E10']},
    ]
    limitations = [
        entry('Real-camera counting is not validated for operational use. Historical warehouse output undercounts compared with the visual estimate in the README.',
              'Counting pada kamera nyata belum tervalidasi untuk penggunaan operasional. Keluaran warehouse historis menghitung lebih sedikit daripada perkiraan visual pada README.', ['E13']),
        entry('The dataset has no track IDs, no empty images, and limited target-camera coverage. Unique tracker IDs do not establish unique visitors.',
              'Dataset tidak memiliki track ID, gambar kosong, dan cakupan kamera sasaran yang memadai. ID tracker unik tidak membuktikan pengunjung unik.', ['E03','E08']),
        entry('Store sampling, merging, and training configuration exist, but store-only evaluation, its error gallery, and store progress monitoring are not implemented.',
              'Pengambilan sampel, penggabungan, dan konfigurasi training toko tersedia, tetapi evaluasi khusus toko, galeri error-nya, dan pemantauan progres toko belum diimplementasikan.', ['E11','E14']),
        entry('The counter accepts local video files and lacks explicit device and image-size CLI options. Occupancy is read from tracked results and can be influenced by tracker filtering.',
              'Counter menerima berkas video lokal dan belum memiliki opsi CLI device serta ukuran gambar secara eksplisit. Occupancy dibaca dari hasil tracking dan dapat dipengaruhi penyaringan tracker.', ['E08']),
        entry('The fallback shared ID of -1 can create false crossings. The store learning-rate override can be ignored by optimizer=auto in the installed Ultralytics version.',
              'ID cadangan bersama berupa -1 dapat menghasilkan lintasan palsu. Override learning rate toko dapat diabaikan oleh optimizer=auto pada versi Ultralytics yang terpasang.', ['E16','E17']),
        entry('Legacy dataset paths and some reports are stale. There is no original dependency manifest, automated test suite, or verified deployment configuration.',
              'Path dataset lama dan sebagian laporan sudah tertinggal. Tidak ada manifest dependensi asli, suite tes otomatis, atau konfigurasi deployment terverifikasi.', ['E14','E15','E18']),
        entry('Dataset redistribution rights are unverified. This handoff excludes source imagery, video, annotations, predictions, and all model binaries, so training and inference cannot be reproduced from the ZIP alone.',
              'Hak distribusi dataset belum terverifikasi. Handoff ini mengeluarkan gambar sumber, video, anotasi, prediksi, dan seluruh binary model, sehingga training dan inferensi tidak dapat direproduksi hanya dari ZIP.', ['E19']),
    ]
    proof = [
        {'type':'reported','title':pair('Recorded test-split detector result','Hasil detektor pada split test yang tercatat'),
         'text':pair('Default: mAP50 0.8907, mAP50–95 0.5842, confidence 0.25, 475 images and 1,844 annotated people. mAP was read from the historical report, not recomputed from a new inference run.',
                     'Default: mAP50 0,8907, mAP50–95 0,5842, confidence 0,25, 475 gambar dan 1.844 anotasi orang. mAP dibaca dari laporan historis, bukan dihitung ulang dari inferensi baru.'), 'evidenceIds':['E07'],'assetIds':['asset-02','asset-03']},
        {'type':'observed','title':pair('Cached counting metrics independently checked','Metrik counting tersimpan diperiksa ulang'),
         'text':pair('Recalculation from cached predictions matches the stored counting and precision/recall metrics for seven model/size outputs. Default MAE is 0.640, within-one is 88.0%, and exact-count share is 55.58%. This checks cached outputs, not current live-camera inference.',
                     'Penghitungan ulang dari prediksi tersimpan cocok dengan metrik counting serta precision/recall tersimpan untuk tujuh keluaran model/ukuran. MAE default adalah 0,640, within-one 88,0%, dan proporsi hitungan tepat 55,58%. Ini memeriksa keluaran tersimpan, bukan inferensi kamera langsung saat ini.'), 'evidenceIds':['E06','E07'],'assetIds':['asset-03','asset-04']},
        {'type':'observed','title':pair('Dataset and artifact checks','Pemeriksaan dataset dan artefak'),
         'text':pair('Local checks find 4,746 images, 18,445 boxes, no scene assigned to multiple splits in the manifest, matching COCO/YOLO labels, and matching committed model checksums. The unbundled inputs are identified by hashes.',
                     'Pemeriksaan lokal menemukan 4.746 gambar, 18.445 kotak, tidak ada scene yang ditetapkan pada beberapa split di manifest, label COCO/YOLO yang cocok, dan checksum model tersimpan yang cocok. Input yang tidak disertakan diidentifikasi dengan hash.'), 'evidenceIds':['E03','E12'],'assetIds':[]},
        {'type':'reported','title':pair('Historical local runtime comparison','Perbandingan runtime lokal historis'),
         'text':pair('On the recorded 40-image, 640-pixel benchmark, OpenVINO CPU is about 19.9 FPS and TensorRT GPU about 78.3 FPS. These are local prediction-call measurements, not tracking accuracy or production throughput.',
                     'Pada benchmark tercatat 40 gambar dan 640 piksel, OpenVINO CPU sekitar 19,9 FPS dan TensorRT GPU sekitar 78,3 FPS. Ini pengukuran pemanggilan prediksi lokal, bukan akurasi tracking atau throughput produksi.'), 'evidenceIds':['E10'],'assetIds':['asset-05']},
    ]
    project = {
        'schemaVersion':'1.0','publicationMode':'public','defaultLanguage':'en','languages':['en','id'],
        'slug':'yolo26-cctv-person-counter', 'title':pair('CCTV Person Counter','Penghitung Orang CCTV'),
        'domain':pair('Computer Vision · Retail Analytics','Visi Komputer · Analitik Ritel'),
        'summary':pair('A computer vision pipeline for counting people in recorded indoor CCTV footage. It combines YOLO26s fine-tuning, tracking, line-crossing counts, dataset auditing, and runtime benchmarking, with documented limits on real-camera performance.',
                       'Pipeline visi komputer untuk menghitung orang pada rekaman CCTV dalam ruangan. Proyek ini mencakup fine-tuning YOLO26s, tracking, penghitungan lintasan, audit dataset, dan benchmark runtime, dengan keterbatasan performa pada kamera nyata yang didokumentasikan.'),
        'stack':['Python','PyTorch','Ultralytics YOLO26s','OpenCV','Supervision','OpenVINO','TensorRT'],
        'purpose':pair('Investigate per-frame person counting and directional footfall from a fixed indoor CCTV viewpoint.',
                       'Mengkaji hitungan orang per frame dan lintasan pengunjung berdasarkan arah dari sudut pandang CCTV indoor tetap.'),
        'targetUsers':pair('Store operators and computer-vision developers evaluating recorded CCTV footage.',
                           'Operator toko dan pengembang visi komputer yang mengevaluasi rekaman CCTV.'),
        'status':pair('Research prototype; store-camera validation incomplete.','Prototipe riset; validasi kamera toko belum lengkap.'),
        'problem':pair('A useful store counter must distinguish the number visible now from line-crossing events and tracker identities. The initial CCTV dataset contained augmented duplicates, mixed categories, and split leakage; strong test scores also failed to predict the behavior of seated and occluded people in warehouse footage.',
                       'Counter toko yang berguna perlu membedakan jumlah yang terlihat saat ini dari peristiwa melintasi garis dan identitas tracker. Dataset CCTV awal memuat duplikasi augmentasi, kategori campuran, dan kebocoran split; skor test tinggi juga belum memprediksi perilaku orang duduk dan tertutup pada rekaman warehouse.'),
        'personalContribution':pair('Naufal Firdaus designed and implemented the custom project pipeline, including dataset auditing and cleaning, experiment configuration, evaluation, export and runtime comparisons, video counting, and store-data preparation. Authorship and the victoriavals account identity were confirmed by the owner. The project uses third-party libraries, pretrained YOLO26s weights, and a Roboflow dataset; the detector is fine-tuned rather than trained from scratch.',
                                    'Naufal Firdaus merancang dan mengimplementasikan pipeline custom proyek, termasuk audit dan pembersihan dataset, konfigurasi eksperimen, evaluasi, ekspor dan perbandingan runtime, counting video, serta persiapan data toko. Kepengarangan dan identitas akun victoriavals dikonfirmasi oleh pemilik. Proyek memakai library pihak ketiga, bobot YOLO26s pretrained, dan dataset Roboflow; detektor melalui fine-tuning, bukan training dari awal.'),
        'contributionEvidence':{'type':'reported','source':'owner-confirmation','evidenceIds':['E01']},
        'features':features, 'engineeringFocus':focus,
        'architecture':pair('Standalone Python CLI stages exchange files. The development branch audits and cleans data, fine-tunes pretrained YOLO26s, evaluates outputs, and exports models. The video branch loads a model backend, tracks detections with ByteTrack or BoT-SORT, updates a Supervision LineZone, and writes CSV, JSON, and optional annotated video. A static report server is a viewing helper, not an inference API.',
                            'Tahap CLI Python mandiri bertukar berkas. Cabang pengembangan mengaudit dan membersihkan data, melakukan fine-tuning YOLO26s pretrained, mengevaluasi keluaran, dan mengekspor model. Cabang video memuat backend model, melacak deteksi dengan ByteTrack atau BoT-SORT, memperbarui Supervision LineZone, dan menulis CSV, JSON, serta video anotasi opsional. Server laporan statis adalah helper tampilan, bukan API inferensi.'),
        'implementation':implementations,'technicalDecisions':decisions,'proof':proof,'limitations':limitations,
        'aiDetails':{
            'input':pair('COCO-format training images and local recorded video files.','Gambar training berformat COCO dan berkas rekaman video lokal.'),
            'preprocessing':pair('Histogram-based duplicate audit, person-category mapping, scene-grouped splits, normalized YOLO boxes, and model image preparation handled by Ultralytics.',
                                 'Audit duplikasi berbasis histogram, pemetaan kategori person, split berbasis scene, kotak YOLO ternormalisasi, dan persiapan gambar model melalui Ultralytics.'),
            'model':pair('Single-class, end-to-end YOLO26s person detector initialized from pretrained weights.','Detektor person YOLO26s end-to-end satu kelas yang diinisialisasi dari bobot pretrained.'),
            'training':pair('Fine-tuning experiments are implemented and historical run artifacts are present. Training from random initialization is not the workflow.',
                            'Eksperimen fine-tuning diimplementasikan dan artefak run historis tersedia. Training dari inisialisasi acak bukan alur yang digunakan.'),
            'inference':pair('Ultralytics detection/tracking with a selected PyTorch or exported runtime; the counter requests 640-pixel inference and confidence 0.25 by default.',
                             'Deteksi/tracking Ultralytics dengan runtime PyTorch atau ekspor yang dipilih; counter meminta inferensi 640 piksel dan confidence 0,25 secara default.'),
            'output':pair('Person boxes, tracking IDs, per-frame tracked-result counts, directional crossing totals, CSV/JSON summaries, and optional video overlays.',
                          'Kotak orang, ID tracking, hitungan hasil tracking per frame, total lintasan berdasarkan arah, ringkasan CSV/JSON, dan overlay video opsional.'),
            'retrieval':None,
            'retrievalExplanation':pair('No retrieval system, language model, or vector database is implemented.','Tidak ada sistem retrieval, model bahasa, atau database vektor yang diimplementasikan.')},
        'sources':[
            {'id':'source-code','label':pair('Python source at reviewed commit','Source Python pada commit yang ditinjau'), 'url':f'{REPO_URL}/tree/{COMMIT}/tools','type':'source','verifiedPublic':True},
            {'id':'source-readme','label':pair('Project README at reviewed commit','README proyek pada commit yang ditinjau'), 'url':f'{REPO_URL}/blob/{COMMIT}/README.md','type':'documentation','verifiedPublic':True},
            {'id':'package-evidence','label':pair('Sanitized aggregate evidence','Bukti agregat yang disanitasi'),'packagePath':'technical/aggregate-evidence.json','type':'evidence'},
        ],
        'demo':None,'demoUnavailableReason':pair('No public application demo was verified. The local HTML pages are historical report viewers with omitted media dependencies.',
                                               'Tidak ada demo aplikasi publik yang terverifikasi. Halaman HTML lokal adalah penampil laporan historis dengan dependensi media yang dikeluarkan.'),
        'coverAsset':'asset-01','recommendedWebsiteAssets':['asset-01','asset-02','asset-05'],
        'snapshot':evidence['snapshot'],
        'licenseStatus':{'repositoryLicenseSpdx':None,
            'customMaterialDistribution':pair('The owner authorized preparation and sharing of this custom-code portfolio handoff. No new license or general third-party reuse grant is assigned by this export.',
                                              'Pemilik mengizinkan penyiapan dan pembagian handoff portfolio kode custom ini. Ekspor ini tidak menetapkan lisensi baru atau izin umum penggunaan ulang oleh pihak ketiga.'),
            'datasetAndModelRedistribution':'unverified-excluded'},
    }
    write_json('project.json',project)
    for lang in ('en','id'):
        write_text(f'case-study.{lang}.md',make_case_study(project,evidence,lang))
    make_readme(project)
    make_technical(project,evidence)
    make_internal(project,evidence)
    make_export_notes(source_records)


def make_case_study(p,e,lang):
    is_en=lang=='en'
    titles = ['Overview','Problem','Personal contribution','Features and workflow','Engineering focus','Architecture','Implementation','Technical decisions','Evidence and results','Limitations'] if is_en else ['Gambaran umum','Masalah','Kontribusi pribadi','Fitur dan alur penggunaan','Fokus engineering','Arsitektur','Implementasi','Keputusan teknis','Bukti dan hasil','Keterbatasan']
    lines=[f'# {p["title"][lang]}', '', p['summary'][lang], '', p['status'][lang], '']
    for n,title in enumerate(titles,1):
        lines += [f'## {n}. {title}','']
        if n==1: lines += [p['purpose'][lang], '', p['targetUsers'][lang], '',
                           ('This is an offline computer-vision project composed of Python command-line programs. Its report pages present experiment outputs; they are not a deployed store-management application.' if is_en else 'Ini adalah proyek visi komputer offline yang terdiri dari program command-line Python. Halaman laporannya menyajikan keluaran eksperimen; bukan aplikasi manajemen toko yang sudah di-deploy.'),'']
        elif n==2: lines += [p['problem'][lang],'']
        elif n==3: lines += [p['personalContribution'][lang],'']
        elif n==4:
            lines += ['- '+f['text'][lang] for f in p['features']]
            lines += ['',('A user first prepares authorized footage and a matching single-person model, then selects a counting line. The counter processes successive frames and produces counts and optional overlays. Draft labels for adaptation require manual correction before training.' if is_en else 'Pengguna terlebih dahulu menyiapkan rekaman yang berhak digunakan dan model satu kelas person yang sesuai, lalu menentukan garis hitung. Counter memproses frame berurutan dan menghasilkan hitungan serta overlay opsional. Label draft untuk adaptasi perlu dikoreksi manual sebelum training.'),'',
                      ('Occupancy, crossing totals, and cumulative tracker IDs are separate outputs. The same person can receive a new tracking ID after a lost association, so cumulative IDs are not validated unique-visitor counts.' if is_en else 'Occupancy, total lintasan, dan ID tracker kumulatif merupakan keluaran terpisah. Orang yang sama dapat memperoleh ID baru setelah asosiasinya hilang, sehingga ID kumulatif bukan hitungan pengunjung unik yang tervalidasi.'),'']
        elif n==5: lines += ['- '+f['text'][lang] for f in p['engineeringFocus']] + ['']
        elif n==6:
            lines += [p['architecture'][lang],'', '![{}](assets/01-person-counter-pipeline.webp)'.format('Code-derived pipeline diagram' if is_en else 'Diagram pipeline berdasarkan kode'),'',
                      ('The diagram is reconstructed documentation of code relationships, not a screenshot of a running deployment.' if is_en else 'Diagram merupakan dokumentasi yang disusun dari hubungan dalam kode, bukan screenshot deployment yang berjalan.'),'']
        elif n==7:
            for f in p['implementation']:
                lines += [f'**{f["feature"][lang]}.** {f["explanation"][lang]}', '', '`'+'`, `'.join(f['paths'])+'`','']
        elif n==8:
            for f in p['technicalDecisions']:
                lines += [f'**{f["decision"][lang]}.** {f["rationale"][lang]} {f["tradeoff"][lang]}','']
        elif n==9:
            lines += [('| Model | Confidence | mAP50 | Counting MAE | Within ±1 |' if is_en else '| Model | Confidence | mAP50 | MAE hitungan | Dalam ±1 |'), '|---|---:|---:|---:|---:|']
            for name in ('baseline','default','tuned','indoor'):
                r=e['evaluation'][name]; c=r['reportedCounting']; lines += [f'| {name} | {r["confidence"]:.2f} | {r["reportedDetection"]["mAP50"]:.3f} | {c["MAE"]:.3f} | {c["within1_pct"]:.1f}% |']
            lines += ['',('The table reports historical test results on 475 images with 1,844 boxes. Confidence was tuned on the test set, so these are exploratory operating-point results. The mAP figures were not regenerated during this handoff.' if is_en else 'Tabel melaporkan hasil test historis pada 475 gambar dengan 1.844 kotak. Confidence di-tuning pada test, sehingga ini merupakan hasil titik operasi eksploratif. Angka mAP tidak dihasilkan ulang saat handoff ini.'),'']
            for f in p['proof']:
                lines += [f'**{f["title"][lang]}.** {f["text"][lang]}','']
            lines += [('Historical field check: a 900-frame warehouse demo averaged 2.35 tracked-result detections and 40 cumulative track IDs. The README describes 5–7 visible people and about six individuals by visual inspection. This is evidence of a domain gap, not a labeled tracking-accuracy measurement.' if is_en else 'Pemeriksaan lapangan historis: demo warehouse 900 frame mencatat rata-rata 2,35 deteksi hasil tracking dan 40 ID track kumulatif. README menyebut 5–7 orang terlihat dan sekitar enam individu berdasarkan inspeksi visual. Ini bukti kesenjangan domain, bukan pengukuran akurasi tracking berlabel.'),'',
                      ('No new training, model inference, or complete video-counting run was performed for this portfolio export. Source checks and cached-output verification are recorded separately in the evidence ledger.' if is_en else 'Tidak ada training, inferensi model, atau run counting video lengkap yang baru untuk ekspor portfolio ini. Pemeriksaan source dan verifikasi keluaran tersimpan dicatat terpisah dalam daftar bukti.'),'']
        else: lines += ['- '+f['text'][lang] for f in p['limitations']] + ['']
    lines += [('Source and evidence: [reviewed Python source]('+p['sources'][0]['url']+'), [aggregate evidence](technical/aggregate-evidence.json), and [evidence ledger](internal/evidence.md).' if is_en else 'Source dan bukti: [source Python yang ditinjau]('+p['sources'][0]['url']+'), [bukti agregat](technical/aggregate-evidence.json), dan [daftar bukti](internal/evidence.md).')]
    return '\n'.join(lines)


def make_readme(p):
    write_text('README.md',f'''# CCTV Person Counter — portfolio handoff / handoff portfolio

## English

This public-mode handoff is prepared for Naufal Firdaus and the agent maintaining his portfolio website. Read `project.json` first, then the English or Indonesian case study. The structured public text uses paired `en` and `id` values; English is primary.

### Website curation

- Start with **asset-01** (pipeline cover), **asset-02** (domain and density results), and **asset-05** (local runtime comparison). Their reasoning and exact files are in [assets-manifest.json](assets-manifest.json).
- Copy only intentionally selected website text and optimized visual files. **Do not automatically copy `source-code/`, `technical/`, or `internal/` into the website's public directory.** The folder named `internal/` is safe to share, but serves the receiving agent rather than the site audience.
- Preserve evidence qualifications in captions. Local benchmarks, cached predictions, and historical warehouse observations must not become production claims.
- Use the 8:5 cover without cropping its text. Use `contain` for detail images and charts. Original chart labels are primarily Indonesian; paired English/Indonesian alt text and captions provide their context.
- There is no verified public demo. Do not create a live-demo button or imply a web UI exists. Use the commit-pinned source link labeled Python source.
- The owner confirmed authorship of the custom pipeline. Credit the pretrained model, Roboflow data, and third-party libraries separately. Do not call the detector training-from-scratch or assign a new code license.

### Package contents and reproduction

The source snapshot contains **27 text files**: 16 Python modules, four historical HTML report pages, four root documentation/Git files, two dataset YAML configurations, and one export note. It preserves relative structure with reviewed path placeholders. [source-manifest.json](source-manifest.json) records every exported file, size, hash, and transformation.

Dataset images, annotations, image lists, prediction caches, model binaries, recorded footage, raw logs, installed dependencies, private configuration, and unpublished manuscripts are excluded. The HTML pages are source examples with missing media, not self-contained demo pages. Read [source-code/EXPORT-NOTES.md](source-code/EXPORT-NOTES.md) before following their legacy instructions.

The ZIP alone cannot reproduce training or inference. Authorized data and separately obtained compatible weights are required. [Development documentation](technical/development.md) provides conditional commands, configuration steps, and observed dependency versions; no original dependency lockfile exists.

### Evidence and publication boundaries

Snapshot: branch `main`, commit `{COMMIT}`, reviewed {DATE} in Asia/Jakarta. The tracked working tree was clean; additional ignored local artifacts were read for verification but are not exported. Aggregate input hashes identify those artifacts without disclosing frame filenames or recording paths.

The public GitHub repository has no declared license. Owner authorization covers preparing and sharing this custom-material handoff; it does not establish third-party dataset/model rights or assign a general reuse license. No source imagery, identifiable people, QR codes, or recording frames are included. Consult [publication review](internal/publication-review.md), [missing information](internal/missing-information.md), and the final machine-readable verification report before publication.

## Bahasa Indonesia

Handoff mode publik ini disiapkan untuk Naufal Firdaus dan agent pengelola website portfolio-nya. Baca `project.json`, kemudian studi kasus Inggris atau Indonesia. Teks publik terstruktur menggunakan pasangan `en` dan `id`; bahasa Inggris menjadi versi utama.

### Kurasi website

- Mulai dari **asset-01** (sampul pipeline), **asset-02** (hasil menurut domain dan kepadatan), serta **asset-05** (perbandingan runtime lokal). Alasan dan berkas tepatnya tersedia pada [assets-manifest.json](assets-manifest.json).
- Salin hanya teks website dan visual optimal yang sengaja dipilih. **Jangan otomatis menyalin `source-code/`, `technical/`, atau `internal/` ke direktori publik website.** Folder `internal/` aman dibagikan, tetapi berfungsi untuk agent penerima, bukan audiens situs.
- Pertahankan batas bukti pada caption. Benchmark lokal, prediksi tersimpan, dan observasi warehouse historis tidak boleh menjadi klaim produksi.
- Gunakan sampul 8:5 tanpa memotong teks. Gunakan `contain` untuk gambar detail dan grafik. Label grafik asli terutama berbahasa Indonesia; alt text dan caption Inggris/Indonesia memberi konteksnya.
- Tidak ada demo publik terverifikasi. Jangan membuat tombol demo langsung atau menyiratkan adanya UI web. Gunakan tautan source pada commit tetap dengan label source Python.
- Pemilik mengonfirmasi kepengarangan pipeline custom. Beri atribusi terpisah untuk model pretrained, data Roboflow, dan library pihak ketiga. Jangan menyebut training detektor dari awal atau menetapkan lisensi kode baru.

### Isi paket dan reproduksi

Snapshot source berisi **27 berkas teks**: 16 modul Python, empat halaman HTML laporan historis, empat berkas dokumentasi/Git root, dua konfigurasi YAML dataset, dan satu catatan ekspor. Struktur relatif dipertahankan dengan placeholder path yang ditinjau. [source-manifest.json](source-manifest.json) mencatat setiap berkas ekspor, ukuran, hash, dan transformasinya.

Gambar dataset, anotasi, daftar gambar, cache prediksi, binary model, rekaman video, log mentah, dependensi terpasang, konfigurasi privat, dan naskah belum terbit dikeluarkan. Halaman HTML adalah contoh source dengan media yang tidak disertakan, bukan halaman demo mandiri. Baca [source-code/EXPORT-NOTES.md](source-code/EXPORT-NOTES.md) sebelum mengikuti petunjuk lamanya.

ZIP saja tidak dapat mereproduksi training atau inferensi. Data yang berhak digunakan dan bobot kompatibel yang diperoleh terpisah diperlukan. [Dokumentasi pengembangan](technical/development.md) menyediakan perintah bersyarat, langkah konfigurasi, dan versi dependensi yang diamati; tidak ada lockfile dependensi asli.

### Bukti dan batas publikasi

Snapshot: branch `main`, commit `{COMMIT}`, ditinjau {DATE} dalam zona Asia/Jakarta. Working tree terlacak bersih; artefak lokal tambahan yang diabaikan Git dibaca untuk verifikasi tetapi tidak diekspor. Hash input agregat mengidentifikasi artefak tersebut tanpa mengungkap nama berkas frame atau path rekaman.

Repository GitHub publik tidak memiliki lisensi yang dideklarasikan. Izin pemilik mencakup penyiapan dan pembagian handoff materi custom ini; izin tersebut tidak menetapkan hak dataset/model pihak ketiga atau lisensi penggunaan ulang umum. Tidak ada gambar sumber, orang yang dapat dikenali, QR, atau frame rekaman di dalam paket. Periksa [tinjauan publikasi](internal/publication-review.md), [informasi yang kurang](internal/missing-information.md), dan laporan verifikasi akhir yang dapat dibaca mesin sebelum publikasi.
''')


def make_export_notes(records):
    write_text('source-code/EXPORT-NOTES.md', '''# Export notes / Catatan ekspor

## English

This is a reviewed, privacy-adjusted source snapshot, not a full repository archive. The 16 Python modules and four HTML report sources are present. No application feature was fixed or changed. Machine-specific examples and a companion-project path were replaced with explicit placeholders; dataset YAML uses `<DATASET_ROOT>`. Exact per-file changes and original/exported hashes are in the parent source manifest.

`README.md`, `PANDUAN.md`, and the HTML reports are historical source documents. Some instructions and figures are stale: store evaluation and its error gallery are not implemented, the progress monitor does not include store, and the dataset report describes an earlier 4,886-image state rather than the current 4,746-image state. Model and dataset binaries, raw labels/annotations, prediction caches, raw run logs, report photos, and video dependencies were deliberately omitted.

Consequently, HTML image/video references may resolve to files absent from this snapshot; this is documented omission, not a working standalone report app. Some original report image references were already inconsistent with local figure locations. The source manifest lists omitted local HTML references. Do not publish these report pages as a live demo without a separate review and authorized media.

Run commands only after replacing placeholders, obtaining authorized matching data/weights, and reading the technical documentation. The counter currently has no RTSP support or explicit device CLI flag; its shared -1 tracker fallback and the store optimizer setting remain known limitations. The export does not assign a code license. The owner authorized this handoff; third-party rights remain separate.

## Bahasa Indonesia

Ini snapshot source yang ditinjau dan disesuaikan untuk privasi, bukan arsip seluruh repository. Seluruh 16 modul Python dan empat source laporan HTML disertakan. Tidak ada fitur aplikasi yang diperbaiki atau diubah. Contoh path mesin dan path proyek pendamping diganti dengan placeholder eksplisit; YAML dataset menggunakan `<DATASET_ROOT>`. Perubahan per berkas serta hash asli/ekspor tercatat di manifest source pada folder induk.

`README.md`, `PANDUAN.md`, dan laporan HTML merupakan dokumen source historis. Sebagian instruksi dan gambar tertinggal: evaluasi toko serta galeri error-nya belum diimplementasikan, pemantau progres belum mencakup toko, dan laporan dataset menjelaskan keadaan lama 4.886 gambar, bukan keadaan saat ini 4.746 gambar. Binary model dan dataset, label/anotasi mentah, cache prediksi, log run mentah, foto laporan, dan dependensi video sengaja dikeluarkan.

Akibatnya, referensi gambar/video HTML dapat mengarah ke berkas yang tidak ada pada snapshot; ini penghilangan yang didokumentasikan, bukan aplikasi laporan mandiri yang berjalan. Sebagian referensi gambar laporan asli juga sudah tidak sesuai dengan lokasi gambar lokal. Manifest source mencantumkan referensi HTML lokal yang dikeluarkan. Jangan memublikasikan halaman laporan tersebut sebagai demo langsung tanpa tinjauan terpisah dan media berizin.

Jalankan perintah hanya setelah mengganti placeholder, memperoleh data/bobot sesuai yang berhak digunakan, dan membaca dokumentasi teknis. Counter saat ini tidak mendukung RTSP atau flag CLI device eksplisit; fallback tracker -1 bersama dan setelan optimizer toko tetap menjadi keterbatasan yang diketahui. Ekspor ini tidak menetapkan lisensi kode. Pemilik mengizinkan handoff ini; hak pihak ketiga tetap terpisah.
''')
    p=DEST/'source-code/EXPORT-NOTES.md'
    records.append({'path':'source-code/EXPORT-NOTES.md','repositoryPath':None,'bytes':p.stat().st_size,
                    'sha256':sha(p.read_bytes()),'originalSha256':None,'origin':'new-export-documentation',
                    'redacted':False,'transformations':[], 'review':{'textReviewed':True,'secretScan':'passed',
                    'distributionBasis':'new-documentation-authorized-by-owner'}})
    from html.parser import HTMLParser
    class Refs(HTMLParser):
        def __init__(self): super().__init__(); self.paths=[]
        def handle_starttag(self,tag,attrs):
            for k,v in attrs:
                if k in ('src','href','poster') and v and not v.startswith(('#','http:','https:','data:')):
                    self.paths.append(v)
    refs=[]
    for rec in records:
        p=DEST/rec['path']
        if p.suffix=='.html':
            parser=Refs(); parser.feed(p.read_text(encoding='utf8'))
            for ref in parser.paths:
                if not (p.parent/ref).exists(): refs.append({'from':rec['path'],'reference':ref,'status':'omitted-or-original-unresolved-media','reason':'Historical source retained; media excluded or original path inconsistent.'})
    write_json('source-manifest.json',{'schemaVersion':'1.0','snapshotCommit':COMMIT,'sourceRoot':'source-code',
        'fileCount':len(records),'pythonModuleCount':16,'historicalHtmlSourceCount':4,
        'scope':pair('All custom Python tools and historical report HTML sources, selected documentation and configuration; not the full repository or a runnable data/model bundle.',
                     'Seluruh tool Python custom dan source HTML laporan historis, dokumentasi serta konfigurasi terpilih; bukan seluruh repository atau bundel data/model yang dapat dijalankan.'),
        'files':sorted(records,key=lambda r:r['path']),'omittedHtmlReferences':refs,
        'exclusions':[
            {'category':'dataset-and-annotations','impact':pair('Rebuilding training and evaluation requires separately authorized raw data and label artifacts.','Membangun ulang training dan evaluasi memerlukan data mentah dan artefak label yang berhak digunakan secara terpisah.')},
            {'category':'model-binaries-and-compiled-graphs','impact':pair('Inference and exports require separately authorized matching weights and runtime-specific exports.','Inferensi dan ekspor memerlukan bobot sesuai dan ekspor khusus runtime yang berhak digunakan secara terpisah.')},
            {'category':'identifiable-images-video-and-media-galleries','impact':pair('Historical HTML reports have omitted media; no footage or people photographs are redistributed.','Media laporan HTML historis tidak disertakan; tidak ada rekaman atau foto orang yang didistribusikan ulang.')},
            {'category':'raw-logs-prediction-caches-and-dataset-manifests','impact':pair('Only reviewed aggregate evidence and charts are supplied; raw rerendering inputs are absent.','Hanya bukti agregat dan grafik yang ditinjau disediakan; input mentah untuk render ulang tidak tersedia.')},
            {'category':'git-installed-dependencies-caches-private-config-agent-notes-unpublished-manuscripts-adjacent-repositories','impact':pair('No history, secrets, installed software, unpublished writing, or companion implementation is bundled.','Tidak ada riwayat, rahasia, perangkat lunak terpasang, tulisan belum terbit, atau implementasi pendamping yang dibundel.')}
        ]})


def make_technical(p,e):
    from technical_content import write_technical
    write_technical(p,e)


def make_internal(p,e):
    from review_content import write_review_documents
    write_review_documents(p,e)
