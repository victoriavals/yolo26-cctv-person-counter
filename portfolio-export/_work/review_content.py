"""Safe evidence and publication-review records; no raw private input is copied."""
from build_export import write_text, DATE, COMMIT


def write_review_documents(p,e):
    rows=[
      ('E01','reported / owner-confirmed','Custom authorship, account identity, and handoff distribution authorization / Kepengarangan custom, identitas akun, dan izin distribusi handoff',
       'Owner statement in this session on '+DATE+'; not inferred from Git author names / Pernyataan pemilik pada sesi ini; bukan disimpulkan dari nama penulis Git'),
      ('E02','implemented + reported','Duplicate audit and original dataset defects / Audit duplikasi dan cacat dataset awal',
       'tools/audit_duplicates.py:23–25, 70–115; tools/clean_cctv.py:43–54; reportedRawAudit in aggregate-evidence.json. Audit counts are historical; clustering was not rerun / Jumlah audit historis; clustering tidak dijalankan ulang'),
      ('E03','observed local artifacts','Current split counts, COCO/YOLO alignment, manifest scene separation / Jumlah split saat ini, kesesuaian COCO/YOLO, pemisahan scene manifest',
       'Local annotations, manifest and YOLO labels read in memory; dataset and inputArtifactHashes in aggregate-evidence.json; tools/clean_cctv.py:155–186, 264–336. Raw artifacts excluded / Artefak mentah dikeluarkan'),
      ('E04','implemented','Fine-tuning settings and experiment definitions / Setelan fine-tuning dan definisi eksperimen',
       'tools/train.py:25–113; model is initialized from pretrained weights; COMMON and RUNS are inspected source / Model diinisialisasi dari bobot pretrained; COMMON dan RUNS diperiksa dari source'),
      ('E05','reported historical artifacts','Training epochs and run duration / Epoch training dan durasi run',
       'runs/person/{default,tuned,indoor}/results.csv and run_meta.json read locally; summarized in historicalTraining; raw logs excluded / Dibaca lokal, diringkas pada historicalTraining; log mentah dikeluarkan'),
      ('E06','implemented + observed cached recomputation','Counting formulas and cached metrics / Rumus counting dan metrik tersimpan',
       'tools/evaluate.py:115–155, 192–254; seven local JSON/NPZ outputs; recomputedCountingAndPR and maxAbsoluteRecomputeDelta in aggregate-evidence.json. No model inference / Tidak ada inferensi model'),
      ('E07','reported detection + observed cached counting','Recorded model results / Hasil model tercatat',
       'reports/eval/{baseline,default,tuned,indoor,default_512,default_416,default_320}.json; reportedDetection is historical, mAPRecomputed=false; counting recomputation matches saved values / Deteksi historis, counting dihitung ulang sesuai nilai tersimpan'),
      ('E08','implemented','Tracking/counting interfaces and output meaning / Antarmuka tracking/counting dan arti keluaran',
       'tools/count_people.py:36–60, 82–103, 124–139; installed Ultralytics tracking callback narrows results to returned tracks when present / Callback tracking Ultralytics terpasang menyaring hasil menjadi track yang dikembalikan bila tersedia'),
      ('E09','implemented','Export staging, dimensions, runtime targeting / Pemisahan ekspor, dimensi, target runtime',
       'tools/export.py:112–158; tools/benchmark_backends.py:43–49, 120–147, 176–239. No fresh export was executed / Tidak ada ekspor baru yang dijalankan'),
      ('E10','reported historical artifacts','Local runtime results and sample-limited parity / Hasil runtime lokal dan kesamaan terbatas sampel',
       'reports/eval/backends_default.json and reports/eval/backend_compare/*.json; only aggregate fields exported. 40-image benchmark differs from 240-frame video sample / Hanya kolom agregat diekspor. Benchmark 40 gambar berbeda dari sampel video 240 frame'),
      ('E11','implemented; actual store run unverified','Store-data preparation and configured fine-tuning / Persiapan data toko dan konfigurasi fine-tuning',
       'tools/prepare_labeling.py:43–59, 101–195; tools/merge_store_data.py:34–41, 79–130; tools/train.py:93–112. No store dataset or store run is present / Dataset atau run toko tidak tersedia'),
      ('E12','observed local artifact checks','Checksums and actual OpenVINO shapes / Checksum dan shape OpenVINO aktual',
       'weights/SHA256SUMS verified against four local files; default PyTorch copies compared; XML Parameter shapes parsed. modelIntegrity records hashes without binaries / modelIntegrity mencatat hash tanpa binary'),
      ('E13','reported historical demo + visual estimate','Warehouse undercount and identity fragmentation / Undercount warehouse dan fragmentasi identitas',
       'reports/tracking/warehouse_summary.json aggregate values; README.md:7–41 visual estimates. No recording or recording path bundled; no ground-truth tracking accuracy / Rekaman dan path tidak dibundel; tidak ada akurasi tracking ber-ground-truth'),
      ('E14','planned / not implemented','Store-only evaluation, gallery and monitoring gaps / Kekurangan evaluasi, galeri, dan pemantauan khusus toko',
       'tools/evaluate.py:39–44; tools/make_eval_figures.py:36–38; tools/status.py:19; legacy PANDUAN.md:182–226 promises more than these implementations / Panduan lama menjanjikan lebih dari implementasi tersebut'),
      ('E15','implemented interfaces; absent service layer','CLI-only application and static report viewer / Aplikasi CLI dan penampil laporan statis',
       'tools/serve_reports.py:78–94; source tree inspected; no application API/database/deployment manifest or original dependency manifest/test suite / Tree source diperiksa; tidak ada API/database aplikasi, manifest deployment, manifest dependensi asli, atau suite tes'),
      ('E16','observed synthetic check','Shared invalid tracker ID can cause a false crossing / ID tracker tidak valid bersama dapat memicu lintasan palsu',
       'tools/count_people.py:93–97 plus installed Supervision LineZone; one synthetic geometry frame with two boxes on opposite sides, both ID -1, produces one out event. focusedChecks records exact values / focusedChecks mencatat nilai tepat'),
      ('E17','observed installed source + repository configuration','Store lr0 override is not guaranteed under auto optimizer / Override lr0 toko tidak terjamin dengan optimizer auto',
       'tools/train.py:32–61, 93–101; installed Ultralytics cfg/default.yaml optimizer=auto and engine/trainer.py explicitly ignores manual lr0 in auto mode. No store training performed / Tidak ada training toko yang dijalankan'),
      ('E18','observed inconsistency','Stale dataset report and absolute roots / Laporan dataset tertinggal dan root absolut',
       'reports/stats.json reports 4,886 images/18,659 boxes; current annotations report 4,746/18,445. Copied data YAML roots replaced with placeholders / Root YAML salinan diganti placeholder'),
      ('E19','unverified rights; exclusion applied','Dataset/model/media redistribution / Distribusi ulang dataset/model/media',
       'README.md:421–426 and .agents/conventions.md:138–143; dataset license not established. GitHub repository metadata license=null; no local LICENSE. Owner confirmed custom-code/chart distribution only / Pemilik mengonfirmasi distribusi kode custom/grafik'),
    ]
    table='\n'.join('| '+' | '.join(row)+' |' for row in rows)
    write_text('internal/evidence.md',f'''# Evidence ledger / Daftar bukti

## Snapshot and classification / Snapshot dan klasifikasi

Reviewed {DATE}, Asia/Jakarta. Branch `main`, commit `{COMMIT}`. Tracked working tree was clean before export. Public GitHub visibility and `license=null` were checked read-only through the GitHub connector. No repository setting was changed. Local ignored images, labels, weights, and graph files were inspected for verification; the commit does not uniquely identify them, so aggregate input fingerprints are included. No raw diff or Git history metadata is bundled.

Ditinjau {DATE}, Asia/Jakarta. Branch `main`, commit `{COMMIT}`. Working tree terlacak bersih sebelum ekspor. Visibilitas GitHub publik dan `license=null` diperiksa secara baca-saja melalui konektor GitHub. Tidak ada setelan repository yang diubah. Gambar, label, bobot, dan graph lokal yang diabaikan Git diperiksa untuk verifikasi; commit tidak mengidentifikasinya secara unik sehingga fingerprint input agregat disertakan. Tidak ada diff mentah atau metadata riwayat Git yang dibundel.

| Classification | Meaning / Arti |
|---|---|
| implemented | Visible in inspected code; not necessarily executed in this review / Terlihat pada kode, belum tentu dijalankan pada tinjauan ini |
| observed | An explicit local check or direct artifact inspection was executed; scope is stated / Pemeriksaan lokal atau inspeksi artefak dijalankan dengan cakupan yang dijelaskan |
| reported | Historical documentation/artifact or an attributed owner statement / Dokumentasi/artefak historis atau pernyataan pemilik dengan atribusi |
| planned | Configuration/intent exists but the stated end result is unfinished / Konfigurasi/niat tersedia tetapi hasil yang dinyatakan belum selesai |
| unverified | Required evidence, permission, or measurement is unavailable / Bukti, izin, atau pengukuran yang dibutuhkan tidak tersedia |

## Claim mapping / Pemetaan klaim

Paths below refer to original repository locations; exported source uses the same relative paths beneath `source-code/`. Unbundled artifacts are referenced for provenance only; no external recording link is provided. Line numbers locate the inspected snapshot and can differ from future revisions.

Path berikut mengacu pada lokasi repository asli; source ekspor memakai path relatif yang sama di bawah `source-code/`. Artefak yang tidak dibundel dirujuk hanya sebagai asal bukti; tidak ada tautan rekaman eksternal. Nomor baris menunjuk snapshot yang diperiksa dan dapat berbeda pada revisi berikutnya.

| ID | Evidence class / Kelas bukti | Claim / Klaim | Source and boundary / Sumber dan batas |
|---|---|---|---|
{table}

## Executed checks and limits / Pemeriksaan yang dijalankan dan batasnya

- Python syntax is parsed without training. COCO/YOLO label alignment and image counts are checked on available local data. Counting, precision, recall, F1, and bias are recomputed from seven cached prediction outputs, with zero difference from saved values. Detection mAP is historical and is not recomputed.
- Four committed model checksum entries and the equality of the two default PyTorch copies are checked. Two OpenVINO XML input shapes are parsed. No model binary is exported.
- One synthetic Supervision check reproduces the invalid-ID crossing behavior. A source check of the installed optimizer explains the low-learning-rate limitation. A synthetic 300-frame time-block input confirms the rounding behavior.
- Package syntax, JSON, manifests, all image dimensions/hashes/metadata, document links, source hashes, secret/PII patterns, ZIP entries, and original-repository integrity are checked by the final packaging validator. Exact outcomes are in `validation-report.json`.
- No new training, detector inference, complete video processing, browser demo, production deployment, automated repository test suite, or code-coverage measurement is claimed. Existing report figures are historical outputs; the cover is newly authored documentation.

- Sintaks Python diperiksa tanpa training. Kesesuaian label COCO/YOLO dan jumlah gambar diperiksa pada data lokal yang tersedia. Counting, precision, recall, F1, dan bias dihitung ulang dari tujuh keluaran prediksi tersimpan, tanpa selisih dari nilai laporan. mAP deteksi bersifat historis dan tidak dihitung ulang.
- Empat entri checksum model tersimpan serta kesamaan dua salinan PyTorch default diperiksa. Dua shape input XML OpenVINO diparsing. Tidak ada binary model yang diekspor.
- Satu pemeriksaan Supervision sintetis mereproduksi lintasan akibat ID tidak valid. Pemeriksaan source optimizer terpasang menjelaskan keterbatasan learning rate rendah. Input blok waktu sintetis 300 frame mengonfirmasi perilaku pembulatan.
- Sintaks paket, JSON, manifest, seluruh dimensi/hash/metadata gambar, tautan dokumen, hash source, pola rahasia/PII, entri ZIP, dan integritas repository asli diperiksa validator pengemasan akhir. Hasil tepatnya tersedia pada `validation-report.json`.
- Tidak ada klaim training baru, inferensi detektor, pemrosesan video lengkap, demo browser, deployment produksi, suite tes repository otomatis, atau pengukuran code coverage. Grafik laporan adalah keluaran historis; sampul merupakan dokumentasi baru.

Aggregate record / Catatan agregat: [aggregate-evidence.json](../technical/aggregate-evidence.json).
''')
    write_text('internal/missing-information.md','''# Missing information / Informasi yang belum tersedia

## English

The package can be handed to a website agent within its stated scope. These gaps limit claims and reproduction; they are not hidden publication approvals:

1. **Dataset and trained-model distribution rights.** The exact upstream dataset terms and any additional requirements for trained/exported model redistribution have not been established. Dataset, media, annotations, caches, and weights are excluded. A direct authoritative dataset license/project link would be needed before reconsidering those exclusions.
2. **Code reuse license.** The owner authorized this portfolio handoff, but did not choose a general repository license. No license is invented. If downloadable public source reuse terms are desired, the owner must decide them separately while considering dependencies.
3. **Target-camera evidence.** No labeled store-camera test set, track-ID ground truth, calibrated line-count reference, store-only evaluation output, or completed store run exists in the inspected snapshot. The target deployment hardware and operational acceptance results are unknown.
4. **Public demo and application screenshots.** No verified public demo exists. This repository is a CLI pipeline with historical report pages, not a web application. No desktop/mobile application screenshot is supplied; the gallery uses project charts and a clearly labeled code-derived diagram.
5. **Reproducible environment.** Installed versions were observed, but an original dependency manifest/lockfile is absent. Availability and compatibility of those wheels on other platforms were not tested.

Authorship and ownership of the custom project are **resolved by the owner's confirmation**: Naufal Firdaus owns the victoriavals account and created the custom project. This does not transfer authorship of the imported libraries, pretrained YOLO architecture/weights, or Roboflow source data.

No additional answer is required to receive this scoped ZIP. Additional permissions and measurements would be needed to expand it into a footage/model distribution or a production-performance claim.

## Bahasa Indonesia

Paket dapat diserahkan kepada agent website sesuai cakupan yang dinyatakan. Kekurangan berikut membatasi klaim dan reproduksi; bukan persetujuan publikasi tersembunyi:

1. **Hak distribusi dataset dan model terlatih.** Ketentuan dataset upstream yang tepat serta persyaratan tambahan distribusi ulang model terlatih/ekspor belum ditetapkan. Dataset, media, anotasi, cache, dan bobot dikeluarkan. Tautan lisensi/proyek dataset yang berwenang diperlukan sebelum mempertimbangkan kembali pengecualian tersebut.
2. **Lisensi penggunaan ulang kode.** Pemilik mengizinkan handoff portfolio ini, tetapi belum memilih lisensi repository umum. Tidak ada lisensi yang dikarang. Jika ketentuan penggunaan ulang source publik yang dapat diunduh diinginkan, pemilik perlu menentukannya terpisah dengan mempertimbangkan dependensi.
3. **Bukti kamera sasaran.** Snapshot yang diperiksa belum memiliki test set kamera toko berlabel, ground truth track ID, referensi hitungan garis terkalibrasi, keluaran evaluasi khusus toko, atau run toko selesai. Hardware deployment sasaran dan hasil penerimaan operasional belum diketahui.
4. **Demo publik dan screenshot aplikasi.** Tidak ada demo publik terverifikasi. Repository ini berupa pipeline CLI dengan halaman laporan historis, bukan aplikasi web. Screenshot aplikasi desktop/mobile tidak disediakan; galeri menggunakan grafik proyek dan diagram berdasarkan kode yang diberi label jelas.
5. **Lingkungan reproduktif.** Versi terpasang diamati, tetapi manifest/lockfile dependensi asli tidak ada. Ketersediaan dan kompatibilitas wheel tersebut pada platform lain tidak diuji.

Kepengarangan dan kepemilikan proyek custom **sudah dijelaskan melalui konfirmasi pemilik**: Naufal Firdaus adalah pemilik akun victoriavals dan pembuat proyek custom. Hal ini tidak memindahkan kepengarangan library yang diimpor, arsitektur/bobot YOLO pretrained, atau data sumber Roboflow.

Tidak diperlukan jawaban tambahan untuk menerima ZIP sesuai cakupan ini. Izin dan pengukuran tambahan dibutuhkan untuk memperluasnya menjadi distribusi rekaman/model atau klaim performa produksi.
''')
    write_text('internal/publication-review.md',f'''# Publication review / Tinjauan publikasi

## English

### Scope and authorization

This handoff is prepared in PUBLIC mode for Naufal Firdaus. The owner explicitly confirmed ownership of the victoriavals account, authorship of the custom project, and the right to share its custom code and charts. The inspected snapshot is branch `main`, commit `{COMMIT}`, reviewed {DATE}. Adjacent repositories, private configuration, and unpublished writing are outside scope. No commit, push, upload, deployment, or visibility change was performed.

### Included and reviewed

The package contains bilingual portfolio text, technical documentation, aggregate evidence, 27 source text files, six main WebP visuals and six previews. Source starts from an explicit 26-file tracked whitelist rather than a recursive workspace copy. The added export note records limitations. The website agent must curate text/images and must not automatically publish source or review folders.

All source/document text is subject to full-file pattern scanning and targeted human inspection of paths, URLs, identifiers, report references, and claims. Every selected chart was viewed, including titles, plot labels and outer edges. The generated cover and all exported main/preview images require final full-frame visual review before archive release. Final status, counts, and executed checks are recorded in `validation-report.json`.

### Redactions and omissions

Machine-specific project examples, dataset roots, and a companion-project path are replaced with explicit placeholders on export copies. Original source files are untouched. Per-file transforms and original/exported hashes appear in the source manifest; removed path values are not reproduced in this review.

No dataset images, annotation files, frame filename lists, prediction arrays, model/graph binaries, footage, contact sheets, error galleries with people, raw run logs, notebook outputs, `.env`, credentials, private keys, cookies, installed dependencies, Git history, agent notes, or unpublished manuscripts are included. Source report HTML retains relative references to omitted media; these are explicitly listed in the source manifest and are not external links to excluded recordings. There are no notebooks or SQL dumps in the selected source whitelist.

Image files are decoded to clean pixel images and re-encoded without EXIF, XMP, location, comments, or source-device metadata. No faces or documents were edited with AI; no person images were used. The cover is a technical diagram, not a fabricated application UI. Four charts preserve existing project plots; the backend chart was re-rendered from the same historical numbers to separate overlapping labels and place sample limits inside the image. No new timing was measured.

### Privacy and links

The only intended personal name and account handle are the owner's authorized public identity. Generic hardware model names and standard loopback/font paths are retained where technically relevant; no device serial numbers or personal user-directory paths are included. Source links are pinned to the reviewed public code/README. Official package/font and XML namespace URLs present in legacy source are checked for credential-bearing userinfo or access tokens. No direct dataset, excluded image, recording, artifact-sharing, or QR link is supplied.

Regex scanning does not establish image safety. The release therefore also requires visual inspection of all final images. Findings without a safe correction would cause the file to be excluded rather than marked reviewed.

### Distribution and claim limits

There is no declared repository license, and the upstream dataset license is unverified. The export does not assign an open-source license or grant third-party rights. Dependencies are imported, not vendored. Historical scores, sample-specific backend matching, cached-output recomputation, owner statements, implemented functionality, and unfinished features are explicitly separated. No production readiness, code coverage, live demonstration, or unique-visitor accuracy is claimed.

## Bahasa Indonesia

### Cakupan dan izin

Handoff ini disiapkan dalam mode PUBLIC untuk Naufal Firdaus. Pemilik mengonfirmasi kepemilikan akun victoriavals, kepengarangan proyek custom, serta hak membagikan kode dan grafik custom-nya. Snapshot yang diperiksa adalah branch `main`, commit `{COMMIT}`, ditinjau {DATE}. Repository pendamping, konfigurasi privat, dan tulisan belum terbit berada di luar cakupan. Tidak ada commit, push, unggahan, deployment, atau perubahan visibilitas.

### Isi dan tinjauan

Paket memuat teks portfolio bilingual, dokumentasi teknis, bukti agregat, 27 berkas source teks, enam visual WebP utama, dan enam preview. Source berasal dari whitelist eksplisit 26 berkas terlacak, bukan salinan workspace rekursif. Catatan ekspor tambahan menjelaskan keterbatasan. Agent website perlu mengkurasi teks/gambar dan tidak boleh otomatis memublikasikan folder source atau tinjauan.

Seluruh teks source/dokumen diperiksa dengan pemindaian pola berkas penuh dan inspeksi manusia terarah pada path, URL, identitas, referensi laporan, serta klaim. Setiap grafik terpilih sudah dilihat, termasuk judul, label plot, dan tepinya. Sampul baru serta seluruh gambar utama/preview ekspor memerlukan tinjauan visual akhir sebelum rilis arsip. Status akhir, jumlah, dan pemeriksaan yang dijalankan tercatat pada `validation-report.json`.

### Redaksi dan pengecualian

Contoh path proyek khusus mesin, root dataset, dan path proyek pendamping diganti dengan placeholder eksplisit pada salinan ekspor. Berkas source asli tidak disentuh. Transformasi per berkas serta hash asli/ekspor tersedia pada manifest source; nilai path yang dihapus tidak disalin ulang dalam tinjauan ini.

Tidak ada gambar dataset, anotasi, daftar nama frame, array prediksi, binary model/graph, rekaman, contact sheet, galeri error berisi orang, log run mentah, output notebook, `.env`, kredensial, kunci privat, cookie, dependensi terpasang, riwayat Git, catatan agent, atau naskah belum terbit. HTML source laporan mempertahankan referensi relatif ke media yang dikeluarkan; referensi tersebut dicantumkan pada manifest source dan bukan tautan eksternal menuju rekaman yang dikecualikan. Tidak ada notebook atau SQL dump dalam whitelist source terpilih.

Berkas gambar didekode menjadi piksel bersih dan dienkode ulang tanpa EXIF, XMP, lokasi, komentar, atau metadata perangkat sumber. Tidak ada wajah atau dokumen yang diedit dengan AI; tidak ada gambar orang yang dipakai. Sampul merupakan diagram teknis, bukan UI aplikasi buatan. Empat grafik mempertahankan plot proyek yang tersedia; grafik backend dirender ulang dari angka historis yang sama untuk memisahkan label bertumpuk dan memasukkan batas sampel ke dalam gambar. Tidak ada pengukuran waktu baru.

### Privasi dan tautan

Satu-satunya nama pribadi dan handle akun yang disengaja adalah identitas publik pemilik yang diizinkan. Nama model hardware generik dan path loopback/font standar dipertahankan jika relevan secara teknis; tidak ada nomor seri perangkat atau path direktori pengguna pribadi. Tautan source dipatok pada kode/README publik yang ditinjau. URL paket/font resmi dan namespace XML pada source lama diperiksa terhadap userinfo kredensial atau token akses. Tidak ada tautan dataset, gambar yang dikeluarkan, rekaman, berbagi artefak, atau QR langsung.

Pemindaian regex tidak membuktikan keamanan gambar. Karena itu, rilis juga memerlukan inspeksi visual semua gambar akhir. Temuan yang tidak dapat dikoreksi secara aman menyebabkan berkas dikeluarkan, bukan ditandai selesai ditinjau.

### Batas distribusi dan klaim

Tidak ada lisensi repository yang dideklarasikan dan lisensi dataset upstream belum terverifikasi. Ekspor tidak menetapkan lisensi open source atau memberikan hak pihak ketiga. Dependensi diimpor, bukan divendorkan. Skor historis, kecocokan backend khusus sampel, penghitungan ulang keluaran tersimpan, pernyataan pemilik, fungsi yang diimplementasikan, dan fitur yang belum selesai dibedakan secara eksplisit. Tidak ada klaim kesiapan produksi, code coverage, demo langsung, atau akurasi pengunjung unik.
''')
