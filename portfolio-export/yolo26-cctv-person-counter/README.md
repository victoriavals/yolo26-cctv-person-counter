# CCTV Person Counter — portfolio handoff / handoff portfolio

## English

This public-mode handoff is prepared for Naufal Firdaus and the agent maintaining his portfolio website. Read `project.json` first, then the English or Indonesian case study. The structured public text uses paired `en` and `id` values; English is primary.

### Website curation

- Start with **asset-01** (pipeline cover), **asset-07** (one actual model-use frame), **asset-02** (domain and density results), and **asset-05** (local runtime comparison). Their reasoning and exact files are in [assets-manifest.json](assets-manifest.json).
- Copy only intentionally selected website text and optimized visual files. **Do not automatically copy `source-code/`, `technical/`, or `internal/` into the website's public directory.** The folder named `internal/` is safe to share, but serves the receiving agent rather than the site audience.
- Preserve evidence qualifications in captions. Local benchmarks, cached predictions, and historical warehouse observations must not become production claims.
- Use the 8:5 cover without cropping its text. Use `contain` for detail images and charts. Original chart labels are primarily Indonesian; paired English/Indonesian alt text and captions provide their context.
- There is no verified public demo. Do not create a live-demo button or imply a web UI exists. Use the commit-pinned source link labeled Python source.
- The owner confirmed authorship of the custom pipeline. Credit the pretrained model, Roboflow data, and third-party libraries separately. Do not call the detector training-from-scratch or assign a new code license.

### Package contents and reproduction

The source snapshot contains **27 text files**: 16 Python modules, four historical HTML report pages, four root documentation/Git files, two dataset YAML configurations, and one export note. It preserves relative structure with reviewed path placeholders. [source-manifest.json](source-manifest.json) records every exported file, size, hash, and transformation.

Raw dataset images, annotations, image lists, prediction caches, model binaries, recorded footage, raw logs, installed dependencies, private configuration, and unpublished manuscripts are excluded. Exactly one redacted single-frame model-use illustration is provided separately as asset-07. The HTML pages are source examples with missing media, not self-contained demo pages. Read [source-code/EXPORT-NOTES.md](source-code/EXPORT-NOTES.md) before following their legacy instructions.

The ZIP alone cannot reproduce training or inference. Authorized data and separately obtained compatible weights are required. [Development documentation](technical/development.md) provides conditional commands, configuration steps, and observed dependency versions; no original dependency lockfile exists.

### Evidence and publication boundaries

Snapshot: branch `main`, commit `624a840b2de026585d7da4bd9bfeb5f92f8984fd`, reviewed 2026-11-16 in Asia/Jakarta. The tracked working tree was clean; additional ignored local artifacts were read for verification but are not exported. Aggregate input hashes identify those artifacts without disclosing frame filenames or recording paths.

The public GitHub repository has no declared license. Owner authorization covers preparing and sharing this custom-material handoff; it does not establish third-party dataset/model rights or assign a general reuse license. One selected model-use frame is included at the user's explicit direction, with its visible facial profile and camera text redacted. Other source imagery and recordings are excluded. The upstream dataset license remains undefined; no broader redistribution clearance is inferred. Consult [publication review](internal/publication-review.md), [missing information](internal/missing-information.md), and the final machine-readable verification report before publication.

## Bahasa Indonesia

Handoff mode publik ini disiapkan untuk Naufal Firdaus dan agent pengelola website portfolio-nya. Baca `project.json`, kemudian studi kasus Inggris atau Indonesia. Teks publik terstruktur menggunakan pasangan `en` dan `id`; bahasa Inggris menjadi versi utama.

### Kurasi website

- Mulai dari **asset-01** (sampul pipeline), **asset-07** (satu frame penggunaan model sebenarnya), **asset-02** (hasil menurut domain dan kepadatan), serta **asset-05** (perbandingan runtime lokal). Alasan dan berkas tepatnya tersedia pada [assets-manifest.json](assets-manifest.json).
- Salin hanya teks website dan visual optimal yang sengaja dipilih. **Jangan otomatis menyalin `source-code/`, `technical/`, atau `internal/` ke direktori publik website.** Folder `internal/` aman dibagikan, tetapi berfungsi untuk agent penerima, bukan audiens situs.
- Pertahankan batas bukti pada caption. Benchmark lokal, prediksi tersimpan, dan observasi warehouse historis tidak boleh menjadi klaim produksi.
- Gunakan sampul 8:5 tanpa memotong teks. Gunakan `contain` untuk gambar detail dan grafik. Label grafik asli terutama berbahasa Indonesia; alt text dan caption Inggris/Indonesia memberi konteksnya.
- Tidak ada demo publik terverifikasi. Jangan membuat tombol demo langsung atau menyiratkan adanya UI web. Gunakan tautan source pada commit tetap dengan label source Python.
- Pemilik mengonfirmasi kepengarangan pipeline custom. Beri atribusi terpisah untuk model pretrained, data Roboflow, dan library pihak ketiga. Jangan menyebut training detektor dari awal atau menetapkan lisensi kode baru.

### Isi paket dan reproduksi

Snapshot source berisi **27 berkas teks**: 16 modul Python, empat halaman HTML laporan historis, empat berkas dokumentasi/Git root, dua konfigurasi YAML dataset, dan satu catatan ekspor. Struktur relatif dipertahankan dengan placeholder path yang ditinjau. [source-manifest.json](source-manifest.json) mencatat setiap berkas ekspor, ukuran, hash, dan transformasinya.

Gambar dataset mentah, anotasi, daftar gambar, cache prediksi, binary model, rekaman video, log mentah, dependensi terpasang, konfigurasi privat, dan naskah belum terbit dikeluarkan. Tepat satu ilustrasi penggunaan model pada satu frame dengan redaksi disediakan terpisah sebagai asset-07. Halaman HTML adalah contoh source dengan media yang tidak disertakan, bukan halaman demo mandiri. Baca [source-code/EXPORT-NOTES.md](source-code/EXPORT-NOTES.md) sebelum mengikuti petunjuk lamanya.

ZIP saja tidak dapat mereproduksi training atau inferensi. Data yang berhak digunakan dan bobot kompatibel yang diperoleh terpisah diperlukan. [Dokumentasi pengembangan](technical/development.md) menyediakan perintah bersyarat, langkah konfigurasi, dan versi dependensi yang diamati; tidak ada lockfile dependensi asli.

### Bukti dan batas publikasi

Snapshot: branch `main`, commit `624a840b2de026585d7da4bd9bfeb5f92f8984fd`, ditinjau 2026-11-16 dalam zona Asia/Jakarta. Working tree terlacak bersih; artefak lokal tambahan yang diabaikan Git dibaca untuk verifikasi tetapi tidak diekspor. Hash input agregat mengidentifikasi artefak tersebut tanpa mengungkap nama berkas frame atau path rekaman.

Repository GitHub publik tidak memiliki lisensi yang dideklarasikan. Izin pemilik mencakup penyiapan dan pembagian handoff materi custom ini; izin tersebut tidak menetapkan hak dataset/model pihak ketiga atau lisensi penggunaan ulang umum. Satu frame penggunaan model terpilih disertakan atas instruksi eksplisit pengguna, dengan profil wajah yang terlihat dan teks kamera ditutup. Gambar sumber lain dan rekaman dikeluarkan. Lisensi dataset upstream tetap belum ditentukan; tidak disimpulkan adanya izin distribusi yang lebih luas. Periksa [tinjauan publikasi](internal/publication-review.md), [informasi yang kurang](internal/missing-information.md), dan laporan verifikasi akhir yang dapat dibaca mesin sebelum publikasi.

## Model-use example / Contoh penggunaan model

The seven-visual gallery includes one actual single-frame inference result; use `contain` to preserve the complete frame. [View the example and code](technical/person-counter-example.md).

Galeri tujuh visual mencakup satu hasil inferensi satu frame yang sebenarnya; gunakan `contain` agar frame tetap utuh. [Lihat contoh dan kode](technical/person-counter-example.md).
