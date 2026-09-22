# Panduan: dari sini sampai model jalan di CCTV toko Anda

Status sekarang: model sudah dilatih dan diuji, perkakasnya lengkap. Yang belum
ada cuma satu — **rekaman dari kamera toko Anda sendiri**. Tanpa itu, semua angka
yang sudah diukur mengukur domain orang lain.

Bukti bahwa ini bukan formalitas: model dengan mAP50 **0,891** di split test-nya
sendiri melewatkan **60–80% orang** di rekaman CCTV gudang nyata. Menaikkan
resolusi dari 640 ke 1600 piksel tidak memperbaikinya sama sekali. Yang kurang
bukan piksel, tapi contoh bentuk orang duduk dan tertutup meja.

Perkiraan waktu Anda: **1 jam merekam + 3–5 jam melabeli + 1 jam training.**

---

## Langkah 1 — Rekam CCTV toko Anda

Ini satu-satunya langkah yang tidak bisa diwakilkan.

**Berapa lama:** minimal 30 menit, idealnya 60 menit total.

Hitungannya: 300 frame dengan jarak minimum 2 detik butuh setidaknya 10 menit
rekaman yang benar-benar berubah. Tapi target pelabelan mengharapkan 25% frame
berisi 6 orang atau lebih, dan momen seramai itu jarang di toko kecil. Jadi
rekam lebih panjang supaya momen ramainya tertangkap.

**Kapan:** pecah jadi 4 sesi @15 menit, bukan sekali 60 menit.

| Sesi | Kenapa perlu |
|---|---|
| Pagi, baru buka | cahaya rendah, lampu baru menyala, toko sepi |
| Jam sibuk | **paling penting** — ini kondisi yang bikin penghitung meleset |
| Siang/sore | cahaya matahari masuk, bayangan bergerak |
| Menjelang tutup | lampu campur, sebagian mati |

**Pastikan ada di dalam rekaman:**

- [ ] Orang **duduk** — di kursi, di kasir, di bangku tunggu
- [ ] Orang **membungkuk** — mengambil barang di rak bawah
- [ ] Orang **tertutup separuh** — di balik rak, meja, tumpukan barang
- [ ] Orang **membelakangi kamera**
- [ ] Orang **berhenti lama** di satu tempat, bukan cuma berjalan lewat
- [ ] Momen **ramai** — paling tidak sekali, walau cuma 2 menit
- [ ] Anak kecil kalau memang ada pengunjung anak

Empat poin pertama adalah celah yang sudah terukur. Kalau tidak ada di rekaman,
model tidak akan pernah belajar mengenalinya.

**Kamera jangan digeser** selama perekaman maupun setelahnya. Semua yang dilatih
mengasumsikan sudut pandang tetap. Kalau kamera dipindah, pelabelan harus diulang.

**Format:** mp4/mkv apa pun. Resolusi asli, jangan dikecilkan.

---

## Langkah 2 — Salin video ke mesin ini

Taruh di mana saja, misalnya `D:\ML\person-counter\rekaman\`. Gabungkan keempat
sesi jadi satu file kalau bisa, atau proses satu-satu dengan `--name` berbeda.

> **Penting soal path:** OpenCV di Windows tidak mengerti path gaya Git Bash.
> Pakai `D:/ML/...`, jangan `/d/ML/...` — kalau salah, video gagal dibuka tanpa
> pesan error yang jelas.

---

## Langkah 3 — Ambil frame dan buat label draft

```bash
cd D:\ML\person-counter
python tools/prepare_labeling.py --source D:/ML/person-counter/rekaman/toko.mp4 --n 300
```

Hasilnya di `data/store/toko/`: `images/`, `labels/` (draft), `classes.txt`,
`manifest.csv`.

Script ini menyaring frame supaya setiap frame yang Anda labeli berbobot:
jarak waktu minimum antar frame, frame yang tidak berubah dibuang, dan komposisi
kepadatannya diatur (15% kosong, 30% 1–2 orang, 30% 3–5 orang, 25% 6+ orang).

**Perhatikan output ini:**

```
kandidat unik: 99  (nyaris tak berubah, dibuang: 0)
sebaran kotak draft: {'0': 0, '1-2': 16, '3-5': 36, '6+': 8}
```

- Kalau muncul `CATATAN: hanya N frame lolos` → turunkan `--min-change` (default
  2.0) atau rekam lebih lama.
- Kalau bucket `6+` nol → rekaman Anda tidak punya momen ramai. Rekam lagi di
  jam sibuk; ini bagian yang paling menentukan.

Mau mulai dari nol tanpa draft? Tambahkan `--no-predraw`.

---

## Langkah 4 — Labeli frame-nya

Ini langkah terlama. Anggarkan **3–5 jam untuk 300 frame** (30–60 detik/frame).

### Pilih alat

Saya memverifikasi ketiganya tersedia, tapi **belum menguji jalannya di mesin ini**
— alat GUI tidak bisa saya uji dari terminal:

| Alat | Kelebihan | Kekurangan |
|---|---|---|
| **CVAT** (Docker) | impor label YOLO langsung, gratis, lokal | perlu Docker |
| **labelImg** | paling ringan, baca `.txt` YOLO apa adanya | `pip install labelImg` + PyQt5; versinya 1.8.6 (2021), mungkin bermasalah di Python 3.11 |
| **Label Studio** | aktif dikembangkan (1.23.0) | draft YOLO harus dikonversi ke format JSON-nya dulu |
| **Roboflow** (web) | UI paling nyaman, Anda sudah punya akun | **rekaman toko Anda terunggah ke server pihak ketiga** |

Untuk pekerjaan ini — satu kelas, 300 frame, draft sudah dalam format YOLO —
**CVAT atau labelImg paling pas** karena membaca `.txt` langsung tanpa konversi.

Soal Roboflow: rekaman itu berisi wajah pelanggan dan karyawan Anda. Mengunggahnya
ke layanan pihak ketiga adalah keputusan privasi, bukan keputusan teknis. Saya
sebutkan supaya Anda memutuskannya sadar, bukan karena kebetulan itu yang termudah.

### Aturan pelabelan — baca ini sebelum mulai

**Kotak draft berasal dari model yang terbukti melewatkan orang duduk dan
tertutup meja.** Kalau Anda hanya menghapus kotak yang salah dan tidak pernah
menambahkan yang terlewat, Anda mengajari model tepat titik butanya sendiri.

- [ ] **Tambahkan** kotak untuk setiap orang yang draft-nya lewatkan — terutama
      yang duduk, membungkuk, dan tertutup separuh
- [ ] **Hapus** kotak yang bukan orang (manekin, poster, refleksi di kaca)
- [ ] **Rapikan** kotak yang posisinya melenceng
- [ ] Orang tertutup separuh → kotak hanya pada bagian yang **terlihat**
- [ ] Orang di tepi frame, walau cuma separuh badan → tetap diberi kotak
- [ ] Refleksi orang di kaca/lantai → **jangan** diberi kotak
- [ ] Foto atau poster orang → **jangan** diberi kotak
- [ ] Konsisten: kalau Anda memutuskan satu kasus, putuskan sama sepanjang 300 frame

Konsistensi lebih penting daripada kesempurnaan. Aturan yang diterapkan sama di
semua frame lebih berguna daripada aturan sempurna yang berubah di tengah jalan.

**Export sebagai format YOLO**, timpa isi `data/store/toko/labels/`.

---

## Langkah 5 — Gabungkan dengan dataset yang ada

```bash
python tools/merge_store_data.py --store data/store/toko
```

Hasilnya di `data/merged/`: dataset lama (4.746 foto) + frame toko Anda, plus dua
config.

Frame Anda dibagi per **blok waktu berurutan**, bukan acak — frame yang berjarak
beberapa detik itu mirip, dan pembagian acak akan membocorkannya antara train dan
test. Itu persis cacat yang membuat split Roboflow asli tidak berarti.

**Perhatikan peringatan ini:**

```
PERINGATAN: hanya N frame toko di test. Di bawah ~50 frame,
angka mAP-nya terlalu berisik untuk dipercaya.
```

Kalau muncul, labeli lebih banyak frame. Di bawah 50 frame test, angkanya bisa
bergeser jauh hanya karena satu-dua frame.

---

## Langkah 6 — Fine-tune

```bash
python tools/train.py store
```

Perkiraan **30–60 menit** di RTX 4060 Ti.

Run ini sengaja beda dari training sebelumnya: mulai dari `best.pt` (bukan COCO
lagi), 60 epoch bukan 150, dan learning rate 0,001 bukan 0,01. Alasannya, bobotnya
sudah dekat dengan yang diinginkan — tugasnya menambahkan satu bentuk yang kurang,
bukan melatih ulang dari awal. Learning rate penuh akan menghapus yang sudah
dipelajari.

Pantau dari terminal lain:

```bash
python tools/status.py --watch
```

---

## Langkah 7 — Evaluasi, dan cara membacanya

```bash
python tools/evaluate.py --all
python tools/make_eval_figures.py
```

Ada **dua angka** dan keduanya harus Anda lihat:

| Diukur di | Artinya |
|---|---|
| `data/merged/data.yaml` (test gabungan) | perbandingan adil dengan run sebelumnya |
| `data/merged/data_store_test.yaml` (**hanya frame toko**) | **angka yang berlaku di lapangan** |

Yang kedua itu yang menentukan. Kalau keduanya berbeda jauh, yang benar selalu
yang dari frame toko Anda.

**Metrik yang dipakai untuk memutuskan: MAE hitungan, bukan mAP.** Produknya
menghitung orang, jadi galat jumlah orang per frame yang penting. Threshold
optimal untuk MAE dan untuk F1 sering jatuh di titik berbeda — `evaluate.py`
melaporkan keduanya, pilih yang MAE.

**Ambang layak pakai untuk penghitung toko:**

| Metrik | Target |
|---|---|
| MAE hitungan | di bawah 0,5 orang/frame |
| Akurat ±1 orang | di atas 90% |
| Bias | antara −0,3 dan +0,3 |
| Recall pada frame ramai | di atas 0,85 |

Pembanding: sebelum fine-tune toko, angkanya MAE 0,64 dan ±1 sebesar 88,0% —
tapi itu di data publik, bukan di toko Anda.

**Lihat juga galeri error-nya**, jangan cuma angkanya:
`reports/eval/figures/errors_store.png` menunjukkan false positive yang paling
diyakini model dan orang terbesar yang terlewat. Itu yang tercepat memberi tahu
*kenapa* model gagal, bukan cuma seberapa sering.

---

## Langkah 8 — Pasang

Pilih backend sesuai mesin di toko:

| Mesin toko | Backend | Kecepatan di 640px |
|---|---|---|
| Ada GPU NVIDIA | **TensorRT** | 12,8 ms · 78 FPS |
| Hanya CPU | **OpenVINO** | 50,2 ms · 20 FPS |
| — | ~~ONNX~~ | 696 ms · 1,4 FPS — jangan |

Export:

```bash
python tools/benchmark_backends.py --run store
```

Jalankan penghitungnya:

```bash
python tools/count_people.py --source D:/ML/person-counter/rekaman/toko.mp4 \
  --line 0,0.6,1,0.6 --max-frames 900 --name toko
```

Sesuaikan `--line` dengan posisi pintu masuk toko Anda. Formatnya
`x1,y1,x2,y2` dalam 0–1, jadi tidak bergantung resolusi. Contoh: `0,0.6,1,0.6`
adalah garis horizontal di 60% tinggi frame.

**Jangan setel tracker sebelum deteksinya benar.** Pada pengujian sebelumnya
tracker melaporkan 40 ID unik untuk sekitar 6 orang — itu gejala deteksi yang
putus-putus, bukan masalah setelan tracker. Perbaiki deteksi dulu.

---

## Kalau hasilnya masih kurang

Urut dari yang paling mungkin membantu:

1. **Labeli lebih banyak frame** — 300 → 600. Ini hampir selalu jawabannya.
2. **Periksa konsistensi label Anda.** Buka ulang 20 frame acak; kalau Anda
   menemukan keputusan yang tidak konsisten, modelnya juga bingung.
3. **Tambah rekaman di kondisi yang gagal.** Kalau error menumpuk saat ramai,
   rekam lebih banyak jam sibuk — bukan menambah frame sepi.
4. **Naikkan ke model lebih besar** (`yolo26m`) — hanya kalau tiga langkah di
   atas sudah dilakukan. Model lebih besar tidak menambal data yang kurang.

Jangan mulai dari langkah 4. Kapasitas model bukan penghambatnya; contoh bentuk
orang duduk yang penghambatnya.

---

## Checklist ringkas

- [ ] Rekam 4 sesi × 15 menit, lintas jam, termasuk jam sibuk
- [ ] Pastikan ada orang duduk, membungkuk, dan tertutup separuh
- [ ] Salin ke mesin, pakai path `D:/...`
- [ ] `python tools/prepare_labeling.py --source <video> --n 300`
- [ ] Cek bucket `6+` tidak nol
- [ ] Labeli di CVAT/labelImg — **tambahkan** yang terlewat, jangan cuma menghapus
- [ ] Export format YOLO ke `data/store/<nama>/labels/`
- [ ] `python tools/merge_store_data.py --store data/store/<nama>`
- [ ] Cek tidak ada peringatan "di bawah 50 frame test"
- [ ] `python tools/train.py store`
- [ ] `python tools/evaluate.py --all` — baca angka **frame toko**
- [ ] MAE di bawah 0,5 dan ±1 di atas 90%? Lanjut pasang. Belum? Labeli lebih banyak.
