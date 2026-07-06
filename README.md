# YouTube Sentiment Analysis with DistilBERT & SVM

Proyek ini dirancang untuk mengumpulkan, membersihkan, dan menganalisis sentimen komentar dari video YouTube menggunakan pendekatan kombinasi Deep Learning (**DistilBERT**) untuk pelabelan otomatis dan Machine Learning (**Linear SVM**) untuk klasifikasi.

---

## 🚀 Fitur Utama & Penjelasan Kode

Berikut adalah urutan logika kodingan per fitur utama yang digunakan dalam proyek ini:

### 🔍 1. Scraping Komentar YouTube
Fitur ini menggunakan library `youtube_comment_downloader` untuk mengambil komentar secara massal berdasarkan daftar URL video yang ditentukan dan menyimpannya ke bentuk data mentah (`.csv`).

<pre><code>
def scrape(video_urls, max_comments):
    downloader = YoutubeCommentDownloader()
    rows = []
    for url in video_urls:
        logging.info(f"Memulai scraping untuk URL: {url}")
        try:
            comments = downloader.get_comments_from_url(url, sort_by=1)
            count = 0
            for c in comments:
                rows.append({
                    "video_url": url,
                    "author": c.get("author"),
                    "comment": c.get("text"),
                    "likes": c.get("votes"),
                    "time": c.get("time")
                })
                count += 1
                if count >= max_comments:
                    break
            logging.info(f"Berhasil mengumpulkan {count} komentar dari {url}")
        except Exception as e:
            logging.error(f"Gagal melakukan scraping pada {url}: {e}")
        time.sleep(1)
    
    if not rows:
        logging.warning("Tidak ada komentar yang berhasil di-scrape. Menggunakan data kosong.")
        
    df = pd.DataFrame(rows)
    output_path = os.path.join(OUTPUT_DIR, "1_raw_comments.csv")
    df.to_csv(output_path, index=False, encoding="utf-8-sig")
    logging.info(f"Data mentah disimpan ke {output_path}")
    return df
</code></pre>

### 🧹 2. Preprocessing Teks (Pembersihan Bahasa Inggris)
Komentar yang didapat dibersihkan dari komponen yang mengganggu analisis seperti tautan (URL), sebutan akun (@mentions), angka, simbol, serta membuang kata umum yang tidak memiliki makna (*Stopwords* bahasa Inggris).

<pre><code>
def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", " ", text)  # Hapus URL
    text = re.sub(r"@\w+", " ", text)           # Hapus mentions
    text = re.sub(r"[^a-zA-Z\s]", " ", text)     # Hapus angka & simbol
    # Stopword removal
    text = " ".join(w for w in text.split() if w not in ENGLISH_STOP_WORDS)
    return text.strip()
</code></pre>

### 🏷️ 3. Pseudo-Labeling Menggunakan DistilBERT
Karena data komentar dari scraping tidak memiliki label (Positif/Negatif), fitur ini memanfaatkan model Deep Learning pra-latih `distilbert-base-uncased-finetuned-sst-2-english` untuk memberikan label otomatis (*pseudo-label*).

<pre><code>
def pseudo_label(df):
    logging.info("Memulai proses pseudo-labeling dengan DistilBERT...")
    if df.empty:
        logging.warning("Data kosong, melewati proses pseudo-labeling.")
        df["sentiment"] = []
        return df

    clf = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
    labels = []
    
    for idx, t in enumerate(df["clean_comment"]):
        r = clf(t[:512])[0]
        labels.append("Positive" if r["label"] == "POSITIVE" else "Negative")
        if (idx + 1) % 100 == 0:
            logging.info(f"Telah melabeli {idx + 1} komentar...")
            
    df["sentiment"] = labels
    output_path = os.path.join(OUTPUT_DIR, "3_pseudo_labels.csv")
    df.to_csv(output_path, index=False, encoding="utf-8-sig")
    logging.info(f"Pseudo-labeling selesai. Hasil disimpan ke {output_path}")
    return df
</code></pre>

### 🤖 4. Ekstraksi TF-IDF & Pelatihan Model SVM
Teks bersih diubah menjadi matriks angka menggunakan `TfidfVectorizer`, kemudian data dibagi (*split*) sebesar 80:20 untuk melatih algoritma `LinearSVC` (SVM) dalam mengklasifikasikan arah sentimen.

<pre><code>
def train_svm(df):
    if df.empty or "sentiment" not in df.columns:
        logging.error("Data tidak valid untuk training model.")
        return

    logging.info("Memulai ekstraksi fitur dengan TF-IDF Vectorizer...")
    vectorizer = TfidfVectorizer(max_features=5000)
    X = vectorizer.fit_transform(df["clean_comment"])
    y = df["sentiment"]
    
    if len(y.unique()) < 2:
        logging.error("Training dibatalkan karena data hanya memiliki 1 jenis kelas sentimen.")
        return

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    logging.info("Melatih model Linear SVM...")
    model = LinearSVC()
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
</code></pre>

---

## 📁 Struktur Folder & Output
Setelah program dijalankan, semua aset hasil analisis akan otomatis terorganisir seperti berikut:
- `sentimen.py` : Skrip Python utama untuk menjalankan seluruh proses.
- `requirements.txt` : Daftar pustaka (dependencies) yang dibutuhkan.
- `process.log` : Catatan log proses eksekusi program.
- `output/` : Folder hasil yang berisi berkas data `.csv`, model tersimpan (`svm_model.pkl`), serta aset visualisasi (`pie_chart.png`, `confusion_matrix.png`, `wordcloud_positive.png`, dll).

---

## 🛠️ Cara Menjalankan Proyek di Laptop Kamu

1. **Clone Repositori ini:**
<pre><code>git clone https://github.com/AlAkbar44/Youtube-Sentimen-Analysis.git</code></pre>

2. **Masuk ke Folder Proyek:**
<pre><code>cd Youtube-Sentimen-Analysis</code></pre>

3. **Instal Seluruh Library yang Dibutuhkan:**
<pre><code>pip3 install -r requirements.txt</code></pre>

4. **Jalankan Program Utama:**
<pre><code>python3 sentimen.py</code></pre>

*Catatan: Gunakan perintah `pip3` dan `python3` jika kamu menjalankan proyek ini menggunakan sistem operasi macOS.*
