import os
import re
import time
import logging
import pickle
import pandas as pd
import matplotlib.pyplot as plt

from youtube_comment_downloader import YoutubeCommentDownloader
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score, 
    precision_recall_fscore_support, 
    confusion_matrix, 
    ConfusionMatrixDisplay,
    classification_report
)
from transformers import pipeline
from wordcloud import WordCloud

# 1. Konfigurasi Logging Proses
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("process.log", encoding="utf-8"),
        logging.StreamHandler()
    ]
)

VIDEO_URLS = [
    "https://youtube.com/shorts/oYAnEH0JoqI?si=6ahP2uQo3P6J87nK",
    "https://youtu.be/-aHFboq0XNw?si=PPVxjQVH4cWa7meL",
    "https://youtu.be/KEDZ2Id0mg4?si=Wi1IcGLMi8Uf3YiL"
]

MAX_COMMENTS_PER_VIDEO = 1000
OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 2. Fitur: Scraping komentar YouTube
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

# 3. Fitur: English text preprocessing & Stopword removal
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

def preprocess(df):
    logging.info("Memulai tahap preprocessing teks...")
    if df.empty or "comment" not in df.columns:
        logging.error("DataFrame kosong atau tidak memiliki kolom 'comment'!")
        return df

    df = df.dropna(subset=["comment"]).copy()
    df["clean_comment"] = df["comment"].apply(clean_text)
    df = df[df["clean_comment"].str.len() > 2]
    
    output_path = os.path.join(OUTPUT_DIR, "2_clean_comments.csv")
    df.to_csv(output_path, index=False, encoding="utf-8-sig")
    logging.info(f"Preprocessing selesai. Data bersih disimpan ke {output_path}")
    return df

# 4. Fitur: DistilBERT pseudo-labeling
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

# 5. Fitur: Modeling (TF-IDF & SVM), Evaluasi, Visualisasi, & Save Model
def train_svm(df):
    if df.empty or "sentiment" not in df.columns:
        logging.error("Data tidak valid untuk training model.")
        return

    logging.info("Memulai ekstraksi fitur dengan TF-IDF Vectorizer...")
    # Fitur: TF-IDF Vectorizer
    vectorizer = TfidfVectorizer(max_features=5000)
    X = vectorizer.fit_transform(df["clean_comment"])
    y = df["sentiment"]
    
    # Cek variasi kelas sebelum splitting
    if len(y.unique()) < 2:
        logging.error("Training dibatalkan karena data hanya memiliki 1 jenis kelas sentimen.")
        return

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Fitur: Linear SVM
    logging.info("Melatih model Linear SVM...")
    model = LinearSVC()
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    
    # Fitur Evaluasi: Accuracy, Precision, Recall, F1-Score
    acc = accuracy_score(y_test, pred)
    p, r, f, _ = precision_recall_fscore_support(y_test, pred, average="weighted")
    
    logging.info("=== EVALUASI MODEL ===")
    logging.info(f"Accuracy : {acc:.4f}")
    logging.info(f"Precision: {p:.4f}")
    logging.info(f"Recall   : {r:.4f}")
    logging.info(f"F1-Score : {f:.4f}")
    
    # Fitur: Classification Report (Dicetak ke Log)
    report = classification_report(y_test, pred)
    logging.info(f"\nClassification Report:\n{report}")
    
    # Fitur Visualisasi: Confusion Matrix
    logging.info("Membuat grafik Confusion Matrix...")
    cm = confusion_matrix(y_test, pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=model.classes_)
    disp.plot(cmap="Blues")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "confusion_matrix.png"), dpi=150)
    plt.close()
    
    # Fitur Visualisasi: Pie Chart Distribusi Sentimen
    logging.info("Membuat grafik Pie Chart sentimen...")
    df["sentiment"].value_counts().plot(kind="pie", autopct="%1.1f%%", ylabel="", colors=["#4CAF50", "#F44336"])
    plt.title("Sentiment Distribution (Pie)")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "pie_chart.png"), dpi=150)
    plt.close()

    # Fitur Visualisasi: Bar Chart Jumlah Sentimen
    logging.info("Membuat grafik Bar Chart sentimen...")
    df["sentiment"].value_counts().plot(kind="bar", color=["#4CAF50", "#F44336"])
    plt.title("Sentiment Count (Bar)")
    plt.xlabel("Sentiment")
    plt.ylabel("Total Comments")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "bar_chart.png"), dpi=150)
    plt.close()

    # Fitur Visualisasi: WordCloud (Positive & Negative)
    logging.info("Membuat visualisasi WordCloud...")
    for sentiment_type in ["Positive", "Negative"]:
        text_subset = " ".join(df[df["sentiment"] == sentiment_type]["clean_comment"])
        if text_subset.strip():
            wc = WordCloud(width=800, height=400, background_color="white", max_words=100).generate(text_subset)
            plt.figure(figsize=(10, 5))
            plt.imshow(wc, interpolation="bilinear")
            plt.axis("off")
            plt.title(f"{sentiment_type} Comments WordCloud")
            plt.tight_layout()
            plt.savefig(os.path.join(OUTPUT_DIR, f"wordcloud_{sentiment_type.lower()}.png"), dpi=150)
            plt.close()
        else:
            logging.warning(f"Teks untuk sentimen {sentiment_type} kosong, WordCloud dilewati.")

    # Fitur: Menyimpan model .pkl (Menyimpan Model & Vectorizer sekaligus)
    model_path = os.path.join(OUTPUT_DIR, "svm_model.pkl")
    vec_path = os.path.join(OUTPUT_DIR, "tfidf_vectorizer.pkl")
    
    with open(model_path, "wb") as f_model:
        pickle.dump(model, f_model)
    with open(vec_path, "wb") as f_vec:
        pickle.dump(vectorizer, f_vec)
        
    logging.info(f"Model berhasil disimpan di: {model_path}")
    logging.info(f"Vectorizer berhasil disimpan di: {vec_path}")

if __name__ == "__main__":
    logging.info("=== PROGRAM SENTIMEN ANALISIS DIMULAI ===")
    raw = scrape(VIDEO_URLS, MAX_COMMENTS_PER_VIDEO)
    clean = preprocess(raw)
    labeled = pseudo_label(clean)
    train_svm(labeled)
    logging.info("=== PROGRAM SELESAI DENGAN SUKSES ===")