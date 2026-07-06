YouTube Sentiment Analysis with DistilBERT & SVM

Proyek ini dirancang untuk mengumpulkan, membersihkan, dan menganalisis sentimen komentar dari video YouTube menggunakan pendekatan kombinasi Deep Learning (DistilBERT) untuk pelabelan otomatis dan Machine Learning (Linear SVM) untuk klasifikasi.

🚀 Fitur Utama & Penjelasan Kode

Berikut adalah penjelasan logika kodingan per fitur utama yang digunakan dalam proyek ini:

<details>
<summary>🔍 <b>1. Scraping Komentar YouTube</b> (Klik untuk melihat kode)</summary>

Fitur ini menggunakan library `youtube_comment_downloader` untuk mengambil komentar secara massal berdasarkan daftar URL video yang ditentukan dan menyimpannya ke bentuk data mentah (`.csv`).
```python
def scrape(video_urls, max_comments):
    downloader = YoutubeCommentDownloader()
    rows = []
    for url in video_urls:
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
        except Exception as e:
            logging.error(f"Gagal melakukan scraping pada {url}: {e}")
    
    df = pd.DataFrame(rows)
    df.to_csv("output/1_raw_comments.csv", index=False, encoding="utf-8-sig")
    return df
