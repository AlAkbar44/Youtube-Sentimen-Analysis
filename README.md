# 🎥 YouTube Sentiment Analysis with DistilBERT & Linear SVM

This project is an end-to-end Natural Language Processing (NLP) pipeline designed to collect, preprocess, and analyze YouTube comments automatically. It combines **DistilBERT**, a pretrained transformer model, for automatic sentiment pseudo-labeling with **Linear Support Vector Machine (Linear SVM)** for sentiment classification. The project also generates visualizations and exports processed datasets for further analysis.

---

# 🚀 Features

- Scrape YouTube comments from one or multiple videos
- Automatic text preprocessing and cleaning
- URL, mentions, numbers, and special character removal
- English stopword removal
- Automatic pseudo-labeling using DistilBERT
- TF-IDF feature extraction
- Sentiment classification using Linear SVM
- Model evaluation using Confusion Matrix
- Sentiment distribution visualization (Pie Chart)
- Word Cloud generation for Positive and Negative comments
- Automatic export of processed datasets, trained models, and visualizations
- Detailed execution logging for monitoring the analysis process

---

# 📈 Workflow

```text
YouTube URLs
      │
      ▼
Comment Scraping
      │
      ▼
Text Cleaning
(URLs, Mentions, Symbols,
Numbers, Stopwords)
      │
      ▼
Pseudo Labeling
(DistilBERT)
      │
      ▼
TF-IDF Vectorization
      │
      ▼
Linear SVM Training
      │
      ▼
Model Evaluation
(Accuracy, Classification Report,
Confusion Matrix)
      │
      ▼
Visualization
(Pie Chart & Word Clouds)
      │
      ▼
Export Results
(CSV Files, Images,
Trained Model)
```

---

# 🛠 Technologies

- Python
- Transformers (Hugging Face)
- DistilBERT
- Scikit-learn
- Linear Support Vector Machine (Linear SVM)
- TF-IDF Vectorizer
- Pandas
- NumPy
- Matplotlib
- WordCloud
- youtube-comment-downloader
- Pickle
- Logging

---

# 📌 Code Overview

Below is the implementation workflow and code snippets for each major component of the project.

## 🔍 1. YouTube Comment Scraping

This module uses the `youtube-comment-downloader` library to collect comments from one or multiple YouTube videos and store them as a raw CSV dataset.

<pre><code>
def scrape(video_urls, max_comments):
    downloader = YoutubeCommentDownloader()
    rows = []
    for url in video_urls:
        logging.info(f"Starting scraping for URL: {url}")
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
            logging.info(f"Collected {count} comments from {url}")
        except Exception as e:
            logging.error(f"Failed to scrape {url}: {e}")
        time.sleep(1)

    if not rows:
        logging.warning("No comments were collected.")

    df = pd.DataFrame(rows)
    output_path = os.path.join(OUTPUT_DIR, "1_raw_comments.csv")
    df.to_csv(output_path, index=False, encoding="utf-8-sig")
    logging.info(f"Raw dataset saved to {output_path}")
    return df
</code></pre>

---

## 🧹 2. Text Preprocessing

Collected comments are cleaned by removing URLs, mentions, numbers, special characters, and English stopwords before feature extraction.

<pre><code>
def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"@\w+", " ", text)
    text = re.sub(r"[^a-zA-Z\s]", " ", text)

    text = " ".join(
        w for w in text.split()
        if w not in ENGLISH_STOP_WORDS
    )

    return text.strip()
</code></pre>

---

## 🏷️ 3. Automatic Pseudo Labeling with DistilBERT

Since scraped comments are unlabeled, the project uses the pretrained `distilbert-base-uncased-finetuned-sst-2-english` model to generate automatic sentiment labels.

<pre><code>
def pseudo_label(df):
    logging.info("Starting pseudo labeling with DistilBERT...")

    clf = pipeline(
        "sentiment-analysis",
        model="distilbert-base-uncased-finetuned-sst-2-english"
    )

    labels = []

    for t in df["clean_comment"]:
        result = clf(t[:512])[0]
        labels.append(
            "Positive"
            if result["label"] == "POSITIVE"
            else "Negative"
        )

    df["sentiment"] = labels
    return df
</code></pre>

---

## 🤖 4. TF-IDF Feature Extraction & Linear SVM Training

Cleaned text is transformed into TF-IDF vectors before training a Linear Support Vector Machine classifier using an 80:20 train-test split.

<pre><code>
def train_svm(df):
    vectorizer = TfidfVectorizer(max_features=5000)

    X = vectorizer.fit_transform(df["clean_comment"])
    y = df["sentiment"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    model = LinearSVC()
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
</code></pre>

---

# 📂 Project Structure & Output

After execution, the project automatically organizes all generated files into the following structure:

```text
.
├── sentimen.py
├── requirements.txt
├── process.log
├── README.md
└── output/
    ├── 1_raw_comments.csv
    ├── 2_clean_comments.csv
    ├── 3_pseudo_labels.csv
    ├── svm_model.pkl
    ├── confusion_matrix.png
    ├── pie_chart.png
    ├── wordcloud_positive.png
    ├── wordcloud_negative.png
    └── ...
```

---

# ⚙ Installation

### 1. Clone the Repository

```bash
git clone https://github.com/AlAkbar44/Youtube-Sentimen-Analysis.git
```

### 2. Navigate to the Project Directory

```bash
cd Youtube-Sentimen-Analysis
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Application

```bash
python sentimen.py
```

> **macOS/Linux users:** Use `python3` and `pip3` if required by your Python installation.

---

# 📊 Generated Outputs

The pipeline automatically generates:

- Raw YouTube comments dataset
- Cleaned dataset
- Pseudo-labeled dataset
- Trained Linear SVM model
- Confusion Matrix
- Classification Report
- Pie Chart
- Positive Word Cloud
- Negative Word Cloud
- Execution log

---

# 👨‍💻 Author

**Al Akbar Himawan**

Junior Data Analyst | Python | SQL | Machine Learning | NLP | Data Visualization

- GitHub: https://github.com/AlAkbar44
- LinkedIn: https://www.linkedin.com/in/alakbarhimawan
- Email: himawanalakbar6@gmail.com
