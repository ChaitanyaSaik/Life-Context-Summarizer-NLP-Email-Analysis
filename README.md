# 📧 Life Context Summarizer: NLP Pipeline for Emails

## 📌 Project Overview
The **Life Context Summarizer** is a Natural Language Processing (NLP) project designed to process, analyze, and summarize large volumes of emails.  
It uses advanced ML and NLP models to provide insights such as email summaries, sentiment, clustering, and action hints.  
This system is designed to reduce information overload by automatically organizing and prioritizing important communications.

---

## 🚀 Features
- 📥 **Data Collection**: Automated parsing of the [Enron Email Dataset](https://www.cs.cmu.edu/~enron/).
- 🧹 **Data Cleaning**: Removes duplicate emails, headers, and noise for cleaner analysis.
- ✂️ **Summarization**: Uses **BART Transformer** and **NLTK** for extractive and abstractive summarization.
- 😀 **Sentiment Analysis**: Classifies email sentiment (positive/negative/neutral) using **DistilBERT**.
- 🔑 **Keyword Extraction**: Regex and frequency-based extraction of key terms.
- 🧩 **Topic Clustering**: Groups emails using **TF-IDF** + **KMeans**.
- 📊 **Visualization**: Multiple data visualizations (Sentiment Distribution, Keywords, Email Lengths, Top Senders, etc.).
- 🏷 **Email Classification**: Logistic Regression classifier for “Important” vs. “Non-Important” emails.
- 📆 **Action Hint Detection**: Extracts dates, deadlines, and tasks for better task management.
- 🔝 **Priority Scoring**: Assigns a score to prioritize critical emails.
- 📚 **Final Digest**: Creates a structured digest for easy consumption.

---

## 📂 Project Workflow
1. **Data Collection**: Download and parse Enron Dataset.
2. **Data Cleaning**: Remove headers, duplicates, and empty content.
3. **Summarization**: Generate short summaries for each email.
4. **Sentiment Analysis**: Predict sentiment polarity.
5. **Keyword Extraction**: Identify important terms.
6. **Topic Clustering**: Group emails by themes.
7. **Classification**: Logistic Regression classifier for email importance.
8. **NER & Action Hints**: Extract names, dates, and deadlines.
9. **Priority Scoring**: Rank emails based on urgency.
10. **Visualization & Reporting**: Generate plots and summary digest.

---

## 🛠️ Tech Stack
| Component                | Technology Used                                      |
|--------------------------|-----------------------------------------------------|
| Language                 | Python 3.10+                                        |
| Notebook Environment     | Jupyter Notebook / Google Colab                     |
| NLP Libraries            | `transformers`, `nltk`, `spacy`, `rouge-score`     |
| ML Libraries             | `scikit-learn`, `pandas`, `numpy`                   |
| Visualization            | `matplotlib`, `seaborn`, `wordcloud`                |
| Dataset                  | [Enron Email Dataset](https://www.cs.cmu.edu/~enron/)|

---

## 📊 Visualizations
1. Sentiment Distribution (Bar Chart)
2. Top Keywords (Bar Chart)
3. Email Length Distribution (Histogram)
4.  Top Senders (Bar Chart)
5.  Emails Per Topic (Cluster Plot)



   <img width="619" height="468" alt="image" src="https://github.com/user-attachments/assets/10a76814-ab40-4d4b-b672-102ffd55fc41" />

   <img width="731" height="447" alt="image" src="https://github.com/user-attachments/assets/f3abef4c-26bf-46f7-a2f3-5c582209b86f" />

   <img width="693" height="382" alt="image" src="https://github.com/user-attachments/assets/c05d1e44-2132-4662-bc7f-4a509acd0780" />

   <img width="730" height="446" alt="image" src="https://github.com/user-attachments/assets/ae3a8cd0-31ed-4e73-b8aa-b4e33f42c076" />

    <img width="603" height="475" alt="image" src="https://github.com/user-attachments/assets/ec754692-9562-4737-ad55-aac6bf400432" />

---

## 📁 File Structure
```
Life-Context-Summarizer/
│
├── notebooks/
│   └── full_project_source_code_nlp.ipynb    # Jupyter Notebook
├── data/
│   └── enron_mail_20110402.tgz               # Dataset
├── images/
│   └── visualizations.png                    # Charts and Plots
├── README.md                                 # Documentation

```

---

## 📦 Installation
```bash
# Clone repository
git clone https://github.com/yourusername/Life-Context-Summarizer.git
cd Life-Context-Summarizer

# Install dependencies
pip install numpy pandas matplotlib seaborn transformers nltk spacy rouge-score scikit-learn wordcloud graphviz dateparser
```

---

## ▶️ Usage
```bash
# Run the notebook
jupyter notebook notebooks/full_project_source_code_nlp.ipynb
```
The notebook will:
- Load Enron dataset
- Summarize and classify emails
- Generate visualizations
- Output a structured email digest

---

## 📈 Future Improvements
- Fine-tune Transformer models for domain-specific summarization.
- Build a web-based UI for daily email digests.
- Add more sophisticated Named Entity Recognition (NER).
- Automate action item extraction with better context understanding.

---

## 📚 References
1. See, A., Liu, P. J., & Manning, C. D. (2017). Get To The Point: Summarization with Pointer-Generator Networks. ACL. https://doi.org/10.18653/v1/p17-1099  
2. Devlin, J., Chang, M. W., Lee, K., & Toutanova, K. (2019). BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. NAACL-HLT.  
3. Ranganathan, J., & Abuka, G. (2022). Text Summarization using Transformer Model. IEEE SNAMS.  
4. Pedregosa, F. et al. (2011). Scikit-learn: Machine Learning in Python. Journal of Machine Learning Research, 12, 2825-2830.  
5. Wolf, T. et al. (2020). Transformers: State-of-the-Art Natural Language Processing. EMNLP.  

---

## 👨‍💻 Author
**Chaitanya Sai Kurapati**  
🎓 Course Project | NLP & ML  
📧 Email: chaitanyasaikurapati@gmail.com

## 👨‍💻 Co-Author
**Bandi Dheeraj**  
🎓 Course Project | NLP & ML  
📧 Email: bandisunny11@gmail.com


---

⭐ If you like this project, don't forget to star this repo!
