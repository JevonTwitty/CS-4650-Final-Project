# AI-Generated Text Detection
### Final Project for **CS 4650: Natural Language Processing**

This repository contains the code for our **AI-generated Text Detection** project. We utilize Logistic Regression, LSTMs, SVMs, and pretrained BERT models to compare their performance in AI detection across several datasets.

---

## 1. Problem Statement and Project Goals
With the rise of large language models (LLMs) in recent years, there has been an influx in AI-generated content in all forms of media, whether that be image, text, or more recently, video. While AI has proven to be beneficial in many contexts, it’s also prone to misuse and can be harmful in different ways. In academic settings, its misuse may impede a student's ability to learn course content, and in general, it can also be used for the dissemination of misinformation, whether malicious or not. As such, the rise in LLMs and AI-generated content has given rise to a new and equally important field, AI Content Detection. Our proposed project sits firmly in this field and aims to compare different AI content detection methods and compare their performance across the popular LLMs (ChatGPT, Gemini, Claude, etc.) and whether or not these LLMs can avoid our detection when given adversarial prompt instructions. We hope our results will inform viewers about the current field of AI Content detection and give insights on how ML can be used for such purposes. 

## 2. Methodology
To assess the detectability of AI-generated text, we will implement several models and compare simple ML architectures with more modern deep learning architectures, allowing us to gauge the level at which AI artifacts become detectable. 

We plan to initially evaluate the performance of simple methods such as logistic regression and support vector machines (SVMs) with TF-IDF vectorization. 

Next, we will implement an LSTM using pretrained GloVe embeddings. 

Finally, we will fine-tune a pretrained DeBERTa model on this AI-generated text detection task.

---

## 3. Installation

First, ensure you have [PyTorch](https://pytorch.org/get-started/locally/) installed for your specific hardware (CUDA, MPS, or CPU). Then, install the remaining dependencies:

```bash
pip install -r requirements.txt
```

---

## 4. Download Datasets

We train and evaluate our models on a diverse set of datasets.

### **Datasets**
Download the following datasets and place the `.csv` files in the `data/` directory.

| Dataset Source | Description | Link |
| :--- | :--- | :--- |
| **AI vs. Human Text** | Large-scale balanced corpus | [Kaggle](https://www.kaggle.com/datasets/shanegerami/ai-vs-human-text) |
| **Multi-Mode Detection** | Multi-generator (GPT/Llama) | [Kaggle](https://www.kaggle.com/datasets/bertnardomariouskono/ai-generated-text-detection-multi-model) |
| **AI Text Detection** | Mixed academic & creative writing | [Kaggle](https://www.kaggle.com/datasets/deepaksingh2510/ai-text-detection-dataset) |
| **DAIGT V2** | High-quality prompt-engineered data | [Kaggle](https://www.kaggle.com/datasets/thedrcat/daigt-v2-train-dataset) |

### **Pre-trained Embeddings**
We support both GloVe and FastText. Place these in `data/embeddings/`.

* **GloVe:** [nlp.stanford.edu/projects/glove](https://nlp.stanford.edu/projects/glove/)
* **FastText:** [fasttext.cc/docs/en/english-vectors.html](https://fasttext.cc/docs/en/english-vectors.html)

> **Note:** We provide pre-computed `.npy` embedding matrices for each dataset. Using these avoids the need to rebuild the matrix from the raw GloVe/FastText files.

---

## 5. Suggested Directory Structure

```text
├── data/
│   ├── ai_vs_human_text.csv
│   ├── daigt_v2_train.csv
│   └── embeddings/
│       ├── fast_text/
│       └── GloVe/
├── data_provider/
│   └── data_loader.py
├── BERT/
│   ├── BERT.ipynb
│   └── BERT.py
├── LogisticRegression/
│   ├── logistic_regression.ipynb
│   └── logistic_regression.py
├── LSTM/
│   ├── LSTM.ipynb
│   ├── LSTM.py
│   └── (files for embedding matrices)
├── SVM/
│   ├── SVM.ipynb
│   └── SVM.py
├── .gitignore
├── .README.md
└── requirements.txt
```

---
