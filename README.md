# CS 4650 Final Project

## Problem Statement and Project Goals
With the rise of large language models (LLMs) in recent years, there has been an influx in AI-generated content in all forms of media, whether that be image, text, or more recently, video. While AI has proven to be beneficial in many contexts, it’s also prone to misuse and can be harmful in different ways. In academic settings, its misuse may impede a student's ability to learn course content, and in general, it can also be used for the dissemination of misinformation, whether malicious or not. As such, the rise in LLMs and AI-generated content has given rise to a new and equally important field, AI Content Detection. Our proposed project sits firmly in this field and aims to compare different AI content detection methods and compare their performance across the popular LLMs (ChatGPT, Gemini, Claude, etc.) and whether or not these LLMs can avoid our detection when given adversarial prompt instructions. We hope our results will inform viewers about the current field of AI Content detection and give insights on how ML can be used for such purposes. 

## Methodology
To assess the detectability of AI-generated text, we will implement several models and compare simple ML architectures with more modern deep learning architectures, allowing us to gauge the level at which AI artifacts become detectable. 

We plan to initially evaluate the performance of simple methods such as logistic regression and support vector machines (SVMs) with TF-IDF vectorization. 

Next, we will implement an LSTM using pretrained GloVe embeddings. 

Finally, we will fine-tune a pretrained DeBERTa model on this AI-generated text detection task.

## Datasets
We will be using three different kaggle datasets to ensure we have a diverse distribution of “AI” and “human” text. If able, we may also look into generating custom datasets of our own to test newer models and those not present in these three datasets. 

https://www.kaggle.com/datasets/shanegerami/ai-vs-human-text

https://www.kaggle.com/datasets/bertnardomariouskono/ai-generated-text-detection-multi-mode

https://www.kaggle.com/datasets/deepaksingh2510/ai-text-detection-dataset
