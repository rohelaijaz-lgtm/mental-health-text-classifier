Mental Health Text Classifier
A natural language processing (NLP) project that classifies text statements into mental health categories using machine learning. Built in Python using scikit-learn and pandas.
**🔗 Live Demo:** [https://mental-health-text-classifier-b5ek4evz6gbmpm93h2ywpg.streamlit.app/]

Background and Motivation
Mental health stigma remains a significant barrier to help-seeking behaviour, not only in Asia but across Europe as well. Many individuals experiencing mental health difficulties turn to anonymous online communities before ever consulting a professional. They find community, shared experience, and a sense of being understood in spaces like Reddit.
This project explores whether machine learning can detect patterns in self-expressed mental health text, not to diagnose, but to offer a structured reflection of what someone's words may indicate, and to encourage professional consultation.

What This Project Does
A Logistic Regression classifier trained on 26,350 real forum statements categorises input text into one of 7 clinical labels:

Suicidal
Anxiety
Depression
Bipolar
Personality Disorder
Stress
Normal


Dataset

Source: Sentiment Mental Health Dataset (Kaggle)
Size: 26,350 statements
Columns: statement (text), status (label)
Labels: Self-selected based on community membership, not formal clinical diagnosis


Methods

Text preprocessing — lowercasing, URL removal, punctuation and number removal, whitespace normalisation
TF-IDF Vectorisation (max 10,000 features)
80/20 train-test split
Logistic Regression classifier
Evaluation using accuracy, precision, recall, and F1-score


Results
| Category | Precision | Recall | F1 |
|---|---|---|---|
| Normal | 0.92 | 0.98 | 0.95 |
| Anxiety | 0.78 | 0.82 | 0.80 |
| Bipolar | 0.83 | 0.66 | 0.74 |
| Suicidal | 0.69 | 0.82 | 0.75 |
| Personality Disorder | 0.82 | 0.75 | 0.78 |
| Depression | 0.60 | 0.57 | 0.58 |
| Stress | 0.90 | 0.59 | 0.71 |


## Model Comparison

| Model | Accuracy |
|---|---|
| LightGBM | 0.7769 |
| Logistic Regression | 0.7682 |
| SVM | 0.7614 |
| XGBoost | 0.7557 |
| Neural Network | 0.7076 |
| Random Forest | 0.7021 |
| KNN | 0.1277 |

Choosing between ML models is analogous to selecting diagnostic tests — each has different sensitivity and specificity for different conditions. Just as a blood culture is the gold standard but resource-intensive, more complex models like LightGBM achieve marginally higher accuracy but at the cost of interpretability. For clinical applications where explainability matters, Logistic Regression remains a defensible choice.

Critically, LightGBM and Logistic Regression achieved identical Suicidal recall (0.82) — the most clinically important metric in this application. Minimising missed suicidal statements matters more than overall accuracy.


Usage
# Classify any new text statement
print(classify_statement("I feel completely alone and no one understands me"))
# Output: Suicidal

print(classify_statement("I can't stop worrying about everything all the time"))
# Output: Anxiety

print(classify_statement("I went for a run today and feeling pretty good"))
# Output: Normal

Clinical Limitations

Labels reflect self-identification, not clinical diagnosis — a form of misclassification bias
Not a diagnostic tool — predictions should never replace professional clinical assessment
Class imbalance — Stress is underrepresented, affecting model performance for that category
Depression recall is 57% — nearly half of depression cases are missed by the current model
Suicidal recall is 82% — approximately 1 in 5 suicidal statements is not correctly identified
Self-diagnosis risk — users must be clearly informed that this tool is a reflective aid, not a diagnosis
Dataset limitations — English language, Reddit-sourced, may not generalise to other populations or languages


Intended Use
This tool is not a substitute for professional help. It is intended as a reflective aid for individuals who already engage with online mental health communities,  offering a structured perspective on their self-expressed experiences and encouraging professional consultation.
If patient-centred care is a genuine commitment, the language patients use to describe their own experiences deserves clinical attention especially in spaces where they feel safe enough to be honest.

Disclaimer
This project is for educational and portfolio purposes only. It is not a clinical tool and must not be used for diagnosis or treatment decisions. If you or someone you know is struggling, please seek professional support.

Tech Stack

Python
pandas
scikit-learn
matplotlib
Google Colab


Author
Rohel — Physician | MSc Public Health (Epidemiology & Biostatistics) | MSc Digital Health candidate
