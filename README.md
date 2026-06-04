# Intelligent Hybrid Recommendation System for Profile-Based Matching
# Intelligent Hybrid Recommendation System for Profile-Based Matching

A career-goal-based professional matching system built as a first-year 
Data Science major project.

## What it does
Recommends the Top-5 most compatible users based on:
- Professional profile similarity (TF-IDF + Cosine Similarity)
- MBTI personality compatibility
- Location proximity
- Logistic Regression trained on Accept/Reject feedback

## Key methodological features
- User-level train/test split (zero data leakage)
- Majority-class DummyClassifier baseline
- StandardScaler + class_weight='balanced'
- 5-fold stratified cross-validation
- Mann-Whitney U test for recommendation quality validation

## Results
| Baseline accuracy | 47.7% |
| LR Accuracy | 58.3% |
| ROC-AUC (CV) | 64.7%.% ± 7.2% |

Honest finding: the three features have limited predictive power on a 
75-user dataset.

## Stack
Python · scikit-learn · NLTK · scipy · pandas · matplotlib · seaborn

## How to run
1. Open the `.ipynb` file in Google Colab
2. Upload both CSV files when prompted in Section 2
3. Run all cells top to bottom
