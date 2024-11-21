# Credit Card Fraud Detection

This repository contains a machine learning models aimed at detecting credit card fraud using an imbalanced dataset. The main focus was to achieve the best possible **F1-score** while also considering the **PR AUC score** as a secondary metric.

## Project Overview

Credit card fraud detection is a critical task that involves identifying fraudulent transactions in real-time. The dataset used in this project was highly imbalanced, with only a small fraction of the transactions being fraudulent. To address this challenge, several techniques and models were applied, including data resampling, feature scaling, and various classification algorithms.

---

## Key Features

- **Imbalanced Dataset Handling:** Both undersampling and oversampling techniques were explored. Undersampling harmed performance, particularly for the majority class, while oversampling led to slight improvements.
- **Privacy-Preserving Data:** Most features were nameless and scaled for privacy, except for `Time` and `Amount`. All features were rescaled for consistency.
- **Metrics:** The primary evaluation metric was the **F1-score**, focusing on the minority class. The **PR AUC score** was also reported for additional insights.
- **Model Comparison:** Performance of multiple machine learning models was compared on train, validation, and test datasets.

---

## Dataset

The dataset contains anonymized credit card transaction data. Key features include:

- **Time:** Seconds elapsed between the transaction and the first transaction in the dataset.
- **Amount:** Transaction amount.
- **Anonymized Features:** Remaining features were scaled and unnamed for privacy.

---

## Methods

1. **Data Preprocessing:**
   - Rescaling all features to ensure uniformity.
   - Addressing class imbalance using oversampling and undersampling techniques.

2. **Model Training and Evaluation:**
   - Tested multiple classifiers: Logistic Regression, Random Forest, Neural Network (MLP), XGBoost, and Voting Classifier.
   - Compared performance on the imbalanced dataset (without oversampling) and oversampled dataset.

3. **Performance Metrics:**
   - **F1-Score**: Captures the balance between precision and recall for the minority class.
   - **PR AUC**: Provides insights into the model's ability to distinguish between classes in an imbalanced setting.

---

## Results

### Without Oversampling

| Model                | Train F1-Score | Val F1-Score | Test F1-Score | Test PR AUC |
|----------------------|----------------|--------------|---------------|-------------|
| Logistic Regression  | 0.770          | 0.753        | 0.726         | 0.726       |
| Random Forest        | 0.983          | 0.840        | 0.849         | 0.842       |
| Neural Network       | 0.964          | 0.817        | 0.834         | 0.839       |
| XGBoost              | 1.000          | 0.877        | 0.885         | 0.866       |
| Voting Classifier    | 1.000          | 0.864        | 0.886         | 0.873       |

### With Oversampling

| Model                | Train F1-Score | Val F1-Score | Test F1-Score | Test PR AUC |
|----------------------|----------------|--------------|---------------|-------------|
| Logistic Regression  | 0.801          | 0.775        | 0.783         | 0.718       |
| Random Forest        | 0.995          | 0.859        | 0.856         | 0.829       |
| Neural Network       | 0.980          | 0.789        | 0.822         | 0.842       |
| XGBoost              | 1.000          | 0.880        | 0.889         | 0.864       |
| Voting Classifier    | 1.000          | 0.881        | 0.890         | 0.869       |

### Insights:
- **Undersampling** negatively impacted model performance, especially on the majority class.
- **Oversampling** slightly improved performance but introduced marginal gains.

---

## Conclusion

- **Best Model:** Voting Classifier achieved the highest F1-score and PR AUC across all datasets without/with oversampling.
- **Key Takeaways:**
  - Addressing class imbalance is crucial for fraud detection.
  - Oversampling can help, but other techniques (e.g., cost-sensitive learning) should also be explored.

---

## Installation and Usage

1. Clone the repository:
   ```bash
   git clone https://github.com/MO-87/CC-Fraud-Detection-with-Imbalanced-Data.git
   ```
2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the project:
   ```bash
   python main.py
   ```

---

## Future Work

- Experiment with cost-sensitive algorithms to handle imbalance.
- Explore additional preprocessing techniques for feature scaling.
- Investigate alternative metrics like MCC (Matthews Correlation Coefficient).

---

## Acknowledgments

- Dataset: Provided as part of a credit card fraud detection challenge in **Dr. Mostafa Saad** ML Course.
- Libraries Used: `scikit-learn`, `XGBoost`, `pandas`, etc.

Feel free to contribute or raise issues for further improvements!
