# loan_default_prediction_ml
Credit risk scoring model with LightGBM and cross-validation for a competetion
# Loan Default Prediction – Credit Risk Scoring with Machine Learning

This project develops a binary classification model to predict whether a borrower will default on a loan within 12 months. The focus is on building a robust feature engineering pipeline and evaluating model performance using cross-validated AUC.

Note: The original dataset was from a private competition and is not included in this repository.  
The code is written as a template and can be adapted to any similar loan/default dataset.

---

## 1. Techniques and Tools

- Language: Python  
- Libraries: `pandas`, `numpy`, `scikit-learn`, `lightgbm`  
- Main methods:
  - Feature engineering on tabular credit/financial data
  - Encoding categorical variables
  - Stratified K-Fold cross-validation
  - Model evaluation using AUC

---

## 2. Repository Structure

- `loan_default_model.py`  
  - Loads training and test data  
  - Applies column dropping and basic cleaning  
  - Creates engineered features (e.g., total income, debt-to-income ratio, experience in months)  
  - Encodes categorical variables using label encoding  
  - Trains a LightGBM classifier with cross-validation  
  - Generates a submission file with predicted default probabilities

---

## 3. How to Use the Code (Template)

1. Prepare your own CSV files:

   - Training data containing features and a binary target column  
   - Test data containing the same features (without target)  
   - Submission template file with an identifier column and target column placeholder

2. Update the configuration at the top of `loan_default_model.py`:

   - `TRAIN_PATH` – path to your training data  
   - `TEST_PATH` – path to your test data  
   - `SUB_TEMPLATE_PATH` – path to your submission template  
   - `OUTPUT_PATH` – desired output file name  
   - `TARGET_COL` – name of your target column

3. Install required packages:

   ```bash
   pip install pandas numpy scikit-learn lightgbm
