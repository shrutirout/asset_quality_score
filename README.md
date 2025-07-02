📂 Dataset Information

Due to GitHub's file size limitations, the full LendingClub dataset (~1.2 GB) is not included here.
- 🔗 The full dataset can be downloaded from: https://www.kaggle.com/datasets/adarshsng/lending-club-loan-data-csv
- 🛠️ To reproduce the full version, use the preprocessing notebook `1_EDA.ipynb` on the raw CSV.
- All the raw/processed datasets are present at this link for your review: https://drive.google.com/drive/folders/1WU4JOSu-9CW0uW7kT4SZ-hCSxcSI7yoa?usp=drive_link
- X_trained, X_test, y_train, y_test are stored in this link for your review: https://drive.google.com/drive/folders/1HirZ5G30Hy7RV8MOGq59SubC0pB3y2uP?usp=drive_link

Project Directory is as follows: 

asset_quality_score/
├── data/
│   ├── raw/                         # 🔹 Raw CSV data (e.g., loan.csv from Kaggle)
│   ├── processed/                   # 🔹 Cleaned dataset after EDA (processed_loan_csv.csv)
│
├── notebooks/
│   ├── 1_EDA.ipynb                  # 🧹 Data cleaning, missing value treatment, feature engineering
│   ├── 2_feature_analysis.ipynb    # 📊 Feature correlation, pruning, and transformation
│   ├── 3_baseline_models.ipynb     # 🤖 Logistic Regression, Decision Tree
│   ├── 4_advanced_models_xgboost_rf.ipynb  # 🌲 XGBoost and Random Forest + hyperparameter tuning
│   ├── 5_svm_knn_logistic_scaled.ipynb     # 📏 Logistic Regression with tuning, k-NN and SVM + SMOTE + scaling
│   ├── 6_tensorflow_pytorch.ipynb         # 🧠 Deep learning using TensorFlow and PyTorch
│   ├── 7_final_comparison.ipynb           # 📈 Comparison of all models on ROC, precision, recall, F1
|   ├── 8_asset_quality_score.ipynb         # Calculating the asset quality score using calibration and .predict_proba
│   └── 9_scoring_explainability.ipynb     # ✅ Final scoring, calibration, SHAP/LIME explainability
│
├── outputs/
│   ├── baseline results in a JSON file
│   ├── 7 other models
│   └── lime_explanation.html      # 💡 LIME HTML explanation
│   ├── X_train.csv                 # 🧪 Train/test splits
│   ├── X_test.csv
│   ├── y_train.csv
│   ├── y_test.csv
│
├── requirements.txt               # 📦 Python dependencies for full reproducibility
├── README.md                      # 📘 Project overview, methodology, results, and references

#### 📊 1. Exploratory Data Analysis (EDA)

#### ✅ Dataset Overview:
* Source: Lending Club loan dataset
* Target Variable: `loan_status` converted to `loan_status_binary`
  * `0`: Good loans (Fully Paid, Current)
  * `1`: Bad loans (Defaulted, Charged Off, etc.)

#### ✅ Column Pruning:
* Removed 40+ irrelevant or non-impactful columns like `id`, `url`, `next_pymnt_d`, hardship-related fields, etc.

#### ✅ Target Variable Summary:
* Good Loans: \~86.86%
* Bad Loans (Defaults): \~13.14%

#### ✅ Missing Value Analysis:
* Performed detailed analysis by missingness level:
  * **High Missingness (>30%)**: Dropped columns.
  * **Moderate**: Filled with median.
  * **Low**: Median/zero/mode fill.
  * **Very Low**: Custom fill based on semantics.

#### ✅ Feature Engineering & Cleaning:
* Converted `term` to numeric
* Mapped `grade` and `sub_grade` to ordinal values
* Cleaned `emp_length`, converted to numeric
* Parsed `issue_d` and `earliest_cr_line`, created `credit_history_length`

#### ✅ Output:
* Final preprocessed file saved as: `Preprocessing_Loan_CSV.csv`

### 🔍 2. Feature Analysis & Selection

#### ✅ Numerical Features:

* Initially **63 numerical columns** were identified.
* Correlation with the target variable (`loan_status_binary`) was calculated.
* Low-correlation features (correlation < `0.01`) were removed, such as:

  * `emp_length`, `chargeoff_within_12_mths`, `tot_coll_amt`, `delinq_amnt`.
  * At the end, **59 numerical features** were retained.

#### ✅ Statistical Analysis:

* Summary statistics (`mean`, `std`, percentiles, `skew`, `kurtosis`) were generated to evaluate distribution characteristics.
* Useful for identifying skewed or heavy-tailed variables that may benefit from transformation.

#### ✅ Categorical Feature Encoding:
* The number of categorical features recognized were 10.
* Initial categorical features: `home_ownership`, `verification_status`, `issue_d`, `loan_status`, `purpose`, `zip_code`, `addr_state`, `earliest_cr_line`, `initial_list_status`, `application_type`.
* Removed irrelevant or high-cardinality columns: `zip_code`, `issue_d`, `loan_status`, `earliest_cr_line`.
* Remaining categorical columns (`addr_state`, `purpose`, etc.) were **label encoded** using `LabelEncoder`.

#### 📊 Visualization
To better understand the behavior of defaulted and non-defaulted loans, we performed extensive visual exploration of key features. The goal was to identify trends, outliers, and variables with strong predictive power for asset quality modeling.

#### 📌 Correlation Heatmap of Top Features
The most insightful visualization was a **correlation heatmap** capturing the relationships between the top 20 numerical features and the binary target variable `loan_status_binary`.

**Key Observations:**
* **Top Correlated Predictors of Default:**
  * `sub_grade`, `grade`, and `int_rate` exhibited the highest positive correlation (\~0.21–0.23) with default probability, making them strong risk indicators.
  * `total_rec_prncp` and `last_pymnt_amnt` were negatively correlated, suggesting that higher repayment amounts reduce default likelihood.

* **Multicollinearity:**
  * `sub_grade`, `grade`, and `int_rate` are highly collinear, which is expected since they reflect creditworthiness tiers.
  * A cluster of credit-limit related features (`total_bc_limit`, `tot_hi_cred_lim`, `avg_cur_bal`) showed strong internal correlation, emphasizing redundancy among financial strength indicators.

* **Feature Pruning Strategy:**
  * Features with near-zero correlation to the target or high redundancy were dropped to reduce noise and improve model efficiency.

#### 📉 Other Key Visualizations (Explored During EDA)

* Interest Rate Analysis: Defaulted loans were associated with significantly higher `int_rate`, as shown through **boxplots**, **violin plots**, and **histograms**.
* Loan Amount & Installments: Higher `loan_amnt` and `installment` values were slightly more common among defaulted borrowers, although not as predictive.
* Loan Term: A clear difference in default rates was visible across loan terms, with longer-term loans showing a marginally higher default tendency.
* Borrower Behavior & Credit History: Features like `revol_util`, `bc_util`, and `inq_last_6mths` provided meaningful differentiation between risky and safe profiles.
* Repayment Features:`total_rec_late_fee` and `total_rec_int` visualizations demonstrated that defaulted borrowers often had poor repayment history or very low interest recovery.
* Demographics & Income: Features like `annual_inc` and `dti` (debt-to-income) were explored, but their predictive contribution was moderate.

## 🔎 Outlier Detection & Treatment

* **13 key features** were analyzed for outliers, including:
  * `loan_amnt`, `annual_inc`, `dti`, `int_rate`, `installment`
  * Utilization metrics: `revol_util`, `revol_bal`, `bc_util`
  * Credit limits and account metrics: `total_acc`, `open_acc`, `avg_cur_bal`, `tot_hi_cred_lim`, `total_bc_limit`

* Features with the **highest proportion of outliers**:

  | Feature          | % Outliers |
  | ---------------- | ---------- |
  | `revol_bal`      | 6.06%      |
  | `total_bc_limit` | 6.04%      |
  | `avg_cur_bal`    | 5.67%      |
  | `annual_inc`     | 4.87%      |

* **Impact on Default Rate:**

  * Outliers in `int_rate`, `revol_util`, and `bc_util` showed a **massive increase** in default rate (↑ \~16.2%)
  * Others like `loan_amnt`, `annual_inc` had lower or inverse impacts

### ✅ Outlier Treatment Strategy

| Risk Level    | Feature Examples                                      | Strategy                                  |
| ------------- | ----------------------------------------------------- | ----------------------------------------- |
| **High-risk** | `int_rate`, `revol_util`, `bc_util`, `dti`            | Capped at IQR bounds (preserve risk info) |
| **Low-risk**  | `loan_amnt`, `annual_inc`, `total_acc`, `avg_cur_bal` | Winsorized at 1st–99th percentiles        |

This dual strategy allowed us to retain useful risk-related outliers while normalizing long-tail distributions in low-impact features.

## 🛠️ Feature Engineering

To enhance model expressiveness, several **domain-specific features** were engineered using financial ratios and borrower behavior patterns:

| Feature Name           | Description                                                             |
| ---------------------- | ----------------------------------------------------------------------- |
| `payment_burden`       | Installment-to-monthly-income ratio                                     |
| `loan_income_ratio`    | Loan amount relative to annual income                                   |
| `utilization_ratio`    | Revolving balance to credit limit usage                                 |
| `active_account_ratio` | Proportion of active open accounts                                      |
| `risk_flag_count`      | Count of borrower red flags (high DTI, delinquencies, high utilization) |
| `delinquency_score`    | Aggregated delinquency and public record risk                           |
| `credit_quality`       | Score inversely related to credit grade                                 |

After preprocessing, the dataset had:

* ✅ **2.26 million records**
* ✅ **74 total features** (post-engineering and capping)

#### ✅ Final Outcome:

* Cleaned and encoded feature set ready for modeling.
* The output DataFrame includes **clean, normalized, robust numerical and encoded categorical variables** correlated with the target.
* Ensures the model trains on the most informative features only.
* Final preprocessed file saved as: `processed_loan_csv.csv`

## 🧠 Model Development & Evaluation

To predict the likelihood of default and assign an asset quality score, we trained and evaluated multiple machine learning and deep learning models using cleaned and feature-engineered data.

### ⚙️ Data Preparation

* **Target Variable**: `loan_status_binary`
  (0 = Non-default, 1 = Default)

* **Train-Test Split**:
  * Stored in:
    * `outputs/X_train.csv`
    * `outputs/X_test.csv`
    * `outputs/y_train.csv`
    * `outputs/y_test.csv`

* **Scaling**:
  * `StandardScaler()` applied to:
    * Logistic Regression
    * SVM
    * k-NN
    * Deep Learning (TensorFlow, PyTorch)
  * Not applied to tree-based models (XGBoost, RF, DT)

* **Handling Class Imbalance**:
  * Dataset is heavily imbalanced (only \~13% default)
  * **SMOTE** applied for:
    * Logistic Regression (SMOTE)
    * SVM
  because they are susceptible to class imbalance and Distance-based.

### 🤖 Models Trained

| Category           | Models                                                                         |
| ------------------ | ------------------------------------------------------------------------------ |
| **Traditional ML** | Logistic Regression, Logistic Regression (SMOTE), Decision Tree, Random Forest |
| **Distance-based** | k-NN                                                                           |
| **Kernel Methods** | SVM                                                                            |
| **Ensemble**       | XGBoost                                                                        |
| **Deep Learning**  | TensorFlow MLP, PyTorch MLP                                                    |

### 🔧 Hyperparameter Tuning

* **XGBoost**:
  * GridSearchCV used for:
    * `max_depth`, `learning_rate`, `n_estimators`, `subsample`, `colsample_bytree`
  * Best model saved at:
    `outputs/best_xgboost.pkl`

* **Deep Learning (TF & PyTorch)**:
  * Optimized for:
    * Epochs
    * Hidden layers
    * Dropout rates
    * Early stopping
  * Used ReLU activations, Adam optimizer, batch normalization

### 📈 Evaluation Metrics
Models were evaluated using:
* Accuracy
* Precision
* Recall
* F1 Score
* AUC-ROC

### 📊 Final Model Comparison

| Model                       | Accuracy | Precision | Recall | F1 Score | AUC    |
| --------------------------- | -------- | --------- | ------ | -------- | ------ |
| **TensorFlow**              | 0.9610   | 0.9379    | 0.7261 | 0.8185   | 0.9657 |
| **PyTorch**                 | 0.9530   | 0.9006    | 0.6884 | 0.7804   | 0.9561 |
| **XGBoost**                 | 0.9415   | 0.9388    | 0.5534 | 0.6964   | 0.9551 |
| Logistic Regression (SMOTE) | 0.8080   | 0.3674    | 0.8087 | 0.5052   | 0.8925 |
| Logistic Regression         | 0.9074   | 0.7628    | 0.3425 | 0.4727   | 0.8903 |
| SVM                         | 0.9016   | 0.7873    | 0.2587 | 0.3895   | 0.8879 |
| Random Forest               | 0.8981   | 0.9013    | 0.1793 | 0.2991   | 0.8870 |
| Decision Tree               | 0.8856   | 0.6672    | 0.1134 | 0.1938   | 0.7783 |
| k-NN                        | 0.8755   | 0.4513    | 0.1256 | 0.1965   | 0.6730 |


### ✅ Final Model Selection: **XGBoost**

Although the **TensorFlow model** demonstrated slightly higher recall and F1 scores, we ultimately selected **XGBoost** for final deployment due to a combination of practical, interpretability, and scoring reasons:
* **Interpretability**:
  XGBoost is inherently more interpretable compared to deep learning. It seamlessly integrates with tools like **SHAP** and **LIME**, allowing us to explain **why a loan was marked risky** — a crucial requirement for lending systems.
* **Probability-Based Scoring**:
  XGBoost’s `.predict_proba()` method produces **well-calibrated default probabilities** which we used to derive **asset quality scores** on a 0–100 scale. These scores aligned meaningfully with observed default densities.
* **Calibration Compatibility**:
  XGBoost responds well to **probability calibration techniques** like **Isotonic Regression** and **Platt Scaling**, improving the reliability of score-based thresholds (e.g., score > 95 = excellent quality). This enhanced the separation between defaulted and non-defaulted loan distributions.

* **Performance vs Practicality**:

  | Metric         | TensorFlow | XGBoost  |
  | -------------- | ---------- | -------- |
  | F1 Score       | **0.8185** | 0.6964   |
  | AUC-ROC        | **0.9657** | 0.9551   |
  | Training Speed | ❌ Slower   | ✅ Faster |
  | Explainability | ❌ Limited  | ✅ High   |

* **Efficiency**:
  XGBoost is significantly faster to train and tune, especially with cross-validation and large tabular data like this LendingClub dataset.

✅ **Summary**: XGBoost provided an ideal trade-off between predictive strength, model interpretability, scoring calibration, and speed — making it the optimal choice for production scoring.

## 📈 **Probability Scoring and Calibration Workflow**

### 1️⃣ Initial Probability Prediction
After training the **XGBoost** model, we generated *raw default probabilities* using:

```python
default_probs = model_xgb.predict_proba(X_scaled)[:, 1]
```

To create a more intuitive scoring scale, these probabilities were inverted (since *higher probability = higher risk*) and scaled to a **0–100 Asset Quality Score**:

✅ **Outcome:**
* Each loan received a continuous score between 0 (worst quality, highest risk) and 100 (best quality, lowest risk).
* **Observation:**
  The model was **underconfident**—even genuinely high-risk loans often received mid-range scores rather than very low ones, as visible in the first graph.
---

### 2️⃣ Probability Calibration
To correct this underconfidence and improve the reliability of the scores, we applied **Isotonic Regression calibration**:

  * It is non-parametric and can flexibly map predicted probabilities to observed frequencies without assuming a logistic shape.
  * Improves how well predicted probabilities align with actual default rates.

* Recomputed the asset scores:
  The second density plot showed **better separation**: Defaulted loans were much more concentrated in the lowest deciles.
---

### 3️⃣ Decile Binning

To further simplify interpretation, the calibrated scores were divided into **10 deciles**:

* **Decile 1** = worst 10% of loans by predicted quality.
* **Decile 10** = best 10%.

### 4️⃣ Assigning Integer Scores and Grades

To create even simpler business-friendly labels:

* **Integer Asset Quality Score:** 1–10 (based on decile).
* **Grades:**

| Decile | Grade                 |
| ------ | --------------------- |
| 1      | E - Extremely Risky   |
| 2      | D - Risky             |
| 3–4    | C - Medium Quality    |
| 5–6    | B - Good Quality      |
| 7–10   | A - Excellent Quality |

### 5️⃣ Saving Final Outputs

The enriched dataset included:
* Raw calibrated probabilities
* Continuous 0–100 asset quality scores
* Integer decile scores
* Categorical grades

## 🧠 **Model Explainability**

**1️⃣ SHAP Analysis (Global Interpretability)**
* **Top Positive Contributors to Default Risk:**
  * Low `total_rec_prncp` (principal repaid)
  * Low `last_pymnt_amnt` (recent payment)
  * High `int_rate` (interest rate)
  * Poor `sub_grade` and `grade`
  * High `installment` amounts
* Features like `total_rec_late_fee` and `payment_burden` also contributed to increased risk.
* Colors in the summary plot:
  * **Red:** High feature values (e.g., high interest rate)
  * **Blue:** Low feature values
* The SHAP values show how each feature pushes the prediction towards default or non-default.


**2️⃣ LIME Analysis (Local Interpretability)**
* For a single observation (index `100`):

  * The prediction was **70% default probability**.
  * LIME identified the top features pushing the probability towards default (e.g., high `grade`, high `payment_burden`) and non-default (e.g., higher `last_pymnt_amnt`).
  * The LIME explanation was saved as an **interactive HTML file** for easy inspection.









