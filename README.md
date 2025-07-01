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

#### ✅ Statistical Analysis:

* Summary statistics (`mean`, `std`, percentiles, `skew`, `kurtosis`) were generated to evaluate distribution characteristics.
* Useful for identifying skewed or heavy-tailed variables that may benefit from transformation.

#### ✅ Categorical Feature Encoding:
* The number of categorical features recognized were 10.
* Initial categorical features: `home_ownership`, `verification_status`, `issue_d`, `loan_status`, `purpose`, `zip_code`, `addr_state`, `earliest_cr_line`, `initial_list_status`, `application_type`.
* Removed irrelevant or high-cardinality columns: `zip_code`, `issue_d`, `loan_status`, `earliest_cr_line`.
* Remaining categorical columns (`addr_state`, `purpose`, etc.) were **label encoded** using `LabelEncoder`.

At the end, **59 numerical features** were retained.

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

Interest Rate Analysis: Defaulted loans were associated with significantly higher `int_rate`, as shown through **boxplots**, **violin plots**, and **histograms**.
Loan Amount & Installments: Higher `loan_amnt` and `installment` values were slightly more common among defaulted borrowers, although not as predictive.
Loan Term: A clear difference in default rates was visible across loan terms, with longer-term loans showing a marginally higher default tendency.
Borrower Behavior & Credit History: Features like `revol_util`, `bc_util`, and `inq_last_6mths` provided meaningful differentiation between risky and safe profiles.
Repayment Features:`total_rec_late_fee` and `total_rec_int` visualizations demonstrated that defaulted borrowers often had poor repayment history or very low interest recovery.
Demographics & Income: Features like `annual_inc` and `dti` (debt-to-income) were explored, but their predictive contribution was moderate.

#### ✅ Final Outcome:

* Cleaned and encoded feature set ready for modeling.
* The output DataFrame includes **robust numerical and encoded categorical variables** correlated with the target.
* Ensures the model trains on the most informative features only.
* Final preprocessed file saved as: `processed_loan_csv.csv`



