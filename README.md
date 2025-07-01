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



