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

