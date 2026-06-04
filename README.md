# King County House Sales Analysis

Exploratory data analysis and regression modeling for house sale prices in King County, Washington, USA. This project uses Python to inspect housing features, visualize relationships with sale price, and evaluate baseline predictive models.

## Project Overview

This notebook was completed as an IBM data science final project. The analysis studies residential house sales in King County, including Seattle, from May 2014 to May 2015. The main target variable is `price`, and the project explores how property attributes such as living area, grade, waterfront status, location, and number of bathrooms relate to market value.

## Objectives

- Load and inspect the King County housing dataset.
- Clean missing values in selected columns.
- Explore price patterns with descriptive statistics and visualizations.
- Identify features most strongly related to house price.
- Build regression models for price prediction.
- Compare model performance using R-squared scores.

## Dataset

- File: `data/housing.csv`
- Records: 21,613 house sales
- Columns: 22, including an index column
- Geographic scope: King County, Washington, USA
- Time period: May 2014 to May 2015
- Target variable: `price`

Key features include:

- `bedrooms`, `bathrooms`, `floors`
- `sqft_living`, `sqft_lot`, `sqft_above`, `sqft_basement`
- `waterfront`, `view`, `condition`, `grade`
- `yr_built`, `yr_renovated`
- `zipcode`, `lat`, `long`
- `sqft_living15`, `sqft_lot15`

## Workflow

1. Import Python libraries.
2. Load the housing dataset.
3. Inspect data types, summary statistics, and missing values.
4. Drop non-predictive identifier columns.
5. Fill missing bedroom and bathroom values using column means.
6. Analyze distributions and relationships with visualizations.
7. Evaluate correlations with `price`.
8. Train linear regression, polynomial regression pipeline, and Ridge regression models.
9. Score models using R-squared.

## Methods Used

- Data cleaning and preprocessing
- Exploratory data analysis
- Correlation analysis
- Boxplots and regression plots
- Linear regression
- Polynomial feature transformation
- Ridge regression
- Train/test splitting
- R-squared model evaluation

## Results

Saved notebook outputs include the following R-squared scores:

| Model | Features | Evaluation | R-squared |
| --- | --- | --- | --- |
| Linear Regression | `sqft_living` | Full dataset | 0.4929 |
| Linear Regression | Selected housing features | Full dataset | 0.6577 |
| Pipeline: StandardScaler + PolynomialFeatures + LinearRegression | Selected housing features | Full dataset | 0.7512 |
| Ridge Regression | Selected housing features | Test set | 0.6479 |
| Polynomial Ridge Regression | Degree-2 transformed selected features | Test set | 0.7003 |

The strongest modeling results in the saved notebook come from polynomial feature engineering combined with regression-based estimators.

## Technology Stack

Primary language: Python

Main libraries:

- pandas: data loading, inspection, cleaning, and tabular analysis
- NumPy: numerical operations
- matplotlib: plot rendering
- seaborn: statistical visualizations
- scikit-learn: preprocessing, pipelines, linear regression, Ridge regression, and model evaluation
- Jupyter Notebook: interactive analysis environment

Estimated project library focus:

| Area | Approx. Share |
| --- | ---: |
| pandas / data wrangling | 35% |
| scikit-learn / modeling | 30% |
| seaborn + matplotlib / visualization | 25% |
| NumPy / numerical support | 10% |

These percentages describe the notebook's practical library usage, not GitHub's automatic language statistics.

## Repository Structure

```text
.
├── data/
│   └── housing.csv
├── notebooks/
│   └── House_Sales_in_King_Count_USA.ipynb
├── .gitattributes
├── .gitignore
├── LICENSE
├── README.md
└── requirements.txt
```

## How To Run

Clone the repository:

```bash
git clone https://github.com/46n/king-county-house-sales-analysis.git
cd king-county-house-sales-analysis
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Open the notebook:

```bash
jupyter notebook notebooks/House_Sales_in_King_Count_USA.ipynb
```

The notebook is configured to load `data/housing.csv` from the repository. If the local file is unavailable, it can fall back to the original course download step.

## Recommended GitHub Topics

`python` `data-science` `machine-learning` `real-estate` `housing-prices` `king-county` `exploratory-data-analysis` `regression` `pandas` `seaborn` `scikit-learn` `jupyter-notebook`

## Future Improvements

- Add cross-validation results for model comparison.
- Tune Ridge regularization parameters.
- Add more feature engineering around location and renovation age.
- Export key charts as images for the README.
- Build a small house price prediction app.

## License

This project is licensed under the MIT License.
