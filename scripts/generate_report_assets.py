from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "housing.csv"
FIGURES_DIR = ROOT / "assets" / "figures"
TABLES_DIR = ROOT / "assets" / "tables"


FEATURES = [
    "floors",
    "waterfront",
    "lat",
    "bedrooms",
    "sqft_basement",
    "view",
    "bathrooms",
    "sqft_living15",
    "sqft_above",
    "grade",
    "sqft_living",
]


def load_clean_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    df = df.drop(columns=["id", "Unnamed: 0"])
    df["bedrooms"] = df["bedrooms"].fillna(df["bedrooms"].mean())
    df["bathrooms"] = df["bathrooms"].fillna(df["bathrooms"].mean())
    return df


def save_waterfront_boxplot(df: pd.DataFrame) -> None:
    plt.figure(figsize=(8, 5))
    sns.boxplot(
        x="waterfront",
        y="price",
        hue="waterfront",
        data=df,
        palette=["#4C78A8", "#F58518"],
        legend=False,
    )
    plt.title("House Price Distribution by Waterfront View")
    plt.xlabel("Waterfront View (0 = No, 1 = Yes)")
    plt.ylabel("Sale Price (USD)")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "q4_waterfront_price_boxplot.png", dpi=160)
    plt.close()


def save_sqft_above_regplot(df: pd.DataFrame) -> None:
    plt.figure(figsize=(8, 5))
    sns.regplot(
        x="sqft_above",
        y="price",
        data=df,
        scatter_kws={"alpha": 0.22, "s": 12, "color": "#4C78A8"},
        line_kws={"color": "#D62728", "linewidth": 2},
    )
    plt.title("Relationship Between Above-Ground Area and House Price")
    plt.xlabel("Above-Ground Square Footage")
    plt.ylabel("Sale Price (USD)")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "q5_sqft_above_price_regplot.png", dpi=160)
    plt.close()


def save_correlation_assets(df: pd.DataFrame) -> None:
    correlations = (
        df.select_dtypes(include="number")
        .corr(numeric_only=True)["price"]
        .drop("price")
        .sort_values(ascending=False)
    )
    correlations.to_frame("correlation_with_price").to_csv(
        TABLES_DIR / "price_correlations.csv"
    )

    top = correlations.head(10).sort_values()
    plt.figure(figsize=(8, 5))
    top.plot(kind="barh", color="#59A14F")
    plt.title("Top Positive Correlations with House Price")
    plt.xlabel("Correlation")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "top_price_correlations.png", dpi=160)
    plt.close()


def calculate_model_scores(df: pd.DataFrame) -> pd.DataFrame:
    x = df[FEATURES]
    y = df["price"]

    linear_sqft = LinearRegression()
    linear_sqft.fit(df[["sqft_living"]], y)

    linear_features = LinearRegression()
    linear_features.fit(x, y)

    pipeline = Pipeline(
        [
            ("scale", StandardScaler()),
            ("polynomial", PolynomialFeatures(include_bias=False)),
            ("model", LinearRegression()),
        ]
    )
    pipeline.fit(x, y)

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.15, random_state=1
    )

    ridge = Ridge(alpha=0.1)
    ridge.fit(x_train, y_train)

    poly = PolynomialFeatures(degree=2)
    x_train_poly = poly.fit_transform(x_train)
    x_test_poly = poly.transform(x_test)

    ridge_poly = Ridge(alpha=0.1)
    ridge_poly.fit(x_train_poly, y_train)

    return pd.DataFrame(
        [
            {
                "criteria": "Q6",
                "model": "Linear Regression",
                "features": "sqft_living",
                "evaluation": "Full dataset",
                "r_squared": linear_sqft.score(df[["sqft_living"]], y),
            },
            {
                "criteria": "Q7",
                "model": "Linear Regression",
                "features": "Selected housing features",
                "evaluation": "Full dataset",
                "r_squared": linear_features.score(x, y),
            },
            {
                "criteria": "Q8",
                "model": "StandardScaler + PolynomialFeatures + LinearRegression",
                "features": "Selected housing features",
                "evaluation": "Full dataset",
                "r_squared": pipeline.score(x, y),
            },
            {
                "criteria": "Q9",
                "model": "Ridge Regression, alpha=0.1",
                "features": "Selected housing features",
                "evaluation": "Test set",
                "r_squared": ridge.score(x_test, y_test),
            },
            {
                "criteria": "Q10",
                "model": "Degree-2 PolynomialFeatures + Ridge, alpha=0.1",
                "features": "Selected housing features",
                "evaluation": "Test set",
                "r_squared": ridge_poly.score(x_test_poly, y_test),
            },
        ]
    )


def save_model_assets(df: pd.DataFrame) -> None:
    scores = calculate_model_scores(df)
    scores.to_csv(TABLES_DIR / "model_scores.csv", index=False)

    plt.figure(figsize=(9, 5))
    chart = scores.sort_values("r_squared", ascending=True)
    plt.barh(chart["criteria"] + " - " + chart["model"], chart["r_squared"], color="#4C78A8")
    plt.xlim(0, 0.85)
    plt.xlabel("R-squared")
    plt.title("Regression Model Performance")
    for index, value in enumerate(chart["r_squared"]):
        plt.text(value + 0.01, index, f"{value:.3f}", va="center")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "model_r2_comparison.png", dpi=160)
    plt.close()


def save_summary_tables(df: pd.DataFrame) -> None:
    floor_counts = df["floors"].value_counts().sort_index().to_frame("house_count")
    floor_counts.to_csv(TABLES_DIR / "floor_counts.csv")

    dataset_profile = pd.DataFrame(
        [
            {"metric": "Rows", "value": len(df)},
            {"metric": "Columns after cleanup", "value": df.shape[1]},
            {"metric": "Median sale price", "value": round(df["price"].median(), 2)},
            {"metric": "Average sale price", "value": round(df["price"].mean(), 2)},
            {"metric": "Minimum sale price", "value": round(df["price"].min(), 2)},
            {"metric": "Maximum sale price", "value": round(df["price"].max(), 2)},
            {"metric": "Missing bedrooms after imputation", "value": int(df["bedrooms"].isna().sum())},
            {"metric": "Missing bathrooms after imputation", "value": int(df["bathrooms"].isna().sum())},
        ]
    )
    dataset_profile.to_csv(TABLES_DIR / "dataset_profile.csv", index=False)


def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")

    df = load_clean_data()
    save_waterfront_boxplot(df)
    save_sqft_above_regplot(df)
    save_correlation_assets(df)
    save_model_assets(df)
    save_summary_tables(df)


if __name__ == "__main__":
    main()
