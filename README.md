# Vehicle Price Estimate

A used-car project that estimates a fair price for a vehicle and the probability that it sells within 30 days.

- **Price model:** regression (Linear, Ridge, Lasso, ElasticNet, Polynomial). The best one is chosen by 5-fold cross-validated RMSE.
- **Sale model:** logistic regression (`sold` = sold within 30 days).
- **Hypothesis tests:** transmission vs. price, age vs. price, damage record vs. price.
- **Final report:** printed in the terminal at the end of the analysis, using the real outputs of the run.

The dataset (`data/car_sales_english.csv`, 1500 rows) is synthetic and was generated with `data/data_generator.py`.

## Project structure

```
main.py                    entry point
data/                      CSV dataset and the data generator
database/                  SQLite database (car_sales.db) and DatabaseManager
analysis/                  hypothesis tests, feature engineering, final report
model/                     price model, sale model, Car classes
eda/data_analysis.ipynb    exploratory data analysis notebook
```

## Setup

Developed with Python 3.15.

```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Run

From the project root:

```
python main.py
```

The program will:

1. Run 10 SQL queries and the 3 hypothesis tests.
2. Train and compare the regression models, then train the sale model.
3. Print the final report.
4. Ask for a vehicle's details and print its fair price and 30-day sale probability.

## Results

Results from one run on the 1500-row dataset (80/20 train/test split, `random_state=42`):

| Model | CV RMSE | Test R2 |
|---|---|---|
| Linear Regression | 132,447 | 0.9078 |
| Ridge | 132,381 | 0.9076 |
| Lasso | 132,445 | 0.9078 |
| ElasticNet | 209,348 | 0.7597 |
| **Polynomial Regression** | **122,974** | **0.9262** |

- Polynomial Regression has the lowest CV RMSE and is the selected price model. Ridge and Lasso are almost the same as Linear Regression. ElasticNet is clearly worse with its default parameters.
- BMW and Mercedes have the highest average prices. Age and mileage affect price the most (age and mileage are almost perfectly correlated, r = 0.9975).
- All three hypothesis tests (transmission, age, damage record vs. price) reject the null hypothesis at alpha = 0.05.
- The sale model (logistic regression) reaches 0.6333 accuracy, 0.7262 recall and 0.6893 F1 on the test set.

The full report is printed by `main.py`.

## Rebuilding the database

`main.py` reads `database/car_sales.db`, which is already included. If it is missing, rebuild it from the CSV:

```python
import pandas as pd
from database.db_manager import DatabaseManager

db = DatabaseManager()
db.create_tables()
db.import_data_from_csv(pd.read_csv("data/car_sales_english.csv"))
```

## Notes

- Running the notebook in `eda/` needs Jupyter, which is not listed in `requirements.txt`.
- Since the data is synthetic, the results do not describe the real market.
