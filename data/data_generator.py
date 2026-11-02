import os
import numpy as np
import pandas as pd


def generate_car_sales_dataset(
    n_samples: int = 1500,
    output_path: str = "car_sales.csv",
    random_state: int = 42,
) -> pd.DataFrame:
  """Generates a synthetic used car dataset fully compliant with the 30-day
  observation period (censoring) and data leakage prevention rules.
  """
  np.random.seed(random_state)

  brand_segment = {
      "Toyota": ["C", "B", "SUV"],
      "BMW": ["D", "C", "SUV"],
      "Renault": ["B", "C"],
      "Fiat": ["B", "C"],
      "Volkswagen": ["C", "D", "SUV"],
      "Ford": ["B", "C", "SUV"],
      "Hyundai": ["B", "C", "SUV"],
      "Mercedes": ["D", "C", "SUV"],
      "Peugeot": ["B", "C", "SUV"],
      "Honda": ["C", "SUV"],
  }

  brands = list(brand_segment.keys())
  cities = [
      "Istanbul",
      "Ankara",
      "Izmir",
      "Bursa",
      "Antalya",
      "Adana",
      "Konya",
      "Kocaeli",
  ]
  fuel_types = ["Gasoline", "Diesel", "Hybrid", "Electric"]
  transmission_types = ["Manual", "Automatic"]
  owner_options = ["1", "2", "3 or more"]

  data = []

  for _ in range(n_samples):
    brand = np.random.choice(brands)
    segment = np.random.choice(brand_segment[brand])
    age = np.random.randint(1, 16)

    mileage = int(np.random.normal(loc=age * 14500, scale=4500))
    mileage = max(5000, round(mileage, -3))

    fuel = np.random.choice(fuel_types, p=[0.42, 0.38, 0.12, 0.08])

    if fuel == "Electric":
      engine = 0.0
      transmission = "Automatic"
    elif fuel == "Hybrid":
      engine = np.random.choice([1.5, 1.6, 1.8, 2.0])
      transmission = "Automatic"
    else:
      engine = np.random.choice(
          [1.0, 1.2, 1.3, 1.4, 1.5, 1.6, 2.0],
          p=[0.08, 0.12, 0.15, 0.20, 0.20, 0.15, 0.10],
      )
      transmission = np.random.choice(transmission_types, p=[0.35, 0.65])

    if age <= 3:
      owner_count = np.random.choice(owner_options, p=[0.75, 0.20, 0.05])
    elif age <= 7:
      owner_count = np.random.choice(owner_options, p=[0.35, 0.45, 0.20])
    else:
      owner_count = np.random.choice(owner_options, p=[0.10, 0.35, 0.55])

    if np.random.rand() < 0.55:
      damage_record = 0
    else:
      damage_record = int(np.random.exponential(scale=35000))
      damage_record = round(max(2500, min(damage_record, 250000)), -2)

    # STEP 1: Calculate the expected fair market value of the vehicle (Does NOT include future info)
    base_price = {"B": 850000, "C": 1200000, "D": 1850000, "SUV": 1650000}[
        segment
    ]
    if brand in ["BMW", "Mercedes"]:
      base_price *= 1.35
    elif brand in ["Fiat", "Renault", "Hyundai"]:
      base_price *= 0.88

    fair_price = (
        base_price * ((0.94) ** age)
        - (mileage * 0.9)
        - (damage_record * 1.1)
        + (100000 if transmission == "Automatic" else 0)
    )
    fair_price = max(350000, fair_price)

    # The price set by the seller (Can be 15% cheaper or 20% more expensive than the fair price)
    price_multiplier = np.random.normal(loc=1.0, scale=0.08)
    price = int(round(max(300000, fair_price * price_multiplier), -4))

    # STEP 2: Determine the probability of being sold within 30 days (Sold) based on vehicle features
    # Lower price compared to fair price, automatic transmission, and low damage increase the sale probability
    sales_score = 0.55
    sales_score += (
        1.0 - (price / fair_price)
    ) * 2.2  # A reasonable price accelerates the sale
    sales_score += 0.10 if transmission == "Automatic" else -0.05
    sales_score += 0.08 if damage_record == 0 else -0.10
    sales_score += 0.05 if owner_count == "1" else -0.05

    # Clip the probability between 0.05 and 0.95
    sales_probability = np.clip(sales_score, 0.05, 0.95)
    sold = 1 if np.random.rand() < sales_probability else 0

    # STEP 3: Assign Listing Days according to the observation period rule
    # We do not generate unsuccessful listings that haven't reached 30 days yet (to prevent Censoring error)
    if sold == 1:
      # Those sold within 30 days: Sold between day 1 and 30
      days_listed = np.random.randint(1, 31)
    else:
      # Those not sold within 30 days: Listings that have completed the 30-day observation period (31 - 90 days)
      days_listed = np.random.randint(31, 91)

    city = np.random.choice(
        cities, p=[0.35, 0.18, 0.14, 0.09, 0.08, 0.06, 0.05, 0.05]
    )

    data.append([
        brand,
        age,
        mileage,
        engine,
        price,
        days_listed,
        sold,
        fuel,
        transmission,
        damage_record,
        owner_count,
        city,
        segment,
    ])

  columns = [
      "Brand",
      "Age",
      "Mileage",
      "Engine",
      "Price (TRY)",
      "Days Listed",
      "Sold",
      "Fuel Type",
      "Transmission",
      "Damage Record",
      "Owner Count",
      "City",
      "Vehicle Segment",
  ]

  df = pd.DataFrame(data, columns=columns)

  os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
  df.to_csv(output_path, index=False, encoding="utf-8-sig")

  print(
      f"[INFO] Rule-compliant dataset with {n_samples} rows has been saved as '{output_path}'."
  )
  return df


if __name__ == "__main__":
    # Generate and save 1500 rows of data
    df_cars = generate_car_sales_dataset(
        n_samples=1500,
        output_path="car_sales_english.csv"
    )

    # Show the first 5 rows of the generated dataset
    print(df_cars.head())

    # Show the general structure of the dataset
    print("\nDataset Info:")
    print(df_cars.info())