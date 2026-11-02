import time
import database.db_manager as db_manager
from model.price_analyzer import train_and_evaluate_model
from analysis.hypothesis import (price_damage_record_difference_test,
                                 age_and_price_difference_test,
                                 transmission_price_difference_test)
from model.sale_predictor import prediction_model
from model.car import Car, CarPriceAnalyzer, CarSalePredictor
from analysis.report import print_final_report


def get_validated_input(prompt, cast_type=str, valid_options=None, min_val=None):
    while True:
        try:
            user_input = input(prompt).strip()

            if not user_input:
                print("Error: This field cannot be left empty.")
                continue

            value = cast_type(user_input)

            if valid_options and cast_type == str:
                matched_option = next((opt for opt in valid_options if opt.lower() == str(value).lower()), None)
                if matched_option:
                    value = matched_option
                else:
                    print(f"Error: Invalid brand/option. Valid options are: {', '.join(valid_options)}")
                    continue

            # Numerical inputs: Check against valid options and minimum limits
            if valid_options and cast_type != str and value not in valid_options:
                print(f"Error: Invalid input. Valid options are: {', '.join(map(str, valid_options))}")
                continue

            if min_val is not None and value < min_val:
                print(f"Error: Value cannot be less than {min_val}.")
                continue

            return value
        except ValueError:
            print(f"Error: Please enter a valid {cast_type.__name__} value.")


def terminal_application(reg_model, log_model, df):
    print("\n" + "=" * 50)
    print(" USED CAR PRICE AND SALES PREDICTION SYSTEM")
    print("=" * 50 + "\n")

    # Valid options come from the training data so the inputs always match the model
    valid_brands = sorted(df["brand"].unique())
    valid_fuel_types = sorted(df["fuel_type"].unique())
    valid_transmissions = sorted(df["transmission"].unique())
    valid_owner_counts = sorted(df["owner_count"].unique())
    valid_cities = sorted(df["city"].unique())
    valid_segments = sorted(df["vehicle_segment"].unique())

    brand = get_validated_input(f"Brand ({'/'.join(valid_brands)}): ", str, valid_options=valid_brands)
    age = get_validated_input("Age (e.g., 3): ", int, min_val=0)
    mileage = get_validated_input("Mileage (e.g., 45000): ", int, min_val=0)
    engine = get_validated_input("Engine Capacity (e.g., 1.6, 0 for Electric): ", float, min_val=0)
    fuel_type = get_validated_input(f"Fuel Type ({'/'.join(valid_fuel_types)}): ", str,
                                    valid_options=valid_fuel_types)
    transmission = get_validated_input(f"Transmission ({'/'.join(valid_transmissions)}): ", str,
                                       valid_options=valid_transmissions)
    damage_record = get_validated_input("Damage Record (TRY, 0 if none): ", int, min_val=0)
    owner_count = get_validated_input(f"Owner Count ({'/'.join(valid_owner_counts)}): ", str,
                                      valid_options=valid_owner_counts)
    city = get_validated_input(f"City ({'/'.join(valid_cities)}): ", str, valid_options=valid_cities)
    vehicle_segment = get_validated_input(f"Vehicle Segment ({'/'.join(valid_segments)}): ", str,
                                          valid_options=valid_segments)
    asking_price = get_validated_input("Seller's Asking Price (TRY): ", int, min_val=1)

    car = Car(brand, age, mileage, engine, fuel_type, transmission,
              damage_record, owner_count, city, vehicle_segment, asking_price)
    df_input = car.to_dataframe()

    print("\nCalculating predictions...\n")
    time.sleep(1)

    predicted_price = CarPriceAnalyzer(reg_model).predict_price(df_input)
    sale_probability = CarSalePredictor(log_model).calculate_sale_probability(df_input)

    price_diff = asking_price - predicted_price

    # 4. Print Results
    print("=" * 50)
    print(f"Vehicle Info         : {brand}, {age} Years Old, {transmission}")
    print(f"Seller's Asking Price: {asking_price:,.2f} TRY")
    print(f"Model's Fair Price   : {predicted_price:,.2f} TRY")
    print("-" * 50)
    if price_diff > 0:
        print(f"Price Evaluation     : The seller is asking {abs(price_diff):,.2f} TRY MORE than the fair price.")
    else:
        print(f"Price Evaluation     : The seller is asking {abs(price_diff):,.2f} TRY LESS than the fair price.")
    print(f"30-Day Sale Prob.    : {sale_probability:.1f}%")
    print("=" * 50)


def run_analysis_and_tests(db, cursor):
    questions = {
        "1. Vehicles priced above 1,000,000 TL": """
                                                 SELECT *
                                                 FROM CarListings
                                                 WHERE price > 1000000;
                                                 """,
        "2. Vehicles with less than 50,000 kilometers on the mileage": """
                                                                       SELECT *
                                                                       FROM CarListings
                                                                       WHERE mileage < 50000;
                                                                       """,
        "3. Vehicle numbers by brand": """
                                       SELECT brand, COUNT(*)
                                       FROM CarListings
                                       GROUP BY brand;
                                       """,
        "4. Average price for each brand": """
                                           SELECT brand, ROUND(AVG(price), 2) AS average_price
                                           FROM CarListings
                                           GROUP BY brand;
                                           """,
        "5. List the 5 brands with the highest average price": """
                                                               SELECT brand, ROUND(AVG(price), 2) AS average_price
                                                               FROM CarListings
                                                               GROUP BY brand
                                                               ORDER BY average_price DESC LIMIT 5;
                                                               """,
        "6. Vehicles with automatic transmission and under 5 years old": """
                                                                         SELECT *
                                                                         FROM CarListings
                                                                         WHERE transmission = 'Automatic'
                                                                           AND age < 5;
                                                                         """,
        "7. Number of listings by city": """
                                         SELECT city, COUNT(*) AS listing_count
                                         FROM CarListings
                                         GROUP BY city
                                         ORDER BY listing_count DESC;
                                         """,
        "8. Average sales duration by fuel type": """
                                                  SELECT fuel_type, ROUND(AVG(days_listed), 2) AS average_days
                                                  FROM CarListings
                                                  GROUP BY fuel_type;
                                                  """,
        "9. The proportion of vehicles sold within 30 days": """
                                                             SELECT ROUND(AVG(sold) * 100, 2) AS sold_percentage
                                                             FROM CarListings;
                                                             """,
        "10. Vehicles that are less than 5 years old and have not been sold within 30 days": """
                                                                                             SELECT *
                                                                                             FROM CarListings
                                                                                             WHERE age < 5
                                                                                               AND sold = 0;
                                                                                             """
    }

    for question, answer in questions.items():
        print(f"\n--- {question} ---")
        result_df = db.execute_query(answer)
        print(result_df)

    print("\n--- Transmission - Price Test Results ---")
    trans_result = transmission_price_difference_test(cursor)
    for key, value in trans_result.items():
        print(f"{key:<35}: {value}")

    print("\n--- Age - Price Test Results ---")
    age_result = age_and_price_difference_test(cursor)
    for key, value in age_result.items():
        print(f"{key:<35}: {value}")

    print("\n--- Damage Count - Price Test Results ---")
    damage_result = price_damage_record_difference_test(cursor)
    for key, value in damage_result.items():
        print(f"{key:<35}: {value}")

    return {"transmission": trans_result, "age": age_result, "damage": damage_result}


if __name__ == "__main__":
    db = db_manager.DatabaseManager()
    conn = db.get_connection()
    cursor = conn.cursor()
    df = db.execute_query("SELECT * FROM CarListings")

    hypothesis_results = run_analysis_and_tests(db, cursor)

    print("\n[INFO] Training the price prediction model...")
    trained_reg_model, reg_results = train_and_evaluate_model(df)

    print("\n[INFO] Training the sales probability model...")
    trained_log_model, log_metrics = prediction_model(df)

    print_final_report(df, hypothesis_results, reg_results, log_metrics)

    terminal_application(trained_reg_model, trained_log_model, df)