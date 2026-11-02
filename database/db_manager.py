import sqlite3
import pandas as pd
import os


class DatabaseManager:
    def __init__(self, db_path="database/car_sales.db"):
        self.db_path = db_path
        os.makedirs(os.path.dirname(os.path.abspath(self.db_path)), exist_ok=True)

    def get_connection(self):
        return sqlite3.connect(self.db_path)

    def create_tables(self):
        conn = self.get_connection()
        cursor = conn.cursor()

        cursor.execute("""CREATE TABLE IF NOT EXISTS CarListings
                          (
                              list_id INTEGER PRIMARY KEY AUTOINCREMENT,
                              brand TEXT,
                              age INTEGER,
                              mileage INTEGER,
                              engine FLOAT,
                              price INTEGER,
                              days_listed INTEGER,
                              sold BOOLEAN,
                              fuel_type TEXT,
                              transmission TEXT,
                              damage_record INTEGER,
                              owner_count TEXT,
                              city TEXT,
                              vehicle_segment TEXT
                          )
                       """)

        conn.commit()
        conn.close()
        print("The database and table were created successfully.")

    def import_data_from_csv(self, df: pd.DataFrame):
        df = df.copy()

        #Map DataFrame column names to the SQLite table column names
        column_mapping = {
            "Brand": "brand",
            "Age": "age",
            "Mileage": "mileage",
            "Engine": "engine",
            "Price (TRY)": "price",
            "Days Listed": "days_listed",
            "Sold": "sold",
            "Fuel Type": "fuel_type",
            "Transmission": "transmission",
            "Damage Record": "damage_record",
            "Owner Count": "owner_count",
            "City": "city",
            "Vehicle Segment": "vehicle_segment"
        }
        df = df.rename(columns=column_mapping)

        valid_columns = list(column_mapping.values())
        df = df[[col for col in valid_columns if col in df.columns]]

        conn = self.get_connection()
        try:
            conn.execute("DELETE FROM CarListings")
            conn.commit()

            df.to_sql("CarListings", con=conn, if_exists="append", index=False)
            conn.commit()
            print(f"Data import completed successfully! ({len(df)} rows added)")
        except Exception as e:
            print(f"ACTUAL ERROR: {repr(e)}")
        finally:
            conn.close()

    def execute_query(self, query: str):
        with self.get_connection() as conn:
            return pd.read_sql_query(query, conn)

    def execute_update(self, query: str):
        conn = self.get_connection()
        cursor = conn.cursor()
        try:
            cursor.execute(query)
            conn.commit()
        except Exception as e:
            print(f"Update error: {repr(e)}")
        finally:
            conn.close()