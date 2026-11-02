import pandas as pd

class Car:

    def __init__(self, brand, age, mileage, engine, fuel_type, transmission,
                 damage_record, owner_count, city, vehicle_segment, asking_price):
        self.__brand = brand
        self.__age = age
        self.__mileage = mileage
        self.__engine = engine
        self.__fuel_type = fuel_type
        self.__transmission = transmission
        self.__damage_record = damage_record
        self.__owner_count = owner_count
        self.__city = city
        self.__vehicle_segment = vehicle_segment
        self.__asking_price = asking_price

    def get_brand(self): return self.__brand
    def get_age(self): return self.__age
    def get_mileage(self): return self.__mileage
    def get_engine(self): return self.__engine
    def get_fuel_type(self): return self.__fuel_type
    def get_transmission(self): return self.__transmission
    def get_damage_record(self): return self.__damage_record
    def get_owner_count(self): return self.__owner_count
    def get_city(self): return self.__city
    def get_vehicle_segment(self): return self.__vehicle_segment
    def get_asking_price(self): return self.__asking_price

    def to_dataframe(self): return pd.DataFrame({
        "brand": [self.__brand],
        "age": [self.__age],
        "mileage": [self.__mileage],
        "engine": [self.__engine],
        "fuel_type": [self.__fuel_type],
        "transmission": [self.__transmission],
        "damage_record": [self.__damage_record],
        "owner_count": [self.__owner_count],
        "city": [self.__city],
        "vehicle_segment": [self.__vehicle_segment],
        "price": [self.__asking_price]
    })

class CarPriceAnalyzer:

    def __init__(self,model):
        self.__model = model

    def predict_price(self, preprocessed_df):
        df_for_pred=preprocessed_df.drop(columns=['price'],errors='ignore')
        return self.__model.predict(df_for_pred)[0]

class CarSalePredictor:
    def __init__(self,log_model):
        self.__model = log_model

    def calculate_sale_probability(self,preprocessed_df):
        return self.__model.predict_proba(preprocessed_df)[0][1]*100

