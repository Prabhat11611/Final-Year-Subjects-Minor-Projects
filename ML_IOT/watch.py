import pandas as pd
from datetime import datetime, timedelta

class HealthKitManager:
    def __init__(self):
        self.is_health_kit_available = self.check_health_kit_availability()
        self.types_to_read = [
            "heart_rate",
            "step_count",
            "respiratory_rate",
            "blood_pressure_systolic",
            "heart_rate_variability_sdnn"
        ]
    
    def check_health_kit_availability(self):
        return True

    def request_authorization(self):
        if not self.is_health_kit_available:
            raise Exception("HealthKit is not available on this device")
        
        return True

    def fetch_health_data(self, start_date, end_date):
        all_data = {
            "HeartRate": self.fetch_quantity_data("heart_rate", start_date, end_date),
            "Steps": self.fetch_quantity_data("step_count", start_date, end_date),
            "RespirationRate": self.fetch_quantity_data("respiratory_rate", start_date, end_date),
            "BloodPressure": self.fetch_quantity_data("blood_pressure_systolic", start_date, end_date),
            "HeartRateVariability": self.fetch_quantity_data("heart_rate_variability_sdnn", start_date, end_date)
        }
        return all_data

    def fetch_quantity_data(self, identifier, start_date, end_date):
        if identifier not in self.types_to_read:
            return []

        num_samples = (end_date - start_date).days * 24 * 60
        return [round(60 + (i % 5), 2) for i in range(num_samples)]

    def export_to_csv(self, data, file_name):
        timestamps = [datetime.now() + timedelta(minutes=i) for i in range(len(data["HeartRate"]))]
        csv_data = {
            "Timestamp": timestamps,
            "HeartRate": data["HeartRate"],
            "Steps": data["Steps"],
            "RespirationRate": data["RespirationRate"],
            "BloodPressure": data["BloodPressure"],
            "HeartRateVariability": data["HeartRateVariability"]
        }
        
        df = pd.DataFrame(csv_data)
        file_path = f"{watch}.csv"
        df.to_csv(file_path, index=False)
        return file_path

if __name__ == "__main__":
    health_manager = HealthKitManager()
    health_manager.request_authorization()
    
    start_date = datetime.now() - timedelta(days=7)
    end_date = datetime.now()
    
    health_data = health_manager.fetch_health_data(start_date, end_date)
    print(health_data)
    
    file_path = health_manager.export_to_csv(health_data, "watch.csv")
    print(f"Data exported to {file_path}")
