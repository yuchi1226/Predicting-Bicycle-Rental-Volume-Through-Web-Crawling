import tkinter as tk
from tkinter import ttk
import requests
import predict_2 as predict_2

class BikeRentalPredictor:
    def __init__(self):
        self.window = tk.Tk()
        self.window.title("Bike Rental Prediction System")
        self.window.geometry("400x250")
        
        # 創建輸入欄位
        self.create_input_fields()
        
        # 創建預測按鈕
        self.predict_button = ttk.Button(self.window, text="Predict", command=self.predict)
        self.predict_button.pack(pady=10)
        
        # 顯示結果的標籤
        self.result_label = ttk.Label(self.window, text="")
        self.result_label.pack(pady=10)
        
        # 新增自動獲取天氣按鈕
        self.weather_button = ttk.Button(self.window, text="Get Current Weather", command=self.get_weather)
        self.weather_button.pack(pady=5)
        
    def create_input_fields(self):
        # 輸入欄位
        ttk.Label(self.window, text="Temperature (°C):").pack()
        self.temp_entry = ttk.Entry(self.window)
        self.temp_entry.pack()
        
        ttk.Label(self.window, text="Humidity (%):").pack()
        self.humidity_entry = ttk.Entry(self.window)
        self.humidity_entry.pack()
        
        ttk.Label(self.window, text="Wind Speed (m/s):").pack()
        self.windspeed_entry = ttk.Entry(self.window)
        self.windspeed_entry.pack()
        
    def get_weather(self):
        try:
            API_KEY = "CWA-940152A9-E5DE-4CA3-BE4F-909FF22CD714"
            location = "彰化縣"
            url = "https://opendata.cwa.gov.tw/api/v1/rest/datastore/F-D0047-091?Authorization=CWA-940152A9-E5DE-4CA3-BE4F-909FF22CD714&LocationName=%E5%BD%B0%E5%8C%96%E7%B8%A3"
        
            response = requests.get(url)
            '''
            params = {
                "Authorization": API_KEY,
                "locationName": location
            }
            response = requests.get(url, params=params, timeout=10, verify=True)
            '''
            if response.status_code == 200:
                data = response.json()
                weather_data = data['records']['location'][0]['weatherElement']
                
                # 解析資料
                temp = next(item['elementValue'] for item in weather_data if item['elementName'] == 'TEMP')
                humidity = next(item['elementValue'] for item in weather_data if item['elementName'] == 'HUMD')
                windspeed = next(item['elementValue'] for item in weather_data if item['elementName'] == 'WDSD')
                
                # 更新輸入欄位
                self.temp_entry.delete(0, tk.END)
                self.temp_entry.insert(0, temp)
                
                self.humidity_entry.delete(0, tk.END)
                self.humidity_entry.insert(0, float(humidity) * 100)  # 轉換為百分比
                
                self.windspeed_entry.delete(0, tk.END)
                self.windspeed_entry.insert(0, windspeed)
                
            else:
                self.result_label.config(text="Unable to fetch weather data")
                
        except requests.exceptions.ConnectionError:
            self.result_label.config(text="Connection failed: Please check your internet connection")
        except requests.exceptions.Timeout:
            self.result_label.config(text="Request timeout: Server response took too long") 
        except Exception as e:
            print(e)
            self.result_label.config(text="An error occurred, please try again later")
    
    def predict(self):
        try:
            # 預設值
            hour = 4
            visibility = 2000
            dew_point_temp = -18.6
            solar_radiation = 0
            rainfall = 0
            snowfall = 0
            season = 'Winter'
            holiday = 'No Holiday'
            functioning_day = 'Yes'

            # 獲取輸入值
            temp = float(self.temp_entry.get())
            humidity = float(self.humidity_entry.get())
            windspeed = float(self.windspeed_entry.get())
            print(temp, humidity, windspeed)

            predicted_count = predict_2.predict_rented_bike_count(hour, temp, humidity, windspeed, visibility, dew_point_temp,
                                            solar_radiation, rainfall, snowfall, season, holiday, functioning_day)
            self.result_label.config(text=f"Predicted rental count: {round(predicted_count)} bikes")
            
        except ValueError:
            self.result_label.config(text="Please enter valid numbers")
    
    def run(self):
        self.window.mainloop()

if __name__ == "__main__":
    app = BikeRentalPredictor()
    app.run()
