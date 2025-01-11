# Bike_Rental_Prediction

## 項目展示
![](https://github.com/yuchi1226/Predicting-Bicycle-Rental-Volume-Through-Web-Crawling/blob/main/%E8%9E%A2%E5%B9%95%E6%93%B7%E5%8F%96%E7%95%AB%E9%9D%A2%202025-01-11%20163619.png)

## 功能

1. 以"Seoul Bike Sharing Demand Prediction"為資料集，預測租借數量(Rented Bike Count)

## 架構

1. 語言:python
2. 開發環境:Linux
3. UI界面: tkinter
4. 功能擴充: 支援台灣即時天氣資料預測

## 使用方法

1. [到網站取得台灣天氣使用API](https://opendata.cwa.gov.tw/dataset/climate/O-A0003-001)

2. 安裝必要套件:

   ``` python
   pip install tk pandas numpy scikit-learn requests
   ```

3. 執行 main.py

4. 點擊predict按鈕，從predict_2.py進行預測，並將預測的租借數量(Rented Bike Count)傳回