# 📈 Stock-price-prediction📉

Stock Price Prediction Web Tool
This repository contains a web application for stock price prediction using time series analysis. The tool allows users to input a stock symbol and the number of days they wish to forecast, providing an accurate prediction of future stock prices along with a visual representation of the forecasted prices.

**Features**
📊**User Input:** Enter a stock symbol (e.g., AAPL for Apple, GOOGL for Alphabet) and specify the number of days for the forecast.

🕰️**Time Series Analysis:** Utilizes advanced time series algorithms to predict future stock prices based on historical data.

💹**Price Forecast:** Displays the predicted stock prices for the specified number of days.

📉**Graphical Visualization:** Provides a graph that visually represents the historical data and the forecasted prices for easy analysis.

📋**Requirements:**

Flask==2.3.3

Flask-SQLAlchemy==3.1.1

Flask-Login==0.6.3

yfinance==0.2.36

pandas==2.1.4

scikit-learn==1.3.2

numpy==1.26.2

tensorflow==2.15.0

plotly==5.18.0

Werkzeug==3.0.1

requests==2.31.0

🛠️**Installation**
Clone the repository and install the required dependencies:

git clone https://github.com/SANS002/stock-price-prediction.git

cd stock-price-prediction>>
python -m venv venv
.\venv\Scripts\activate

>>pip install -r requirements.txt

🗂️**Setup**
Create a folder called templates and move your HTML files to that folder.

🚀**Usage**
Run the Flask application:
>>python app.py

Open your web browser and navigate to http://127.0.0.1:5000 to use the stock price prediction tool.

🔍**Example**
Enter the stock symbol (e.g., AAPL).

Specify the number of days to forecast (e.g., 30 days).

Click the **"Predict"** button to view the forecasted stock prices and the graph

🤝**Contributing**
Feel free to submit issues, fork the repository, and send pull requests. Contributions are welcome
