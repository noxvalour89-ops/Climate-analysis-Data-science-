import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

def load_nasa_data():
    """Fetches Global Land-Ocean Temperature Index from NASA."""
    url = "https://data.giss.nasa.gov/gistemp/tabledata_v4/GLB.Ts+dSST.csv"
    try:
        df = pd.read_csv(url, skiprows=1)
        # Filter out missing values (NASA uses '***')
        df = df[df['J-D'] != '***'].copy()
        df['Year'] = df['Year'].astype(int)
        df['Anomaly'] = df['J-D'].astype(float)
        return df[['Year', 'Anomaly']]
    except Exception as e:
        print(f"Error: {e}")
        return None

def main():
    df = load_nasa_data()
    if df is None: return

    # Prepare Model
    X = df[['Year']].values
    y = df['Anomaly'].values
    model = LinearRegression().fit(X, y)
    
    # Predict 30 years into the future
    last_year = df['Year'].max()
    future = np.array([[y] for y in range(last_year + 1, last_year + 31)])
    preds = model.predict(future)

    # Visualization
    plt.figure(figsize=(10, 6))
    plt.style.use('seaborn-v0_8-muted')
    plt.scatter(X, y, color='#3498db', s=15, alpha=0.6, label='NASA Observed Data')
    plt.plot(X, model.predict(X), color='#e74c3c', label='Historical Trend')
    plt.plot(future, preds, '--', color='#f39c12', label='2050 Projection')
    
    plt.title(f"Global Temperature Trends & Projections (Data: NASA GISTEMP v4)")
    plt.xlabel("Year")
    plt.ylabel("Temperature Anomaly (°C)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('nasa_climate_report.png', dpi=300)
    print("Analysis complete. Plot saved as 'nasa_climate_report.png'.")

if __name__ == "__main__":
    main()
