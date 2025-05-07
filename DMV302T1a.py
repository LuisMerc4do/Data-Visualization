# Weather Data Analysis and Visualization
# ---------------------------------------
# This script analyzes weather data from DMVA3T2.csv
# Key features:
# - Temperature overview and trends
# - Temperature anomaly heatmap
# - Cumulative rainfall comparison
# - Climate extremes detection

# Importing required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from matplotlib.colors import TwoSlopeNorm
import matplotlib.dates as mdates

# ------------------------------
# STEP 1: Load and prepare the dataset
# ------------------------------
def load_data():
    # Naming columns to match dataset structure
    columns = [
        'Date', 'Avg_Temp', 'Min_Temp', 'Max_Temp',
        'Historical_Avg_Min', 'Historical_Avg_Max',
        'Historical_Lowest', 'Historical_Highest',
        'Rainfall', 'Historical_Avg_Rain', 'Historical_Highest_Rain'
    ]
    
    # Reading the CSV file
    df = pd.read_csv('DMVA3T2.csv', header=None, names=columns)
    
    # Converting date column to datetime format
    df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%y')
    
    # Adding time components for grouping
    df['Day'] = df['Date'].dt.day
    df['Month'] = df['Date'].dt.month
    
    # Calculating anomalies
    df['Avg_Temp_Deviation'] = df['Avg_Temp'] - (df['Historical_Avg_Min'] + df['Historical_Avg_Max']) / 2
    df['7d_Avg_Temp'] = df['Avg_Temp'].rolling(window=7, center=True).mean()
    df['Cumulative_Rainfall'] = df['Rainfall'].cumsum()
    df['Cumulative_Historical_Rain'] = df['Historical_Avg_Rain'].cumsum()
    
    # Flagging extreme days
    df['Extreme_Temp_Day'] = (abs(stats.zscore(df['Avg_Temp'])) > 2)
    df['Extreme_Rain_Day'] = (df['Rainfall'] > df['Historical_Highest_Rain'] * 0.7)
    
    return df

# ------------------------------
# STEP 2: Temperature trend overview
# ------------------------------
def plot_temperature_trends(df):
    plt.figure(figsize=(12, 6))
    
    # Plotting temperature lines
    plt.plot(df['Date'], df['Avg_Temp'], label='Average Temp', color='green')
    plt.plot(df['Date'], df['Max_Temp'], label='Max Temp', color='red', alpha=0.6)
    plt.plot(df['Date'], df['Min_Temp'], label='Min Temp', color='blue', alpha=0.6)
    plt.plot(df['Date'], df['7d_Avg_Temp'], label='7-Day Avg Temp', color='black', linewidth=2)
    
    # Historical averages
    plt.plot(df['Date'], df['Historical_Avg_Max'], 'r--', alpha=0.4, label='Hist. Avg Max')
    plt.plot(df['Date'], df['Historical_Avg_Min'], 'b--', alpha=0.4, label='Hist. Avg Min')
    
    # Highlighting extreme days
    extreme = df[df['Extreme_Temp_Day']]
    plt.scatter(extreme['Date'], extreme['Avg_Temp'], color='orange', s=50, label='Extreme Temp')
    
    plt.title('Temperature Trends Over the Year')
    plt.ylabel('Temperature (°C)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b'))
    plt.tight_layout()
    plt.savefig('plot_temperature_trends.png', dpi=300)

# ------------------------------
# STEP 3: Heatmap for temperature anomalies
# ------------------------------
def plot_temp_anomalies(df):
    pivot = df.pivot_table(index='Day', columns='Month', values='Avg_Temp_Deviation', aggfunc='mean')
    plt.figure(figsize=(10, 8))
    
    norm = TwoSlopeNorm(vmin=-5, vcenter=0, vmax=5)
    sns.heatmap(pivot, cmap='RdBu_r', norm=norm, cbar_kws={'label': 'Anomaly (°C)'})
    
    plt.title('Temperature Anomaly Heatmap')
    plt.xlabel('Month')
    plt.ylabel('Day of Month')
    plt.xticks(np.arange(12) + 0.5, ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                                     'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'], rotation=0)
    plt.tight_layout()
    plt.savefig('plot_temp_anomalies.png', dpi=300)

# ------------------------------
# STEP 4: Compare cumulative rainfall
# ------------------------------
def plot_cumulative_rain(df):
    plt.figure(figsize=(12, 6))
    
    # Actual vs Historical
    plt.plot(df['Date'], df['Cumulative_Rainfall'], label='Actual Rainfall', color='blue')
    plt.plot(df['Date'], df['Cumulative_Historical_Rain'], label='Historical Average', linestyle='--', color='navy')
    
    # Add annotation showing percentage difference
    final_actual = df['Cumulative_Rainfall'].iloc[-1]
    final_expected = df['Cumulative_Historical_Rain'].iloc[-1]
    diff_percent = ((final_actual - final_expected) / final_expected) * 100
    plt.annotate(f'{final_actual:.1f}mm ({diff_percent:.1f}% vs historical)', 
                 xy=(df['Date'].iloc[-1], final_actual), xytext=(-120, 30),
                 textcoords='offset points', ha='right', bbox=dict(boxstyle='round', fc='lightyellow'))
    
    plt.title('Cumulative Rainfall Comparison')
    plt.ylabel('Total Rainfall (mm)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b'))
    plt.tight_layout()
    plt.savefig('plot_cumulative_rain.png', dpi=300)

# ------------------------------
# STEP 5: Identify and show extreme weather events
# ------------------------------
def plot_extremes(df):
    high = df[df['Max_Temp'] > df['Historical_Highest']]
    low = df[df['Min_Temp'] < df['Historical_Lowest']]
    rain = df[df['Rainfall'] > df['Historical_Highest_Rain']]
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    
    # Temperature extremes
    ax1.plot(df['Date'], df['Max_Temp'], color='red', alpha=0.5)
    ax1.plot(df['Date'], df['Min_Temp'], color='blue', alpha=0.5)
    ax1.plot(df['Date'], df['Historical_Highest'], 'r--', alpha=0.6, label='Hist High')
    ax1.plot(df['Date'], df['Historical_Lowest'], 'b--', alpha=0.6, label='Hist Low')
    ax1.scatter(high['Date'], high['Max_Temp'], color='darkred', s=70, marker='^', label='New High')
    ax1.scatter(low['Date'], low['Min_Temp'], color='darkblue', s=70, marker='v', label='New Low')
    ax1.set_title('Temperature Extremes')
    ax1.set_ylabel('Temp (°C)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Rainfall extremes
    ax2.bar(df['Date'], df['Rainfall'], color='skyblue', alpha=0.7)
    ax2.plot(df['Date'], df['Historical_Highest_Rain'], 'b--', alpha=0.6, label='Hist Max Rain')
    ax2.bar(rain['Date'], rain['Rainfall'], color='purple', label='New Rain Record')
    ax2.set_title('Rainfall Extremes')
    ax2.set_ylabel('Rainfall (mm)')
    ax2.set_xlabel('Date')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    ax2.xaxis.set_major_locator(mdates.MonthLocator())
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
    plt.tight_layout()
    plt.savefig('plot_extremes.png', dpi=300)

# ------------------------------
# STEP 6: Run the full analysis
# ------------------------------
def main():
    print("Loading and analyzing data...")
    df = load_data()
    
    plot_temperature_trends(df)
    plot_temp_anomalies(df)
    plot_cumulative_rain(df)
    plot_extremes(df)
    
    print("Analysis complete. All plots saved.")

# Only run the script if this is the main file
if __name__ == '__main__':
    main()
