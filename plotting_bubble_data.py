#plotting_bubble_data

#RELIES ON bubble_data_collection_cvs_function.py

import matplotlib.pyplot as plt
import pandas as pd

def plotting(merged):
    assert isinstance(merged, pd.DataFrame), "merged must be a pandas DataFrame"
    assert not merged.empty, "merged DataFrame cannot be empty"
    assert 'date' in merged.columns, "merged must contain 'date' column"
    assert 'bubble_index' in merged.columns, "merged must contain 'bubble_index' column"
    assert 'tic' in merged.columns, "merged must contain 'tic' column"
    assert 'prc' in merged.columns, "merged must contain 'prc' column"
    assert 'saleq' in merged.columns, "merged must contain 'saleq' column"
    assert 'mktcap' in merged.columns, "merged must contain 'mktcap' column"
    
    ts = merged.groupby("date")["bubble_index"].mean()

    plt.figure(figsize=(12,5))
    plt.plot(ts, label="EV Bubble Index")
    plt.axhline(0, linestyle="--")
    plt.axhline(2, color="red", linestyle="--", label="Bubble Threshold")
    plt.title("EV Bubble Index Over Time")
    plt.xlabel("Year")
    plt.ylabel("Bubble Index (Z-score)")
    plt.legend()
    plt.show()

    company = "NIO"

    df = merged[merged["tic"] == company]

    plt.figure(figsize=(12,5))
    plt.plot(df["date"], df["prc"], label="Price", linewidth=2)
    plt.plot(df["date"], df["saleq"], label="Sales (quarterly)", linewidth=2)
    plt.legend()
    plt.title(f"{company}: Price vs Sales")
    plt.show()

    # Filter TSLA data
    tsla = merged[merged['tic'] == 'TSLA'].copy()

    # Ensure data is sorted by date
    tsla = tsla.sort_values('date')

    # Calculate growth rates
    tsla['mktcap_growth'] = tsla['mktcap'].pct_change()  # daily/quarterly % change
    tsla['sales_growth'] = tsla['saleq'].pct_change()
    tsla['ps_ratio'] = tsla['mktcap'] / tsla['saleq']

    # Optional: Z-score the PS ratio for bubble visualization
    tsla['ps_z'] = (tsla['ps_ratio'] - tsla['ps_ratio'].mean()) / tsla['ps_ratio'].std()

    # Plot
    plt.figure(figsize=(14,6))

    plt.plot(tsla['date'], tsla['ps_ratio'], label='PS ratio', color='blue', linewidth=2)
    plt.plot(tsla['date'], tsla['sales_growth'], label='Sales Growth', color='green', linestyle='--')
    plt.plot(tsla['date'], tsla['mktcap_growth'], label='Market Cap Growth', color='orange', linestyle='--')

    plt.title("Tesla: PS Ratio vs Sales & Market Cap Growth")
    plt.xlabel("Date")
    plt.ylabel("Value / Growth Rate")
    plt.legend()
    plt.grid(True)
    plt.show()

    # Filter for Tesla
    tsla = merged[merged['tic'] == 'TSLA'].copy()

    # Compute P/S ratio if not already computed
    tsla['ps'] = tsla['mktcap'] / tsla['saleq']

    # Optional: compute normalized values for easier comparison
    tsla['mktcap_norm'] = tsla['mktcap'] / tsla['mktcap'].max()
    tsla['saleq_norm'] = tsla['saleq'] / tsla['saleq'].max()
    tsla['ps_norm'] = tsla['ps'] / tsla['ps'].max()

    # Plot
    plt.figure(figsize=(12,6))
    plt.plot(tsla['date'], tsla['mktcap_norm'], label='Market Cap (normalized)')
    plt.plot(tsla['date'], tsla['saleq_norm'], label='Sales (normalized)')
    plt.plot(tsla['date'], tsla['ps_norm'], label='P/S Ratio (normalized)')
    plt.xlabel('Date')
    plt.ylabel('Normalized Value')
    plt.title('Tesla: Market Cap, Sales, and P/S Ratio Over Time')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Filter for Tesla only
    tesla = merged[merged['tic'] == 'TSLA'].copy()

    # Calculate P/S ratio if not already done
    tesla['ps'] = tesla['mktcap'] / tesla['saleq']

    # Compute percentage change from previous row
    tesla['ps_pct_change'] = tesla['ps'].pct_change() * 100  # in %

    # Optional: smooth using rolling window to make spike clearer
    tesla['ps_pct_change_smooth'] = tesla['ps_pct_change'].rolling(window=5).mean()

    # Plot
    plt.figure(figsize=(12,6))
    plt.plot(tesla['date'], tesla['ps_pct_change_smooth'], label='P/S % change (smoothed)')
    plt.axhline(0, color='gray', linestyle='--')
    plt.title('Tesla P/S Ratio % Change Over Time')
    plt.xlabel('Date')
    plt.ylabel('% Change')
    plt.legend()
    plt.show()


plotting(merg)
