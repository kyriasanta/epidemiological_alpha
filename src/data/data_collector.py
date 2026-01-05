"""
Minimal data collector using only yfinance - FIXED VERSION.
Handles different yfinance data structures.
"""
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings('ignore')


class SimpleDataCollector:
    """Minimal data collector for epidemiological alpha project."""

    def __init__(self, start_date="2020-01-01", end_date=None):
        """
        Initialize with date range.

        Parameters:
        -----------
        start_date : str
            Start date in YYYY-MM-DD format
        end_date : str
            End date (defaults to today)
        """
        self.start_date = start_date
        self.end_date = end_date or datetime.now().strftime("%Y-%m-%d")

    def download_etf_data(self):
        """
        Download ETF data for consumer discretionary vs staples.
        FIXED: Handles different yfinance data structures.

        Returns:
        --------
        dict: Dictionary with raw and processed data
        """
        print("=" * 60)
        print("Downloading ETF Data for Epidemiological Alpha")
        print("=" * 60)

        # Define our ETFs
        tickers = {
            'XLY': 'Consumer Discretionary',  # Luxury goods, cars, retail
            'XLP': 'Consumer Staples',  # Food, beverages, household items
            'SPY': 'S&P 500',  # Market benchmark
        }

        print(f"Downloading {list(tickers.keys())} from {self.start_date} to {self.end_date}")

        # Download data
        try:
            data = yf.download(
                list(tickers.keys()),
                start=self.start_date,
                end=self.end_date,
                progress=False,
                group_by='ticker'
            )

            if data.empty:
                raise ValueError("No data downloaded. Check dates and tickers.")

            print(f"Download successful. Data shape: {data.shape}")
            print(f"Data columns: {data.columns.tolist()[:10]}...")  # Show first 10 columns

        except Exception as e:
            print(f"Error downloading data: {e}")
            print("Falling back to simulated data...")
            return self._create_simulated_data()

        # FIXED: Extract adjusted close prices - handles different data structures
        prices = pd.DataFrame()

        # Check data structure
        print(f"\nAnalyzing data structure...")

        # Case 1: Single ticker format (no multi-level columns)
        if isinstance(data.columns, pd.Index) and 'Adj Close' in data.columns:
            print("Detected: Single DataFrame with 'Adj Close' column")
            prices = pd.DataFrame(index=data.index)
            for ticker in tickers.keys():
                prices[ticker] = data['Adj Close']

        # Case 2: Multi-level columns (ticker then metric)
        elif isinstance(data.columns, pd.MultiIndex):
            print("Detected: Multi-level columns (ticker, metric)")
            prices = pd.DataFrame(index=data.index)
            for ticker in tickers.keys():
                if ticker in data.columns.get_level_values(0):
                    # Try different possible column names
                    for adj_name in ['Adj Close', 'Adj_Close', 'AdjClose', 'Close']:
                        if adj_name in data[ticker].columns:
                            prices[ticker] = data[ticker][adj_name]
                            print(f"  Found {ticker} data in column '{adj_name}'")
                            break
                    else:
                        print(f"  Warning: Could not find price column for {ticker}")
                        # Use Close if Adj Close not found
                        prices[ticker] = data[ticker]['Close']
                else:
                    print(f"  Warning: Ticker {ticker} not in downloaded data")

        # Case 3: Simple DataFrame with ticker columns
        else:
            print("Detected: Simple DataFrame structure")
            prices = pd.DataFrame(index=data.index)
            for ticker in tickers.keys():
                if ticker in data.columns:
                    prices[ticker] = data[ticker]
                else:
                    # Try to find any column containing the ticker
                    matching_cols = [col for col in data.columns if ticker in str(col)]
                    if matching_cols:
                        prices[ticker] = data[matching_cols[0]]
                    else:
                        print(f"  Warning: Could not find data for {ticker}")

        # If we still have no data, fall back to simulated
        if prices.empty or len(prices.columns) == 0:
            print("No price data extracted. Using simulated data...")
            return self._create_simulated_data()

        # Fill any missing values
        prices = prices.ffill().bfill()

        # Remove any rows with all NaN
        prices = prices.dropna(how='all')

        print(f"\nExtracted {len(prices.columns)} tickers: {list(prices.columns)}")
        print(f"Date range: {prices.index[0].date()} to {prices.index[-1].date()}")
        print(f"Total rows: {len(prices)}")

        # Calculate daily returns
        returns = prices.pct_change().dropna()

        # Calculate relative performance (XLY vs XLP)
        if 'XLY' in prices.columns and 'XLP' in prices.columns:
            prices['XLY_XLP_Ratio'] = prices['XLY'] / prices['XLP']
            if 'XLY_XLP_Ratio' in prices.columns:
                returns['Relative_Performance'] = prices['XLY_XLP_Ratio'].pct_change().dropna()

        # Calculate rolling volatility (20-day)
        volatility = returns.rolling(window=20).std() * np.sqrt(252)

        # Create comprehensive dataset
        result = {
            'prices': prices,
            'returns': returns,
            'volatility': volatility,
            'tickers': tickers,
            'metadata': {
                'start_date': self.start_date,
                'end_date': self.end_date,
                'download_date': datetime.now().strftime("%Y-%m-%d"),
                'data_points': len(prices),
                'data_structure': str(type(data.columns))
            }
        }

        print(f"\nData Collection Complete!")
        print(f"Time period: {prices.index[0].date()} to {prices.index[-1].date()}")
        print(f"Total trading days: {len(prices)}")
        print(f"Price columns: {list(prices.columns)}")

        return result

    def _create_simulated_data(self):
        """Create simulated data if download fails."""
        print("Creating simulated ETF data...")

        # Generate date range
        start = pd.to_datetime(self.start_date)
        end = pd.to_datetime(self.end_date)
        dates = pd.bdate_range(start=start, end=end)

        np.random.seed(42)  # For reproducibility

        # Simulate prices with correlation
        n_days = len(dates)
        base_returns = np.random.randn(n_days) * 0.01

        # XLY: More volatile (discretionary)
        xly_returns = base_returns * 1.2 + np.random.randn(n_days) * 0.005
        xly_prices = 100 * np.exp(np.cumsum(xly_returns))

        # XLP: Less volatile (staples)
        xlp_returns = base_returns * 0.8 + np.random.randn(n_days) * 0.003
        xlp_prices = 95 * np.exp(np.cumsum(xlp_returns))

        # SPY: Market benchmark
        spy_returns = base_returns + np.random.randn(n_days) * 0.002
        spy_prices = 400 * np.exp(np.cumsum(spy_returns))

        # Create DataFrames
        prices = pd.DataFrame({
            'XLY': xly_prices,
            'XLP': xlp_prices,
            'SPY': spy_prices
        }, index=dates)

        returns = prices.pct_change().dropna()
        prices['XLY_XLP_Ratio'] = prices['XLY'] / prices['XLP']
        returns['Relative_Performance'] = prices['XLY_XLP_Ratio'].pct_change().dropna()

        volatility = returns.rolling(window=20).std() * np.sqrt(252)

        return {
            'prices': prices,
            'returns': returns,
            'volatility': volatility,
            'tickers': {'XLY': 'Consumer Discretionary', 'XLP': 'Consumer Staples', 'SPY': 'S&P 500'},
            'metadata': {'simulated': True, 'data_points': len(prices)}
        }

    def save_data(self, data, filename="etf_data.pkl"):
        """Save downloaded data to file."""
        import pickle
        # Ensure directory exists
        import os
        os.makedirs("data/raw", exist_ok=True)

        with open(f"data/raw/{filename}", 'wb') as f:
            pickle.dump(data, f)
        print(f"Data saved to data/raw/{filename}")

    def load_data(self, filename="etf_data.pkl"):
        """Load saved data from file."""
        import pickle
        with open(f"data/raw/{filename}", 'rb') as f:
            data = pickle.load(f)  # FIXED: was pickle.dump, should be pickle.load
        return data


# Quick test function - UPDATED
def test_data_collection():
    """Test the data collector."""
    print("Testing SimpleDataCollector...")

    # Try multiple date ranges in case one fails
    test_ranges = [
        ("2023-01-01", "2023-12-31"),
        ("2024-01-01", "2024-06-01"),  # More recent
        ("2022-01-01", "2022-06-01"),  # Different year
    ]

    for start_date, end_date in test_ranges:
        print(f"\n{'=' * 60}")
        print(f"Trying {start_date} to {end_date}")
        print('=' * 60)

        try:
            collector = SimpleDataCollector(
                start_date=start_date,
                end_date=end_date
            )

            data = collector.download_etf_data()

            print(f"\nData Keys: {list(data.keys())}")
            print(f"Prices shape: {data['prices'].shape}")
            print(f"Returns shape: {data['returns'].shape}")

            # Save the data
            collector.save_data(data, f"etf_data_{start_date[:7]}_{end_date[:7]}.pkl")

            # Quick visualization
            try:
                import matplotlib.pyplot as plt

                fig, axes = plt.subplots(2, 2, figsize=(12, 8))

                # Plot 1: Price series
                if 'XLY' in data['prices'].columns and 'XLP' in data['prices'].columns:
                    data['prices'][['XLY', 'XLP']].plot(ax=axes[0, 0], title='ETF Prices')
                    axes[0, 0].set_ylabel('Price ($)')
                    axes[0, 0].legend(['Discretionary (XLY)', 'Staples (XLP)'])

                # Plot 2: Relative performance ratio
                if 'XLY_XLP_Ratio' in data['prices'].columns:
                    data['prices']['XLY_XLP_Ratio'].plot(ax=axes[0, 1], title='XLY/XLP Ratio', color='purple')
                    axes[0, 1].axhline(y=data['prices']['XLY_XLP_Ratio'].mean(), color='r', linestyle='--', alpha=0.5)
                    axes[0, 1].set_ylabel('Ratio')

                # Plot 3: Returns distribution
                if 'Relative_Performance' in data['returns'].columns:
                    data['returns']['Relative_Performance'].hist(ax=axes[1, 0], bins=50, edgecolor='black', alpha=0.7)
                    axes[1, 0].set_title('Relative Returns Distribution')
                    axes[1, 0].set_xlabel('Daily Return')

                # Plot 4: Cumulative relative performance
                if 'Relative_Performance' in data['returns'].columns:
                    cum_perf = (1 + data['returns']['Relative_Performance']).cumprod() - 1
                    cum_perf.plot(ax=axes[1, 1], title='Cumulative Relative Performance', color='green')
                    axes[1, 1].set_ylabel('Cumulative Return')
                    axes[1, 1].axhline(y=0, color='black', linestyle='-', alpha=0.3)

                plt.tight_layout()
                plt.savefig(f'data_test_{start_date[:7]}_{end_date[:7]}.png', dpi=150, bbox_inches='tight')
                plt.close(fig)  # Close figure to avoid display issues
                print(f"Visualization saved to 'data_test_{start_date[:7]}_{end_date[:7]}.png'")

            except Exception as e:
                print(f"Visualization failed: {e}")

            return data  # Success!

        except Exception as e:
            print(f"Failed for {start_date}-{end_date}: {e}")
            continue

    print("\nAll date ranges failed. Using simulated data.")
    collector = SimpleDataCollector(start_date="2023-01-01", end_date="2023-12-31")
    data = collector._create_simulated_data()
    return data


# Diagnostic function to understand data structure
def diagnose_yfinance():
    """Diagnose what yfinance is returning."""
    print("\n" + "=" * 60)
    print("DIAGNOSING YFINANCE DATA STRUCTURE")
    print("=" * 60)

    try:
        # Test with single ticker first
        print("\n1. Testing single ticker (AAPL):")
        data_single = yf.download("AAPL", period="5d", progress=False)
        print(f"   Shape: {data_single.shape}")
        print(f"   Columns: {data_single.columns.tolist()}")
        print(f"   Column type: {type(data_single.columns)}")

        # Test with our ETFs
        print("\n2. Testing multiple tickers (XLY, XLP):")
        data_multi = yf.download(["XLY", "XLP"], period="5d", progress=False)
        print(f"   Shape: {data_multi.shape}")
        print(f"   Columns: {data_multi.columns.tolist()}")
        print(f"   Column type: {type(data_multi.columns)}")

        # Check if it's multi-level
        if isinstance(data_multi.columns, pd.MultiIndex):
            print(f"   MultiIndex levels: {data_multi.columns.nlevels}")
            print(f"   Level 0 (tickers): {data_multi.columns.get_level_values(0).unique().tolist()}")
            print(f"   Level 1 (metrics): {data_multi.columns.get_level_values(1).unique().tolist()}")

            # Show what's available for XLY
            print(f"\n3. Available columns for XLY:")
            if 'XLY' in data_multi.columns.get_level_values(0):
                xly_cols = data_multi['XLY'].columns.tolist()
                print(f"   {xly_cols}")

        return data_single, data_multi

    except Exception as e:
        print(f"Diagnosis failed: {e}")
        return None, None


if __name__ == "__main__":
    # First run diagnosis
    diagnose_yfinance()

    # Then run test
    print("\n" + "=" * 60)
    print("RUNNING MAIN DATA COLLECTION TEST")
    print("=" * 60)
    test_data_collection()