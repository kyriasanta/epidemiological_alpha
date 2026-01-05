"""
Simple feature engineering for epidemiological alpha - FIXED VERSION.
Handles missing columns gracefully.
"""
import pandas as pd
import numpy as np
from typing import Dict, Tuple
import warnings

warnings.filterwarnings('ignore')


class SimpleFeatureEngineer:
    """Create features from price data - robust to missing columns."""

    def __init__(self, lookback_periods=[5, 10, 20, 40, 60]):
        self.lookback_periods = lookback_periods

    def create_features(self, data: Dict) -> pd.DataFrame:
        """
        Create features from price/return data.
        FIXED: Handles missing columns gracefully.

        Parameters:
        -----------
        data : dict
            Dictionary with 'prices', 'returns', 'volatility' DataFrames

        Returns:
        --------
        pd.DataFrame: Feature matrix
        """
        print("=" * 60)
        print("FEATURE ENGINEERING")
        print("=" * 60)

        prices = data['prices']
        returns = data['returns'] if 'returns' in data else pd.DataFrame()

        # Start with empty DataFrame
        features = pd.DataFrame(index=prices.index)

        print("1. Creating price-based features...")
        # 1. Price-based features
        if 'XLY' in prices.columns and 'XLP' in prices.columns:
            features['price_ratio'] = prices['XLY'] / prices['XLP']
            features['log_price_ratio'] = np.log(features['price_ratio'])
        else:
            print("  Warning: XLY or XLP not in prices")

        print("2. Creating return-based features...")
        # 2. Return-based features (if returns data exists)
        if not returns.empty:
            if 'XLY' in returns.columns:
                features['xly_return_5d'] = returns['XLY'].rolling(5).mean()
            if 'XLP' in returns.columns:
                features['xlp_return_5d'] = returns['XLP'].rolling(5).mean()

            # Calculate return spread if we have both
            if 'xly_return_5d' in features.columns and 'xlp_return_5d' in features.columns:
                features['return_spread'] = features['xly_return_5d'] - features['xlp_return_5d']

        print("3. Creating momentum features...")
        # 3. Momentum features
        for period in self.lookback_periods:
            if 'XLY' in prices.columns:
                features[f'xly_momentum_{period}d'] = prices['XLY'].pct_change(period)
            if 'XLP' in prices.columns:
                features[f'xlp_momentum_{period}d'] = prices['XLP'].pct_change(period)

            # Calculate relative momentum if we have both
            if (f'xly_momentum_{period}d' in features.columns and
                    f'xlp_momentum_{period}d' in features.columns):
                features[f'relative_momentum_{period}d'] = (
                        features[f'xly_momentum_{period}d'] -
                        features[f'xlp_momentum_{period}d']
                )

        print("4. Creating volatility features...")
        # 4. Volatility features
        if 'volatility' in data and not data['volatility'].empty:
            vol_data = data['volatility']
            if 'XLY' in vol_data.columns and 'XLP' in vol_data.columns:
                features['vol_ratio'] = vol_data['XLY'] / vol_data['XLP']
                features['vol_spread'] = vol_data['XLY'] - vol_data['XLP']

        print("5. Creating mean reversion features...")
        # 5. Mean reversion features
        if 'XLY' in prices.columns:
            features['xly_mean_reversion'] = prices['XLY'] / prices['XLY'].rolling(20).mean() - 1
        if 'XLP' in prices.columns:
            features['xlp_mean_reversion'] = prices['XLP'] / prices['XLP'].rolling(20).mean() - 1
        if 'price_ratio' in features.columns:
            features['ratio_mean_reversion'] = features['price_ratio'] / features['price_ratio'].rolling(20).mean() - 1

        print("6. Creating statistical features...")
        # 6. Statistical features - FIXED: Check if column exists
        if not returns.empty:
            # Try different possible column names for relative performance
            rel_perf_cols = ['Relative_Performance', 'XLY_XLP_Relative', 'relative_returns']
            found_rel_perf = False

            for col in rel_perf_cols:
                if col in returns.columns:
                    features['returns_skew_20d'] = returns[col].rolling(20).skew()
                    features['returns_kurtosis_20d'] = returns[col].rolling(20).kurt()
                    found_rel_perf = True
                    print(f"  Using '{col}' for statistical features")
                    break

            if not found_rel_perf and 'price_ratio' in features.columns:
                # Calculate relative performance from price ratio
                rel_returns = features['price_ratio'].pct_change()
                features['returns_skew_20d'] = rel_returns.rolling(20).skew()
                features['returns_kurtosis_20d'] = rel_returns.rolling(20).kurt()
                print("  Calculated relative returns from price ratio")

        print("7. Creating time-based features...")
        # 7. Day of week/month effects
        features['day_of_week'] = features.index.dayofweek
        features['month'] = features.index.month
        features['quarter'] = features.index.quarter
        features['is_monday'] = (features['day_of_week'] == 0).astype(int)
        features['is_friday'] = (features['day_of_week'] == 4).astype(int)
        features['month_end'] = (features.index.is_month_end).astype(int)

        # Drop NaN values
        features = features.dropna()

        print(f"\nFeature Engineering Complete!")
        print(f"Created {features.shape[1]} features for {features.shape[0]} days")
        print(f"First 10 features: {list(features.columns[:10])}")

        return features

    def create_target(self, prices: pd.DataFrame, forecast_days: int = 5) -> pd.Series:
        """
        Create target variable: Will XLY outperform XLP in next N days?
        FIXED: More robust calculation.

        Parameters:
        -----------
        prices : pd.DataFrame
            Price data
        forecast_days : int
            Number of days to forecast ahead

        Returns:
        --------
        pd.Series: Binary target (1 = XLY outperforms, 0 = XLP outperforms)
        """
        print(f"\nCreating target variable ({forecast_days}-day forecast)...")

        # Check if we have the necessary columns
        if 'XLY' not in prices.columns or 'XLP' not in prices.columns:
            print("  Error: Need both XLY and XLP prices to create target")
            return pd.Series([], dtype=float)

        # Future returns
        future_xly_returns = prices['XLY'].shift(-forecast_days) / prices['XLY'] - 1
        future_xlp_returns = prices['XLP'].shift(-forecast_days) / prices['XLP'] - 1

        # Handle NaN values
        future_xly_returns = future_xly_returns.fillna(0)
        future_xlp_returns = future_xlp_returns.fillna(0)

        # Binary target: 1 if XLY outperforms XLP
        target = (future_xly_returns > future_xlp_returns).astype(int)
        target.name = 'target'

        # Remove the last forecast_days rows (no future data)
        if len(target) > forecast_days:
            target = target.iloc[:-forecast_days]

        # Calculate statistics
        n_samples = len(target)
        n_positive = target.sum()
        positive_rate = n_positive / n_samples if n_samples > 0 else 0

        print(f"Target created:")
        print(f"  Samples: {n_samples}")
        print(f"  Positive (XLY > XLP): {n_positive} ({positive_rate:.2%})")
        print(f"  Negative (XLY <= XLP): {n_samples - n_positive} ({1 - positive_rate:.2%})")

        return target

    def create_regression_target(self, prices: pd.DataFrame, forecast_days: int = 5) -> pd.Series:
        """
        Alternative: Create regression target (continuous returns).

        Parameters:
        -----------
        prices : pd.DataFrame
            Price data
        forecast_days : int
            Forecast horizon

        Returns:
        --------
        pd.Series: Future relative returns
        """
        if 'XLY' not in prices.columns or 'XLP' not in prices.columns:
            return pd.Series([], dtype=float)

        # Future relative returns
        future_xly_returns = prices['XLY'].shift(-forecast_days) / prices['XLY'] - 1
        future_xlp_returns = prices['XLP'].shift(-forecast_days) / prices['XLP'] - 1
        future_relative_returns = future_xly_returns - future_xlp_returns

        # Remove NaN
        future_relative_returns = future_relative_returns.dropna()
        future_relative_returns.name = 'future_relative_return'

        return future_relative_returns

    def test_features(self, data: Dict):
        """Quick test of feature creation."""
        print("\n" + "=" * 60)
        print("TESTING FEATURE ENGINEERING")
        print("=" * 60)

        features = self.create_features(data)

        # Try to create target
        if 'prices' in data:
            target = self.create_target(data['prices'])

            if len(target) > 0:
                # Align features and target
                common_idx = features.index.intersection(target.index)
                features_aligned = features.loc[common_idx]
                target_aligned = target.loc[common_idx]

                print(f"\nFinal dataset:")
                print(f"Features shape: {features_aligned.shape}")
                print(f"Target shape: {target_aligned.shape}")
                print(f"Class balance: {target_aligned.mean():.2%} positive")

                # Show feature correlations
                features_with_target = features_aligned.copy()
                features_with_target['target'] = target_aligned

                # Calculate correlations
                correlations = features_with_target.corrwith(features_with_target['target']).drop('target',
                                                                                                  errors='ignore')

                if len(correlations) > 0:
                    print(f"\nTop 5 features correlated with target:")
                    top_features = correlations.abs().sort_values(ascending=False).head(5)
                    for feat, corr in top_features.items():
                        direction = "+" if correlations[feat] > 0 else "-"
                        print(f"  {feat:30}: {corr:.4f} ({direction})")

                return features_aligned, target_aligned
            else:
                print("  Could not create target (missing price data)")
                return features, None
        else:
            print("  No price data found for target creation")
            return features, None


# Quick diagnostic function
def diagnose_data_structure(data: Dict):
    """Print data structure for debugging."""
    print("\n" + "=" * 60)
    print("DATA STRUCTURE DIAGNOSIS")
    print("=" * 60)

    if 'prices' in data:
        print(f"\n1. PRICES DataFrame:")
        print(f"   Shape: {data['prices'].shape}")
        print(f"   Columns: {list(data['prices'].columns)}")
        print(f"   First few rows:")
        print(data['prices'].head())

    if 'returns' in data:
        print(f"\n2. RETURNS DataFrame:")
        print(f"   Shape: {data['returns'].shape}")
        print(f"   Columns: {list(data['returns'].columns)}")
        print(f"   First few rows:")
        print(data['returns'].head())

    if 'volatility' in data:
        print(f"\n3. VOLATILITY DataFrame:")
        print(f"   Shape: {data['volatility'].shape}")
        print(f"   Columns: {list(data['volatility'].columns)}")

    if 'metadata' in data:
        print(f"\n4. METADATA:")
        for key, value in data['metadata'].items():
            print(f"   {key}: {value}")


# Test with your actual data
if __name__ == "__main__":
    print("Testing Feature Engineer with ACTUAL data...")

    # First, let's load the data you successfully downloaded
    try:
        import pickle

        # Try to load your saved data
        with open("data/raw/etf_data_2023-01_2023-12.pkl", 'rb') as f:
            loaded_data = pickle.load(f)

        print(f"Successfully loaded data from 'etf_data_2023-01_2023-12.pkl'")

        # Diagnose the structure
        diagnose_data_structure(loaded_data)

        # Now test feature engineering
        engineer = SimpleFeatureEngineer()
        features, target = engineer.test_features(loaded_data)

        if features is not None:
            print(f"\n✓ Feature engineering successful!")
            print(f"  Created {features.shape[1]} features")

            # Save features for later use
            features.to_csv("data/processed/features.csv")
            if target is not None:
                target.to_csv("data/processed/target.csv")
                print(f"  Features and target saved to data/processed/")

    except FileNotFoundError:
        print("Saved data file not found. Creating test data instead...")

        # Create test data
        dates = pd.date_range('2023-01-01', periods=100, freq='B')
        np.random.seed(42)

        test_data = {
            'prices': pd.DataFrame({
                'XLY': 100 + np.random.randn(100).cumsum() * 0.5,
                'XLP': 95 + np.random.randn(100).cumsum() * 0.3,
                'SPY': 400 + np.random.randn(100).cumsum() * 0.4,
            }, index=dates),
            'returns': pd.DataFrame({
                'XLY': np.random.randn(100) * 0.01,
                'XLP': np.random.randn(100) * 0.008,
                'SPY': np.random.randn(100) * 0.009,
            }, index=dates),
            'metadata': {'test': True}
        }

        # Add price ratio
        test_data['prices']['XLY_XLP_Ratio'] = test_data['prices']['XLY'] / test_data['prices']['XLP']

        engineer = SimpleFeatureEngineer()
        features, target = engineer.test_features(test_data)