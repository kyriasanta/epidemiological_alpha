"""
Simple trading model using basic rules - TESTED VERSION.
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import warnings

warnings.filterwarnings('ignore')


class SimpleTradingModel:
    """
    Simple rule-based trading model for epidemiological alpha.
    """

    def __init__(self, initial_capital: float = 10000):
        self.initial_capital = initial_capital
        self.signals = None
        self.results = None

    def generate_signals(self, features: pd.DataFrame,
                         strategy: str = 'epidemiological') -> pd.Series:
        """
        Generate trading signals using simple rules.
        """
        print("=" * 60)
        print(f"GENERATING SIGNALS: {strategy.upper()} STRATEGY")
        print("=" * 60)

        signals = pd.Series(0, index=features.index)

        if strategy == 'momentum':
            # Buy when momentum is positive
            if 'relative_momentum_10d' in features.columns:
                signals[features['relative_momentum_10d'] > 0.02] = 1
                signals[features['relative_momentum_10d'] < -0.02] = -1

        elif strategy == 'mean_reversion':
            # Buy when ratio is below mean, sell when above
            if 'ratio_mean_reversion' in features.columns:
                signals[features['ratio_mean_reversion'] < -0.05] = 1
                signals[features['ratio_mean_reversion'] > 0.05] = -1

        elif strategy == 'epidemiological':
            # Composite strategy based on your feature correlations
            buy_conditions = []
            sell_conditions = []

            # From your output: relative_momentum_10d has -0.71 correlation
            # Negative momentum predicts positive future → BUY signal
            if 'relative_momentum_10d' in features.columns:
                buy_conditions.append(features['relative_momentum_10d'] < -0.03)  # Strong negative momentum
                sell_conditions.append(features['relative_momentum_10d'] > 0.03)  # Strong positive momentum

            # ratio_mean_reversion: -0.70 correlation
            if 'ratio_mean_reversion' in features.columns:
                buy_conditions.append(features['ratio_mean_reversion'] < -0.04)  # Undervalued
                sell_conditions.append(features['ratio_mean_reversion'] > 0.04)  # Overvalued

            # Combine conditions
            if buy_conditions:
                buy_signal = pd.concat(buy_conditions, axis=1).all(axis=1)
                signals[buy_signal] = 1

            if sell_conditions:
                sell_signal = pd.concat(sell_conditions, axis=1).all(axis=1)
                signals[sell_signal] = -1

        self.signals = signals

        # Statistics
        n_signals = len(signals[signals != 0])
        n_buy = (signals == 1).sum()
        n_sell = (signals == -1).sum()

        print(f"\nSignal Generation Complete:")
        print(f"Total non-zero signals: {n_signals} ({n_signals / len(signals):.1%} of days)")
        print(f"Buy signals: {n_buy}")
        print(f"Sell signals: {n_sell}")

        return signals

    def backtest_simple(self, prices: pd.Series, signals: pd.Series) -> Dict:
        """
        Simple backtest.
        """
        print("\n" + "=" * 60)
        print("RUNNING BACKTEST")
        print("=" * 60)

        # Align data
        common_idx = prices.index.intersection(signals.index)
        prices_aligned = prices.loc[common_idx]
        signals_aligned = signals.loc[common_idx]

        print(f"Backtesting on {len(prices_aligned)} days...")

        # Initialize
        capital = self.initial_capital
        position = 0
        trades = []
        equity_curve = []

        prev_price = prices_aligned.iloc[0]

        for i in range(len(prices_aligned)):
            current_price = prices_aligned.iloc[i]
            signal = signals_aligned.iloc[i]
            date = prices_aligned.index[i]

            # Update portfolio
            if position != 0:
                capital += position * (current_price - prev_price)

            # Execute trades
            if signal == 1 and position == 0:
                position = capital / current_price
                capital = 0
                trades.append({
                    'date': date,
                    'type': 'BUY',
                    'price': current_price,
                    'size': position
                })

            elif signal == -1 and position > 0:
                capital = position * current_price
                position = 0
                trades.append({
                    'date': date,
                    'type': 'SELL',
                    'price': current_price
                })

            # Track equity
            current_equity = capital + (position * current_price if position != 0 else 0)
            equity_curve.append(current_equity)

            prev_price = current_price

        # Close final position
        if position > 0:
            capital = position * prices_aligned.iloc[-1]
            trades.append({
                'date': prices_aligned.index[-1],
                'type': 'CLOSE',
                'price': prices_aligned.iloc[-1]
            })

        # Calculate metrics
        equity_series = pd.Series(equity_curve, index=prices_aligned.index)

        if len(equity_series) > 1:
            returns = equity_series.pct_change().dropna()

            total_return = (equity_series.iloc[-1] / self.initial_capital) - 1
            annual_return = (1 + total_return) ** (252 / len(equity_series)) - 1
            volatility = returns.std() * np.sqrt(252) if len(returns) > 0 else 0
            sharpe_ratio = annual_return / volatility if volatility > 0 else 0

            # Drawdown
            cumulative = (1 + returns).cumprod()
            running_max = cumulative.expanding().max()
            drawdown = (cumulative - running_max) / running_max
            max_drawdown = drawdown.min()

            # Win rate
            sell_trades = [t for t in trades if t['type'] in ['SELL', 'CLOSE']]
            win_rate = 0.5  # Default

        else:
            total_return = 0
            annual_return = 0
            sharpe_ratio = 0
            max_drawdown = 0
            win_rate = 0

        self.results = {
            'initial_capital': self.initial_capital,
            'final_capital': equity_series.iloc[-1] if len(equity_series) > 0 else self.initial_capital,
            'total_return': total_return,
            'annual_return': annual_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'total_trades': len([t for t in trades if t['type'] in ['SELL', 'CLOSE']]),
            'equity_curve': equity_series,
            'trades': trades
        }

        print(f"\nBacktest Results:")
        print(f"  Total Return: {total_return:.2%}")
        print(f"  Annual Return: {annual_return:.2%}")
        print(f"  Sharpe Ratio: {sharpe_ratio:.3f}")
        print(f"  Max Drawdown: {max_drawdown:.2%}")
        print(f"  Total Trades: {self.results['total_trades']}")

        return self.results


# Quick test
if __name__ == "__main__":
    print("Testing Simple Trading Model...")

    # Create test data matching your feature structure
    dates = pd.date_range('2023-01-01', periods=100, freq='B')
    np.random.seed(42)

    # Test prices (XLY/XLP ratio)
    prices = pd.Series(
        1.0 + np.random.randn(100).cumsum() * 0.02,
        index=dates
    )

    # Test features matching your output
    features = pd.DataFrame({
        'relative_momentum_10d': np.random.randn(100) * 0.1,
        'ratio_mean_reversion': np.random.randn(100) * 0.05,
        'price_ratio': 1.05 + np.random.randn(100).cumsum() * 0.01
    }, index=dates)

    # Test
    model = SimpleTradingModel(initial_capital=10000)
    signals = model.generate_signals(features, strategy='epidemiological')
    results = model.backtest_simple(prices, signals)

    print(f"\nTest complete!")