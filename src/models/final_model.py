"""
FINAL Simple Trading Model - Ready for interviews.
Simple, working, and demonstrates quant thinking.
"""
import pandas as pd
import numpy as np
from typing import Dict
import warnings

warnings.filterwarnings('ignore')


class FinalTradingModel:
    """
    Final simple model that works well enough for demonstration.
    Uses mean reversion logic based on your feature correlations.
    """

    def __init__(self, initial_capital: float = 10000):
        self.initial_capital = initial_capital
        self.signals = None
        self.results = None

    def generate_signals_simple(self, features: pd.DataFrame) -> pd.Series:
        """
        SUPER SIMPLE signal generation that should work.
        Based on your strong mean reversion correlations.
        """
        print("=" * 60)
        print("SIMPLE MEAN REVERSION SIGNALS")
        print("=" * 60)

        signals = pd.Series(0, index=features.index)

        # SIMPLE RULE: Buy when both conditions indicate mean reversion
        buy_condition = pd.Series(True, index=features.index)
        sell_condition = pd.Series(True, index=features.index)

        # Condition 1: Negative momentum (mean reversion buy signal)
        if 'relative_momentum_10d' in features.columns:
            buy_condition = buy_condition & (features['relative_momentum_10d'] < -0.01)
            sell_condition = sell_condition & (features['relative_momentum_10d'] > 0.01)

        # Condition 2: Undervalued ratio
        if 'ratio_mean_reversion' in features.columns:
            buy_condition = buy_condition & (features['ratio_mean_reversion'] < -0.02)
            sell_condition = sell_condition & (features['ratio_mean_reversion'] > 0.02)

        # Apply signals
        signals[buy_condition] = -1
        signals[sell_condition] = 1

        # Print stats
        n_signals = len(signals[signals != 0])
        print(f"Generated {n_signals} signals ({n_signals / len(signals):.1%} of days)")
        print(f"Buy: {(signals == 1).sum()}, Sell: {(signals == -1).sum()}")

        self.signals = signals
        return signals

    def backtest_simple(self, prices: pd.Series, signals: pd.Series) -> Dict:
        """
        Simple backtest with basic metrics.
        """
        print("\n" + "=" * 60)
        print("SIMPLE BACKTEST")
        print("=" * 60)

        # Align data
        common_idx = prices.index.intersection(signals.index)
        prices = prices.loc[common_idx]
        signals = signals.loc[common_idx]

        # SIMPLE backtest logic
        position = 0  # 0 = out, 1 = long, -1 = short
        capital = self.initial_capital
        equity = [capital]
        trades = []

        for i in range(1, len(prices)):
            current_price = prices.iloc[i]
            prev_price = prices.iloc[i - 1]
            signal = signals.iloc[i]

            # Update existing position
            if position == 1:  # Long position
                capital += (current_price - prev_price) * (capital / prev_price)
            elif position == -1:  # Short position
                capital -= (current_price - prev_price) * (capital / prev_price)

            # Execute new signals
            if signal == 1 and position != 1:  # Buy signal, not already long
                position = 1
                trades.append({'date': prices.index[i], 'action': 'BUY', 'price': current_price})
            elif signal == -1 and position != -1:  # Sell signal, not already short
                position = -1
                trades.append({'date': prices.index[i], 'action': 'SELL', 'price': current_price})
            elif signal == 0 and position != 0:  # Neutral signal, close position
                position = 0
                trades.append({'date': prices.index[i], 'action': 'CLOSE', 'price': current_price})

            equity.append(capital)

        # Calculate metrics
        equity_series = pd.Series(equity, index=prices.index)
        returns = equity_series.pct_change().dropna()

        if len(returns) > 0:
            total_return = (equity_series.iloc[-1] / self.initial_capital) - 1
            sharpe = returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0

            # Simple drawdown
            cumulative = (1 + returns).cumprod()
            running_max = cumulative.expanding().max()
            drawdown = (cumulative - running_max) / running_max
            max_dd = drawdown.min()
        else:
            total_return = 0
            sharpe = 0
            max_dd = 0

        self.results = {
            'total_return': total_return,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_dd,
            'total_trades': len([t for t in trades if t['action'] in ['SELL', 'CLOSE']]),
            'final_capital': equity_series.iloc[-1],
            'equity_curve': equity_series
        }

        print(f"Results:")
        print(f"  Total Return: {total_return:+.2%}")
        print(f"  Sharpe Ratio: {sharpe:+.3f}")
        print(f"  Max Drawdown: {max_dd:+.2%}")
        print(f"  Total Trades: {self.results['total_trades']}")

        return self.results

    def plot_simple(self, prices: pd.Series, save_path='strategy_results.png'):
        """Simple plot of results."""
        try:
            import matplotlib.pyplot as plt

            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

            # Plot 1: Price with signals
            ax1.plot(prices.index, prices.values, label='XLY/XLP Ratio', color='blue', alpha=0.7)

            if self.signals is not None:
                buy_signals = self.signals[self.signals == 1]
                sell_signals = self.signals[self.signals == -1]

                if len(buy_signals) > 0:
                    ax1.scatter(buy_signals.index, prices.loc[buy_signals.index],
                                color='green', s=50, marker='^', label='Buy', zorder=5)
                if len(sell_signals) > 0:
                    ax1.scatter(sell_signals.index, prices.loc[sell_signals.index],
                                color='red', s=50, marker='v', label='Sell', zorder=5)

            ax1.set_title('Trading Signals on XLY/XLP Ratio')
            ax1.set_ylabel('Ratio')
            ax1.legend()
            ax1.grid(True, alpha=0.3)

            # Plot 2: Equity curve
            if self.results and 'equity_curve' in self.results:
                ax2.plot(self.results['equity_curve'].index, self.results['equity_curve'].values,
                         label='Portfolio Value', color='green', linewidth=2)
                ax2.axhline(y=self.initial_capital, color='red', linestyle='--',
                            label=f'Initial ${self.initial_capital:,.0f}')
                ax2.set_title('Portfolio Performance')
                ax2.set_ylabel('Value ($)')
                ax2.set_xlabel('Date')
                ax2.legend()
                ax2.grid(True, alpha=0.3)

            plt.tight_layout()
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"\nPlot saved to {save_path}")

        except ImportError:
            print("Matplotlib not available for plotting")


# Quick test
if __name__ == "__main__":
    print("Testing Final Model...")

    # Simple test data
    dates = pd.date_range('2023-01-01', periods=100, freq='B')
    np.random.seed(42)

    # Create features with clear mean reversion pattern
    features = pd.DataFrame({
        'relative_momentum_10d': np.sin(np.arange(100) * 0.3) * 0.1,  # Oscillating
        'ratio_mean_reversion': np.sin(np.arange(100) * 0.25) * 0.08,
    }, index=dates)

    # Create prices with some trend
    prices = pd.Series(1.0 + np.arange(100) * 0.001 + np.sin(np.arange(100) * 0.2) * 0.05,
                       index=dates)

    model = FinalTradingModel(initial_capital=10000)
    signals = model.generate_signals_simple(features)
    results = model.backtest_simple(prices, signals)

    print("\n✓ Model test complete!")