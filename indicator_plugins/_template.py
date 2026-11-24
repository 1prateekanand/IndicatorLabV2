"""TEMPLATE - files whose names start with an underscore are NOT loaded.

To add one of your own indicators:
  1. Copy this file to e.g. my_indicators.py (no leading underscore).
  2. Decorate each causal signal function with @indicator.
  3. Return a pandas Series aligned to df.index, with values in {-1, 0, 1}.
  4. Use only data up to and including the current bar (rolling windows, ewm, shift(1)).
  5. Press "Reload registry" on the Indicators page, or POST /api/indicators/reload.

The registry runs a self-test on every indicator. If it fails, the indicator is listed as
unavailable with the reason, and it is excluded from search.
"""
import pandas as pd

from indicatorlab.registry import ParamSpec, indicator


@indicator(
    "my_example_indicator",            # unique, lowercase, letters/digits/underscores
    category="Custom",
    params=(
        # ParamSpec(name, default, low, high, kind="int"|"float", step=None)
        ParamSpec("period", 20, 5, 100),
    ),
    description="Replace with a description of your indicator.",
)
def my_example_indicator(df: pd.DataFrame, period: int = 20) -> pd.Series:
    # Example only: long when close is above its simple moving average, short when below.
    sma = df["close"].rolling(period).mean()
    signal = (df["close"] > sma).astype(int) - (df["close"] < sma).astype(int)
    return signal.astype("int8")
