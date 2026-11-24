# Candle data

Put your historical 1-minute OHLCV CSV files here, or anywhere else, and import them with
`POST /api/datasets/import` or the Dataset selector.

Accepted layouts (headers are case-insensitive):

    timestamp,open,high,low,close,volume
    2025-01-06 09:15:00,22000.5,22010.0,21995.0,22004.0,1250

or separate `date` and `time` columns with the same OHLC/volume columns.

Validation rejects: missing columns, non-numeric or non-positive prices, high/low that do not bound
open and close, and fewer than 500 candles. Duplicate timestamps are collapsed.

Generate synthetic test data (random walk, not market data) with `python scripts/generate_sample_data.py`.
