import pandas as pd
import numpy as np
import tushare as ts
from fin_agent.config import Config
from datetime import datetime, timedelta
import json

def get_pro():
    ts.set_token(Config.TUSHARE_TOKEN)
    return ts.pro_api()

def calculate_macd(df, fast_period=12, slow_period=26, signal_period=9):
    """
    Calculate MACD, Signal, and Hist.
    """
    # EMA12
    ema12 = df['close'].ewm(span=fast_period, adjust=False).mean()
    # EMA26
    ema26 = df['close'].ewm(span=slow_period, adjust=False).mean()
    # DIF
    dif = ema12 - ema26
    # DEA
    dea = dif.ewm(span=signal_period, adjust=False).mean()
    # MACD Hist
    macd_hist = (dif - dea) * 2
    
    df['dif'] = dif
    df['dea'] = dea
    df['macd'] = macd_hist
    return df

def calculate_rsi(df, period=14):
    """
    Calculate RSI.
    """
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).ewm(alpha=1/period, adjust=False).mean()
    loss = (-delta.where(delta < 0, 0)).ewm(alpha=1/period, adjust=False).mean()
    
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    return df

def calculate_kdj(df, k_period=9, d_period=3, j_period=3):
    """
    Calculate KDJ.
    """
    low_min = df['low'].rolling(window=k_period).min()
    high_max = df['high'].rolling(window=k_period).max()
    
    rsv = (df['close'] - low_min) / (high_max - low_min) * 100
    
    k_values = []
    d_values = []
    
    k = 50
    d = 50
    
    for r in rsv:
        if pd.isna(r):
            k_values.append(np.nan)
            d_values.append(np.nan)
            continue
        k = (2/3) * k + (1/3) * r
        d = (2/3) * d + (1/3) * k
        k_values.append(k)
        d_values.append(d)
        
    df['k'] = k_values
    df['d'] = d_values
    df['j'] = 3 * df['k'] - 2 * df['d']
    return df

def calculate_boll(df, period=20, std_dev=2):
    """
    Calculate Bollinger Bands.
    """
    df['boll_mid'] = df['close'].rolling(window=period).mean()
    std = df['close'].rolling(window=period).std()
    df['boll_upper'] = df['boll_mid'] + (std * std_dev)
    df['boll_lower'] = df['boll_mid'] - (std * std_dev)
    return df

def detect_patterns(df):
    """
    Detect technical patterns from a dataframe containing indicators.
    Returns a dictionary of detected patterns for the latest date.
    """
    if df.empty or len(df) < 2:
        return {}
        
    # Ensure data is sorted by date ascending for calculation
    # (Though get_technical_indicators uses ascending for calculation, it returns descending or sorts it again inside)
    # Let's ensure we are working with the end of the series (latest data)
    
    # Check sort order. If the first date is newer than the last, it's descending.
    if df.iloc[0]['trade_date'] > df.iloc[-1]['trade_date']:
        # Descending, so row 0 is latest, row 1 is yesterday
        curr = df.iloc[0]
        prev = df.iloc[1]
    else:
        # Ascending, so last row is latest
        curr = df.iloc[-1]
        prev = df.iloc[-2]

    patterns = []
    
    # --- MACD Patterns ---
    # Golden Cross: Previous DIF < DEA and Current DIF > DEA
    if prev['dif'] < prev['dea'] and curr['dif'] > curr['dea']:
        patterns.append("MACD_Golden_Cross (MACD金叉)")
    # Dead Cross: Previous DIF > DEA and Current DIF < DEA
    if prev['dif'] > prev['dea'] and curr['dif'] < curr['dea']:
        patterns.append("MACD_Dead_Cross (MACD死叉)")
        
    # --- KDJ Patterns ---
    # Golden Cross: K crosses above D
    if prev['k'] < prev['d'] and curr['k'] > curr['d']:
        patterns.append("KDJ_Golden_Cross (KDJ金叉)")
    # Dead Cross: K crosses below D
    if prev['k'] > prev['d'] and curr['k'] < curr['d']:
        patterns.append("KDJ_Dead_Cross (KDJ死叉)")
    # Overbought/Oversold
    if curr['k'] > 80 or curr['d'] > 80:
         patterns.append("KDJ_Overbought (KDJ超买)")
    if curr['k'] < 20 or curr['d'] < 20:
         patterns.append("KDJ_Oversold (KDJ超卖)")

    # --- RSI Patterns ---
    if curr['rsi'] > 70:
        patterns.append("RSI_Overbought (RSI超买)")
    if curr['rsi'] < 30:
        patterns.append("RSI_Oversold (RSI超卖)")
        
    # --- Bollinger Bands ---
    if curr['close'] > curr['boll_upper']:
        patterns.append("BOLL_Upper_Break (突破布林上轨)")
    if curr['close'] < curr['boll_lower']:
        patterns.append("BOLL_Lower_Break (跌破布林下轨)")
        
    return {
        "trade_date": curr['trade_date'],
        "patterns": patterns,
        "signals": {
            "macd": "bullish" if curr['dif'] > curr['dea'] else "bearish",
            "rsi": curr['rsi'],
            "kdj": "bullish" if curr['k'] > curr['d'] else "bearish"
        }
    }

def get_technical_indicators(ts_code, start_date=None, end_date=None):
    """
    Get technical indicators (MACD, RSI, KDJ, BOLL) for a stock.
    Returns the last 5 records by default to save token usage, 
    but calculates based on a longer history to ensure accuracy.
    """
    # Fetch enough history for accurate calculation (at least 60-90 days)
    if not end_date:
        end_date = datetime.now().strftime('%Y%m%d')
    
    # Logic: Fetch ~200 days of data to calculate indicators properly
    calc_start_date = (datetime.strptime(end_date, '%Y%m%d') - timedelta(days=365)).strftime('%Y%m%d')
    
    try:
        pro = get_pro()
        df = pro.daily(ts_code=ts_code, start_date=calc_start_date, end_date=end_date)
        
        if df.empty:
            return f"No daily data found for {ts_code} to calculate indicators."
        
        # Sort ascending for calculation
        df = df.sort_values('trade_date', ascending=True).reset_index(drop=True)
        
        # Calculate Indicators
        df = calculate_macd(df)
        df = calculate_rsi(df)
        df = calculate_kdj(df)
        df = calculate_boll(df)
        
        # Format columns
        # Keep trade_date, close, and indicators
        cols = ['trade_date', 'close', 'dif', 'dea', 'macd', 'rsi', 'k', 'd', 'j', 'boll_upper', 'boll_mid', 'boll_lower']
        result_df = df[cols].copy()
        
        # Round values
        for col in cols:
            if col != 'trade_date':
                result_df[col] = result_df[col].round(3)
        
        # Sort descending again for display (newest first)
        result_df = result_df.sort_values('trade_date', ascending=False)
        
        # Filter by requested start_date if provided
        if start_date:
             result_df = result_df[result_df['trade_date'] >= start_date]
        else:
             # Default return last 10 days to avoid token limit overflow
             result_df = result_df.head(10)
             
        return result_df.to_json(orient='records', force_ascii=False)
        
    except Exception as e:
        return f"Error calculating technical indicators: {str(e)}"

def get_technical_patterns(ts_code):
    """
    Identify technical patterns (Golden Cross, Overbought/Oversold, etc.) for a stock based on latest data.
    """
    end_date = datetime.now().strftime('%Y%m%d')
    calc_start_date = (datetime.strptime(end_date, '%Y%m%d') - timedelta(days=365)).strftime('%Y%m%d')
    
    try:
        pro = get_pro()
        df = pro.daily(ts_code=ts_code, start_date=calc_start_date, end_date=end_date)
        
        if df.empty:
            return f"No daily data found for {ts_code}."
            
        # Sort ascending for calculation
        df = df.sort_values('trade_date', ascending=True).reset_index(drop=True)
        
        # Calculate Indicators
        df = calculate_macd(df)
        df = calculate_rsi(df)
        df = calculate_kdj(df)
        df = calculate_boll(df)
        
        # Detect patterns
        result = detect_patterns(df)
        
        return json.dumps(result, ensure_ascii=False)
    except Exception as e:
        return f"Error identifying technical patterns: {str(e)}"

def detect_macd_convergence(df, lookback=5):
    """
    Detect MACD convergence at high or low levels.
    :param df: DataFrame containing MACD indicators (dif, dea, macd)
    :param lookback: Number of periods to look back for convergence detection
    :return: 'high_convergence', 'low_convergence', or None
    """
    if len(df) < lookback + 1:
        return None
    
    # Get recent data (last lookback periods)
    recent_data = df.iloc[-lookback:]
    
    # Check if MACD histogram is converging (getting smaller in absolute value)
    hist_values = recent_data['macd'].abs().values
    
    # Check if histogram is decreasing (converging)
    is_converging = all(hist_values[i] >= hist_values[i+1] for i in range(len(hist_values)-1))
    
    if not is_converging:
        return None
    
    # Check if at high or low level
    last_dif = df.iloc[-1]['dif']
    last_dea = df.iloc[-1]['dea']
    avg_hist = recent_data['macd'].mean()
    
    # Define thresholds for "high" and "low" convergence
    # Using standard deviation to determine significant levels
    macd_std = df['macd'].std()
    if macd_std == 0:
        return None
    
    # High convergence: both DIF and DEA are positive and histogram is converging
    if last_dif > macd_std and last_dea > macd_std:
        return 'high_convergence'
    
    # Low convergence: both DIF and DEA are negative and histogram is converging
    if last_dif < -macd_std and last_dea < -macd_std:
        return 'low_convergence'
    
    return None

def regression_strategy(ts_code, start_date=None, end_date=None):
    """
    Regression Strategy that generates buy/sell signals based on:
    Buy signals:
        1. MACD converging at low levels
        2. RSI < 30 (oversold)
        3. Previous day close within 3% of BOLL lower band
    
    Sell signals (either condition):
        1. RSI > 70 AND price pullback 5% from recent high (profit taking)
        2. MACD Dead Cross (DIF crosses below DEA)
    
    :param ts_code: Stock code
    :param start_date: Start date (YYYYMMDD)
    :param end_date: End date (YYYYMMDD)
    :return: JSON string with signals
    """
    if not end_date:
        end_date = datetime.now().strftime('%Y%m%d')
    
    # Fetch enough history for accurate calculation
    calc_start_date = (datetime.strptime(end_date, '%Y%m%d') - timedelta(days=365)).strftime('%Y%m%d')
    
    try:
        pro = get_pro()
        df = pro.daily(ts_code=ts_code, start_date=calc_start_date, end_date=end_date)
        
        if df.empty:
            return f"No daily data found for {ts_code}."
        
        # Sort ascending for calculation
        df = df.sort_values('trade_date', ascending=True).reset_index(drop=True)
        
        # Calculate Indicators
        df = calculate_macd(df)
        df = calculate_rsi(df)
        df = calculate_boll(df)
        
        signals = []
        
        # Track position state for trailing stop
        in_position = False
        buy_price = 0
        max_price = 0
        
        # Generate signals for each day (starting from day with enough data)
        for i in range(20, len(df)):
            current = df.iloc[i]
            prev = df.iloc[i-1]
            prev2 = df.iloc[i-2]
            
            # Check MACD convergence using lookback data
            lookback_df = df.iloc[max(0, i-10):i+1]
            macd_conv = detect_macd_convergence(lookback_df)
            
            # MACD Dead Cross: Previous DIF > DEA and Current DIF < DEA
            macd_dead_cross = prev['dif'] > prev['dea'] and current['dif'] < current['dea']
            
            # Buy signal conditions
            buy_cond1 = macd_conv == 'low_convergence'
            buy_cond2 = prev['rsi'] < 35  # Previous day RSI < 30
            buy_cond3 = False
            if not pd.isna(prev['boll_lower']) and prev['boll_lower'] != 0:
                # Previous day close within 3% of BOLL lower band
                lower_band = prev['boll_lower']
                price = prev['close']
                buy_cond3 = abs(price - lower_band) / lower_band <= 0.03
            
            buy_signal = all([buy_cond1, buy_cond2, buy_cond3])
            
            # Update position state on buy signal
            if buy_signal and not in_position:
                in_position = True
                buy_price = current['close']
                max_price = current['close']
            
            # Update max price if in position
            if in_position and current['high'] > max_price:
                max_price = current['high']
            
            # Check if price is above BOLL middle band
            above_boll_mid = False
            if not pd.isna(current['boll_mid']) and current['boll_mid'] != 0:
                above_boll_mid = current['close'] > current['boll_mid']
            
            # Sell signal conditions (both require price above BOLL middle band)
            # Condition A: RSI > 70 AND price pullback 5% from max price (while in position) AND above BOLL mid
            sell_cond_a = False
            if in_position and max_price > 0:
                pullback = (max_price - current['close']) / max_price
                sell_cond_a = current['rsi'] > 70 and pullback >= 0.05 and above_boll_mid
            
            # Condition B: MACD Dead Cross AND above BOLL mid
            sell_cond_b = macd_dead_cross and above_boll_mid
            
            # Sell signal: either condition A or B
            sell_signal = sell_cond_a or sell_cond_b
            
            # Reset position on sell signal
            if sell_signal and in_position:
                in_position = False
            
            signal = {
                'trade_date': str(current['trade_date']),
                'close': round(float(current['close']), 2),
                'rsi': round(float(current['rsi']), 2),
                'dif': round(float(current['dif']), 4) if not pd.isna(current['dif']) else None,
                'dea': round(float(current['dea']), 4) if not pd.isna(current['dea']) else None,
                'macd_convergence': macd_conv if macd_conv else '',
                'boll_upper': round(float(current['boll_upper']), 2) if not pd.isna(current['boll_upper']) else None,
                'boll_lower': round(float(current['boll_lower']), 2) if not pd.isna(current['boll_lower']) else None,
                'in_position': 'yes' if in_position else 'no',
                'buy_price': round(float(buy_price), 2) if buy_price > 0 else None,
                'max_price': round(float(max_price), 2) if max_price > 0 else None,
                'buy_signal': 'yes' if buy_signal else 'no',
                'sell_signal': 'yes' if sell_signal else 'no',
                'buy_conditions': {
                    'macd_low_convergence': 'yes' if buy_cond1 else 'no',
                    'rsi_below_30': 'yes' if buy_cond2 else 'no',
                    'near_boll_lower': 'yes' if buy_cond3 else 'no'
                },
                'sell_conditions': {
                    'rsi_above_70_with_pullback': 'yes' if sell_cond_a else 'no',
                    'macd_dead_cross': 'yes' if sell_cond_b else 'no'
                }
            }
            
            signals.append(signal)
        
        # Filter by requested date range
        if start_date:
            signals = [s for s in signals if s['trade_date'] >= start_date]
        
        # Sort by date descending (newest first)
        signals = sorted(signals, key=lambda x: x['trade_date'], reverse=True)
        
        return json.dumps({
            'ts_code': ts_code,
            'signals': signals,
            'strategy_rules': {
                'buy': [
                    'MACD在低位收敛',
                    'RSI指标低于30',
                    '前一天收盘价在布林下轨3%范围内'
                ],
                'sell': [
                    '价格在布林中轨以上',
                    'RSI > 70 且从近期高点回落5% (止盈)',
                    'MACD死叉 (DIF下穿DEA)'
                ]
            }
        }, ensure_ascii=False)
        
    except Exception as e:
        return f"Error running regression strategy: {str(e)}"
