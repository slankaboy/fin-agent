import pandas as pd
import numpy as np
import tushare as ts
from datetime import datetime, timedelta
import json
from fin_agent.config import Config
from fin_agent.tools.technical_indicators import calculate_macd, calculate_rsi, calculate_kdj, calculate_boll

class BacktestEngine:
    def __init__(self, initial_capital=100000, commission=0.0003):
        self.initial_capital = initial_capital
        self.commission = commission
        self.cash = initial_capital
        self.position = 0
        self.history = [] # Trade history
        self.portfolio_values = [] # Daily portfolio values

    def _fetch_data(self, ts_code, start_date, end_date):
        """Fetch daily data using Tushare"""
        try:
            ts.set_token(Config.TUSHARE_TOKEN)
            pro = ts.pro_api()
            
            # Fetch a bit more data before start_date for indicator warm-up
            warmup_start = (datetime.strptime(start_date, '%Y%m%d') - timedelta(days=60)).strftime('%Y%m%d')
            
            df = pro.daily(ts_code=ts_code, start_date=warmup_start, end_date=end_date)
            if df.empty:
                raise ValueError(f"No data found for {ts_code}")
                
            # Sort ascending
            df = df.sort_values('trade_date', ascending=True).reset_index(drop=True)
            return df
        except Exception as e:
            raise e

    def _fetch_limit_data(self, ts_code, start_date, end_date):
        """Fetch limit up/down detail data (limit_list_d) for a stock."""
        try:
            ts.set_token(Config.TUSHARE_TOKEN)
            pro = ts.pro_api()
            warmup_start = (datetime.strptime(start_date, '%Y%m%d') - timedelta(days=10)).strftime('%Y%m%d')
            df = pro.limit_list_d(ts_code=ts_code, start_date=warmup_start, end_date=end_date)
            if df.empty:
                return pd.DataFrame()
            df = df.sort_values('trade_date', ascending=True).reset_index(drop=True)
            return df
        except Exception:
            return pd.DataFrame()

    def _calculate_indicators(self, df, strategy_config):
        """Calculate indicators needed for the strategy"""
        strategy_type = strategy_config.get('type', 'ma_cross')
        
        if strategy_type == 'ma_cross':
            short_window = int(strategy_config.get('short_window', 5))
            long_window = int(strategy_config.get('long_window', 20))
            
            df['short_ma'] = df['close'].rolling(window=short_window).mean()
            df['long_ma'] = df['close'].rolling(window=long_window).mean()
            
        elif strategy_type == 'macd':
            df = calculate_macd(df)
            
        elif strategy_type == 'rsi':
            window = int(strategy_config.get('window', 14))
            df = calculate_rsi(df, period=window)
            
        return df

    def _generate_signal(self, row, prev_row, strategy_config):
        """Generate Buy (1), Sell (-1), or Hold (0) signal"""
        strategy_type = strategy_config.get('type', 'ma_cross')
        
        if prev_row is None:
            return 0
            
        if strategy_type == 'ma_cross':
            # Golden Cross
            if prev_row['short_ma'] <= prev_row['long_ma'] and row['short_ma'] > row['long_ma']:
                return 1
            # Dead Cross
            elif prev_row['short_ma'] >= prev_row['long_ma'] and row['short_ma'] < row['long_ma']:
                return -1
                
        elif strategy_type == 'macd':
            # MACD Golden Cross (DIF crosses above DEA)
            if prev_row['dif'] <= prev_row['dea'] and row['dif'] > row['dea']:
                return 1
            # MACD Dead Cross
            elif prev_row['dif'] >= prev_row['dea'] and row['dif'] < row['dea']:
                return -1
                
        elif strategy_type == 'rsi':
            lower = int(strategy_config.get('lower', 30))
            upper = int(strategy_config.get('upper', 70))
            
            # Oversold -> Buy
            if prev_row['rsi'] >= lower and row['rsi'] < lower:
                return 1
            # Overbought -> Sell
            elif prev_row['rsi'] <= upper and row['rsi'] > upper:
                return -1
                
        return 0

    def run_limit_backtest(self, ts_code, start_date, end_date, strategy_config):
        """
        Backtest strategies based on limit up/down (涨跌停) and failed breaks (炸板).

        Strategies:
        - limit_up_follow: Buy on next open after a limit-up day, sell after N days or stop-loss.
        - limit_up_break: Buy when a stock breaks limit-up (炸板) intraday, expecting reversal.
        - continuous_limit: Buy after N consecutive limit-up days, sell on first non-limit day.
        """
        ts.set_token(Config.TUSHARE_TOKEN)
        pro = ts.pro_api()

        strategy_type = strategy_config.get('type', 'limit_up_follow')
        hold_days    = int(strategy_config.get('hold_days', 3))
        stop_loss    = float(strategy_config.get('stop_loss', -0.05))   # -5%
        take_profit  = float(strategy_config.get('take_profit', 0.10))  # +10%
        min_cont     = int(strategy_config.get('min_continuous', 2))    # for continuous_limit

        # Fetch price data
        df = self._fetch_data(ts_code, start_date, end_date)
        df = df[df['trade_date'] >= start_date].reset_index(drop=True)

        # Fetch limit detail data
        ldf = self._fetch_limit_data(ts_code, start_date, end_date)
        ldf = ldf[ldf['trade_date'] >= start_date].reset_index(drop=True) if not ldf.empty else pd.DataFrame()

        # Build lookup dicts keyed by trade_date
        price_map = {row['trade_date']: row for _, row in df.iterrows()}
        dates = sorted(price_map.keys())

        # limit_map: date -> limit row (only limit-up 'U')
        limit_map = {}
        if not ldf.empty:
            for _, row in ldf.iterrows():
                limit_map[row['trade_date']] = row

        trades = []
        portfolio_values = []
        cash = self.initial_capital
        position = 0
        entry_price = 0.0
        entry_date = None
        hold_count = 0
        consecutive = 0  # consecutive limit-up days counter

        for i, date in enumerate(dates):
            row = price_map[date]
            current_price = float(row['close'])
            lrow = limit_map.get(date)

            # Track consecutive limit-up days
            if lrow is not None and str(lrow.get('limit', '')) == 'U':
                consecutive += 1
            else:
                consecutive = 0

            # ── Manage open position ──────────────────────────────────────────
            if position > 0:
                hold_count += 1
                pnl_pct = (current_price - entry_price) / entry_price

                should_exit = False
                exit_reason = ''

                if pnl_pct <= stop_loss:
                    should_exit = True
                    exit_reason = 'stop_loss'
                elif pnl_pct >= take_profit:
                    should_exit = True
                    exit_reason = 'take_profit'
                elif hold_count >= hold_days:
                    should_exit = True
                    exit_reason = 'hold_days'
                elif strategy_type == 'continuous_limit':
                    # Exit on first day that is NOT a limit-up
                    if lrow is None or str(lrow.get('limit', '')) != 'U':
                        should_exit = True
                        exit_reason = 'limit_broken'

                if should_exit:
                    revenue = position * current_price
                    comm = revenue * self.commission
                    cash += revenue - comm
                    trades.append({
                        'date': date, 'action': 'SELL', 'price': current_price,
                        'shares': position, 'pnl_pct': round(pnl_pct * 100, 2),
                        'reason': exit_reason
                    })
                    position = 0
                    entry_price = 0.0
                    hold_count = 0

            # ── Entry signals ─────────────────────────────────────────────────
            if position == 0 and i + 1 < len(dates):
                next_date = dates[i + 1]
                next_row = price_map.get(next_date)
                if next_row is None:
                    continue
                buy_price = float(next_row['open'])

                enter = False

                if strategy_type == 'limit_up_follow':
                    # Buy next open after a limit-up close
                    if lrow is not None and str(lrow.get('limit', '')) == 'U':
                        enter = True

                elif strategy_type == 'limit_up_break':
                    # 炸板: stock hit limit-up intraday but closed below limit
                    # Detected when open_times > 0 (limit opened) but no 'U' close
                    if lrow is not None:
                        open_times = int(lrow.get('open_times', 0) or 0)
                        is_limit_close = str(lrow.get('limit', '')) == 'U'
                        if open_times > 0 and not is_limit_close:
                            enter = True  # 炸板 reversal buy

                elif strategy_type == 'continuous_limit':
                    # Buy after reaching min_continuous consecutive limit-up days
                    if consecutive >= min_cont:
                        enter = True

                if enter and cash > buy_price:
                    shares = int(cash / (buy_price * (1 + self.commission)) / 100) * 100
                    if shares > 0:
                        cost = shares * buy_price
                        comm = cost * self.commission
                        cash -= cost + comm
                        position = shares
                        entry_price = buy_price
                        entry_date = next_date
                        hold_count = 0
                        trades.append({
                            'date': next_date, 'action': 'BUY', 'price': buy_price,
                            'shares': shares, 'trigger_date': date,
                            'strategy': strategy_type
                        })

            total_value = cash + position * current_price
            portfolio_values.append({'trade_date': date, 'value': total_value})

        # Force close at end
        if position > 0:
            last_price = float(df.iloc[-1]['close'])
            revenue = position * last_price
            comm = revenue * self.commission
            cash += revenue - comm
            trades.append({
                'date': dates[-1], 'action': 'SELL', 'price': last_price,
                'shares': position, 'reason': 'end_of_backtest'
            })

        final_value = cash
        total_return = (final_value - self.initial_capital) / self.initial_capital

        pv_df = pd.DataFrame(portfolio_values)
        max_drawdown = 0.0
        if not pv_df.empty:
            pv_df['cummax'] = pv_df['value'].cummax()
            pv_df['drawdown'] = (pv_df['cummax'] - pv_df['value']) / pv_df['cummax']
            max_drawdown = float(pv_df['drawdown'].max())

        # Win rate
        sell_trades = [t for t in trades if t['action'] == 'SELL' and 'pnl_pct' in t]
        win_rate = (sum(1 for t in sell_trades if t['pnl_pct'] > 0) / len(sell_trades) * 100) if sell_trades else 0

        return {
            "ts_code": ts_code,
            "strategy": strategy_type,
            "params": {k: v for k, v in strategy_config.items() if k != 'type'},
            "initial_capital": self.initial_capital,
            "final_value": round(final_value, 2),
            "total_return_pct": round(total_return * 100, 2),
            "max_drawdown_pct": round(max_drawdown * 100, 2),
            "trades_count": len([t for t in trades if t['action'] == 'BUY']),
            "win_rate_pct": round(win_rate, 2),
            "recent_trades": trades[-6:]
        }

    def run(self, ts_code, start_date, end_date, strategy_config):
        # 1. Prepare Data
        df = self._fetch_data(ts_code, start_date, end_date)
        df = self._calculate_indicators(df, strategy_config)
        
        # Filter data to match requested start_date (after indicator calculation)
        # But we need prev_row, so we keep one extra row before start_date if possible
        mask = df['trade_date'] >= start_date
        # Find index of first row matching mask
        if not mask.any():
            return {"error": "No data in requested date range"}
            
        start_idx = mask.idxmax()
        if start_idx > 0:
            start_idx -= 1 # Keep one prior row for signal calc
            
        # Iterate
        prev_row = None
        
        for index, row in df.iloc[start_idx:].iterrows():
            # Skip rows before actual start_date for TRADING, but use for signal
            is_trading_period = row['trade_date'] >= start_date
            
            current_price = row['close']
            
            # Record Portfolio Value (Daily close)
            if is_trading_period:
                total_value = self.cash + (self.position * current_price)
                self.portfolio_values.append({
                    'trade_date': row['trade_date'],
                    'value': total_value
                })
            
            # Generate Signal
            signal = self._generate_signal(row, prev_row, strategy_config)
            
            # Execute Trade
            if is_trading_period and signal != 0:
                # Buy
                if signal == 1 and self.cash > 0:
                    # Buy max shares (lots of 100)
                    max_shares = int(self.cash / (current_price * (1 + self.commission)) / 100) * 100
                    if max_shares > 0:
                        cost = max_shares * current_price
                        comm = cost * self.commission
                        self.cash -= (cost + comm)
                        self.position += max_shares
                        self.history.append({
                            'date': row['trade_date'],
                            'action': 'BUY',
                            'price': current_price,
                            'shares': max_shares,
                            'commission': comm
                        })
                # Sell
                elif signal == -1 and self.position > 0:
                    # Sell all
                    revenue = self.position * current_price
                    comm = revenue * self.commission
                    self.cash += (revenue - comm)
                    
                    self.history.append({
                        'date': row['trade_date'],
                        'action': 'SELL',
                        'price': current_price,
                        'shares': self.position,
                        'commission': comm
                    })
                    self.position = 0
            
            prev_row = row
            
        # Final Calculation
        final_value = self.cash + (self.position * df.iloc[-1]['close'])
        total_return = (final_value - self.initial_capital) / self.initial_capital
        
        # Calculate Max Drawdown
        pv_df = pd.DataFrame(self.portfolio_values)
        if not pv_df.empty:
            pv_df['cummax'] = pv_df['value'].cummax()
            pv_df['drawdown'] = (pv_df['cummax'] - pv_df['value']) / pv_df['cummax']
            max_drawdown = pv_df['drawdown'].max()
        else:
            max_drawdown = 0
            
        return {
            "ts_code": ts_code,
            "strategy": strategy_config.get('type'),
            "initial_capital": self.initial_capital,
            "final_value": round(final_value, 2),
            "total_return_pct": round(total_return * 100, 2),
            "max_drawdown_pct": round(max_drawdown * 100, 2),
            "trades_count": len(self.history),
            "trades": self.history[-5:] # Return last 5 trades for brevity
        }

def run_backtest(ts_code, strategy="ma_cross", start_date=None, end_date=None, params=None):
    """
    Wrapper for Tool usage.
    params: JSON string or dict of strategy parameters.

    Standard strategies: ma_cross, macd, rsi
    Limit strategies:    limit_up_follow, limit_up_break, continuous_limit
    """
    if not start_date:
        start_date = (datetime.now() - timedelta(days=365)).strftime('%Y%m%d')
    if not end_date:
        end_date = datetime.now().strftime('%Y%m%d')
        
    # Parse params
    strategy_config = {"type": strategy}
    if params:
        if isinstance(params, str):
            try:
                params_dict = json.loads(params)
                strategy_config.update(params_dict)
            except:
                pass
        elif isinstance(params, dict):
            strategy_config.update(params)
            
    engine = BacktestEngine()
    try:
        limit_strategies = ('limit_up_follow', 'limit_up_break', 'continuous_limit')
        if strategy in limit_strategies:
            result = engine.run_limit_backtest(ts_code, start_date, end_date, strategy_config)
        else:
            result = engine.run(ts_code, start_date, end_date, strategy_config)
        return json.dumps(result, ensure_ascii=False, indent=2)
    except Exception as e:
        return f"Error running backtest: {str(e)}"

