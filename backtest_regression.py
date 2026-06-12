#!/usr/bin/env python3
import sys
sys.path.insert(0, '/Users/agiuser/code/fin-agent')

import tushare as ts
from fin_agent.config import Config
from fin_agent.tools.technical_indicators import regression_strategy
import json

def get_pro():
    ts.set_token(Config.TUSHARE_TOKEN)
    return ts.pro_api()

def main():
    print('获取上证A股股票列表...')
    pro = get_pro()
    df = pro.stock_basic(exchange='SSE', list_status='L', fields='ts_code,symbol,name,area,industry,list_date')
    sh_stocks = df.to_dict('records')

    # 取前30支股票
    top_30_stocks = sh_stocks[:30]
    print(f'选取 {len(top_30_stocks)} 支上证股票')

    results = []
    for stock in top_30_stocks:
        ts_code = stock['ts_code']
        name = stock['name']
        print(f'分析 {name} ({ts_code})...')
        
        try:
            result = regression_strategy(ts_code=ts_code)
            if isinstance(result, str) and result.startswith('Error'):
                print(f'  错误: {result}')
                continue
            
            result_data = json.loads(result)
            signals = result_data['signals']
            
            # 统计买入和卖出信号数量
            buy_signals = sum(1 for s in signals if s['buy_signal'] == 'yes')
            sell_signals = sum(1 for s in signals if s['sell_signal'] == 'yes')
            
            # 计算模拟收益（基于信号）
            total_return = 0
            trade_count = 0
            position = False
            buy_price = 0
            
            # 按日期升序处理信号进行回测
            signals_sorted = sorted(signals, key=lambda x: x['trade_date'])
            
            for signal in signals_sorted:
                if signal['buy_signal'] == 'yes' and not position:
                    buy_price = signal['close']
                    position = True
                elif signal['sell_signal'] == 'yes' and position:
                    sell_price = signal['close']
                    if buy_price > 0:
                        total_return += (sell_price - buy_price) / buy_price
                        trade_count += 1
                    position = False
            
            results.append({
                'ts_code': ts_code,
                'name': name,
                'buy_signals': buy_signals,
                'sell_signals': sell_signals,
                'trade_count': trade_count,
                'total_signals': len(signals),
                'total_return': round(total_return * 100, 2)
            })
        except Exception as e:
            print(f'  异常: {str(e)}')

    print()
    print('=' * 80)
    print('回归策略回测结果（上证30支股票 - 近90天）')
    print('=' * 80)
    print(f'{"股票名称":<12} {"代码":<12} {"买入信号":<8} {"卖出信号":<8} {"交易次数":<8} {"收益率(%)":<12}')
    print('-' * 80)

    for r in sorted(results, key=lambda x: x['total_return'], reverse=True):
        return_color = '+' if r['total_return'] > 0 else ''
        print(f'{r["name"]:<12} {r["ts_code"]:<12} {r["buy_signals"]:<8} {r["sell_signals"]:<8} {r["trade_count"]:<8} {return_color}{r["total_return"]:<12}')

    print()
    print(f'总计: {len(results)} 支股票')
    avg_return = sum(r['total_return'] for r in results) / len(results)
    print(f'平均收益率: {avg_return:.2f}%')
    profitable_count = sum(1 for r in results if r['total_return'] > 0)
    print(f'盈利股票数: {profitable_count}/{len(results)}')
    best_return = max(r['total_return'] for r in results)
    print(f'最高收益率: {best_return:.2f}%')
    worst_return = min(r['total_return'] for r in results)
    print(f'最低收益率: {worst_return:.2f}%')
    total_trades = sum(r['trade_count'] for r in results)
    print(f'总交易次数: {total_trades}')

if __name__ == '__main__':
    main()
