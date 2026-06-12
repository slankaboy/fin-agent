#!/usr/bin/env python3
import sys
sys.path.insert(0, '/Users/agiuser/code/fin-agent')

from fin_agent.tools.technical_indicators import regression_strategy
import json

# 上证股票列表（手动定义）
sh_stocks = [
    {'ts_code': '600000.SH', 'name': '浦发银行'},
    {'ts_code': '600004.SH', 'name': '白云机场'},
    {'ts_code': '600006.SH', 'name': '东风股份'},
    {'ts_code': '600007.SH', 'name': '中国国贸'},
    {'ts_code': '600008.SH', 'name': '首创环保'},
    {'ts_code': '600009.SH', 'name': '上海机场'},
    {'ts_code': '600010.SH', 'name': '包钢股份'},
    {'ts_code': '600011.SH', 'name': '华能国际'},
    {'ts_code': '600012.SH', 'name': '皖通高速'},
    {'ts_code': '600015.SH', 'name': '华夏银行'},
    {'ts_code': '600016.SH', 'name': '民生银行'},
    {'ts_code': '600018.SH', 'name': '上港集团'},
    {'ts_code': '600019.SH', 'name': '宝钢股份'},
    {'ts_code': '600028.SH', 'name': '中国石化'},
    {'ts_code': '600029.SH', 'name': '南方航空'},
    {'ts_code': '600030.SH', 'name': '中信证券'},
    {'ts_code': '600036.SH', 'name': '招商银行'},
    {'ts_code': '600519.SH', 'name': '贵州茅台'},
    {'ts_code': '601318.SH', 'name': '中国平安'},
    {'ts_code': '601398.SH', 'name': '工商银行'},
    {'ts_code': '601899.SH', 'name': '紫金矿业'},
    {'ts_code': '601328.SH', 'name': '交通银行'},
    {'ts_code': '601988.SH', 'name': '中国银行'},
    {'ts_code': '601628.SH', 'name': '中国人寿'},
    {'ts_code': '600031.SH', 'name': '三一重工'},
    {'ts_code': '601898.SH', 'name': '中煤能源'},
    {'ts_code': '601166.SH', 'name': '兴业银行'},
    {'ts_code': '600585.SH', 'name': '海螺水泥'},
    {'ts_code': '600017.SH', 'name': '日照港'},
    {'ts_code': '600023.SH', 'name': '浙能电力'},
]

def main():
    # 设置180天数据范围
    from datetime import datetime, timedelta
    end_date = datetime.now().strftime('%Y%m%d')
    start_date = (datetime.now() - timedelta(days=180)).strftime('%Y%m%d')
    
    print(f'选取 {len(sh_stocks)} 支上证股票 (分析日期: {start_date} - {end_date})')

    results = []
    for stock in sh_stocks:
        ts_code = stock['ts_code']
        name = stock['name']
        print(f'分析 {name} ({ts_code})...')
        
        try:
            result = regression_strategy(ts_code=ts_code, start_date=start_date, end_date=end_date)
            if isinstance(result, str) and result.startswith('Error'):
                print(f'  错误: {result}')
                continue
            
            result_data = json.loads(result)
            signals = result_data['signals']
            
            buy_signals = sum(1 for s in signals if s['buy_signal'] == 'yes')
            sell_signals = sum(1 for s in signals if s['sell_signal'] == 'yes')
            
            total_return = 0
            trade_count = 0
            position = False
            buy_price = 0
            
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
