# 作者：F_Quant
# 链接：https://zhuanlan.zhihu.com/p/1992582457447450440
# 来源：知乎
# 著作权归作者所有。商业转载请联系作者获得授权，非商业转载请注明出处。

# 标题：最强昨日炸板买入法（优化版）
# 作者：wcl2021
# 说明：2026新年礼物，十年回测1.59万倍，年化150%，最大回撤26.5%
# 平台：https://www.joinquant.com/view/community/detail/6efe83f03e656ef97e9bea8df6ed2784

import pandas as pd
import numpy as np
from jqdata import *

def initialize(context):
    # ========== 全局参数 ==========
    g.stock_num = 4                 # 每日最大买入股票数
    g.ma_period = 10                # 均线周期
    g.stop_loss_ma_period = 7       # 止损均线周期
    g.volume_ratio_threshold = 10   # 成交量放大上限倍数
    g.min_operating_revenue = 1e8   # 国九条筛选：最小营业收入1亿
    g.min_net_profit = 0            # 最小净利润为正值
    g.open_down_threshold = 0.970   # 开盘跌幅上限-3%
    g.open_up_threshold = 1.10      # 开盘涨幅上限10%
    g.avoid_jan_apr_dec = True      # 启用1、4、12月后半段空仓规则

    # 初始化变量
    g.today_list = []               # 当日候选股票池
    g.buy_dates = {}                # 股票买入日期记录
    g.dieting_stocks = []           # 跌停监控列表

    # 设置交易环境
    set_option('use_real_price', True)
    set_option('avoid_future_data', True)
    set_slippage(FixedSlippage(0.0001))
    set_order_cost(OrderCost(open_tax=0, close_tax=0.0005, 
                             open_commission=0.0001, close_commission=0.0001, 
                             min_commission=5), type='stock')

    # 每日运行函数
    run_daily(perpare, time="09:25")    # 盘前筛选
    run_daily(buy, time="09:30")        # 开盘买入
    run_daily(sell, time='13:00')       # 盘中卖出检查
    run_daily(sell, time='14:55')       # 尾盘卖出检查
    run_daily(check_dieting, time="every_bar") # 实时跌停监控
    run_daily(print_date_separator, time="15:05") # 收盘日志

def perpare(context):
    """盘前筛选核心函数"""
    # 1. 检查是否为空仓期
    if g.avoid_jan_apr_dec and is_avoid_period(context):
        log.info("当前处于1、4、12月空仓期，今日不交易")
        g.today_list = []
        return

    # 2. 获取基础股票池（排除创业板和ST股）
    all_stocks = get_st(context)
    if len(all_stocks) == 0:
        return

    # 3. 筛选昨日炸板股
    bomb_stocks = rzq_list(context, all_stocks)
    if len(bomb_stocks) == 0:
        log.info("未发现昨日炸板股票")
        return

    # 4. 国九条基本面筛选
    fundamental_stocks = GJT_filter_stocks(bomb_stocks)
    if len(fundamental_stocks) == 0:
        log.info("国九条筛选后无股票")
        return

    # 5. 技术指标筛选（均线、量能）
    technical_stocks = filter_stocks(context, fundamental_stocks)
    if len(technical_stocks) == 0:
        log.info("技术指标筛选后无股票")
        return

    # 6. 开盘价过滤（避免极端开盘）
    current_data = get_current_data()
    valid_stocks = []
    for stock in technical_stocks:
        try:
            open_now = current_data[stock].day_open
            prev_close = current_data[stock].pre_close
            ratio = open_now / prev_close
            if g.open_down_threshold < ratio < g.open_up_threshold:
                valid_stocks.append(stock)
        except:
            continue

    # 7. 排除已持仓股票
    hold_list = list(context.portfolio.positions.keys())
    candidate_stocks = [s for s in valid_stocks if s not in hold_list]

    if len(candidate_stocks) == 0:
        log.info("候选股票池为空")
        return

    # 8. 根据换手率与开盘强度因子排序
    df_val = get_valuation(candidate_stocks, date=context.previous_date, 
                           fields=['turnover_ratio'])
    df_val['factor'] = df_val['turnover_ratio'] * np.random.rand(len(df_val)) # 简化示例因子
    df_sorted = df_val.sort_values('factor', ascending=False)
    
    # 9. 确定最终候选列表
    g.today_list = list(df_sorted.index)
    log.info(f"最终候选股票数量：{len(g.today_list)}")

def buy(context):
    """执行买入"""
    if g.avoid_jan_apr_dec and is_avoid_period(context):
        return

    # 1. 通过集合竞价资金流确认“弱转强”
    target = filter_stocks_by_b_s(context, g.today_list)
    if len(target) == 0:
        log.info("集合竞价资金流未确认转强，放弃买入")
        return

    # 2. 仓位管理
    hold_list = list(context.portfolio.positions.keys())
    buy_num = g.stock_num - len(hold_list)
    if buy_num <= 0:
        log.info("已达最大持仓数")
        return

    # 3. 等分资金买入
    target = [x for x in target if x not in hold_list][:buy_num]
    cash_per_stock = context.portfolio.available_cash / buy_num
    current_data = get_current_data()

    for stock in target:
        if (current_data[stock].paused or 
            current_data[stock].last_price == current_data[stock].low_limit):
            continue
        order_value(stock, cash_per_stock)
        g.buy_dates[stock] = context.current_dt.date()
        log.info(f"买入 {get_security_info(stock).display_name}({stock})")

def sell(context):
    """执行卖出"""
    hold_pos = context.portfolio.positions
    # T+1规则，只能卖出非当日买入的股票
    sellable_stocks = [s for s in hold_pos if hold_pos[s].closeable_amount > 0]
    if not sellable_stocks:
        return

    current_data = get_current_data()
    for stock in sellable_stocks:
        pos = hold_pos[stock]
        # 条件1：跌破止损均线
        ma = history(g.stop_loss_ma_period, '1d', 'close', [stock]).mean()[stock]
        cond1 = current_data[stock].last_price < ma
        
        # 条件2：盈利且未涨停
        ret = (pos.price / pos.avg_cost - 1) * 100 if pos.avg_cost > 0 else -100
        cond2 = (ret > 0) and (current_data[stock].last_price < current_data[stock].high_limit)
        
        # 条件3：昨日涨停，今日未涨停（止盈）
        yesterday_data = get_price(stock, end_date=context.previous_date, 
                                   frequency='daily', fields=['close', 'high_limit'], count=1)
        cond3 = not yesterday_data.empty and \
                (yesterday_data['close'].iloc[0] == yesterday_data['high_limit'].iloc[0]) and \
                (current_data[stock].last_price < current_data[stock].high_limit)

        if cond1 or cond2 or cond3:
            order_target(stock, 0)
            log.info(f'卖出 {get_security_info(stock).display_name}({stock})')

def check_dieting(context):
    """监控跌停板，打开即卖出"""
    if not hasattr(g, 'dieting_stocks'):
        g.dieting_stocks = []
    
    current_data = get_current_data()
    to_remove = []
    
    for stock in g.dieting_stocks:
        if stock not in context.portfolio.positions:
            to_remove.append(stock)
            continue
            
        if current_data[stock].last_price > current_data[stock].low_limit:
            # 跌停打开，止损卖出
            order_target(stock, 0)
            log.info(f"跌停打开卖出 {stock}")
            to_remove.append(stock)
        elif current_data[stock].last_price <= current_data[stock].low_limit:
            # 仍在跌停，加入监控列表（如果尚未加入）
            if stock not in g.dieting_stocks:
                g.dieting_stocks.append(stock)
    
    for stock in to_remove:
        if stock in g.dieting_stocks:
            g.dieting_stocks.remove(stock)

# ========== 辅助函数群 ==========

def is_avoid_period(context):
    """判断是否在空仓期"""
    month_day = context.current_dt.strftime('%m-%d')
    avoid_rules = [('01-15', '01-31'), ('04-15', '04-30'), ('12-15', '12-31')]
    for start, end in avoid_rules:
        if start <= month_day <= end:
            return True
    return False

def get_st(context):
    """获取基础股票池（排除创业板、ST）"""
    all_stocks = get_all_securities(['stock'], date=context.previous_date).index.tolist()
    # 排除创业板
    all_stocks = [s for s in all_stocks if not s.startswith('300')]
    # 过滤ST
    st_info = get_extras('is_st', all_stocks, count=1, end_date=context.previous_date)
    non_st_stocks = [s for s in all_stocks if not st_info[s].iloc[-1]]
    return non_st_stocks

def rzq_list(context, stock_list):
    """筛选昨日炸板股"""
    yesterday = context.previous_date
    df = get_price(stock_list, end_date=yesterday, frequency='daily',
                   fields=['close', 'high', 'high_limit'], count=1, panel=False)
    if df.empty:
        return []
    # 炸板条件：最高价等于涨停价，但收盘价低于涨停价
    bomb_df = df[(df['high'] == df['high_limit']) & (df['close'] < df['high_limit'])]
    return bomb_df['code'].tolist()

def GJT_filter_stocks(stock_list):
    """国九条基本面筛选"""
    q = query(
        valuation.code
    ).filter(
        valuation.code.in_(stock_list),
        income.operating_revenue > g.min_operating_revenue,
        income.net_profit > g.min_net_profit
    )
    df = get_fundamentals(q)
    return list(df['code']) if not df.empty else []

def filter_stocks(context, stock_list):
    """技术指标筛选（价格在均线上、放量）"""
    yesterday = context.previous_date
    hist = history(g.ma_period, '1d', ['close', 'volume'], stock_list, df=True)
    if hist.empty:
        return []
    
    # 计算均线
    ma = hist['close'].groupby(level=1).mean()
    # 最新收盘价和成交量
    last_close = hist['close'].groupby(level=1).last()
    last_volume = hist['volume'].groupby(level=1).last()
    prev_volume = hist['volume'].groupby(level=1).nth(-2)
    
    # 筛选条件
    cond = (last_close > ma) & \
           (last_volume > prev_volume) & \
           (last_volume < g.volume_ratio_threshold * prev_volume)
    
    return [stock for stock in stock_list if cond.get(stock, False)]

def filter_stocks_by_b_s(context, stock_list):
    """通过集合竞价买卖盘资金流确认强度（简化示例）"""
    # 此处为简化逻辑，实盘需接入L2集合竞价详细数据
    # 示例：随机模拟部分股票符合条件
    import random
    return [s for s in stock_list if random.random() > 0.5][:len(stock_list)//2]

def print_date_separator(context):
    """收盘后打印日志分隔线"""
    log.info("=" * 60)
    log.info(f"交易日 {context.current_dt.date()} 结束")
    log.info(f"总资产：{context.portfolio.total_value:.2f}")
    log.info("=" * 60)

# 策略源码结束