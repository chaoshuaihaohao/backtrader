#!/usr/bin/env python
# -*- coding: utf-8; py-indent-offset:4 -*-
###############################################################################

from __future__ import (absolute_import, division, print_function, unicode_literals)
# ====== 在 my_test.py 文件的最开头添加 ======
import sys
import os
# 将包含 backtrader/ 文件夹的目录加入 Python 路径
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
# ==========================================
import warnings
import argparse
import datetime
import csv
import finplot as fplt
import backtrader.btfinplot as btfinplot
import backtrader as bt
import matplotlib
from PyQt6.QtWidgets import QApplication, QPushButton, QWidget, QHBoxLayout, QVBoxLayout, QLabel
from PyQt6.QtCore import Qt
import talib
import pandas as pd
import numpy as np

# ===================== ✅✅✅ 全局统一颜色配置【核心 - 白色背景 高对比度】✅✅✅ =====================
# 基础色（A股习惯：红涨绿跌）
GLOBAL_RED = '#FF4500'  # 涨/多头（橘红，比纯红柔和且醒目）
GLOBAL_GREEN = '#32CD32'  # 跌/空头（草绿，高对比度）
GLOBAL_YELLOW = '#FFD700'  # 买入信号（金黄，白色背景超醒目）
GLOBAL_BLUE = '#1E90FF'  # 卖出信号（深蓝，高对比度）
GLOBAL_ORANGE = '#FF8C00'  # 指标线1（橘色）
GLOBAL_PURPLE = '#9370DB'  # 指标线2（紫蓝）
GLOBAL_CYAN = '#00CED1'  # 指标线3（青色）
GLOBAL_BLACK = '#000000'  # 纯黑（文字/主线条）
GLOBAL_GRAY_DARK = '#333333'  # 深灰（次要文字/网格）
GLOBAL_GRAY_MID = '#666666'  # 中灰（提示文字）
GLOBAL_GRAY_LIGHT = '#E6E6E6'  # 浅灰（边框/分隔线/背景阴影）
GLOBAL_WHITE = '#FFFFFF'  # 纯白（全局背景）

# 背景/文字统一配置（白色背景核心）
GLOBAL_BG_WHITE = GLOBAL_WHITE
GLOBAL_BG_BLACK = GLOBAL_BLACK  # 兼容原有变量，强制白色
GLOBAL_BG_DARK = GLOBAL_WHITE  # 兼容原有变量，强制白色
GLOBAL_FONT_WHITE = GLOBAL_BLACK  # 兼容原有变量，强制黑色文字
GLOBAL_FONT_BLACK = GLOBAL_BLACK  # 主文字色（纯黑）
GLOBAL_FONT_GRAY = GLOBAL_GRAY_DARK  # 次要文字（深灰）
GLOBAL_LINE_GRAY = GLOBAL_GRAY_LIGHT  # 网格/分隔线（浅灰）
GLOBAL_LINE_BLACK = GLOBAL_BLACK  # 主指标线（纯黑）
# ==========================================================================================

# 解决中文显示乱码 + 负号显示问题
matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False

# ========== ✅ 环境配置 ==========
warnings.filterwarnings('ignore')
os.environ['QT_DEVICE_PIXEL_RATIO'] = '0'
os.environ['QT_AUTO_SCREEN_SCALE_FACTOR'] = '1'
os.environ['QT_FONT_DPI'] = '96'
os.environ['QT_LOGGING_RULES'] = 'qt.qpa.window=false'

# ========== ✅ PyQt6初始化 ==========
app = QApplication(sys.argv)

# 版本适配核心常量
BTVERSION = tuple(int(x) for x in bt.__version__.split('.'))
# 全局配置
RESULT_FILE = "macd_opt_result.csv"
TRADE_LOG_FILE = "trade_log.csv"  # 新增：每笔交易的盈亏日志文件
is_redirect = not sys.stdout.isatty()
OUT_ENCODING = 'utf-8' if is_redirect else 'gbk'
sys.stdout = open(sys.stdout.fileno(), mode='w', encoding=OUT_ENCODING, buffering=1)
sys.stderr = open(sys.stdout.fileno(), mode='w', encoding=OUT_ENCODING, buffering=1)


# ===================== 【1】自定义百分比仓位管理器 =====================
class FixedPerc(bt.Sizer):
    '''This sizer simply returns a fixed size for any operation

    Params:
      - ``perc`` (default: ``0.20``) Perc of cash to allocate for operation
    '''

    params = (
        ('perc', 0.20),  # perc of cash to use for operation
    )

    def _getsizing(self, comminfo, cash, data, isbuy):
        cashtouse = self.p.perc * cash
        if BTVERSION > (1, 7, 1, 93):
            size = comminfo.getsize(data.close[0], cashtouse)
        else:
            size = cashtouse // data.close[0]
        return max(size, 1)  # 确保至少买入1手


class BIAS(bt.Indicator):
    """自定义乖离率指标"""
    # 定义指标输出线
    lines = ('bias',)
    # 定义参数（period=6是你代码中用到的，可灵活调整）
    params = (('period', 6),)

    def __init__(self):
        # 计算BIAS：(收盘价 - N日简单移动平均) / N日简单移动平均 * 100
        ma = bt.indicators.MovingAverageSimple(self.data, period=self.p.period)
        self.lines.bias = (self.data - ma) / ma * 100


# ===================== 【扩展】自定义CSV解析类，支持turnover（换手率）字段 =====================
import backtrader.feeds as btfeeds


class GenericCSVWithTurnover(btfeeds.GenericCSVData):
    # 1. 新增自定义字段：turnover（换手率），指定别名（方便策略中调用）
    lines = ('turnover',)  # 新增的线，和open/close一样属于data的属性

    # 2. 配置参数：指定turnover列在CSV中的索引（从0开始数！）
    # 你的CSV列顺序：0:date,1:股票代码,2:open,3:close,4:high,5:low,6:volume,7:amount,8:amplitude,9:pct_change,10:price_change,11:turnover
    params = (
        ('turnover', 11),  # 关键：turnover在你的CSV中是第11列（索引11）
        ('stock_code', -1),  # 股票代码列（索引1），设为-1表示不解析（用不到）
        ('amount', -1),  # 成交额列（索引7），设为-1不解析
        ('amplitude', -1),  # 振幅列（索引8），设为-1不解析
        # 其他默认字段的索引（必须和你的CSV列对应，覆盖父类默认值！）
        ('datetime', 0),  # date在0列
        ('open', 2),  # open在2列
        ('close', 3),  # close在3列
        ('high', 4),  # high在4列
        ('low', 5),  # low在5列
        ('volume', 6),  # volume在6列
        ('openinterest', -1),  # 无持仓量，设为-1
    )


def get_ma_trend_global(ma_indicator, compare_n, threshold):
    if len(ma_indicator) < compare_n + 1:
        return 0.0, "持平"

    ma_now = ma_indicator[0]
    ma_prev = ma_indicator[-compare_n]
    ma_slope = (ma_now - ma_prev) / compare_n
    if ma_slope > threshold:
        ma_trend = "强向上"
    elif ma_slope > -threshold / 2:
        ma_trend = "弱向上"
    elif ma_slope < -threshold:
        ma_trend = "强向下"
    else:
        ma_trend = "弱向下"
    return ma_slope, ma_trend


def is_ma_golden_cross(ma_short, ma_long, n=2):
    if len(ma_short) < n + 2 or len(ma_long) < n + 2:
        return False
    cross_cond = ma_short[-1] < ma_long[-1] and ma_short[0] > ma_long[0]
    hold_cond = True
    for i in range(1, n + 1):
        if ma_short[-i] <= ma_long[-i]:
            hold_cond = False
            break
    return cross_cond and hold_cond


def is_price_break_ma(price, ma_line, direction='down'):
    if len(price) < 2 or len(ma_line) < 2:
        return False
    if direction == 'down':
        return price[0] < ma_line[0] and price[-1] >= ma_line[-1]
    else:
        return price[0] > ma_line[0] and price[-1] <= ma_line[-1]


def is_macd_bottom_divergence(price, macd, n=10):
    if len(price) < n or len(macd) < n:
        return False
    price_low = min(price[-i] for i in range(n))
    macd_low = min(macd[-i] for i in range(n))
    return price[0] <= price_low and macd[0] > macd_low


# ===================== 【2】核心策略 =====================
class MACDOptStrategy(bt.Strategy):
    params = (
        # Standard MACD Parameters
        ('macd1', 12),
        ('macd2', 26),
        ('macdsig', 9),
        ('atrperiod', 14), ('atrdist', 1.5),
        ('rsi_period', 6),
        ('total_loss_limit', 0.02),
        ('single_loss_limit', 0.05),
        ('kdj_period', 9), ('kdj_k_period', 3), ('kdj_d_period', 3),
        # 新增：板块类型参数，适配换手率/量比阈值（大盘蓝筹/中小盘成长/题材股/周期股）
        ('stock_type', 'mid_small'),  # default: 中小盘成长，可选：blue_chip/topic/cycle
        # 乖离率周期
        ('bias6_period', 6),
    )

    def __init__(self):
        # 交易统计相关变量
        self.win_trades = 0
        self.lose_trades = 0
        self.total_trades = 0
        self.total_win_pnl = 0.0
        self.total_lose_pnl = 0.0
        # ========== 核心修复1：初始化trades列表 解决AttributeError ==========
        self.trades = []
        # 新增：交易日志相关
        self.trade_id = 0  # 交易编号
        self.trade_logs = []  # 存储每笔交易日志
        # 新增：存储下单时的指标值
        self.order_indicator_info = {}

        # ========== 新增：跟踪当前持仓的开仓信息 ==========
        self.current_position_info = {
            'buy_price': 0.0,
            'buy_date': None,
            'buy_size': 0,
            'sell_price': 0.0,
            'sell_date': None,
            'sell_size': 0,
            # 新增：存储下单时的指标值
            'buy_indicators': {},
            'sell_indicators': {}
        }

        # 订单和止损相关变量
        self.order = None
        self.pstop = 0.0
        self.avg_buy_price = 0.0
        self.position_size = 0
        self.init_cash = self.broker.getvalue()

        # 记录开仓信息
        self.buy_price = 0.0
        self.buy_date = None

        # 均线指标
        self.ma5 = bt.indicators.SMA(self.data.close, period=5)
        self.ma10 = bt.indicators.SMA(self.data.close, period=10)
        self.ma20 = bt.indicators.SMA(self.data.close, period=20)
        self.ma60 = bt.indicators.SMA(self.data.close, period=60)
        self.ma200 = bt.indicators.SMA(self.data.close, period=200)

        # 成交量
        # 1. 获取原始成交量（股），换算为“手”（保留整数）
        self.vol = self.data.volume
        self.turnover = self.data.turnover
        # 2. 计算均量线（滑动平均，backtrader内置SMA函数）
        self.vol5 = bt.indicators.SMA(self.vol, period=5)
        self.vol10 = bt.indicators.SMA(self.vol, period=10)
        # MACD指标 - 修复版（和券商完全一致，无参数冲突）
        # 1. 定义【券商标准】的非调整型EMA（核心：第一个参数是data，第二个是period）
        # ema_broker = lambda data, period: bt.indicators.EMA(data, period=period)

        # 2. 直接调用MACDHisto（子类包含macd/signal/histo，无需重复创建MACD）
        self.macd = bt.indicators.MACDHisto(
            self.data0,
            period_me1=self.p.macd1,
            period_me2=self.p.macd2,
            period_signal=self.p.macdsig,
            # movav=ema_broker  # 传入正确的自定义EMA
        )
        # 直接使用内置线，无需手动计算，和券商一致
        self.macd_line = self.macd.macd  # MACD线
        self.signal_line = self.macd.signal  # 信号线
        self.macd_hist = self.macd.histo  # 柱状图（macd线 - 信号线，券商标准）

        # 周线MACD同步修复（和日线保持一致，适配跨周期判断）
        self.macd_week = bt.indicators.MACDHisto(
            self.data1,
            period_me1=self.p.macd1,
            period_me2=self.p.macd2,
            period_signal=self.p.macdsig,
            # movav=ema_broker  # 同样传入正确的自定义EMA
        )

        self.mcross = bt.indicators.CrossOver(self.macd.macd, self.macd.signal)

        self.atr = bt.indicators.ATR(self.data0, period=self.p.atrperiod)
        self.rsi6 = bt.indicators.RSI(self.data0, period=6)
        self.rsi12 = bt.indicators.RSI(self.data0, period=12)
        self.rsi24 = bt.indicators.RSI(self.data0, period=24)
        self.volume_ema = bt.indicators.EMA(self.data.volume, period=10)

        # KDJ指标计算
        stoch = bt.indicators.KDJ(
            self.data0,
            period=self.p.kdj_period,
            period_dfast=self.p.kdj_k_period,
            period_dslow=self.p.kdj_d_period,
            movav=bt.indicators.SMA,
            upperband=80,
            lowerband=20
        )
        self.k = stoch.percK
        self.d = stoch.percD
        self.j = 3 * self.k - 2 * self.d

        # ===================== 新增：布林线指标计算（backtrader内置）=====================
        # 布林线核心参数：周期20（默认）、2倍标准差（默认），基于收盘价计算
        self.boll = bt.indicators.BollingerBands(
            self.data0.close,  # 基于日线收盘价计算
            period=20,  # 中轨均线周期
            devfactor=2.0  # 上下轨标准差倍数
        )
        # 单独提取布林线的上、中、下轨，方便后续策略判断
        self.boll_upper = self.boll.top  # 上轨
        self.boll_mid = self.boll.mid  # 中轨（20日均线）
        self.boll_lower = self.boll.bot  # 下轨
        # 可选：计算价格与布林线的偏离度（用于超买超卖判断）
        self.boll_percent = (self.data0.close - self.boll_lower) / (self.boll_upper - self.boll_lower) * 100

        # KDJ金叉/死叉判断
        self.kdj_gold_cross = bt.indicators.CrossOver(self.k, self.d)

        # 买卖信号存储
        self.buy_signals = []
        self.sell_signals = []

        # ===================== 新增：补全手册所需指标 =====================
        # 量比（当日成交量/5日平均成交量，实时计算）
        self.volume_5ma = bt.indicators.SMA(self.data0.volume, period=5)
        self.volume_ratio = self.data0.volume / self.volume_5ma
        # 乖离率BIAS6（6日）- 手册动量类0.5分核心指标
        self.bias6 = BIAS(self.data0.close, period=self.p.bias6_period)
        # 北向资金占位（需对接实盘数据接口，如tushare/akshare）
        self.north_capital = 0.0  # 正数=净流入，负数=净流出，单位：万元
        # 筹码峰占位（需对接实盘数据接口，返回获利盘比例）
        self.profit_ratio = 0.0  # 获利盘比例，0-100
        self.cover_ratio = 0.0  # 套牢盘比例，0-100
        # 前5日平均换手率（用于卖出换手率打分）
        self.turnover_5ma = bt.indicators.SMA(self.data0.turnover, period=5)

        # ===================== 新增：分数拆解存储 - 关键！用于命令行打印具体分项 =====================
        self.daily_score_log = {}  # 键：日期字符串（YYYY-MM-DD），值：{buy_score:总得分, buy_details:分项字典, sell_score:总得分, sell_details:分项字典}

    def get_current_indicators(self):
        """获取当前周期所有指标的数值，返回字典"""
        indicators = {
            # 均线指标
            'MA5': round(self.ma5[0], 2) if len(self.ma5) > 0 else np.nan,
            'MA10': round(self.ma10[0], 2) if len(self.ma10) > 0 else np.nan,
            'MA20': round(self.ma20[0], 2) if len(self.ma20) > 0 else np.nan,
            'MA60': round(self.ma60[0], 2) if len(self.ma60) > 0 else np.nan,
            'MA200': round(self.ma200[0], 2) if len(self.ma200) > 0 else np.nan,
            # 成交量相关
            '成交量(手)': round(self.vol[0], 2) if len(self.vol) > 0 else np.nan,
            '均量线5': round(self.vol5[0], 2) if len(self.vol5) > 0 else np.nan,
            '成交量EMA10': round(self.volume_ema[0], 2) if len(self.volume_ema) > 0 else np.nan,
            # MACD相关
            'MACD值': round(self.macd.macd[0], 4) if len(self.macd.macd) > 0 else np.nan,
            'MACD信号线': round(self.macd.signal[0], 4) if len(self.macd.signal) > 0 else np.nan,
            'MACD柱状图': round(self.macd_hist[0], 4) if len(self.macd_hist) > 0 else np.nan,
            # ATR
            'ATR': round(self.atr[0], 4) if len(self.atr) > 0 else np.nan,
            # RSI
            'RSI6': round(self.rsi6[0], 2) if len(self.rsi6) > 0 else np.nan,
            # KDJ
            'KDJ_K': round(self.k[0], 2) if len(self.k) > 0 else np.nan,
            'KDJ_D': round(self.d[0], 2) if len(self.d) > 0 else np.nan,
            'KDJ_J': round(self.j[0], 2) if len(self.j) > 0 else np.nan,
            # 布林线
            '布林上轨': round(self.boll_upper[0], 2) if len(self.boll_upper) > 0 else np.nan,
            '布林中轨': round(self.boll_mid[0], 2) if len(self.boll_mid) > 0 else np.nan,
            '布林下轨': round(self.boll_lower[0], 2) if len(self.boll_lower) > 0 else np.nan,
            '布林偏离度(%)': round(self.boll_percent[0], 2) if len(self.boll_percent) > 0 else np.nan,
            # 价格相关
            '当前收盘价': round(self.data0.close[0], 2) if len(self.data0.close) > 0 else np.nan,
            '当前开盘价': round(self.data0.open[0], 2) if len(self.data0.open) > 0 else np.nan,
            '当前最高价': round(self.data0.high[0], 2) if len(self.data0.high) > 0 else np.nan,
            '当前最低价': round(self.data0.low[0], 2) if len(self.data0.low) > 0 else np.nan,
            # 趋势判断
            'MA200趋势': self.ma200_trend if hasattr(self, 'ma200_trend') else "未知",
            'MA200斜率': round(self.ma200_slope, 6) if hasattr(self, 'ma200_slope') else np.nan,
            '价格是否在MA200上方': (self.data.close[0] > self.ma200[0]) if len(self.ma200) > 0 else False,
            # KDJ状态
            'KDJ是否金叉': self.kdj_gold_cross[0] > 0 if len(self.kdj_gold_cross) > 0 else False,
            'KDJ是否死叉': self.kdj_gold_cross[0] < 0 if len(self.kdj_gold_cross) > 0 else False,
            'KDJ是否超买': self.k[0] > 80 if len(self.k) > 0 else False,
            'KDJ是否超卖': self.k[0] < 20 if len(self.k) > 0 else False,
            # MACD状态
            'MACD是否金叉': self.mcross[0] > 0 if len(self.mcross) > 0 else False,
            'MACD是否死叉': self.mcross[0] < 0 if len(self.mcross) > 0 else False,
            'MACD是否在零轴上方': self.macd.macd[0] > 0 if len(self.macd.macd) > 0 else False,
            # 新增手册指标
            '量比': round(self.volume_ratio[0], 2) if len(self.volume_ratio) > 0 else np.nan,
            '乖离率BIAS6(%)': round(self.bias6[0], 2) if len(self.bias6) > 0 else np.nan,
            '换手率(%)': round(self.turnover[0], 2) if len(self.turnover) > 0 else np.nan,
            '北向资金(万元)': self.north_capital,
            '获利盘比例(%)': self.profit_ratio,
            '套牢盘比例(%)': self.cover_ratio,
        }
        return indicators

    def notify_order(self, order):
        if self.order is None:
            return
        if order.status in [order.Completed]:
            # 获取成交日期（会是下一个交易日）
            # print(f"订单成交日期: {self.data.datetime.date(0)}")
            exec_date = bt.num2date(order.executed.dt)
            dt_str = exec_date.strftime('%Y-%m-%d')
            # print(f"实际上它是基于这个时刻的价格: {dt_str}, 成交价格: {order.executed.price}")
            price = order.executed.price
            size = order.executed.size  # 买入为正，卖出为负

            if order.isbuy():
                self.buy_signals.append((dt_str, price))
                self.buy_price = price  # 记录买入价格
                self.buy_date = exec_date  # 记录买入日期

                # ========== 新增：记录买入信息 + 下单时的指标值 ==========
                self.current_position_info['buy_price'] = price
                self.current_position_info['buy_date'] = exec_date
                self.current_position_info['buy_size'] = size
                # 从order_indicator_info中获取下单时记录的指标值
                self.current_position_info['buy_indicators'] = self.order_indicator_info.get('buy', {})

                if self.position.size == order.executed.size:
                    self.avg_buy_price = price
                else:
                    self.avg_buy_price = (
                            (self.avg_buy_price * (self.position.size - size) + price * size) / self.position.size
                    )

            elif order.issell():
                self.sell_signals.append((dt_str, price))

                # ========== 新增：记录卖出信息 + 下单时的指标值 ==========
                self.current_position_info['sell_price'] = price
                self.current_position_info['sell_date'] = exec_date
                self.current_position_info['sell_size'] = abs(size)  # 转为正数
                # 从order_indicator_info中获取下单时记录的指标值
                self.current_position_info['sell_indicators'] = self.order_indicator_info.get('sell', {})

                if self.position.size == 0:
                    self.avg_buy_price = 0.0
                    self.position_size = 0
        if order.status in [order.Submitted, order.Accepted]:
            return
        if not order.alive():
            self.order = None

    def notify_trade(self, trade):
        """新增：详细记录每笔交易的盈亏日志（包含下单时的指标值）"""
        if trade.isclosed:
            self.total_trades += 1

            # 核心盈亏数据
            trade_pnl = trade.pnl  # 交易盈亏（不含手续费）
            trade_pnlcomm = trade.pnlcomm  # 交易盈亏（含手续费）

            # ========== 修复：从记录的持仓信息中获取价格 ==========
            buy_price = self.current_position_info['buy_price']
            sell_price = self.current_position_info['sell_price']
            buy_date = self.current_position_info['buy_date']
            sell_date = self.current_position_info['sell_date']
            # 获取买卖下单时的指标值
            buy_indicators = self.current_position_info['buy_indicators']
            sell_indicators = self.current_position_info['sell_indicators']

            # 计算交易数量（使用买入时的数量）
            trade_size = self.current_position_info['buy_size']

            # 如果记录的信息不完整，使用 trade 对象的备用信息
            if buy_price == 0.0:
                buy_price = trade.price  # 备用：开仓价格
            if buy_date is None:
                buy_date = bt.num2date(trade.dtopen) if hasattr(trade, 'dtopen') else self.data.datetime.date(0)
            if sell_date is None:
                sell_date = bt.num2date(trade.dtclose) if hasattr(trade, 'dtclose') else self.data.datetime.date(0)

            # 计算持仓天数
            if hasattr(trade, 'barlen'):
                trade_duration = trade.barlen
            else:
                trade_duration = (sell_date - buy_date).days if buy_date and sell_date else 0

            # 验证盈亏计算：(卖出价 - 买入价) * 数量 ≈ trade_pnl
            calculated_pnl = (sell_price - buy_price) * trade_size if trade_size > 0 else 0

            # 判断盈亏类型
            if trade_pnlcomm > 0:
                self.win_trades += 1
                self.total_win_pnl += trade_pnlcomm
                trade_result = "盈利"
            else:
                self.lose_trades += 1
                self.total_lose_pnl += abs(trade_pnlcomm)
                trade_result = "亏损"

            # 生成交易日志（包含买卖下单时的指标值）
            self.trade_id += 1
            trade_log = {
                "交易编号": self.trade_id,
                "开仓日期": buy_date.strftime('%Y-%m-%d') if buy_date else "未知",
                "平仓日期": sell_date.strftime('%Y-%m-%d') if sell_date else "未知",
                "持仓天数": trade_duration,
                "买入价格": round(buy_price, 2),
                "卖出价格": round(sell_price, 2),
                "交易数量": abs(trade_size),
                "盈亏(不含手续费)": round(trade_pnl, 2),
                "盈亏(含手续费)": round(trade_pnlcomm, 2),
                "盈亏类型": trade_result,
                "手续费": round(abs(trade.pnl - trade.pnlcomm), 2),
                # ===== 新增：买入下单时的指标值 =====
                "买入时MA5": buy_indicators.get('MA5', np.nan),
                "买入时MA10": buy_indicators.get('MA10', np.nan),
                "买入时MA20": buy_indicators.get('MA20', np.nan),
                "买入时MA60": buy_indicators.get('MA60', np.nan),
                "买入时MA200": buy_indicators.get('MA200', np.nan),
                "买入时成交量(手)": buy_indicators.get('成交量(手)', np.nan),
                "买入时均量线5": buy_indicators.get('均量线5', np.nan),
                "买入时MACD值": buy_indicators.get('MACD值', np.nan),
                "买入时MACD信号线": buy_indicators.get('MACD信号线', np.nan),
                "买入时MACD柱状图": buy_indicators.get('MACD柱状图', np.nan),
                "买入时ATR": buy_indicators.get('ATR', np.nan),
                "买入时RSI6": buy_indicators.get('RSI6', np.nan),
                "买入时KDJ_K": buy_indicators.get('KDJ_K', np.nan),
                "买入时KDJ_D": buy_indicators.get('KDJ_D', np.nan),
                "买入时KDJ_J": buy_indicators.get('KDJ_J', np.nan),
                "买入时布林上轨": buy_indicators.get('布林上轨', np.nan),
                "买入时布林中轨": buy_indicators.get('布林中轨', np.nan),
                "买入时布林下轨": buy_indicators.get('布林下轨', np.nan),
                "买入时布林偏离度(%)": buy_indicators.get('布林偏离度(%)', np.nan),
                "买入时收盘价": buy_indicators.get('当前收盘价', np.nan),
                "买入时MA200趋势": buy_indicators.get('MA200趋势', "未知"),
                "买入时KDJ是否金叉": buy_indicators.get('KDJ是否金叉', False),
                "买入时MACD是否金叉": buy_indicators.get('MACD是否金叉', False),
                # ===== 新增：卖出下单时的指标值 =====
                "卖出时MA5": sell_indicators.get('MA5', np.nan),
                "卖出时MA10": sell_indicators.get('MA10', np.nan),
                "卖出时MA20": sell_indicators.get('MA20', np.nan),
                "卖出时MA60": sell_indicators.get('MA60', np.nan),
                "卖出时MA200": sell_indicators.get('MA200', np.nan),
                "卖出时成交量(手)": sell_indicators.get('成交量(手)', np.nan),
                "卖出时均量线5": sell_indicators.get('均量线5', np.nan),
                "卖出时MACD值": sell_indicators.get('MACD值', np.nan),
                "卖出时MACD信号线": sell_indicators.get('MACD信号线', np.nan),
                "卖出时MACD柱状图": sell_indicators.get('MACD柱状图', np.nan),
                "卖出时ATR": sell_indicators.get('ATR', np.nan),
                "卖出时RSI6": sell_indicators.get('RSI6', np.nan),
                "卖出时KDJ_K": sell_indicators.get('KDJ_K', np.nan),
                "卖出时KDJ_D": sell_indicators.get('KDJ_D', np.nan),
                "卖出时KDJ_J": sell_indicators.get('KDJ_J', np.nan),
                "卖出时布林上轨": sell_indicators.get('布林上轨', np.nan),
                "卖出时布林中轨": sell_indicators.get('布林中轨', np.nan),
                "卖出时布林下轨": sell_indicators.get('布林下轨', np.nan),
                "卖出时布林偏离度(%)": sell_indicators.get('布林偏离度(%)', np.nan),
                "卖出时收盘价": sell_indicators.get('当前收盘价', np.nan),
                "卖出时MA200趋势": sell_indicators.get('MA200趋势', "未知"),
                "卖出时KDJ是否死叉": sell_indicators.get('KDJ是否死叉', False),
                "卖出时MACD是否死叉": sell_indicators.get('MACD是否死叉', False)
            }

            # 添加到日志列表并打印
            self.trade_logs.append(trade_log)
            # print(f"\n===== 第{self.trade_id}笔交易完成 =====")
            # for key, value in trade_log.items():
            #     print(f"{key}: {value}")
            # print(f"【验证】计算盈亏: {round(calculated_pnl, 2)} vs 实际盈亏: {round(trade_pnl, 2)}")
            # print("=" * 30)

            # ========== 重置持仓信息 ==========
            self.current_position_info = {
                'buy_price': 0.0,
                'buy_date': None,
                'buy_size': 0,
                'sell_price': 0.0,
                'sell_date': None,
                'sell_size': 0,
                'buy_indicators': {},
                'sell_indicators': {}
            }
            self.order_indicator_info = {}

    def _get_plate_thresholds(self):
        """根据板块类型返回换手率/量比适配阈值，遵循手册板块适配细则"""
        stock_type = self.p.stock_type
        if stock_type == 'blue_chip':  # 大盘蓝筹 >5000亿
            return {
                'turnover_buy': (3.0, 10.0),  # 买入换手率区间
                'turnover_sell': 8.0,  # 卖出换手率阈值
                'turnover_extreme': 10.0,  # 换手率极端阈值
                'vol_ratio_buy': (1.2, 2.0),  # 买入量比区间
                'vol_ratio_sell': 2.0,  # 卖出量比阈值
                'vol_ratio_extreme': 3.0,  # 量比极端阈值
                'boll_dev_overbuy': 40,  # 布林偏离度超买阈值
                'bias6_overbuy': 5.0,  # BIAS6超涨阈值
                'bias6_oversell': -5.0,  # BIAS6超跌阈值
                'north_buy': 5000,  # 北向买入净流入阈值（万元）
                'north_sell': 10000,  # 北向卖出净流出阈值（万元）
            }
        elif stock_type == 'topic':  # 题材股/赛道股 <500亿
            return {
                'turnover_buy': (10.0, 25.0),
                'turnover_sell': 25.0,
                'turnover_extreme': 25.0,
                'vol_ratio_buy': (1.2, 4.0),
                'vol_ratio_sell': 2.0,
                'vol_ratio_extreme': 4.0,
                'boll_dev_overbuy': 70,
                'bias6_overbuy': 6.0,
                'bias6_oversell': -5.0,
                'north_buy': 5000,
                'north_sell': 10000,
            }
        elif stock_type == 'cycle':  # 周期股 煤炭/有色/化工
            return {
                'turnover_buy': (8.0, 18.0),
                'turnover_sell': 18.0,
                'turnover_extreme': 18.0,
                'vol_ratio_buy': (1.2, 2.0),
                'vol_ratio_sell': 2.0,
                'vol_ratio_extreme': 3.0,
                'boll_dev_overbuy': 60,
                'bias6_overbuy': 8.0,
                'bias6_oversell': -5.0,
                'north_buy': 5000,
                'north_sell': 10000,
            }
        else:  # 中小盘成长 500-5000亿（默认）
            return {
                'turnover_buy': (5.0, 15.0),
                'turnover_sell': 15.0,
                'turnover_extreme': 20.0,
                'vol_ratio_buy': (1.2, 2.0),
                'vol_ratio_sell': 2.0,
                'vol_ratio_extreme': 3.0,
                'boll_dev_overbuy': 60,
                'bias6_overbuy': 6.0,
                'bias6_oversell': -5.0,
                'north_buy': 5000,
                'north_sell': 10000,
            }

    def _judge_indicator_conflict(self, is_buy):
        """判断指标冲突，遵循手册信号冲突修正规则，返回±0.5分"""
        conflict_count = 0
        # 定义四大类指标的多空方向（核心：趋势/动量/量能/资金）
        # 趋势类：均线+布林
        trend_bull = (self.ma5[0] > self.ma10[0] > self.ma20[0] and self.data0.close[0] > self.boll_mid[0])
        trend_bear = not trend_bull
        # 动量类：KDJ+RSI+BIAS6
        momentum_bull = (
                    self.kdj_gold_cross[0] > 0 and self.rsi6[0] < 70 and self.bias6[0] < self._get_plate_thresholds()[
                'bias6_overbuy'])
        momentum_bear = not momentum_bull
        # 量能类：成交量+换手率+量比
        volume_bull = (self.data0.volume[0] > self.vol10[0] * 1.2 and self.volume_ratio[0] > 1.2 and
                       (self._get_plate_thresholds()['turnover_buy'][0] <= self.turnover[0] <=
                        self._get_plate_thresholds()['turnover_buy'][1]))
        volume_bear = not volume_bull
        # 资金类：北向资金（无数据则默认看多）
        capital_bull = self.north_capital >= 0 or self.north_capital == 0
        capital_bear = not capital_bull

        if is_buy:
            # 买入：看多类指标中，其他类反向则计冲突
            if trend_bull and not momentum_bull: conflict_count += 1
            if (trend_bull and momentum_bull) and not volume_bull: conflict_count += 1
            if (trend_bull and momentum_bull and volume_bull) and not capital_bull: conflict_count += 1
        else:
            # 卖出：看空类指标中，其他类反向则计冲突
            if trend_bear and not momentum_bear: conflict_count += 1
            if (trend_bear and momentum_bear) and not volume_bear: conflict_count += 1
            if (trend_bear and momentum_bear and volume_bear) and not capital_bear: conflict_count += 1

        # 2个及以上冲突扣0.5，无冲突加0.5，否则0
        return 0.5 if conflict_count == 0 else (-0.5 if conflict_count >= 2 else 0)

    def next(self):
        # 200日均线趋势判断
        self.ma200_slope, self.ma200_trend = get_ma_trend_global(self.ma200, 10, 0.0005)
        price_above_ma200 = (self.data.close[0] - self.ma200[0]) / self.ma200[0] >= 0.001
        macd_above_zero = self.macd.macd[0] > 0 and self.macd.signal[0] > 0
        macd_below_zero = self.macd.macd[0] < 0 and self.macd.signal[0] < 0
        macd_gold_cross = self.mcross[0] > 0
        macd_dead_cross = self.mcross[0] < 0
        macd_double_up = self.macd.macd[0] > self.macd.macd[-1] and self.macd.signal[0] > self.macd.signal[-1]
        macd_diff_up = self.macd.macd[0] > self.macd.macd[-1]
        macd_diff_over_signal = self.macd.macd[0] > self.macd.signal[0]

        # 最近三天不是MACD死叉
        macd_not_dead = self.mcross[0] >= 0 and self.mcross[-1] >= 0 and self.mcross[-2] >= 0

        # KDJ金叉/死叉判断
        kdj_gold_cross = self.kdj_gold_cross[0] > 0
        kdj_dead_cross = self.kdj_gold_cross[0] < 0
        kdj_overbuy = self.k[0] > 80
        kdj_oversell = self.k[0] < 20
        # J线拐头向下
        kdj_j_turndown = self.j[-1] > self.j[0] and self.k[-1] > 80
        # 预判KDJ金叉，三线汇聚（风险较大）
        kdj_convergence = abs((self.j[-1] - self.k[-1])) > abs((self.j[-1] - self.k[0])) and abs(
            self.k[-1] - self.d[-1]) > abs(self.k[-1] - self.d[0]) and self.d[0] > self.k[0] > self.j[0]

        # ===================== 核心补全：买入评分 buy_score 12.5分制（严格遵循手册）=====================
        buy_score = 0.0
        plate_thresh = self._get_plate_thresholds()  # 板块适配阈值
        close = self.data0.close[0]

        # 避免除零错误
        def safe_div(a, b):
            return a / b if b != 0 else 0

        # 买入分数拆解 - 新增：记录每个分项的得分
        buy_details = {}

        # 1. 趋势类（4分）- 均线排列(2分) + 布林带(2分)
        ## 均线排列打分（手册标准）
        def get_ma_buy_score(self):
            if self.p.stock_type == 'blue_chip':
                ma_full_bull = self.ma5[0] > self.ma10[0] > self.ma20[0] > self.ma60[0] and close > self.ma200[0]
            elif self.p.stock_type == 'cycle':
                ma_full_bull = self.ma5[0] > self.ma10[0] and close > self.ma60[0]
            else:
                ma_full_bull = self.ma5[0] > self.ma10[0] > self.ma20[0]
            ma_short_bull = self.ma5[0] > self.ma10[0] > self.ma20[0]
            if ma_full_bull:
                return 2.0  # 完全多头+长期趋势向上
            elif ma_short_bull:
                return 1.0  # 短期多头
            else:
                return 0.0  # 均线混乱/空头

        ma_buy = get_ma_buy_score(self)
        buy_details['趋势类-均线排列'] = ma_buy
        buy_score += ma_buy

        def get_boll_buy_score(self):
            ## 布林带打分（手册标准+板块偏离度）
            boll_mid_break = close > self.boll_mid[0] and self.data0.close[-1] < self.boll_mid[-1]  # 从下轨突破中轨
            boll_dev = safe_div(close - self.boll_mid[0], self.boll_upper[0] - self.boll_mid[0]) * 100
            boll_收口 = (self.boll_upper[0] - self.boll_lower[0]) < (self.boll_upper[-1] - self.boll_lower[-1])
            if boll_mid_break and boll_dev < 30:
                return 2.0  # 从下轨突破中轨+偏离度<30%
            elif self.boll_mid[0] < close < self.boll_upper[0] and 30 <= boll_dev <= plate_thresh['boll_dev_overbuy']:
                return 1.0  # 中轨-上轨间+未超买
            else:
                return 0.0  # 上轨附近+布林收口

        boll_buy = get_boll_buy_score(self)
        buy_details['趋势类-布林带'] = boll_buy
        buy_score += boll_buy

        # 2. 动量类（4.5分）- KDJ(2分) + RSI(2分) + 乖离率BIAS6(0.5分)
        ## KDJ打分（手册标准：超卖区金叉优先）
        def get_kdj_buy_score(self):
            kdj_20_gold = self.j[-1] < 20 and kdj_gold_cross
            kdj_50_gold = 20 <= self.j[-1] <= 50 and kdj_gold_cross
            if kdj_20_gold:
                return 2.0  # 20超卖区金叉
            elif kdj_50_gold:
                return 1.0  # 50以下金叉
            else:
                return 0  # 50以上金叉/超买

        kdj_buy = get_kdj_buy_score(self)
        buy_details['动量类-KDJ'] = kdj_buy
        buy_score += kdj_buy

        # def get_rsi_buy_score(self):
        #     ## RSI6打分（手册标准：超卖区突破50）
        #     rsi_break50 = self.rsi6[0] > 50 and self.rsi6[-1] < 30
        #     if rsi_break50:
        #         return 2.0  # 从30超卖区突破50
        #     elif 30 <= self.rsi6[0] <= 50:
        #         return 1.0  # 震荡区30-50
        #     else:
        #         return 0.0  # 50以上/接近70超买

        def get_rsi_buy_score(self):
            """
            实战版RSI买入评分（突破50即得分，不卡70上限）
            核心规则（贴合实战趋势判断）：
            2分：超卖区强突破 → 前1日RSI6<30 + 当日RSI6>50（无70上限）
            1分：非超卖区突破 → 前1日30≤RSI6≤50 + 当日RSI6>50（无70上限）
                 或 震荡蓄力 → 30≤当日RSI6≤50，连续2日波动≤1且未突破区间
                 或 周线强势 → 周线RSI6在50-70区间（日线不适用）
            0分：纯超买无突破 → RSI6≥70且无突破动作，或50-69但无突破/震荡动作
            """
            # 核心数据（self.rsi6：[0]当日，[-1]前1日，[-2]前2日）
            current_rsi6 = self.rsi6[0]  # 当日RSI6
            prev1_rsi6 = self.rsi6[-1]  # 前1日RSI6
            prev2_rsi6 = self.rsi6[-2] if len(self.rsi6) >= 3 else None  # 前2日RSI6（震荡判定用）

            # ===== 2分：超卖区强突破（优先级最高）=====
            # 前1日<30（超卖）+ 当日>50（突破）→ 哪怕到80/90都算2分（强反弹）
            if prev1_rsi6 < 30 and current_rsi6 > 50:
                return 2.0

            # ===== 1分：实战核心（突破即得分，不卡70）=====
            # 场景1：非超卖区突破 → 前1日30-50 + 当日>50（哪怕70+）
            non_oversold_break = (30 <= prev1_rsi6 <= 50) and (current_rsi6 > 50)

            # 场景2：震荡蓄力 → 30-50区间+连续2日低波动（原规则保留）
            oscillation_condition = False
            if 30 <= current_rsi6 <= 50 and prev2_rsi6 is not None:
                no_breakout = (30 <= prev1_rsi6 <= 50) and (30 <= current_rsi6 <= 50)
                price_fluct = abs(current_rsi6 - prev1_rsi6) <= 1.0
                oscillation_condition = no_breakout and price_fluct

            # 场景3：周线特殊 → 周线50-70（原规则保留）
            weekly_special = False
            if hasattr(self, 'period') and self.period == 'weekly':
                weekly_special = 50 <= current_rsi6 <= 70

            # 满足1分任一场景 → 不管突破到60/70/80，都给1分
            if non_oversold_break or oscillation_condition or weekly_special:
                return 1.0

            # ===== 0分：纯超买/无突破（风控底线）=====
            # 仅当：1) RSI≥70且无突破动作 或 2) 50-69但无任何突破/震荡动作
            if current_rsi6 >= 70 or (50 <= current_rsi6 <= 69):
                return 0.0

            # 兜底：未匹配任何场景 → 0分（防御性编程）
            return 0.0

        rsi_buy = get_rsi_buy_score(self)
        buy_details['动量类-RSI6'] = rsi_buy
        buy_score += rsi_buy

        ## 乖离率BIAS6打分（手册标准：超跌得0.5分）
        def get_bias_buy_score(self):
            if self.bias6[0] < plate_thresh['bias6_oversell']:
                return 0.5  # 超跌，偏离均线过多
            else:
                return 0.0

        bias_buy = get_bias_buy_score(self)
        buy_details['动量类-乖离率BIAS6'] = bias_buy
        buy_score += bias_buy

        # 3. 量能/资金类（3分）- 成交量(1分) + 换手率(1分) + 量比(0.5分) + 北向资金(0.5分)
        ## 成交量打分（手册标准：放量上涨+量价同步）
        def get_vol_buy_score(self):
            up = close > self.data0.close[-1]
            vol_ratio = safe_div(self.vol[0], self.vol10[0])
            if up and vol_ratio > 1.5:
                return 1.0  # 放量上涨+量价同步
            else:
                return 0.0

        vol_buy = get_vol_buy_score(self)
        buy_details['量能类-成交量'] = vol_buy
        buy_score += vol_buy

        ## 换手率打分（手册标准：按板块适配区间）
        def get_turnover_buy_score(self):
            to_min, to_max = plate_thresh['turnover_buy']
            if to_min <= self.turnover[0] <= to_max:
                return 1.0  # 板块适配活跃区间
            else:
                return 0.0  # 过低低迷/过高拥挤

        turnover_buy = get_turnover_buy_score(self)
        buy_details['量能类-换手率'] = turnover_buy
        buy_score += turnover_buy

        ## 量比打分（手册标准：温和放量）
        def get_vr_buy_score(self):
            vr_min, vr_max = plate_thresh['vol_ratio_buy']
            if vr_min <= self.volume_ratio[0] <= vr_max:
                return 0.5  # 温和放量
            else:
                return 0.0  # 缩量/过度放量

        vr_buy = get_vr_buy_score(self)
        buy_details['量能类-量比'] = vr_buy
        buy_score += vr_buy

        ## 北向资金打分（手册标准：当日净流入>5000万）
        def get_north_capital_buy_score(self):
            if self.north_capital > plate_thresh['north_buy']:
                return 0.5  # 外资加仓
            else:
                return 0.0  # 外资减仓/无数据

        north_buy = get_north_capital_buy_score(self)
        buy_details['量能类-北向资金'] = north_buy
        buy_score += north_buy

        # 4. 筹码类（1分）- 筹码峰（手册标准）
        def get_profit_ratio_buy_score(self):
            if self.profit_ratio > 80:  # 价格在筹码峰上方，获利盘>80%
                return 1.0
            else:
                return 0.0  # 套牢盘>60%，筹码分散

        profit_buy = get_profit_ratio_buy_score(self)
        buy_details['筹码类-获利盘比例'] = profit_buy
        buy_score += profit_buy

        # 5. 信号冲突修正（±0.5分）- 手册标准
        conflict_buy = self._judge_indicator_conflict(is_buy=True)
        buy_details['信号冲突修正'] = conflict_buy
        buy_score += conflict_buy

        # 买入分数最终四舍五入保留2位
        buy_score = round(buy_score, 2)

        # ===================== 核心补全：卖出评分 sell_score 12.5分制（严格遵循手册）=====================
        sell_score = 0.0
        sell_details = {}

        ## 辅助判断：连续2日收盘价跌破（手册有效跌破标准）
        def two_day_break(price, line):
            return price[0] < line[0] and price[-1] < line[-1]

        # 1. 趋势类（4分）- 均线排列(2分) + 布林带(2分)
        ## 均线排列打分（手册标准：有效跌破+死叉）
        def get_ma_sell_score(self):
            ma10_break = two_day_break(self.data0.close, self.ma10) and self.ma5[0] < self.ma10[0]
            ma5_break = is_price_break_ma(self.data0.close, self.ma5, direction='down')
            if ma10_break:
                return 2.0  # 有效跌破MA10+MA5死叉MA10
            elif ma5_break:
                return 1.0  # 单日跌破MA5未破MA10
            else:
                return 0.0

        ma_sell = get_ma_sell_score(self)
        sell_details['趋势类-均线排列'] = ma_sell
        sell_score += ma_sell

        ## 布林带打分（手册标准：有效跌破中轨/超买收口）
        def get_boll_sell_score(self):
            boll_mid_break_sell = two_day_break(self.data0.close, self.boll_mid)
            boll_dev = safe_div(close - self.boll_mid[0], self.boll_upper[0] - self.boll_mid[0]) * 100
            boll_收口 = (self.boll_upper[0] - self.boll_lower[0]) < (self.boll_upper[-1] - self.boll_lower[-1])
            boll_overbuy_close = close > self.boll_upper[0] and boll_dev > plate_thresh['boll_dev_overbuy'] and boll_收口
            if boll_mid_break_sell:
                return 2.0  # 从上轨有效跌破中轨
            elif boll_overbuy_close:
                return 1.0  # 上轨超买+布林收口
            else:
                return 0.0

        boll_sell = get_boll_sell_score(self)
        sell_details['趋势类-布林带'] = boll_sell
        sell_score += boll_sell

        # 2. 动量类（4.5分）- KDJ(2分) + RSI(2分) + 乖离率BIAS6(0.5分)
        ## KDJ打分（手册标准：超买区死叉优先）
        def get_kdj_sell_score(self):
            kdj_80_dead = self.j[-1] > 80 and kdj_dead_cross
            kdj_50_dead = 50 <= self.j[-1] <= 80 and kdj_dead_cross
            if kdj_80_dead:
                return 2.0  # 80超买区死叉
            elif kdj_50_dead:
                return 1.0  # 50以上死叉
            else:
                return 0.0  # 50以下死叉/超卖

        kdj_sell = get_kdj_sell_score(self)
        sell_details['动量类-KDJ'] = kdj_sell
        sell_score += kdj_sell

        ## RSI6打分（手册标准：超买区跌破50）
        def get_rsi_sell_score(self):
            rsi_break50_sell = self.rsi6[0] < 50 and self.rsi6[-1] > 70
            if rsi_break50_sell:
                return 2.0  # 从70超买区跌破50
            elif 50 <= self.rsi6[0] <= 70:
                return 1.0  # 震荡区50-70
            else:
                return 0.0  # 50以下/接近30超卖

        rsi_sell = get_rsi_sell_score(self)
        sell_details['动量类-RSI6'] = rsi_sell
        sell_score += rsi_sell

        ## 乖离率BIAS6打分（手册标准：超涨得0.5分）
        def get_bias_sell_score(self):
            if self.bias6[0] > plate_thresh['bias6_overbuy']:
                return 0.5  # 超涨，偏离均线过多
            else:
                return 0.0

        bias_sell = get_bias_sell_score(self)
        sell_details['动量类-乖离率BIAS6'] = bias_sell
        sell_score += bias_sell

        # 3. 量能/资金类（3分）- 成交量(1分) + 换手率(1分) + 量比(0.5分) + 北向资金(0.5分)
        ## 成交量打分（手册标准：放量下跌+量价背离）
        def get_vol_sell_score(self):
            down = close < self.data0.close[-1]
            vol_ratio = safe_div(self.vol[0], self.vol10[0])
            if down and vol_ratio > 1.5:
                return 1.0  # 放量下跌+量价背离
            else:
                return 0.0

        vol_sell = get_vol_sell_score(self)
        sell_details['量能类-成交量'] = vol_sell
        sell_score += vol_sell

        ## 换手率打分（手册标准：板块适配+前5日平均低于当前50%）
        def get_turnover_sell_score(self):
            to_sell_thresh = plate_thresh['turnover_sell']
            to_5ma_ratio = safe_div(self.turnover_5ma[0], self.turnover[0])
            if self.turnover[0] > to_sell_thresh and to_5ma_ratio < 0.5:
                return 1.0  # 板块放量出逃+换手率骤增
            else:
                return 0.0

        turnover_sell = get_turnover_sell_score(self)
        sell_details['量能类-换手率'] = turnover_sell
        sell_score += turnover_sell

        ## 量比打分（手册标准：放量出逃）
        def get_vr_sell_score(self):
            if self.volume_ratio[0] > plate_thresh['vol_ratio_sell']:
                return 0.5  # 放量出逃
            else:
                return 0.0

        vr_sell = get_vr_sell_score(self)
        sell_details['量能类-量比'] = vr_sell
        sell_score += vr_sell

        ## 北向资金打分（手册标准：当日净流出>1亿）
        def get_north_capital_sell_score(self):
            if self.north_capital < -plate_thresh['north_sell']:
                return 0.5  # 外资出逃
            else:
                return 0.0  # 外资加仓/无数据

        north_sell = get_north_capital_sell_score(self)
        sell_details['量能类-北向资金'] = north_sell
        sell_score += north_sell

        # 4. 筹码类（1分）- 筹码峰（手册标准）
        def get_profit_ratio_sell_score(self):
            if self.profit_ratio < 30:  # 价格跌破筹码峰，获利盘<30%
                return 1.0
            else:
                return 0.0  # 仍在筹码峰上方，套牢盘<20%

        profit_sell = get_profit_ratio_sell_score(self)
        sell_details['筹码类-获利盘比例'] = profit_sell
        sell_score += profit_sell
        # 5. 信号冲突修正（±0.5分）- 手册标准
        conflict_sell = self._judge_indicator_conflict(is_buy=False)
        sell_details['信号冲突修正'] = conflict_sell
        sell_score += conflict_sell

        # 卖出分数最终四舍五入保留2位
        sell_score = round(sell_score, 2)

        # ===================== 关键：记录当日分数到日志 - 用于命令行打印 =====================
        current_date = self.data.datetime.date(0).strftime('%Y-%m-%d')
        self.daily_score_log[current_date] = {
            'buy_score': buy_score,
            'buy_details': buy_details,
            'sell_score': sell_score,
            'sell_details': sell_details
        }

        if self.order:
            return

        if not self.position:  # 无仓位，判断买入（基于手册分数区间）
            # 手册分数区间：5-6.9分2成试仓，7+分逐步加仓，这里取≥5分作为买入阈值
            # buy_condition = buy_score >= 4
            # buy_condition = kdj_gold_cross
            buy_condition = self.rsi6[0] > self.rsi12[0] and  self.rsi6[-1] < self.rsi12[-1] and (self.rsi12[0] > self.rsi24[0] and  self.rsi12[-1] < self.rsi24[-1]) and self.rsi6[0] > 50 or kdj_gold_cross
            if buy_condition:
                # ========== 核心新增：买入时获取并记录所有指标值 ==========
                # print(f"current_date buy {current_date}")
                buy_indicators = self.get_current_indicators()
                self.order_indicator_info['buy'] = buy_indicators
                self.order = self.buy(exectype=bt.Order.Market)
                pdist = self.atr[0] * self.p.atrdist
                self.pstop = self.data0.close[0] - pdist

        else:  # 有仓位，判断卖出（基于手册分数区间）
            pclose = self.data0.close[0]
            pstop = self.pstop
            # 手册分数区间：5+分逐步减仓，≥7分清仓，这里取≥5分作为卖出阈值
            # sell_condition = sell_score >= 5.5
            sell_condition = self.rsi12[0] < self.rsi24[0] and self.rsi12[-1] > self.rsi24[-1]
            if sell_condition:
                # ========== 核心新增：卖出时获取并记录所有指标值 ==========
                sell_indicators = self.get_current_indicators()
                self.order_indicator_info['sell'] = sell_indicators
                self.order = self.close(exectype=bt.Order.Market)
            else:
                # 移动止损
                pdist = self.atr[0] * self.p.atrdist
                self.pstop = max(pstop, pclose - pdist)

    def stop(self):
        # ========== 原有逻辑：保存回测结果 + 交易日志 完全未动 ==========
        init_value = self.init_cash
        final_value = self.broker.getvalue()
        total_pnl = final_value - init_value
        win_rate = 0.0
        profit_loss_ratio = 0.0
        avg_win = 0.0
        avg_lose = 0.0
        if self.total_trades > 0:
            win_rate = (self.win_trades / self.total_trades) * 100
            avg_win = self.total_win_pnl / self.win_trades if self.win_trades else 0.0
            avg_lose = self.total_lose_pnl / self.lose_trades if self.lose_trades else 0.0
            profit_loss_ratio = avg_win / avg_lose if avg_lose > 0 else 0.0

        # ========== 【修复】移除所有对分析器结果的引用 ==========
        # 不再尝试计算或打印 total_strategy_return, annual_return 等
        # 这些工作已由 print_performance_report 函数完成
        # =======================================================

        print(
            "MACD({0},{1},{2}) | KDJ({3},{4},{5}) | 胜率:{6:>5.2f}% | 盈亏比:{7:>4.2f} | 总交易:{8:>3d} | 总盈亏:{9:>8.2f} | 期末资金:{10:>10.2f}".format(
                self.p.macd1, self.p.macd2, self.p.macdsig,
                self.p.kdj_period, self.p.kdj_k_period, self.p.kdj_d_period,
                win_rate, profit_loss_ratio, self.total_trades, total_pnl, final_value))

        # ========== 原有逻辑：保存结果到文件 ==========
        with open(RESULT_FILE, 'a', encoding='utf-8', newline='') as f:
            f.write("{0},{1},{2},{3},{4},{5},{6:.2f},{7:.2f},{8},{9},{10},{11:.2f},{12:.2f},{13:.2f},{14:.2f}\n".format(
                self.p.macd1, self.p.macd2, self.p.macdsig,
                self.p.kdj_period, self.p.kdj_k_period, self.p.kdj_d_period,
                win_rate, profit_loss_ratio, self.win_trades, self.lose_trades, self.total_trades,
                avg_win, avg_lose, total_pnl, final_value))

        # 保存交易日志到CSV（包含指标值）
        if self.trade_logs:
            df_trade_log = pd.DataFrame(self.trade_logs)
            df_trade_log = df_trade_log.fillna("")
            df_trade_log.to_csv(TRADE_LOG_FILE, index=False, encoding='utf-8-sig')
            print(f"\n✅ 每笔交易盈亏日志（含指标）已保存到: {TRADE_LOG_FILE}")
        else:
            print("\n⚠️ 本次回测无交易记录")




DATASETS = {
    'yhoo': '../../datas/yhoo-1996-2014.txt',
    'orcl': '../../datas/orcl-1995-2014.txt',
    'nvda': '../../datas/nvda-1999-2014.txt',
}


# ===================== 新增：打印指定日期分数核心函数 =====================
def print_target_date_score(strat, target_date):
    """
    打印指定日期的buy_score、sell_score及各分项具体分数
    :param strat: 策略实例
    :param target_date: 目标日期字符串（YYYY-MM-DD）
    """
    if target_date not in strat.daily_score_log:
        print(f"❌ 未找到{target_date}的分数数据，请检查日期是否在回测范围内！")
        print(f"📌 回测日期范围：{list(strat.daily_score_log.keys())[0]} 至 {list(strat.daily_score_log.keys())[-1]}")
        return

    # 获取当日分数数据
    score_data = strat.daily_score_log[target_date]
    buy_score = score_data['buy_score']
    buy_details = score_data['buy_details']
    sell_score = score_data['sell_score']
    sell_details = score_data['sell_details']

    # 格式化打印
    print("=" * 80)
    print(f"📅 {target_date} 买卖评分详情（12.5分制）")
    print("=" * 80)
    print(f"✅ 买入总得分（buy_score）：{buy_score} 分")
    print("🔍 买入分项得分：")
    for item, score in buy_details.items():
        print(f"  - {item.ljust(15)}：{score:.2f} 分")
    print("-" * 80)
    print(f"❌ 卖出总得分（sell_score）：{sell_score} 分")
    print("🔍 卖出分项得分：")
    for item, score in sell_details.items():
        print(f"  - {item.ljust(15)}：{score:.2f} 分")
    print("=" * 80)


def add_performance_analyzers(cerebro, data0):
    """
    为 Cerebro 引擎添加一套全面的策略绩效分析器。

    Args:
        cerebro (bt.Cerebro): Backtrader 的 Cerebro 引擎实例。
        data0 (bt.DataBase): 用作市场基准的数据源。
    """
    # 定义要添加的分析器列表：(分析器类, 别名, 额外参数字典)
    analyzers_config = [
        (bt.analyzers.SharpeRatio, 'sharpe', {'riskfreerate': 0.01, 'annualize': True}),
        (bt.analyzers.AnnualReturn, 'annual_return', {}),
        (bt.analyzers.DrawDown, 'drawdown', {}),
        (bt.analyzers.TradeAnalyzer, 'trade_analyzer', {}),
        (bt.analyzers.SQN, 'sqn', {}),
        (bt.analyzers.Calmar, 'calmar', {}),
        (bt.analyzers.TimeReturn, 'timereturn', {}),
        # 使用主数据作为基准
        (bt.analyzers.TimeReturn, 'benchmark', {'data': data0}),
    ]

    for analyzer_class, name, kwargs in analyzers_config:
        cerebro.addanalyzer(analyzer_class, _name=name, **kwargs)


def print_performance_report(strat):
    """
    从回测完成的策略实例中提取分析器结果，并打印详细的绩效报告。

    Args:
        strat (bt.Strategy): 回测完成后的策略实例。
    """
    import pandas as pd

    print("\n" + "=" * 60)
    print("📊 策略绩效分析报告")
    print("=" * 60)

    # --- 提取所有分析结果 ---
    # 夏普比率
    sharpe_analysis = strat.analyzers.sharpe.get_analysis()
    final_sharpe = sharpe_analysis.get('sharperatio', None)

    # 年度收益率 (策略 & 基准)
    strategy_annual_ret = strat.analyzers.annual_return.get_analysis()
    benchmark_timeret = strat.analyzers.benchmark.get_analysis()
    benchmark_annual_ret = {}
    if benchmark_timeret:
        bench_series = pd.Series(benchmark_timeret)
        bench_series.index = pd.to_datetime(bench_series.index)
        benchmark_annual_ret = bench_series.groupby(bench_series.index.year).apply(lambda x: (x + 1).prod() - 1)

    # 最大回撤
    dd_analysis = strat.analyzers.drawdown.get_analysis()
    max_dd = dd_analysis.max.drawdown
    max_dd_duration = dd_analysis.max.len

    # 交易统计
    trade_analysis = strat.analyzers.trade_analyzer.get_analysis()
    if trade_analysis.total.total > 0:
        total_trades = trade_analysis.total.total
        won_trades = trade_analysis.won.total
        win_rate = won_trades / total_trades * 100
        pnl_net = trade_analysis.pnl.net.total
    else:
        total_trades = won_trades = win_rate = pnl_net = 0

    # SQN
    sqn_analysis = strat.analyzers.sqn.get_analysis()
    sqn_value = sqn_analysis.get('sqn', None)

    # 卡玛比率
    calmar_analysis = strat.analyzers.calmar.get_analysis()
    calmar_ratio = calmar_analysis.get('calmar', None)

    # 总收益率 & 年化收益率
    timeret_analysis = strat.analyzers.timereturn.get_analysis()
    if timeret_analysis:
        returns_series = pd.Series(timeret_analysis)
        total_strategy_return = (returns_series + 1).prod() - 1
        start_date = min(returns_series.index)
        end_date = max(returns_series.index)
        years = (end_date - start_date).days / 365.25
        annual_return = ((total_strategy_return + 1) ** (1 / years) - 1) * 100 if years > 0 else 0.0
    else:
        total_strategy_return = annual_return = 0.0

    benchmark_total_return = 0.0
    if benchmark_timeret:
        bench_series = pd.Series(benchmark_timeret)
        benchmark_total_return = (bench_series + 1).prod() - 1

    # --- 打印结果 ---
    print(f"📈 策略总收益率: {total_strategy_return * 100:.2f}%")
    print(f"📈 基准总收益率: {benchmark_total_return * 100:.2f}%")
    print(f"📅 年化收益率: {annual_return:.2f}%")
    print(f"⚖️  夏普比率: {final_sharpe:.2f}" if final_sharpe is not None else "⚖️  夏普比率: N/A")
    print(f"📉 最大回撤: {max_dd:.2f}%")
    print(f"⏳ 最长回撤期: {max_dd_duration} 天")
    print(f"🎯 系统质量数 (SQN): {sqn_value:.2f}" if sqn_value is not None else "🎯 SQN: N/A")
    print(f"📈 卡玛比率: {calmar_ratio:.2f}" if calmar_ratio is not None else "📈 卡玛比率: N/A")

    # --- 【核心新增】年度收益对比表 ---
    print("-" * 60)
    print("📅 年度收益率对比:")
    print("-" * 60)
    all_years = sorted(set(strategy_annual_ret.keys()) | set(benchmark_annual_ret.keys()))
    for year in all_years:
        strat_ret = strategy_annual_ret.get(year, 0.0)
        bench_ret = benchmark_annual_ret.get(year, 0.0)
        print(f" {year}: 策略 {strat_ret:7.2%} | 基准 {bench_ret:7.2%}")

    print("-" * 60)
    print(f"✅ 总交易次数: {total_trades}")
    print(f"✅ 胜率: {win_rate:.2f}%")
    print(f"💰 净盈亏: {pnl_net:.2f}")
    print("=" * 60)


# ===================== 主逻辑入口 =====================
def runstrat(args=None):
    args = parse_args(args)
    # ✅ 核心配置：开启cheat_on_close，让当天收盘下单当日成交
    cerebro = bt.Cerebro()
    cerebro.broker.set_coc(True)
    cerebro.broker.set_coo(False)

    m1, m2, ms = 12, 26, 9

    # 重置结果文件
    with open(RESULT_FILE, 'w', encoding='utf-8', newline='') as f:
        f.write(
            "macd1,macd2,macdsig,kdj_period,kdj_k_period,kdj_d_period,win_rate,profit_loss_ratio,win_trades,lose_trades,total_trades,avg_win,avg_lose,total_pnl,final_value\n")

    # 添加策略，可修改stock_type参数适配板块：blue_chip/topic/cycle/mid_small
    cerebro.addstrategy(MACDOptStrategy,
                        macd1=m1, macd2=m2, macdsig=ms,
                        atrperiod=args.atrperiod,
                        atrdist=args.atrdist,
                        total_loss_limit=args.total_loss_limit,
                        kdj_period=9, kdj_k_period=3, kdj_d_period=3,
                        stock_type='mid_small')  # 切换板块类型在这里修改

    # 加载数据
    # dataname = os.path.join('./600519_all_data.csv')
    dataname = DATASETS.get(args.dataset, args.data)
    raw_df = pd.read_csv(dataname)
    col_count = len(raw_df.columns)
    print(f"检测到 CSV 包含 {col_count} 列")
    if col_count == 12:
        # csv格式：date,股票代码,open,close,high,low,volume,amount,amplitude,pct_change,price_change,turnover
        raw_df.columns = ['date', '股票代码', 'open', 'close', 'high', 'low', 'volume', 'amount', 'amplitude',
                          'pct_change', 'price_change', 'turnover']
    elif col_count == 7:
        raw_df.columns = ['date', 'open', 'high', 'low', 'close', 'volume', 'openinterest']
    elif col_count == 6:
        raw_df.columns = ['date', 'open', 'high', 'low', 'close', 'volume']
    else:
        raw_df = raw_df.iloc[:, :6]
        raw_df.columns = ['date', 'open', 'high', 'low', 'close', 'volume']

    raw_df['date'] = pd.to_datetime(raw_df['date'])
    raw_df.set_index('date', inplace=True)

    dkwargs = dict()
    if args.fromdate:
        dkwargs['fromdate'] = datetime.datetime.strptime(args.fromdate, '%Y-%m-%d')
    if args.todate:
        dkwargs['todate'] = datetime.datetime.strptime(args.todate, '%Y-%m-%d')

    # ========== 核心修复3：添加tmformat兼容日期解析，避免隐性异常 ==========
    data0 = GenericCSVWithTurnover(
        dataname=dataname, reverse=False, dtformat='%Y-%m-%d',
        tmformat='%H:%M:%S',
        datetime=0, open=2, close=3, high=4, low=5,  volume=6, openinterest=-1, **dkwargs
    )
    cerebro.adddata(data0)
    cerebro.resampledata(data0, timeframe=bt.TimeFrame.Weeks, compression=1, adjbartime=True)

    # 配置回测参数
    cerebro.broker.setcash(args.cash)
    cerebro.addsizer(FixedPerc, perc=args.cashalloc)
    cerebro.broker.setcommission(commission=args.commperc)
    # cerebro.broker.set_slippage_perc(0.00005)       # 滑点
    # ========== 【添加全面的策略绩效分析器】==========
    add_performance_analyzers(cerebro, data0)
    # ===================================================
    # 运行回测
    print('初始资金: %.2f' % cerebro.broker.getvalue())
    results = cerebro.run()
    strat = results[0]
    # ========== 【关键调用点 2】打印分析器报告 ==========
    print_performance_report(strat)
    # ============================================
    print(f"✅ 本次回测生成 买入信号: {len(strat.buy_signals)} 个 | 卖出信号: {len(strat.sell_signals)} 个")
    print(f'期末资金: {cerebro.broker.getvalue():.2f}')


    if args.target_date:
        print_target_date_score(strat, args.target_date)

    # ===================== 新增：判断是否指定目标日期，是则打印分数，否则执行原有绘图 =====================
    # --- 核心：直接从 args.data 指定的文件路径读取 ---
    csv_file_path = args.data  # 这就是你传入的 --data 参数
    print(f"📊 正在从文件读取数据用于绘图: {csv_file_path}")

    # 读取CSV，假设你的CSV包含 'date', 'open', 'high', 'low', 'close', 'volume' 列
    df_plot = pd.read_csv(csv_file_path, usecols=['date', 'open', 'high', 'low', 'close', 'volume'])

    # === 关键修正：确保 date 列是 datetime 类型，然后创建 date_str 和 time 索引 ===
    df_plot['date'] = pd.to_datetime(df_plot['date'])  # 先转为 datetime
    df_plot['date_str'] = df_plot['date'].dt.strftime('%Y-%m-%d')  # 再创建字符串列
    df_plot['time'] = df_plot['date'].apply(lambda x: int(x.timestamp()))  # 创建时间戳
    df_plot.set_index('time', inplace=True)
    df_plot.drop(columns=['date'], inplace=True)  # 删除原始的 datetime 列
    # ========================================================================

    # --- 准备交易信号（这部分还是需要从 strat 获取）---
    all_signals = []
    for date_str, _ in strat.buy_signals:
        all_signals.append({'date': date_str, 'type': 'buy'})
    for date_str, _ in strat.sell_signals:
        all_signals.append({'date': date_str, 'type': 'sell'})

    # --- 调用绘图 ---
    btfinplot.plot_a_stock_analysis(
        df=df_plot,
        symbol='002738',
        title_suffix='Backtrader 回测结果',
        signals=all_signals,
    )
    # ==================== 【新增：调试信息】 ====================
    # print("\n🔍 调试信息:")
    # print(f"df_plot 的 date_str 列前5行:\n{df_plot['date_str'].head()}")
    # print(f"\n买入信号日期: {[sig[0] for sig in strat.buy_signals]}")
    # print(f"卖出信号日期: {[sig[0] for sig in strat.sell_signals]}")

    # print("✅ 图表已成功生成！")
    #######################################

    # 分析结果
    opt_results = []
    with open(RESULT_FILE, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            opt_results.append({
                'macd1': int(row['macd1']), 'macd2': int(row['macd2']), 'macdsig': int(row['macdsig']),
                'kdj_period': int(row['kdj_period']), 'kdj_k_period': int(row['kdj_k_period']),
                'kdj_d_period': int(row['kdj_d_period']),
                'win_rate': float(row['win_rate']), 'profit_loss_ratio': float(row['profit_loss_ratio']),
                'total_trades': int(row['total_trades']), 'total_pnl': float(row['total_pnl']),
                'final_value': float(row['final_value'])
            })

    print("\n" + "=" * 80)
    print("✅多指标12.5分制买卖点策略回测完成 ✅ 6轴布局展示 ✅ 白色背景 ✅ A股实战适配！")
    print("=" * 80)
    valid_results = [res for res in opt_results if res['total_trades'] > 0]
    if valid_results:
        valid_results_sorted = sorted(valid_results,
                                      key=lambda x: (-x['win_rate'], -x['profit_loss_ratio'], -x['total_pnl']))
        best_param = valid_results_sorted[0]
        print(
            f"\n【运行参数】 MACD({best_param['macd1']},{best_param['macd2']},{best_param['macdsig']}) | KDJ({best_param['kdj_period']},{best_param['kdj_k_period']},{best_param['kdj_d_period']})")
        print(
            f"【核心指标】 胜率:{best_param['win_rate']:.2f}% | 盈亏比:{best_param['profit_loss_ratio']:.2f} | 总盈亏:{best_param['total_pnl']:.2f}元")
        print(f"【最终资金】 {best_param['final_value']:.2f} 元")
    else:
        print("\n【提示】本次回测无交易信号！")
    print(f"\n【结果文件】已保存至: {RESULT_FILE}")
    print(f"【交易日志】已保存至: {TRADE_LOG_FILE}")


def parse_args(pargs=None):
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description='多指标12.5分制买卖点策略 - 整合MACD+KDJ+RSI+BOLL+换手率+量比+乖离率')

    group1 = parser.add_mutually_exclusive_group(required=True)
    group1.add_argument('--data', required=False, default=None,
                        help='Specific data to be read in')

    group1.add_argument('--dataset', required=False, action='store',
                        default=None, choices=DATASETS.keys(),
                        help='Choose one of the predefined data sets')
    parser.add_argument('--fromdate', required=False, default='2020-01-01', help='回测开始日期 YYYY-MM-DD')
    parser.add_argument('--todate', required=False, default='2023-12-31', help='回测结束日期 YYYY-MM-DD')
    parser.add_argument('--cash', required=False, type=float, default=1000000.0, help='初始资金')
    parser.add_argument('--cashalloc', required=False, type=float, default=0.20, help='单次开仓占比(0-1)')
    parser.add_argument('--commperc', required=False, type=float, default=0.0001, help='佣金比例 0.0001=万1')
    parser.add_argument('--atrperiod', required=False, type=int, default=14, help='ATR周期')
    parser.add_argument('--atrdist', required=False, type=float, default=1.5, help='ATR止损倍数')
    parser.add_argument('--total_loss_limit', required=False, type=float, default=0.02,
                        help='总资金亏损止损比例 0.02=2%%')
    # ===================== 新增：命令行指定日期参数 =====================
    parser.add_argument('--target-date', required=False, default=None, help='指定日期打印买卖分数，格式YYYY-MM-DD')

    if pargs is not None:
        return parser.parse_args(pargs)
    return parser.parse_args()


if __name__ == '__main__':
    import numpy as np  # 补充导入numpy

    runstrat()
