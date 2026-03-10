#!/usr/bin/env python
# ====== my_test.py ======
# -*- coding: utf-8; py-indent-offset:4 -*-
###############################################################################

from __future__ import (absolute_import, division, print_function, unicode_literals)

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
import talib
import pandas as pd
import numpy as np

# 只保留 PyQt5，删除所有 PyQt6
from PyQt5.QtWidgets import QApplication, QPushButton, QWidget, QHBoxLayout, QVBoxLayout, QLabel
from PyQt5.QtCore import Qt, QTimer

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
        # 关键：取价格的绝对值计算仓位，避免负数导致size为负
        price = abs(data.close[0])
        cashtouse = self.p.perc * cash
        if BTVERSION > (1, 7, 1, 93):
            size = comminfo.getsize(price, cashtouse)
        else:
            size = cashtouse // price
        return max(size, 1)  # 确保至少1手


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
        ('stock_code', 1),  # 股票代码列（索引1），设为-1表示不解析（用不到）
        ('amount', -1),  # 成交额列（索引7），设为-1不解析
        ('amplitude', 8),  # 振幅列（索引8），设为-1不解析
        # 其他默认字段的索引（必须和你的CSV列对应，覆盖父类默认值！）
        ('datetime', 0),  # date在0列
        ('open', 2),  # open在2列
        ('close', 3),  # close在3列
        ('high', 4),  # high在4列
        ('low', 5),  # low在5列
        ('volume', 6),  # volume在6列
        ('openinterest', -1),  # 无持仓量，设为-1
    )


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
        ('stock_type', '中盘股'),  # default: 中小盘成长，可选：blue_chip/topic/cycle
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

        # MACD指标
        # 直接调用MACDHisto（子类包含macd/signal/histo，无需重复创建MACD）
        self.macd = bt.indicators.MACDHisto(
            self.data0,
            period_me1=self.p.macd1,
            period_me2=self.p.macd2,
            period_signal=self.p.macdsig,
            # movav=ema_broker  # 传入正确的自定义EMA
        )
        # 直接使用内置线，无需手动计算，和券商一致
        self.macd_diff = self.macd.macd  # MACD线
        self.macd_signal = self.macd.signal  # 信号线
        self.macd_hist = self.macd.histo  # 柱状图（macd线 - 信号线，券商标准）
        self.mcross = bt.indicators.CrossOver(self.macd.macd, self.macd.signal)

        # 周线MACD同步修复（和日线保持一致，适配跨周期判断）
        self.week_macd = bt.indicators.MACDHisto(
            self.data1,
            period_me1=self.p.macd1,
            period_me2=self.p.macd2,
            period_signal=self.p.macdsig,
            # movav=ema_broker  # 同样传入正确的自定义EMA
        )
        self.week_macd_diff = self.week_macd.macd  # MACD线
        self.week_macd_signal = self.week_macd.signal  # 信号线
        self.week_macd_hist = self.week_macd.histo  # 柱状图（macd线 - 信号线，券商标准）

        self.week_mcross = bt.indicators.CrossOver(self.week_macd.macd, self.week_macd.signal)

        self.atr = bt.indicators.ATR(self.data0, period=self.p.atrperiod)
        self.rsi6 = bt.indicators.RSI(self.data0, period=6)
        self.rsi12 = bt.indicators.RSI(self.data0, period=12)
        self.rsi24 = bt.indicators.RSI(self.data0, period=24)
        self.week_rsi6 = bt.indicators.RSI(self.data1, period=6)
        self.week_rsi12 = bt.indicators.RSI(self.data1, period=12)
        self.week_rsi24 = bt.indicators.RSI(self.data1, period=24)
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

        # KDJ金叉/死叉判断
        self.kdj_gold_cross = bt.indicators.CrossOver(self.k, self.d)

        week_stoch = bt.indicators.KDJ(
            self.data1,
            period=self.p.kdj_period,
            period_dfast=self.p.kdj_k_period,
            period_dslow=self.p.kdj_d_period,
            movav=bt.indicators.SMA,
            upperband=80,
            lowerband=20
        )
        self.week_k = week_stoch.percK
        self.week_d = week_stoch.percD
        self.week_j = 3 * self.k - 2 * self.d

        # KDJ金叉/死叉判断
        self.week_kdj_gold_cross = bt.indicators.CrossOver(self.week_k, self.week_d)

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

        # 成交量/换手率
        # 1. 获取原始成交量（股），换算为“手”（保留整数）
        self.vol = self.data.volume
        self.vol5 = bt.indicators.SMA(self.data.volume, period=5)
        self.vol10 = bt.indicators.SMA(self.data.volume, period=10)
        self.turnover = self.data.turnover
        # 前5日平均换手率（用于卖出换手率打分）
        self.turnover_5ma = bt.indicators.SMA(self.data0.turnover, period=5)
        # 量比（当日成交量/5日平均成交量，实时计算）
        self.volume_ratio = self.data0.volume / self.vol5

        # 乖离率BIAS6（6日）- 手册动量类0.5分核心指标
        self.bias6 = BIAS(self.data0.close, period=self.p.bias6_period)
        self.bias12 = BIAS(self.data0.close, period=12)
        self.bias24 = BIAS(self.data0.close, period=24)
        # 北向资金占位（需对接实盘数据接口，如tushare/akshare）

        # OBV
        self.obv = bt.indicators.OnBalanceVolume(self.data)
        # Accum/Dist
        self.ad = bt.indicators.AccumDist(self.data)

        self.north_capital = 0.0  # 正数=净流入，负数=净流出，单位：万元
        # 筹码峰占位（需对接实盘数据接口，返回获利盘比例）
        self.profit_ratio = 0.0  # 获利盘比例，0-100
        self.cover_ratio = 0.0  # 套牢盘比例，0-100

        # ===================== 新增：分数拆解存储 - 关键！用于命令行打印具体分项 =====================
        self.daily_score_log = {}  # 键：日期字符串（YYYY-MM-DD），值：{buy_score:总得分, buy_details:分项字典, sell_score:总得分, sell_details:分项字典}

        # ===================== 新增：连续3天分数历史存储 =====================
        self.daily_scores_history = []  # 格式：[(date_str, buy_score), ...]，仅保留最近3天
        self.TREND_BONUS_MAX = 1.0  # 趋势最大加分/减分

        # 买卖信号存储
        self.buy_signals = []
        self.sell_signals = []

        # 股票类型参数（适配tradescore）
        if self.p.stock_type == 'blue_chip':
            self.p.stock_type = '大盘股'
        elif self.p.stock_type == 'mid_small':
            self.p.stock_type = '中盘股'
        elif self.p.stock_type == 'topic':
            self.p.stock_type = '小盘股'
        elif self.p.stock_type == 'cycle':
            self.p.stock_type = '小盘股'

    # ===================== 新增：连续3天分数趋势判断函数 =====================
    def _check_3day_score_trend(self):
        """
        检查连续3天的买入分数趋势，返回趋势类型和评分调整值
        返回值：
            trend_type: str - '递增'/'递减'/'震荡'/'不足3天'
            trend_adjust: float - 趋势调整分（-1.0 ~ +1.0）
        """
        if len(self.daily_scores_history) < 3:
            return '不足3天', 0.0

        # 提取最近3天的分数（按时间顺序：第一天、第二天、第三天,day3是今天）
        day1_date, day1_score = self.daily_scores_history[0]
        day2_date, day2_score = self.daily_scores_history[1]
        day3_date, day3_score = self.daily_scores_history[2]

        # 判断趋势
        if day3_score >= day2_score >= day1_score:
            # 连续递增：加分
            return '递增', (abs(day1_score + day2_score)) / 2

        elif day3_score <= day2_score <= day1_score:
            # 连续递减：减分
            return '递减', -(abs(day1_score + day2_score)) / 2

        else:
            return '震荡', 0.0

    # ===================== 新增：打印连续3天分数详情 =====================
    def _print_3day_score_details(self):
        """打印最近3天的分数详情（调试/日志用）"""
        if len(self.daily_scores_history) < 3:
            print(f"⚠️  连续3天分数数据不足（当前仅{len(self.daily_scores_history)}天）")
            return

        print("📊 连续3天分数趋势：")
        for i, (date_str, score) in enumerate(self.daily_scores_history, 1):
            print(f"   第{i}天({date_str}): {score:.2f}分")

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
            '价格在MA200上方': (self.data.close[0] > self.ma200[0]) if len(self.ma200) > 0 else False,
            # KDJ状态
            # KDJ
            'KDJ_K': round(self.k[0], 2) if len(self.k) > 0 else np.nan,
            'KDJ_D': round(self.d[0], 2) if len(self.d) > 0 else np.nan,
            'KDJ_J': round(self.j[0], 2) if len(self.j) > 0 else np.nan,
            '2日KDJ趋势': "下降" if (
                    len(self.k) >= 2 and len(self.j) >= 2 and self.k[-1] > self.k[0] and self.j[-1] > self.j[
                0]) else "上升" if (
                    len(self.k) >= 2 and len(self.j) >= 2 and self.k[-1] < self.k[0] and self.j[-1] < self.j[
                0]) else "震荡/数据不足",
            'KDJ金叉死叉': (btfinplot.get_kdj_status(self)),
            '周KDJ金叉死叉': (btfinplot.week_get_kdj_status(self)),
            'KDJ超买': self.k[0] > 80 if len(self.k) > 0 else False,
            'KDJ超卖': self.k[0] < 20 if len(self.k) > 0 else False,
            # MACD状态
            'MACD状态': btfinplot.get_macd_status(self),
            '周MACD状态': btfinplot.get_week_macd_status(self),
            # 新增手册指标
            '量比': round(self.volume_ratio[0], 2) if len(self.volume_ratio) > 0 else np.nan,
            '乖离率BIAS6(%)': round(self.bias6[0], 2) if len(self.bias6) > 0 else np.nan,
            '换手率(%)': round(self.turnover[0], 2) if len(self.turnover) > 0 else np.nan,
            '北向资金(万元)': self.north_capital,
            '获利盘比例(%)': self.profit_ratio,
            '套牢盘比例(%)': self.cover_ratio,
            '跌破五日均价': self.data.close[0] < self.ma5[0],

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
        """新增：详细记录每笔交易的盈亏日志（包含下单时的指标值 + 总评分）"""
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

            # ========== 新增：获取买卖日期对应的总评分 ==========
            # 转换日期格式为 YYYY-MM-DD（匹配 daily_score_log 的键）
            buy_date_str = buy_date.strftime('%Y-%m-%d') if buy_date else ""
            sell_date_str = sell_date.strftime('%Y-%m-%d') if sell_date else ""
            # 从每日评分日志中提取总评分（处理日期不存在的情况）
            buy_total_score = self.daily_score_log.get(buy_date_str, {}).get('buy_score', np.nan)
            sell_total_score = self.daily_score_log.get(sell_date_str, {}).get('buy_score', np.nan)

            # 计算交易数量（使用buy-的数量）
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

            # 生成交易日志（包含买卖下单时的指标值 + 总评分）
            self.trade_id += 1
            # 生成交易日志（按8大指标分类排列，便于复盘分析）
            # 生成交易日志（按8大指标核心判定条件排列，记录决策逻辑）
            trade_log = {
                "交易编号": self.trade_id,
                "开仓日期": buy_date.strftime('%Y-%m-%d') if buy_date else "未知",
                "平仓日期": sell_date.strftime('%Y-%m-%d') if sell_date else "未知",
                "持仓天数": trade_duration,
                "盈亏类型": trade_result,
                "盈亏(含手续费)": round(trade_pnlcomm, 2),
                "涨幅": f"{round((sell_price - buy_price) / buy_price * 100, 2)}%",

                # ===================== 1. 【MACD】判定逻辑 =====================
                # 核心逻辑：零轴位置 + 柱状图状态 + 背离
                "【1.MACD(买)】zero_position": f"{buy_indicators['MACD状态']['zero_position']}",
                "【1.MACD(买)】cross_status": f"{buy_indicators['MACD状态']['cross_status']}",
                "【1.MACD(买)】last_gold_day": f"{buy_indicators['MACD状态']['last_gold_day']}",
                "【1.MACD(买)】last_dead_day": f"{buy_indicators['MACD状态']['last_dead_day']}",
                "【1.MACD(卖)】zero_position": f"{sell_indicators['MACD状态']['zero_position']}",
                "【1.MACD(卖)】cross_status": f"{sell_indicators['MACD状态']['cross_status']}",
                "【1.MACD(卖)】last_gold_day": f"{sell_indicators['MACD状态']['last_gold_day']}",
                "【1.MACD(卖)】last_dead_day": f"{sell_indicators['MACD状态']['last_dead_day']}",

                "【1.周MACD(买)】zero_position": f"{buy_indicators['周MACD状态']['zero_position']}",
                "【1.周MACD(买)】cross_status": f"{buy_indicators['周MACD状态']['cross_status']}",
                "【1.周MACD(买)】last_gold_day": f"{buy_indicators['周MACD状态']['last_gold_day']}",
                "【1.周MACD(买)】last_dead_day": f"{buy_indicators['周MACD状态']['last_dead_day']}",
                "【1.周MACD(卖)】zero_position": f"{sell_indicators['MACD状态']['zero_position']}",
                "【1.周MACD(卖)】cross_status": f"{sell_indicators['周MACD状态']['cross_status']}",
                "【1.周MACD(卖)】last_gold_day": f"{sell_indicators['周MACD状态']['last_gold_day']}",
                "【1.周MACD(卖)】last_dead_day": f"{sell_indicators['周MACD状态']['last_dead_day']}",



                # ===================== 2. 【KDJ】判定逻辑 =====================
                # 核心逻辑：极值区(J>90/J<10) + 金死叉 + 趋势位置
                "【2.KDJ】J值位置(买/卖)": f"{'极端超卖(J<' + str(buy_indicators.get('KDJ_J', 0)) + ')' if buy_indicators.get('KDJ_J', 0) < 10 else '超卖区' if buy_indicators.get('KDJ_J', 0) < 20 else '震荡/高位'} / {'极端超买(J>' + str(sell_indicators.get('KDJ_J', 0)) + ')' if sell_indicators.get('KDJ_J', 0) > 90 else '超买区' if sell_indicators.get('KDJ_J', 0) > 80 else '震荡/低位'}",
                "【2.KDJ(买)】日线金死叉": f"{buy_indicators.get('KDJ金叉死叉')}",
                "【2.KDJ(卖)】日线金死叉": f"{sell_indicators.get('KDJ金叉死叉')}",
                "【2.KDJ(买)】周线金死叉": f"{buy_indicators.get('周KDJ金叉死叉')}",
                "【2.KDJ(卖)】周线金死叉": f"{sell_indicators.get('周KDJ金叉死叉')}",
                "【2.KDJ】连续极值(买/卖)": f"{'连续超卖' if buy_indicators.get('KDJ_J', 0) < 20 and buy_indicators.get('KDJ_J_prev1', 0) < 20 else '无'} / {'连续超买' if sell_indicators.get('KDJ_J', 0) > 80 and sell_indicators.get('KDJ_J_prev1', 0) > 80 else '无'}",

                # ===================== 3. 【RSI】判定逻辑 =====================
                # 核心逻辑：超买超卖线(20/80) + 背离/金叉
                "【3.RSI】数值区间(买/卖)": f"{'极端超卖(<20)' if buy_indicators.get('RSI6', 0) < 20 else '超卖(20-30)' if buy_indicators.get('RSI6', 0) < 30 else '常态'} / {'极端超买(>80)' if sell_indicators.get('RSI6', 0) > 80 else '超买(70-80)' if sell_indicators.get('RSI6', 0) > 70 else '常态'}",
                "【3.RSI】突破信号(买/卖)": f"{'突破50线金叉' if buy_indicators.get('RSI6', 0) > 50 and buy_indicators.get('RSI6_prev', 0) < 50 else '普通金叉'} / {'跌破50线死叉' if sell_indicators.get('RSI6', 0) < 50 and sell_indicators.get('RSI6_prev', 0) > 50 else '普通死叉'}",

                # ===================== 4. 【BIAS】判定逻辑 =====================
                # 核心逻辑：极端阈值(大盘<-5%等) + 背离
                "【4.BIAS】6日数值(买/卖)": f"{buy_indicators.get('乖离率BIAS6(%)', 0):.2f}% / {sell_indicators.get('乖离率BIAS6(%)', 0):.2f}%",
                "【4.BIAS】状态(买/卖)": f"{'极端超卖/共振' if buy_indicators.get('乖离率BIAS6(%)', 0) < -5 else '回归平衡区'} / {'极端超买/共振' if sell_indicators.get('乖离率BIAS6(%)', 0) > 5 else '回归平衡区'}",
                "【4.BIAS】背离(买/卖)": f"{'BIAS底背离' if buy_indicators.get('BIAS底背离') else '无'} / {'BIAS顶背离' if sell_indicators.get('BIAS顶背离') else '无'}",

                # ===================== 5. 【BOLL】判定逻辑 =====================
                # 核心逻辑：触及轨线 + 中轨趋势
                "【5.BOLL】价格位置(买/卖)": f"{'触及下轨' if buy_indicators.get('布林偏离度(%)', 0) < 5 else '中轨下方'} / {'触及上轨' if sell_indicators.get('布林偏离度(%)', 0) > 95 else '中轨上方'}",
                "【5.BOLL】中轨趋势(买/卖)": f"{'中轨向上' if buy_indicators.get('布林中轨', 0) > buy_indicators.get('布林中轨_prev', 0) else '中轨向下'} / {'中轨向下' if sell_indicators.get('布林中轨', 0) < sell_indicators.get('布林中轨_prev', 0) else '中轨向上'}",
                "【5.BOLL】背离(买/卖)": f"{'BOLL底背离' if buy_indicators.get('BOLL底背离') else '无'} / {'BOLL顶背离' if sell_indicators.get('BOLL顶背离') else '无'}",

                # ===================== 6. 【均线】判定逻辑 =====================
                # 核心逻辑：多空排列 + 价格相对位置
                "【6.均线】排列(买/卖)": f"{'多头排列' if buy_indicators.get('MA5', 0) > buy_indicators.get('MA10', 0) else '空头排列'} / {'空头排列' if sell_indicators.get('MA5', 0) < sell_indicators.get('MA10', 0) else '多头排列'}",
                "【6.均线】价格位置(买/卖)": f"{'站稳MA60' if buy_indicators.get('价格在MA200上方') else 'MA60下方'} / {'跌破MA60' if sell_indicators.get('价格在MA200上方') == False else 'MA60上方'}",
                "【6.均线】背离(买/卖)": f"{'均线底背离' if buy_indicators.get('MA底背离') else '无'} / {'均线顶背离' if sell_indicators.get('MA顶背离') else '无'}",

                # ===================== 7. 【成交量】判定逻辑 =====================
                # 核心逻辑：量价配合 + 极值(地量/天量)
                "【7.成交量】量价关系(买/卖)": f"{'价涨量增' if buy_indicators.get('成交量(手)', 0) > buy_indicators.get('均量线5', 0) else '缩量整理'} / {'价跌量增' if sell_indicators.get('成交量(手)', 0) > sell_indicators.get('均量线5', 0) else '缩量阴跌'}",
                "【7.成交量】极值状态(买/卖)": f"{'地量筑底' if buy_indicators.get('成交量(手)', 0) < buy_indicators.get('均量线5', 0) * 0.5 else '放量突破'} / {'天量见顶' if sell_indicators.get('成交量(手)', 0) > sell_indicators.get('均量线5', 0) * 1.5 else '温和放量'}",

                # ===================== 8. 【换手率】判定逻辑 =====================
                # 核心逻辑：换手区间(低/健康/高) + 股票类型适配
                "【8.换手率】数值(买/卖)": f"{buy_indicators.get('换手率(%)', 0):.2f}% / {sell_indicators.get('换手率(%)', 0):.2f}%",
                "【8.换手率】状态(买/卖)": f"{'低位惜售' if buy_indicators.get('换手率(%)', 0) < 2 else '活跃换手'} / {'高位放量' if sell_indicators.get('换手率(%)', 0) > 10 else '温和'}",
                "【8.换手率】类型适配(买/卖)": f"{self.p.stock_type}(买) / {self.p.stock_type}(卖)",

                # ===================== 9. 【综合位置修正】 =====================
                # 核心逻辑：这是评分手册中决定分数翻倍/减半的关键
                "【9.位置修正】股价阶段(买/卖)": f"{'低位(MA60下)' if buy_price < buy_indicators.get('MA60', 0) else '中位/高位'} / {'高位(创阶段新高)' if sell_price > max(buy_indicators.get('MA60', 0), sell_indicators.get('MA60', 0)) else '中位/低位'}",

                # ===================== 新增：连续3天分数趋势 =====================
                "【10.连续3天分数】买入日趋势": self.daily_scores_history[-1][1] if len(
                    self.daily_scores_history) > 0 else np.nan,
                "【10.连续3天分数】趋势类型": self._check_3day_score_trend()[0],
                "【10.连续3天分数】趋势调整分": self._check_3day_score_trend()[1],
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
        # 添加数据长度检查，避免前期指标计算不准确
        if len(self.data) < 60:  # 确保有足够数据计算MA60等
            return

        # 200日均线趋势判断
        if len(self.ma200) >= 10:
            self.ma200_slope, self.ma200_trend = btfinplot.get_ma_trend_global(self.ma200, 10, 0.0005)
        else:
            self.ma200_slope, self.ma200_trend = 0.0, "未知"

        # ========== 核心修正1：严格按手册计算总评分和决策 ==========
        total_score, decision, details = btfinplot.calculate_total_score(self)

        # 记录当日评分到日志
        current_date = self.data.datetime.date(0).strftime('%Y-%m-%d')
        self.daily_score_log[current_date] = {
            'buy_score': total_score,  # 使用tradescore的总评分
            'buy_details': details,
            'sell_score': -total_score,
            'sell_details': details
        }

        # ===================== 新增：更新连续3天分数历史 =====================
        current_score_entry = (current_date, total_score)
        self.daily_scores_history.append(current_score_entry)
        # 只保留最近3天的分数
        if len(self.daily_scores_history) > 3:
            self.daily_scores_history.pop(0)

        # ===================== 新增：获取连续3天趋势调整分 =====================
        trend_type, trend_adjust = self._check_3day_score_trend()
        # 最终决策分数 = 当日分数 + 趋势调整分
        final_decision_score = total_score

        if self.order:
            return

        # ========== 核心修正2：严格遵循手册的交易决策阈值（使用最终决策分） ==========
        if not self.position:  # 无仓位
            # 强力买入/重仓（≥12分）
            if final_decision_score >= btfinplot.DECISION_THRESHOLDS['强力买入']:
                buy_indicators = self.get_current_indicators()
                self.order_indicator_info['buy'] = buy_indicators
                self.order = self.buy(exectype=bt.Order.Market)
                # 关键：用价格绝对值计算止损价
                pdist = self.atr[0] * self.p.atrdist
                self.pstop = abs(self.data0.close[0]) - pdist  # 取绝对值
                # 确保止损价为正
                self.pstop = max(self.pstop, 0.01)
                print(
                    f"【强力买入】日期：{current_date} | 当日评分：{total_score:.2f} | 趋势({trend_type})调整：{trend_adjust:.2f} | 最终评分：{final_decision_score:.2f} | 决策：{decision} | 细节: {details}")
                self._print_3day_score_details()  # 打印3天分数详情
            # 积极买入/加仓（8~11.9分）
            elif btfinplot.DECISION_THRESHOLDS['积极买入'] <= final_decision_score < btfinplot.DECISION_THRESHOLDS[
                '强力买入']:
                buy_indicators = self.get_current_indicators()
                self.order_indicator_info['buy'] = buy_indicators
                self.order = self.buy(exectype=bt.Order.Market)
                pdist = self.atr[0] * self.p.atrdist
                self.pstop = self.data0.close[0] - pdist
                print(
                    f"【积极买入】日期：{current_date} | 当日评分：{total_score:.2f} | 趋势({trend_type})调整：{trend_adjust:.2f} | 最终评分：{final_decision_score:.2f} | 决策：{decision} | 细节: {details}")
                self._print_3day_score_details()
            # 谨慎买入/低吸（4~7.9分）
            # elif btfinplot.DECISION_THRESHOLDS['谨慎买入'] <= total_score < btfinplot.DECISION_THRESHOLDS['积极买入']:
            elif 6 <= final_decision_score < btfinplot.DECISION_THRESHOLDS['积极买入']:
                buy_indicators = self.get_current_indicators()
                self.order_indicator_info['buy'] = buy_indicators
                self.order = self.buy(exectype=bt.Order.Market)
                pdist = self.atr[0] * self.p.atrdist
                self.pstop = self.data0.close[0] - pdist
                print(
                    f"【谨慎买入】日期：{current_date} | 当日评分：{total_score:.2f} | 趋势({trend_type})调整：{trend_adjust:.2f} | 最终评分：{final_decision_score:.2f} | 决策：{decision} | 细节: {details}")
                self._print_3day_score_details()
            # 观望区间（-3.9~3.9分）
            else:
                # 可选：打印趋势信息（非必须）
                # print(f"【观望】日期：{current_date} | 当日评分：{total_score:.2f} | 趋势({trend_type})调整：{trend_adjust:.2f} | 最终评分：{final_decision_score:.2f} | 决策：{decision} | 细节: {details}")
                pass

        else:  # 有仓位
            pclose = self.data0.close[0]
            pstop = self.pstop

            # 强力卖出/空仓（≤-12分）
            if final_decision_score <= btfinplot.DECISION_THRESHOLDS['强力卖出']:
                sell_indicators = self.get_current_indicators()
                self.order_indicator_info['sell'] = sell_indicators
                self.order = self.close(exectype=bt.Order.Market)
                print(
                    f"【强力卖出】日期：{current_date} | 当日评分：{total_score:.2f} | 趋势({trend_type})调整：{trend_adjust:.2f} | 最终评分：{final_decision_score:.2f} | 决策：{decision} | 细节: {details}")
                self._print_3day_score_details()  # 打印3天分数详情
            # 谨慎卖出/减仓（-8~-4分）
            elif btfinplot.DECISION_THRESHOLDS['强力卖出'] < final_decision_score <= btfinplot.DECISION_THRESHOLDS[
                '谨慎卖出']:
                sell_indicators = self.get_current_indicators()
                self.order_indicator_info['sell'] = sell_indicators
                self.order = self.close(exectype=bt.Order.Market)
                print(
                    f"【谨慎卖出】日期：{current_date} | 当日评分：{total_score:.2f} | 趋势({trend_type})调整：{trend_adjust:.2f} | 最终评分：{final_decision_score:.2f} | 决策：{decision} | 细节: {details}")
                self._print_3day_score_details()
            # 观望区间（-3.9~3.9分），执行移动止损
            else:
                # 移动止损（保留原有逻辑）
                pdist = self.atr[0] * self.p.atrdist
                self.pstop = max(pstop, pclose - pdist)
                # print(f"【持仓观望】日期：{current_date} | 当日评分：{total_score:.2f} | 趋势({trend_type})调整：{trend_adjust:.2f} | 最终评分：{final_decision_score:.2f} | 止损价：{self.pstop:.2f} | 细节: {details}")

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
            print(f"\n✅ 每笔交易盈亏日志（含连续3天分数趋势）已保存到: {TRADE_LOG_FILE}")
        else:
            print("\n⚠️ 本次回测无交易记录")


DATASETS = {
    'yhoo': '../../datas/yhoo-1996-2014.txt',
    'orcl': '../../datas/orcl-1995-2014.txt',
    'nvda': '../../datas/nvda-1999-2014.txt',
}

# ===================== 新增：打印指定日期分数核心函数（新增连续3天展示） =====================
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
        datetime=0, open=2, close=3, high=4, low=5, volume=6, openinterest=-1, **dkwargs
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
    df_plot = pd.read_csv(csv_file_path, usecols=['date', '股票代码', 'open', 'high', 'low', 'close', 'volume', 'turnover'])

    # === 关键修正：确保 date 列是 datetime 类型，然后创建 date_str 和 time 索引 ===
    df_plot['date'] = pd.to_datetime(df_plot['date'])  # 先转为 datetime
    df_plot['date_str'] = df_plot['date'].dt.strftime('%Y-%m-%d')  # 再创建字符串列
    df_plot['time'] = df_plot['date'].apply(lambda x: int(x.timestamp()))  # 创建时间戳
    df_plot.set_index('time', inplace=True)
    df_plot.drop(columns=['date'], inplace=True)  # 删除原始的 datetime 列
    # ========================================================================
    # ===================== 最终修复版：提取股票代码（兼容所有类型） =====================
    stock_code = "未知股票"
    if '股票代码' in df_plot.columns:
        # 过滤空值/空白字符串（先统一转为字符串）
        # 步骤1：将整列转为字符串，避免numpy.int64/numpy.float64类型
        df_plot['股票代码'] = df_plot['股票代码'].astype(str)
        # 步骤2：过滤空值、空白字符串、nan字符串
        valid_codes = df_plot['股票代码'].replace(['', 'nan', 'NaN', 'None'], np.nan).dropna()

        if not valid_codes.empty:
            # 步骤3：取第一个值，转为字符串后去空格
            stock_code = str(valid_codes.iloc[0]).strip()
            # 步骤4：处理浮点数字符串（如600519.0 → 600519）
            if '.' in stock_code:
                stock_code = stock_code.split('.')[0]
            # 步骤5：补全6位前置零（A股代码固定6位）
            if stock_code.isdigit():
                stock_code = stock_code.zfill(6)

    print(f"✅ 从 df_plot 提取到股票代码：{stock_code}")
    # --- 准备交易信号（这部分还是需要从 strat 获取）---
    # ===================== 【替换这部分绘图代码】 =====================

    # 1. 准备绘图数据 (df_plot 你已经处理好了，这里确保有 date_str)
    # ... (你之前的 df_plot 处理代码保留) ...
    # 确保 df_plot 有 date_str 列 (如果没有就加上)
    if 'date_str' not in df_plot.columns:
        # 假设索引是时间戳，或者有 date 列
        # 这里根据你实际情况调整，确保能生成 YYYY-MM-DD 格式的字符串列
        pass

    # 2. 准备交易信号
    all_signals = []
    for date_str, _ in strat.buy_signals:
        all_signals.append({'date': date_str, 'type': 'buy'})
    for date_str, _ in strat.sell_signals:
        all_signals.append({'date': date_str, 'type': 'sell'})

    # 3. 计算周线数据
    df_weekly = btfinplot.resample_to_weekly(df_plot)

    print("正在创建图表窗口...")

    # 4. 创建周线图 (不立即显示)
    btfinplot.plot_a_stock_analysis(
        df=df_weekly,
        symbol=stock_code,
        title_suffix='周线图',
        signals=all_signals,
        is_weekly=True,
        show_immediately=False  # 【关键】False，先不显示
    )

    # 5. 创建日线图 (不立即显示)
    btfinplot.plot_a_stock_analysis(
        df=df_plot,
        symbol=stock_code,
        title_suffix='日线图',
        signals=all_signals,
        is_weekly=False,
        show_immediately=False  # 【关键】False，先不显示
    )

    print("两个窗口已创建，正在显示...")
    print("📈 关闭所有图表窗口后程序将自动退出。")

    # 6. 【终极关键】使用 finplot 原生的 show()
    # 这会显示上面创建的所有窗口，并阻塞程序直到所有窗口都关闭
    btfinplot.fplt.show()

    print("所有窗口已关闭，程序结束。")

    # 这里不需要再调用 qt_app.exec_() 了，fplt.show() 已经处理完了
    # ===================================================================
    # ===================================================================
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
