# btfinplot.py
# !/usr/bin/env python3

from collections import defaultdict
import dateutil.parser
import finplot as fplt
import numpy as np
import pandas as pd
import requests
import os
# 新增：导入pyqtgraph用于创建独立图例
import pyqtgraph as pg
from PyQt5.QtWidgets import QWidget, QGridLayout, QComboBox, QCheckBox, QApplication  # 补全导入
from PyQt5.QtCore import Qt, QTimer  # 新增 QTimer 用于异步创建窗口
import sys  # 新增：管理Qt应用程序

baseurl = 'https://www.bitmex.com/api'
fplt.timestamp_format = '%m/%d/%Y %H:%M:%S.%f'

# ========== 消除QT DPI警告的核心配置 ==========
# 在创建QApplication之前设置环境变量（优先方案）
os.environ.pop('QT_DEVICE_PIXEL_RATIO', None)
os.environ['QT_AUTO_SCREEN_SCALE_FACTOR'] = '1'
os.environ['QT_SCREEN_SCALE_FACTORS'] = ''
os.environ['QT_SCALE_FACTOR'] = '1'

# ========== 简化的Qt属性设置（不启动事件循环） ==========
def init_qt_attributes():
    """在创建QApplication之前设置DPI属性"""
    # 必须在QCoreApplication实例化之前设置这些属性
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)


# ========== 新增：独立图例容器API ==========
def create_legend(ax,
                  name="default",
                  pos="top_left",
                  size=(100, 100),
                  bg_color="white",
                  text_color="black",
                  border_color="#cccccc"):
    """
    创建独立的图例容器（绑定到指定ax）
    :param ax: 绑定的绘图轴（finplot的ax/PlotItem）
    :param name: 图例唯一标识（用于后续获取）
    :param pos: 位置，支持top_left/top_right/bottom_left/bottom_right或(x,y)元组
    :param size: 图例尺寸 (width, height)
    :param bg_color: 背景色
    :param text_color: 文本色
    :param border_color: 边框色
    :return: 创建的LegendItem实例
    """
    # 初始化图例容器
    legend = pg.LegendItem(size=size, offset=(0, 0))
    legend.setParentItem(ax)
    legend._name = name  # 绑定唯一标识

    # 设置样式
    legend._bg_color = pg.mkColor(bg_color)
    legend._text_color = pg.mkColor(text_color)
    legend._border_color = pg.mkColor(border_color)

    # 重写paint方法自定义样式
    def custom_paint(p, *args):
        p.setPen(pg.mkPen(legend._border_color))
        p.setBrush(pg.mkBrush(legend._bg_color))
        p.drawRect(legend.boundingRect())

    legend.paint = custom_paint

    # 设置位置
    set_legend_pos(legend, pos, ax)

    # 存储到ax的自定义属性中（方便后续获取）
    if not hasattr(ax, "_legends"):
        ax._legends = {}
    ax._legends[name] = legend

    return legend


def get_legend(ax, name="default"):
    """
    根据名称获取独立图例容器
    :param ax: 绘图轴
    :param name: 图例唯一标识
    :return: LegendItem实例（不存在返回None）
    """
    if hasattr(ax, "_legends") and name in ax._legends:
        return ax._legends[name]
    return None


def set_legend_pos(legend, pos, ax=None):
    """
    设置图例位置
    :param legend: 图例实例
    :param pos: 位置，支持top_left/top_right/bottom_left/bottom_right或(x,y)元组
    :param ax: 绘图轴（用于计算相对位置）
    """
    if ax is None:
        ax = legend.parentItem()
    if not ax:
        return

    # 获取ax的边界矩形
    ax_rect = ax.boundingRect()
    ax_width = ax_rect.width()
    ax_height = ax_rect.height()
    # 修复：size是属性（tuple），不是方法，去掉括号
    legend_width, legend_height = legend.size

    # 解析位置
    pos_map = {
        "top_left": (10, ax_height - legend_height - 10),
        "top_right": (ax_width - legend_width - 10, ax_height - legend_height - 10),
        "bottom_left": (10, 10),
        "bottom_right": (ax_width - legend_width - 10, 10)
    }
    if isinstance(pos, str) and pos in pos_map:
        x, y = pos_map[pos]
    elif isinstance(pos, (tuple, list)) and len(pos) == 2:
        x, y = pos
    else:
        x, y = pos_map["top_left"]  # 默认左上角

    # 设置位置并锚定
    legend.setPos(x, y)
    legend.anchor((0, 1), (0, 1))  # 锚点对齐图例左上角


def add_legend_item(ax, legend_name, plot_item, label):
    """
    给指定图例添加条目
    :param ax: 绘图轴
    :param legend_name: 图例名称
    :param plot_item: finplot绘制的曲线/图形实例
    :param label: 图例文本
    """
    legend = get_legend(ax, legend_name)
    if legend:
        legend.addItem(plot_item, label)


def local2timestamp(s):
    return int(dateutil.parser.parse(s).timestamp())


def download_price_history(symbol='XBTUSD', start_time='2023-01-01', end_time='2023-10-29', interval_mins=60):
    csv_file_path = r'D:\outsidework\github\stockdata\A_data\002738_qfq_A_data.csv'
    if not os.path.exists(csv_file_path):
        raise FileNotFoundError(f"本地CSV文件不存在：{csv_file_path}，请检查路径是否正确！")

    # ✅ 关键修改：加入 'turnover' 到 usecols（前提是 CSV 里真有这列！）
    required_cols = ['date', 'open', 'close', 'high', 'low', 'volume', 'turnover']
    df = pd.read_csv(csv_file_path, usecols=required_cols)

    df['time'] = df['date'].apply(lambda x: local2timestamp(x))
    start_time = local2timestamp(start_time)
    end_time = local2timestamp(end_time)
    df = df[(df['time'] >= start_time) & (df['time'] <= end_time)]
    df = df.dropna(subset=required_cols)  # 包含 turnover

    return df.set_index('time')[['open', 'close', 'high', 'low', 'volume', 'turnover']]


# ========== 新增：周期重采样函数 ==========
def resample_to_weekly(df_daily):
    """
    将日线数据重采样为周线数据（周一开盘，周五收盘）
    :param df_daily: 日线DataFrame，索引为时间戳（秒），包含open/close/high/low/volume/turnover
    :return: 周线DataFrame
    """
    # 转换索引为datetime
    df = df_daily.copy()
    df.index = pd.to_datetime(df.index, unit='s')

    # 周线重采样规则：
    # - open: 每周第一个值
    # - high: 每周最大值
    # - low: 每周最小值
    # - close: 每周最后一个值
    # - volume: 每周求和
    # - turnover: 每周平均值
    weekly_df = df.resample('W-FRI').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum',
        'turnover': 'mean'
    }).dropna()

    # 转换回时间戳索引（秒）
    weekly_df.index = weekly_df.index.astype('int64') // 10 ** 9
    weekly_df.index.name = 'time'

    # 添加date_str列（用于信号绘制）
    weekly_df['date_str'] = pd.to_datetime(weekly_df.index, unit='s').strftime('%Y-%m-%d')

    return weekly_df


def plot_accumulation_distribution(df, ax):
    mfm_numerator = 2 * df['close'] - df['high'] - df['low']
    mfm_denominator = df['high'] - df['low']

    mfm = np.where(mfm_denominator == 0, 0.0, mfm_numerator / mfm_denominator)
    mf_volume = mfm * df['volume']

    # 关键修正：A/D 线起始值设为第一个有效成交量（同花顺常见做法）
    ad_line = mf_volume.copy()
    ad_line.iloc[0] = df['volume'].iloc[0]  # 或设为 0，两者差异仅在绝对值，不影响趋势
    ad_line = ad_line.cumsum()

    ad_line.plot(ax=ax, legend='Accum买方/Dist卖方', color='#f00000')


def plot_bollinger_bands(df, ax):
    # 同花顺默认参数：20日均线，2.5倍标准差
    window = 20
    num_std = 2.5

    # 1. 计算中轨 (MID)
    mean = df['close'].rolling(window=window).mean()

    # ========== 新增：计算 BIAS (乖离率) ==========
    ma6 = df['close'].rolling(window=6).mean()
    ma12 = df['close'].rolling(window=12).mean()
    ma24 = df['close'].rolling(window=24).mean()

    df['bias6'] = (df['close'] - ma6) / ma6 * 100
    df['bias12'] = (df['close'] - ma12) / ma12 * 100
    df['bias24'] = (df['close'] - ma24) / ma24 * 100
    # ========== 计算逻辑结束 ==========

    # 2. 计算标准差
    stddev = df['close'].rolling(window=window).std(ddof=0)

    # 3. 计算上下轨
    df['boll_hi'] = mean + num_std * stddev
    df['boll_lo'] = mean - num_std * stddev
    df['boll_mid'] = mean  # 确保写回df

    # 4. 绘图
    p0 = df['boll_hi'].plot(ax=ax, color='red', legend='BB Upper')
    p1 = df['boll_lo'].plot(ax=ax, color='green', legend='BB Lower')
    fplt.fill_between(p0, p1, color='#bbb')
    mean.plot(ax=ax, color='blue', style='--', width=1.0)


def plot_ema(df, ax):
    df.close.ewm(span=9).mean().plot(ax=ax, legend='EMA')


def plot_ma(df, ax):
    # 多周期均线（MA5 / 10 / 20 / 60 / 200）
    # 使用局部变量避免污染原df（可选）
    close = df['close']

    # 关键！！！所有均线 加 .dropna()，删除前面无效NaN，不撑大Y轴灰色框
    ma5 = close.rolling(window=5).mean().dropna()
    ma10 = close.rolling(window=10).mean().dropna()
    ma20 = close.rolling(window=20).mean().dropna()
    ma60 = close.rolling(window=60).mean().dropna()
    ma200 = close.rolling(window=200).mean().dropna()

    # 绘制均线并添加到独立MA图例
    curve5 = fplt.plot(ma5, ax=ax, color='black')
    curve10 = fplt.plot(ma10, ax=ax, color='orange')
    curve20 = fplt.plot(ma20, ax=ax, color='red')
    curve60 = fplt.plot(ma60, ax=ax, color='green')  # 紫色
    curve200 = fplt.plot(ma200, ax=ax, color='blue')  # 青色

    # 添加到独立MA图例
    add_legend_item(ax, "ma_legend", curve5, 'MA5')
    add_legend_item(ax, "ma_legend", curve10, 'MA10')
    add_legend_item(ax, "ma_legend", curve20, 'MA20')
    add_legend_item(ax, "ma_legend", curve60, 'MA60')
    add_legend_item(ax, "ma_legend", curve200, 'MA200')


def plot_candlestick(df, ax):
    """
    绘制标准（原始）K线图（非Heikin-Ashi）
    要求 df 包含 'open', 'close', 'high', 'low' 列
    """
    # ===== 1. 绘制K线 =====
    fplt.candlestick_ochl(df[['open', 'close', 'high', 'low']], ax=ax)


def plot_heikin_ashi(df, ax):
    df['h_close'] = (df.open + df.close + df.high + df.low) / 4
    ho = (df.open.iloc[0] + df.close.iloc[0]) / 2
    for i, hc in zip(df.index, df['h_close']):
        df.loc[i, 'h_open'] = ho
        ho = (ho + hc) / 2
    df['h_high'] = df[['high', 'h_open', 'h_close']].max(axis=1)
    df['h_low'] = df[['low', 'h_open', 'h_close']].min(axis=1)
    df[['h_open', 'h_close', 'h_high', 'h_low']].plot(ax=ax, kind='candle')


def plot_heikin_ashi_volume(df, ax):
    df[['h_open', 'h_close', 'volume']].plot(ax=ax, kind='volume')


def plot_volume(df, ax):
    df[['open', 'close', 'volume']].plot(ax=ax, kind='volume')


def plot_on_balance_volume(df, ax):
    obv = pd.Series(index=df.index, dtype='float64')
    obv.iloc[0] = df['volume'].iloc[0]  # 第一天 OBV = 成交量

    for i in range(1, len(df)):
        if df['close'].iloc[i] > df['close'].iloc[i - 1]:
            obv.iloc[i] = obv.iloc[i - 1] + df['volume'].iloc[i]
        elif df['close'].iloc[i] < df['close'].iloc[i - 1]:
            obv.iloc[i] = obv.iloc[i - 1] - df['volume'].iloc[i]
        else:
            obv.iloc[i] = obv.iloc[i - 1]  # 相等时保持不变

    obv.plot(ax=ax, legend='OBV量价关系', color='#008800')


def plot_rsi(df, ax):
    """计算并绘制三个不同周期的RSI指标（RSI6/12/24）"""
    periods = [6, 12, 24]
    colors = ['black', 'orange', 'red']
    curve = {}
    for i, period in enumerate(periods):
        # ===== 为每个周期独立初始化数据（避免跨周期污染）=====
        diff = df['close'].diff().values
        gains = diff.copy()
        losses = -diff.copy()

        with np.errstate(invalid='ignore'):
            gains[(gains < 0) | np.isnan(gains)] = 0.0
            losses[(losses <= 0) | np.isnan(losses)] = 1e-10

        m = (period - 1) / period
        ni = 1 / period

        # ===== 安全处理空切片警告 =====
        # 前period个值设为NaN（RSI有效值从第period+1根K线开始）
        gains[:period] = np.nan
        losses[:period] = np.nan

        # 安全计算初始平均值（过滤NaN后计算）
        valid_gains = gains[period:2 * period]  # 取后续有效数据段
        valid_losses = losses[period:2 * period]
        avg_gain = np.nanmean(valid_gains) if not np.all(np.isnan(valid_gains)) else 0.0
        avg_loss = np.nanmean(valid_losses) if not np.all(np.isnan(valid_losses)) else 1e-10

        # 初始化平滑起点
        if len(gains) > period:
            gains[period] = avg_gain
            losses[period] = avg_loss

        # Wilder's 平滑计算
        for idx in range(period + 1, len(gains)):
            gains[idx] = ni * gains[idx] + m * gains[idx - 1]
            losses[idx] = ni * losses[idx] + m * losses[idx - 1]

        # 计算RSI
        rs = gains / losses
        rsi = 100 - (100 / (1 + rs))

        # 存储并绘制（使用当前周期的列名）
        col_name = f'rsi_{period}'
        df[col_name] = rsi

        # df[col_name].plot(ax=ax, legend=f'RSI({period})', color=colors[i])
        curve[col_name] = df[col_name].plot(ax=ax, color=colors[i], legend=col_name)

        # 添加到独立MA图例
        add_legend_item(ax, "rsi_legend", curve[col_name], f'RSI({period})')

    # 设置坐标轴范围
    fplt.set_y_range(0, 100, ax=ax)

    # 超买超卖区域
    fplt.add_horizontal_band(30, 70, ax=ax)
    # curve = fplt.plot([50] * len(df), ax=ax, color='gray', style='--', width=1.0, legend='强弱线(50)')
    fplt.plot([50] * len(df), ax=ax, color='gray', style='--', width=1.0)
    # 添加到独立MA图例
    # add_legend_item(ax, "rsi_legend", curve, '强弱线(50)')


def plot_macd(df, ax):
    # plot macd with standard colors first
    # 核心修复：提前计算完整MACD序列并存储到df，添加adjust=False对齐A股券商逻辑
    df['macd_line'] = df.close.ewm(span=12, adjust=False).mean() - df.close.ewm(span=26, adjust=False).mean()
    df['signal_line'] = df['macd_line'].ewm(span=9, adjust=False).mean()
    df['macd_diff'] = df['macd_line'] - df['signal_line']

    fplt.volume_ocv(df[['open', 'close', 'macd_diff']], ax=ax, colorfunc=fplt.strength_colorfilter)
    fplt.plot(df['macd_line'], ax=ax, legend='DIFF', color='black')
    fplt.plot(df['signal_line'], ax=ax, legend='Signal', color='red')


def plot_vma(df, ax):
    df.volume.rolling(20).mean().plot(ax=ax, legend='VOL20', color='#c0c030')


def plot_kdj(df, ax):
    """
    使用 calculate_kdj_bt 计算 KDJ 并绘图（完全对齐A股券商逻辑）
    """

    def calculate_kdj_bt(df_plot, period=9, k_period=3, d_period=3):
        """
        完全对齐【A股券商版KDJ】递推逻辑（2/3前值+1/3当前值），适配所有pandas版本
        核心：和backtrader的_KDJBase类next()逻辑1:1复刻，彻底解决数据不一致
        :param df_plot: 原始K线df（含open/high/low/close列，小写）
        :param period: KDJ核心周期（默认9）
        :param k_period: 兼容参数（无实际作用，保留为了传参一致）
        :param d_period: 兼容参数（无实际作用，保留为了传参一致）
        :return: 带kdj_k/kdj_d/kdj_j列的df
        """
        df = df_plot.copy()
        # 若数据量不足，直接填充50（和券商版KDJ初始化一致）
        if len(df) < 1:
            df['kdj_k'] = 50.0
            df['kdj_d'] = 50.0
            df['kdj_j'] = 50.0
            return df

        # Step1：计算RSV - 完全对齐券商版逻辑（手动取N周期高低，除零保护RSV=50）
        df['n_high'] = df['high'].rolling(window=period, min_periods=1).max()  # 至少1个数据，避免空值
        df['n_low'] = df['low'].rolling(window=period, min_periods=1).min()
        # 除零保护：高低价相同时RSV=50（券商标准）
        df['rsv'] = np.where(
            df['n_high'] == df['n_low'],
            50.0,
            100.0 * (df['close'] - df['n_low']) / (df['n_high'] - df['n_low'])
        )

        # Step2：计算K/D - 券商版核心递推逻辑（2/3前值 + 1/3当前值），初始化K/D=50
        df['kdj_k'] = 50.0  # 第一根K线初始化K=50
        df['kdj_d'] = 50.0  # 第一根K线初始化D=50
        for i in range(1, len(df)):
            # K线递推：2/3 * 前一根K + 1/3 * 当前RSV
            df.loc[df.index[i], 'kdj_k'] = (2 / 3) * df.loc[df.index[i - 1], 'kdj_k'] + (1 / 3) * df.loc[
                df.index[i], 'rsv']
            # D线递推：2/3 * 前一根D + 1/3 * 当前K
            df.loc[df.index[i], 'kdj_d'] = (2 / 3) * df.loc[df.index[i - 1], 'kdj_d'] + (1 / 3) * df.loc[
                df.index[i], 'kdj_k']

        # Step3：计算J线 - 券商通用公式 3K - 2D
        df['kdj_j'] = 3 * df['kdj_k'] - 2 * df['kdj_d']

        # 清理中间列，避免干扰后续绘图
        df.drop(columns=['n_high', 'n_low', 'rsv'], inplace=True)

        # 适配新版pandas的空值填充（无method参数，直接用bfill()），兼容所有版本
        df['kdj_k'] = df['kdj_k'].bfill().fillna(50.0)
        df['kdj_d'] = df['kdj_d'].bfill().fillna(50.0)
        df['kdj_j'] = df['kdj_j'].bfill().fillna(50.0)
        return df

    # 1. 调用计算函数（默认参数 period=9, k/d_period=3）
    df_with_kdj = calculate_kdj_bt(df, period=9, k_period=3, d_period=3)

    # 关键：将KDJ列添加到原始df中，供悬停回调使用
    df['kdj_k'] = df_with_kdj['kdj_k']
    df['kdj_d'] = df_with_kdj['kdj_d']
    df['kdj_j'] = df_with_kdj['kdj_j']

    # 2. 绘制 K/D/J 线（使用鲜明且专业的颜色）
    df['kdj_k'].plot(ax=ax, legend='K', color='black')
    df['kdj_d'].plot(ax=ax, legend='D', color='orange')
    df['kdj_j'].plot(ax=ax, legend='J', color='red')

    # 5. 可选：添加超买/超卖区域（半透明）
    fplt.add_horizontal_band(20, 80, ax=ax)


def draw_trade_signals(df_plot, ax, buy_signals=None, sell_signals=None):
    """绘制买卖信号 - 兼容周/月线，动态偏移"""
    if buy_signals is None:
        buy_signals = []
    if sell_signals is None:
        sell_signals = []

    # === 新增：关键检查，确保 df_plot 有 date_str 列 ===
    if 'date_str' not in df_plot.columns:
        print("⚠️ Warning: df_plot 缺少 'date_str' 列，无法绘制交易信号！")
        return
    # ===================================================

    if len(df_plot) < 1 or (not buy_signals and not sell_signals):
        return

    price_offset = 0.05

    # 买入信号（↓ 三角形，放在最低价下方）
    if buy_signals:
        buy_x, buy_y = [], []
        for dt_str, _ in buy_signals:
            # 使用 .loc 和布尔索引更安全
            mask = df_plot['date_str'] == dt_str
            if mask.any():
                # 找到匹配行的索引（即 time 时间戳，现在是秒级）
                idx = df_plot.index[mask][0]
                signal_price = df_plot.loc[idx, 'low'] * (1 - price_offset)
                buy_x.append(idx)  # 直接使用秒级索引
                buy_y.append(signal_price)
        if buy_x:
            # ========== 【关键修改】先绘制，再设置Z值 ==========
            plot_item = fplt.plot(buy_x, buy_y, ax=ax, color='#FFD700',
                                  style='v', width=2.5, legend='买入')
            # ====================================================

    # 卖出信号（↑ 三角形，放在最高价上方）
    if sell_signals:
        sell_x, sell_y = [], []
        for dt_str, _ in sell_signals:
            mask = df_plot['date_str'] == dt_str
            if mask.any():
                idx = df_plot.index[mask][0]
                signal_price = df_plot.loc[idx, 'high'] * (1 + price_offset)
                sell_x.append(idx)  # 直接使用秒级索引
                sell_y.append(signal_price)
        if sell_x:
            # ========== 【关键修改】先绘制，再设置Z值 ==========
            plot_item = fplt.plot(sell_x, sell_y, ax=ax, color='#1E90FF',
                                  style='^', width=2.5, legend='卖出')
            # ====================================================


def append_trade_signals(df_plot, ax, signals):
    """
    向图表追加买卖信号（兼容你的绘图系统）

    :param df_plot: DataFrame，必须包含 'date_str' 列（格式 '%Y-%m-%d'）以及 'high', 'low'
    :param ax: finplot 的绘图轴（ax）
    :param signals: list of dict or tuple，例如：
                    - [{'date': '2023-05-10', 'type': 'buy'}, {'date': '2023-06-15', 'type': 'sell'}]
                    - 或 [('2023-05-10', 'buy'), ('2023-06-15', 'sell')]
    """
    if not signals:
        return

    buy_signals = []
    sell_signals = []

    # 解析 signals 格式
    for sig in signals:
        if isinstance(sig, dict):
            date_str = sig['date']
            sig_type = sig.get('type', '').lower()
        elif isinstance(sig, (tuple, list)) and len(sig) >= 2:
            date_str = sig[0]
            sig_type = str(sig[1]).lower()
        else:
            continue  # 跳过无效格式

        if sig_type in ('buy', 'b', '买入'):
            buy_signals.append((date_str, None))
        elif sig_type in ('sell', 's', '卖出'):
            sell_signals.append((date_str, None))

    # 复用你已有的 draw_trade_signals 函数
    draw_trade_signals(df_plot, ax, buy_signals=buy_signals, sell_signals=sell_signals)


# ========== 公共回调函数（可被外部复用或覆盖） ==========
def default_hover_callback(x, y, df, symbol, interval, hover_label, rsi_hover_label, kdj_hover_label, macd_hover_label,
                           vol_hover_label, boll_hover_label, ax, ax3, ax4, ax2, ax1, ax5):
    """
    默认的鼠标悬停提示回调函数（新增RSI+KDJ+MACD+VOL数值显示）。
    """
    ts_sec = int(x // 1_000_000_000)

    # 1. 更新主图K线信息 (保持原有逻辑不变)
    if ts_sec not in df.index:
        if hover_label is not None:
            hover_label.setText('')
        if rsi_hover_label is not None:
            rsi_hover_label.setText('')
        if kdj_hover_label is not None:
            kdj_hover_label.setText('')
        if macd_hover_label is not None:
            macd_hover_label.setText('')
        if vol_hover_label is not None:
            vol_hover_label.setText('')
        if boll_hover_label is not None:  # 新增：清空BOLL标签
            boll_hover_label.setText('')
        return

    row = df.loc[ts_sec]
    color = 'red' if row.open < row.close else 'green'
    kline_txt = (f'<span style="font-size:13px">{symbol} {interval.upper()}</span> '
                 f'开<span style="color:{color}">{row.open:.2f}</span> '
                 f'收<span style="color:{color}">{row.close:.2f}</span> '
                 f'高<span style="color:{color}">{row.high:.2f}</span> '
                 f'低<span style="color:{color}">{row.low:.2f}</span> '
                 f'换<span style="color:{color}">{row.turnover:.2f}%</span>'
                 f'涨幅<span style="color:{color}">{(row.close - row.open) / row.open * 100:.2f}%</span>')

    if hover_label is not None:
        ax_rect = ax.boundingRect()
        hover_label.setPos(ax_rect.width() - 350, 20)
        hover_label.setText(kline_txt)

    # 2. 更新RSI数值信息 (保持原有逻辑不变)
    rsi_parts = []
    for period in [6, 12, 24]:
        col_name = f'rsi_{period}'
        if col_name in df.columns and pd.notna(row[col_name]):
            rsi_val = row[col_name]
            # 根据RSI值设置颜色（超买红，超卖绿，正常黑）
            if rsi_val > 70:
                rsi_color = 'red'
            elif rsi_val < 30:
                rsi_color = 'green'
            else:
                rsi_color = 'black'
            rsi_parts.append(f'RSI{period}: <span style="color:{rsi_color}">{rsi_val:.1f}</span>')

    rsi_txt = ' | '.join(rsi_parts) if rsi_parts else ''
    if rsi_hover_label is not None:
        ax3_rect = ax3.boundingRect()
        rsi_hover_label.setPos(ax3_rect.width() - 200, 20)
        rsi_hover_label.setText(rsi_txt)

    # 3. 更新KDJ数值信息 (保持原有逻辑不变)
    kdj_parts = []
    for col in ['kdj_k', 'kdj_d', 'kdj_j']:
        if col in df.columns and pd.notna(row[col]):
            kdj_val = row[col]
            # 根据KDJ值设置颜色（超买红，超卖绿，正常黑）
            if kdj_val > 80:
                kdj_color = 'red'
            elif kdj_val < 20:
                kdj_color = 'green'
            else:
                kdj_color = 'black'
            kdj_parts.append(f'{col.upper()}: <span style="color:{kdj_color}">{kdj_val:.1f}</span>')

    kdj_txt = ' | '.join(kdj_parts) if kdj_parts else ''
    if kdj_hover_label is not None:
        ax4_rect = ax4.boundingRect()
        kdj_hover_label.setPos(ax4_rect.width() - 200, 20)
        kdj_hover_label.setText(kdj_txt)

    # 4. 修复：更新MACD数值信息（读取预计算的列，不再重复计算）
    macd_parts = []
    # 检查df是否包含预计算的MACD列
    if all(col in df.columns for col in ['macd_line', 'signal_line', 'macd_diff']):
        # 直接读取提前计算好的值
        macd_line = row['macd_line']
        signal_line = row['signal_line']
        macd_hist = row['macd_diff']

        # 过滤NaN值，避免显示无效数据
        if pd.notna(macd_line) and pd.notna(signal_line) and pd.notna(macd_hist):
            # MACD颜色规则：柱状线为正（红），为负（绿），零轴（黑）
            macd_color = 'red' if macd_hist > 0 else 'green' if macd_hist < 0 else 'black'
            macd_parts.append(f'DIFF: <span style="color:{macd_color}">{macd_line:.3f}</span>')
            macd_parts.append(f'DEA: <span style="color:{macd_color}">{signal_line:.3f}</span>')
            macd_parts.append(f'HIST: <span style="color:{macd_color}">{macd_hist:.3f}</span>')
    macd_txt = ' | '.join(macd_parts) if macd_parts else ''
    if macd_hover_label is not None:
        ax2_rect = ax2.boundingRect()
        macd_hover_label.setPos(ax2_rect.width() - 200, 20)
        macd_hover_label.setText(macd_txt)

    # ========== 5. 新增：更新成交量数值信息 ==========
    vol_parts = []
    current_vol = row['volume']
    # 格式化当前成交量（假设单位是手，转换为万手显示）
    vol_parts.append(f'VOL: <span style="color:blue">{current_vol / 10000:.0f}万</span>')

    # 计算并格式化VOL5 (5日均量线)
    if len(df.loc[:ts_sec]) >= 5:
        vol5 = df['volume'].loc[:ts_sec].iloc[-5:].mean()
        vol_parts.append(f'VOL5: <span style="color:orange">{vol5 / 10000:.0f}万</span>')

    # 计算并格式化VOL10 (10日均量线)
    if len(df.loc[:ts_sec]) >= 10:
        vol10 = df['volume'].loc[:ts_sec].iloc[-10:].mean()
        vol_parts.append(f'VOL10: <span style="color:green">{vol10 / 10000:.0f}万</span>')

    vol_txt = ' | '.join(vol_parts)
    if vol_hover_label is not None:
        ax1_rect = ax1.boundingRect()
        vol_hover_label.setPos(ax1_rect.width() - 200, 20)
        vol_hover_label.setText(vol_txt)

    # ========== 6. 新增：更新布林带(BOLL)数值信息 ==========
    # ========== 6. 更新布林带(BOLL)与乖离率(BIAS)数值信息 ==========
    boll_txt = ''
    # 检查所有需要的列是否存在（包括BIAS列）
    bias_cols = ['bias6', 'bias12', 'bias24']
    if all(col in df.columns for col in ['boll_hi', 'boll_lo', 'boll_mid'] + bias_cols):
        row = df.loc[ts_sec]
        # 确保中轨值有效
        if pd.notna(row['boll_mid']):
            # 定义颜色方案
            colors = {
                'mid': 'blue',
                'up': 'red',
                'lo': 'green',
                'bias6': 'magenta',  # 洋红
                'bias12': 'orange',  # 橙色
                'bias24': 'purple'  # 紫色
            }

            # 格式化数值
            val = {
                'mid': row['boll_mid'],
                'up': row['boll_hi'],
                'lo': row['boll_lo'],
                'b6': row['bias6'],
                'b12': row['bias12'],
                'b24': row['bias24']
            }

            # 构建显示文本：BOLL参数 + BIAS参数
            # 格式：MID: xx | UP: xx | LOW: xx | B6: x% | B12: x% | B24: x%
            boll_txt = (f'MID: <span style="color:{colors["mid"]}">{val["mid"]:.2f}</span> | '
                        f'UP: <span style="color:{colors["up"]}">{val["up"]:.2f}</span> | '
                        f'LOW: <span style="color:{colors["lo"]}">{val["lo"]:.2f}</span> | '
                        f'B6: <span style="color:{colors["bias6"]}">{val["b6"]:+.2f}</span> | '
                        f'B12: <span style="color:{colors["bias12"]}">{val["b12"]:+.2f}</span> | '
                        f'B24: <span style="color:{colors["bias24"]}">{val["b24"]:+.2f}</span>')

    if boll_hover_label is not None:
        ax5_rect = ax5.boundingRect()
        boll_hover_label.setPos(ax5_rect.width() - 350, 20)  # 宽度稍微调宽一点，因为内容变多了
        boll_hover_label.setText(boll_txt)


def default_crosshair_callback(x, y, xtext, ytext, df):
    """
    默认的十字线信息回调函数。
    可被外部替换以自定义Y轴显示。
    """
    ts_sec = int(x // 1_000_000_000)
    if ts_sec in df.index:
        close_price = df.loc[ts_sec, 'close']
        ytext = '%s (close%+.2f)' % (ytext, (y - close_price))
    return xtext, ytext


def create_ctrl_panel(win):
    panel = QWidget(win)
    panel.move(100, 0)
    win.scene().addWidget(panel)
    layout = QGridLayout(panel)

    panel.symbol = QComboBox(panel)
    [panel.symbol.addItem(i + 'USDT') for i in 'BTC ETH XRP DOGE BNB SOL ADA LTC LINK DOT TRX BCH'.split()]
    panel.symbol.setCurrentIndex(1)
    layout.addWidget(panel.symbol, 0, 0)
    panel.symbol.currentTextChanged.connect(change_asset)

    layout.setColumnMinimumWidth(1, 30)

    panel.interval = QComboBox(panel)
    [panel.interval.addItem(i) for i in '1d 4h 1h 30m 15m 5m 1m 1s'.split()]
    panel.interval.setCurrentIndex(6)
    layout.addWidget(panel.interval, 0, 2)
    panel.interval.currentTextChanged.connect(change_asset)

    layout.setColumnMinimumWidth(3, 30)

    panel.indicators = QComboBox(panel)
    [panel.indicators.addItem(i) for i in 'Clean:Few indicators:Moar indicators'.split(':')]
    panel.indicators.setCurrentIndex(1)
    layout.addWidget(panel.indicators, 0, 4)
    panel.indicators.currentTextChanged.connect(change_asset)

    layout.setColumnMinimumWidth(5, 30)

    panel.darkmode = QCheckBox(panel)
    panel.darkmode.setText('Haxxor mode')
    panel.darkmode.setCheckState(pg.Qt.QtCore.Qt.CheckState.Checked)
    panel.darkmode.toggled.connect(dark_mode_toggle)
    layout.addWidget(panel.darkmode, 0, 6)

    return panel


# ========== 核心公共API（修改为不自动show） ==========
def plot_a_stock_analysis(
        df,
        symbol='A股',
        title_suffix='',
        signals=None,
        hover_callback=None,  # 新增参数
        crosshair_callback=None,  # 新增参数
        is_weekly=False,
        show_immediately=False  # 新增：是否立即显示窗口（用于异步控制），默认False
):
    """
    绘制完整的A股多指标分析图（新增RSI+KDJ+MACD动态数值显示）。
    【修改】不再自动调用 fplt.show()，等待最后的统一调用

    参数:
        df (pd.DataFrame): 必须包含列 ['open', 'high', 'low', 'close', 'volume']，且索引为时间戳（秒）。
                           还需包含 'date_str' 列（格式 '%Y-%m-%d'），用于信号绘制。
        symbol (str): 股票代码，用于标题和悬停提示。
        title_suffix (str): 标题后缀，可选。
        signals (list): 交易信号列表，例如：
                        - [{'date': '2023-05-10', 'type': 'buy'}, {'date': '2023-06-15', 'type': 'sell'}]
        hover_callback (callable): 自定义悬停回调，签名为 func(x, y, df, symbol, interval, hover_label, rsi_hover_label, kdj_hover_label, macd_hover_label, ax, ax3, ax4, ax2)
        crosshair_callback (callable): 自定义十字线回调，签名为 func(x, y, xtext, ytext, df)
        is_weekly (bool): 是否为周线图（用于调整标题和周期显示）
        show_immediately (bool): 是否立即显示（默认False，等待最后统一显示）
    """
    interval = 'w' if is_weekly else 'd'  # 周线/日线标识

    # change to b/w coloring templates for next plots
    fplt.candle_bull_color = fplt.volume_bull_color = 'red'  # K线/成交量阳线边框颜色
    fplt.candle_bull_body_color = fplt.volume_bull_body_color = 'white'  # K线/成交量阳线实体颜色
    fplt.candle_bear_color = fplt.candle_bear_body_color = 'green'
    fplt.volume_bear_color = fplt.volume_bear_body_color = 'green'
    fplt.legend_text_color = 'black'
    fplt.legend_background_color = 'gray'

    # 调整标题：区分日线/周线
    period_label = '周线' if is_weekly else '日线'
    full_title = f'A股 {symbol} {period_label} 平均K线图'
    if title_suffix:
        full_title += f' - {title_suffix}'

    # 关键：创建新窗口（每个plot调用创建独立窗口）
    ax, ax1, ax2, ax3, ax4, ax5, ax6, ax7 = fplt.create_plot(full_title, rows=8,
                                                             init_zoom_periods=50 if is_weekly else 100)
    # 不显示第一个ax的网格线
    ax.set_visible(xgrid=False, ygrid=False)

    # ========== 核心修改：创建独立的MA图例容器 ==========
    create_legend(
        ax=ax,
        name="ma_legend",
        pos="top_left",
        size=(120, 50),
        bg_color=fplt.legend_background_color,
        text_color=fplt.legend_text_color
    )

    # ========== 主函数中：创建所有悬停标签 ==========
    # 主图K线悬停标签
    hover_label = fplt.add_legend('', ax=ax)
    hover_label.setPos(800, 20)
    hover_label.setZValue(1000)

    # RSI子图悬停标签
    rsi_hover_label = fplt.add_legend('', ax=ax3)
    rsi_hover_label.setPos(ax3.boundingRect().width() - 200, 20)
    rsi_hover_label.setZValue(1000)

    # KDJ子图悬停标签
    kdj_hover_label = fplt.add_legend('', ax=ax4)
    kdj_hover_label.setPos(ax4.boundingRect().width() - 200, 20)
    kdj_hover_label.setZValue(1000)

    # MACD子图悬停标签
    macd_hover_label = fplt.add_legend('', ax=ax2)
    macd_hover_label.setPos(ax2.boundingRect().width() - 200, 20)
    macd_hover_label.setZValue(1000)

    # ========== 新增：成交量子图悬停标签 ==========
    vol_hover_label = fplt.add_legend('', ax=ax1)
    vol_hover_label.setPos(ax1.boundingRect().width() - 200, 20)
    vol_hover_label.setZValue(1000)

    # ========== 新增：布林带(副图ax5)悬停标签 ==========
    boll_hover_label = fplt.add_legend('', ax=ax5)
    boll_hover_label.setPos(ax5.boundingRect().width() - 250, 20)
    boll_hover_label.setZValue(1000)

    ####### 图层顺序 #######
    # 1.  K线图 (主图)
    # 2.  成交量 (Volume) 及其均量线
    # 3.  MACD (趋势与动能)
    # 4.  RSI (超买超卖)
    # 5.  KDJ (短线买卖信号)
    # 6.  OBV (能量潮，资金流)
    # 7.  布林带 (Bollinger Bands) 或 EMA (短期趋势)
    # price chart

    plot_candlestick(df, ax=ax)
    plot_ma(df, ax)

    plot_heikin_ashi(df, ax5)

    # 注意：必须确保 plot_bollinger_bands 在 plot_ema 之前调用
    # 或者确保 plot_bollinger_bands 内部计算了 df['boll_mid'] 等列
    plot_bollinger_bands(df, ax5)
    plot_ema(df, ax5)

    # volume chart
    plot_heikin_ashi_volume(df, ax1)
    plot_volume(df, ax1)
    plot_vma(df, ax=ax1)

    plot_macd(df, ax2)  # MACD计算后会自动给df添加macd_diff列
    plot_rsi(df, ax3)
    plot_kdj(df, ax4)  # KDJ计算后会自动给df添加kdj_k/kdj_d/kdj_j列

    # some more charts
    plot_accumulation_distribution(df, ax7)
    plot_on_balance_volume(df, ax6)

    # ========== 【关键新增】添加交易信号 ==========
    if signals is not None:
        append_trade_signals(df_plot=df, ax=ax, signals=signals)

    # ========== 设置回调 ==========
    final_hover_cb = hover_callback or default_hover_callback
    final_crosshair_cb = crosshair_callback or default_crosshair_callback

    # 先执行：计算并绘图
    plot_bollinger_bands(df, ax5)  # <--- 这一步会把 boll_mid, boll_hi, boll_lo 写入 df

    # 包装回调函数，注意这里增加了 vol_hover_label 和 ax1
    def wrapped_hover(x, y):
        return final_hover_cb(x, y, df, symbol, interval, hover_label, rsi_hover_label, kdj_hover_label,
                              macd_hover_label, vol_hover_label, boll_hover_label, ax, ax3, ax4, ax2, ax1, ax5)

    def wrapped_crosshair(x, y, xtext, ytext):
        return final_crosshair_cb(x, y, xtext, ytext, df)

    # 绑定鼠标回调到主图 ax，这样移动鼠标时所有子图数据都能更新
    fplt.set_mouse_callback(wrapped_hover, ax=ax, when='hover')
    fplt.add_crosshair_info(wrapped_crosshair, ax=ax)

    fplt.autoviewrestore()

    # 关键：根据参数控制是否立即显示窗口
    if show_immediately:
        fplt.show(qt_exec=False)  # qt_exec=False：不阻塞Qt事件循环


# ========== 空函数（避免运行时错误） ==========
def change_asset(*args):
    pass


def dark_mode_toggle(*args):
    pass


# ========== 新增：异步绘制日线图的函数 ==========
def plot_daily_chart_async(df_daily, symbol, signals):
    """异步绘制日线图（用于QTimer调用）"""
    plot_a_stock_analysis(
        df=df_daily,
        symbol=symbol,
        title_suffix='日线图',
        signals=signals,
        is_weekly=False,
        show_immediately=True
    )


# ========== 主函数：使用 finplot 原生 show() ==========
def main():
    symbol = '002738'

    # 1. 设置Qt属性（不创建App实例，不启动事件循环）
    init_qt_attributes()

    # 2. 下载日线数据
    df_daily = download_price_history(
        symbol=symbol,
        start_time='2014-12-30',  # 你的数据起始日期
        end_time='2026-01-01'  # 数据结束日期
    )
    # 新增：添加date_str列（draw_trade_signals需要）
    df_daily['date_str'] = pd.to_datetime(df_daily.index, unit='s').strftime('%Y-%m-%d')

    # 2. 转换为周线数据
    df_weekly = resample_to_weekly(df_daily)

    # 3. 定义交易信号示例（可选）
    sample_signals = [
        {'date': '2023-01-20', 'type': 'buy'},
        {'date': '2023-06-30', 'type': 'sell'},
        {'date': '2024-03-15', 'type': 'buy'}
    ]

    # 4. 绘制周线图（不立即显示）
    print("正在创建周线图...")
    plot_a_stock_analysis(
        df=df_weekly,
        symbol=symbol,
        title_suffix='周线图',
        signals=sample_signals,
        is_weekly=True,
        show_immediately=False
    )

    # 5. 绘制日线图（不立即显示）
    print("正在创建日线图...")
    plot_a_stock_analysis(
        df=df_daily,
        symbol=symbol,
        title_suffix='日线图',
        signals=sample_signals,
        is_weekly=False,
        show_immediately=False
    )

    print("两个窗口都已创建，正在显示...")

    # ========== 【关键】使用 finplot 原生的 show() ==========
    # 这会显示所有窗口，并阻塞直到所有窗口都关闭
    fplt.show()

    print("所有窗口已关闭，程序退出")
    sys.exit(0)


# 如果直接运行此文件，则执行示例（保留原有逻辑）
if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(f"发生错误: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)