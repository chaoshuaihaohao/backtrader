from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

from . import Indicator, DivByZero


class AD(Indicator):
    '''
    Chaikin Accumulation/Distribution 指标 (A/D)

    由 Marc Chaikin 开发，用于衡量资金流入和流出的指标。

    **核心逻辑**:
        - 计算一个乘数 (Multiplier)，反映收盘价在高低点范围内的位置。
        - 将乘数与成交量相乘，得到资金流。
        - 对资金流进行累加，形成 A/D 线。

    **计算公式**:
        Multiplier = ((Close - Low) - (High - Close)) / (High - Low)
                   = (2 * Close - High - Low) / (High - Low)
        MoneyFlowVolume = Multiplier * Volume
        AD = AD_prev + MoneyFlowVolume

    **信号解读**:
        - A/D 与价格的**背离 (Divergence)** 是最强的信号。
        - 价格创新高，A/D 未创新高 -> 可能见顶 (卖出信号)。
        - 价格创新低，A/D 未创新低 -> 可能见底 (买入信号)。

    注意：
        - 当 High == Low 时，分母为 0。此时我们采用安全除法，将 Multiplier 设为 0。
    '''
    alias = ('AccumDist', 'AccumulationDistribution')
    lines = ('ad',)
    params = (
        ('safediv', True),  # 除零保护，当 High==Low 时
        ('safeval', 0.0),  # High==Low 时的 Multiplier 值
    )

    plotlines = dict(
        ad=dict(color='teal', linewidth=1.0, alpha=0.8),
    )

    def __init__(self):
        # 可以在这里初始化一些通用变量，但主要计算在 next 中
        super(AD, self).__init__()

    def nextstart(self):
        """
        初始化第一根 K 线的值。
        通常设为 0.0。
        """
        self.lines.ad[0] = 0.0
        return True

    def next(self):
        # 获取当前的 High, Low, Close, Volume
        high = self.data.high[0]
        low = self.data.low[0]
        close = self.data.close[0]
        volume = self.data.volume[0]

        # 计算分母 (Range)
        price_range = high - low

        # 安全除法处理
        if price_range == 0:
            # 如果最高价等于最低价，无法计算乘数
            multiplier = self.p.safeval
        else:
            # 计算 Chaikin Multiplier
            # 公式：(2 * Close - High - Low) / (High - Low)
            multiplier = (2 * close - high - low) / price_range

        # 计算当前的资金流体积
        money_flow_volume = multiplier * volume

        # 累加逻辑
        # 获取前一个 A/D 值
        ad_prev = self.lines.ad[-1]

        # 当前 A/D = 前值 + 当前资金流
        ad_current = ad_prev + money_flow_volume

        # 赋值
        self.lines.ad[0] = ad_current


# 兼容性别名
class AccumDist(AD):
    '''Accumulation/Distribution 的别名'''
    pass