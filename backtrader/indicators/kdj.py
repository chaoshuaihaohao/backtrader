from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

from . import Indicator, Max, MovAv, Highest, Lowest, DivByZero


class _KDJBase(Indicator):
    """KDJ基础类-终极修复：解决空列表max()报错+兼容所有数据场景"""
    lines = ('K', 'D', 'J', 'percK', 'percD',)
    params = (
        ('period', 9),  # KDJ核心周期
        ('period_dfast', 3),  # 兼容参数-无实际作用
        ('period_dslow', 3),  # 兼容参数-无实际作用
        ('movav', None),  # 兼容参数-无实际作用
        ('upperband', 80.0),  # 超买线
        ('lowerband', 20.0),  # 超卖线
        ('safe_div', True),  # 除零保护
        ('safe_val', 50.0)  # 除零时RSV=50
    )

    plotlines = dict(
        K=dict(_name='K', color='red', linewidth=1.5, alpha=0.9),
        D=dict(_name='D', color='blue', ls='--', linewidth=1.2, alpha=0.9),
        J=dict(_name='J', color='green', ls=':', linewidth=1.2, alpha=0.9),
        percK=dict(_plotskip=True),
        percD=dict(_plotskip=True),
    )

    def _plotlabel(self):
        return [self.p.period, self.p.period_dfast, self.p.period_dslow]

    def _plotinit(self):
        self.plotinfo.plotyhlines = [self.p.upperband, self.p.lowerband]

    def next(self):
        # 核心修复1：先获取当前收盘价，作为空列表时的兜底值
        close = self.data.close[0] if len(self.data.close) > 0 else 50.0

        # 核心修复2：获取高低价列表，为空则兜底为[close]
        high_list = self.data.high.get(size=self.p.period) or [close]
        low_list = self.data.low.get(size=self.p.period) or [close]

        # 手动计算N周期最高/最低值（此时列表必非空，彻底解决max()空值错误）
        n_high = max(high_list)
        n_low = min(low_list)

        # 计算RSV，除零保护（高低价相同时RSV=50，券商标准）
        if n_high == n_low:
            rsv = self.p.safe_val
        else:
            rsv = 100.0 * (close - n_low) / (n_high - n_low)

        # 券商KDJ核心递推：K/D初始值50，后续2/3前值+1/3当前值
        if len(self) == 1:
            # 第一根K线：初始化K/D=50
            k_current = 50.0
            d_current = 50.0
        else:
            # 后续K线：加权递推
            k_current = (2 / 3) * self.lines.K[-1] + (1 / 3) * rsv
            d_current = (2 / 3) * self.lines.D[-1] + (1 / 3) * k_current

        # 计算J线：3K-2D（券商通用公式）
        j_current = 3 * k_current - 2 * d_current

        # 赋值所有指标线（含兼容percK/percD）
        self.lines.K[0] = k_current
        self.lines.D[0] = d_current
        self.lines.J[0] = j_current
        self.lines.percK[0] = k_current
        self.lines.percD[0] = d_current


class KDJ(_KDJBase):
    """
    A股券商版标准KDJ（终极修复：解决空列表max()报错+无额外导入+完全适配策略）
    放入backtrader/indicators/，可通过bt.indicators.KDJ调用
    test.py无需任何修改，直接运行
    """

    def __init__(self):
        super(KDJ, self).__init__()