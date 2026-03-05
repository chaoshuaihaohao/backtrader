from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import math  # 补上math依赖，解决isnan报错
from . import Indicator  # 复用backtrader indicator基础类


class _OBVBase(Indicator):
    """OBV基础类-终极修复：解决空数据/首根K线/成交量为0等边界问题
    完全对齐A股券商版OBV计算逻辑，适配所有数据场景"""
    lines = ('OBV',)  # 核心指标线：OBV值
    params = (
        ('initial_value', 0.0),  # 首根K线OBV初始值（券商默认=当日成交量，设0也可）
        ('safe_vol', 0.0),       # 成交量为空时的兜底值
        ('plot_ma', False),      # 是否绘制OBV均线（可选）
        ('ma_period', 5),        # OBV均线周期（默认5日）
    )

    plotlines = dict(
        OBV=dict(
            _name='OBV',
            color='orange',
            linewidth=1.3,
            alpha=0.9,
            _fill_gt=('OBV_ma', 'lightgreen'),  # 若开启均线，OBV上穿均线填充绿色
            _fill_lt=('OBV_ma', 'lightcoral')    # OBV下穿均线填充珊瑚色
        ),
        OBV_ma=dict(  # 仅当plot_ma=True时显示
            _name='OBV_MA5',
            color='purple',
            ls='--',
            linewidth=1.0,
            alpha=0.8
        )
    )

    def _plotlabel(self):
        """绘图标签：显示参数"""
        labels = []
        if self.p.plot_ma:
            labels.append(f"OBV-MA{self.p.ma_period}")
        return labels

    def __init__(self):
        """初始化：可选添加OBV均线（和KDJ的初始化风格一致）"""
        super(_OBVBase, self).__init__()
        # 若开启均线，动态添加OBV_ma线（不影响核心计算）
        if self.p.plot_ma:
            self.lines.OBV_ma = self.lines.OBV.rolling(window=self.p.ma_period).mean()

    def next(self):
        """核心计算逻辑：严格遵循A股券商OBV算法，处理所有边界"""
        # 核心修复1：数据兜底（解决空列表/数据缺失报错）
        current_close = self.data.close[0] if len(self.data.close) > 0 else 0.0
        current_vol = self.data.volume[0] if len(self.data.volume) > 0 else self.p.safe_vol

        # 核心修复2：首根K线处理（无历史数据时初始化）
        if len(self) == 1:
            # 首根K线OBV：默认=当日成交量（券商标准），也可设为initial_value
            obv_current = current_vol if current_vol != 0 else self.p.initial_value
            self.lines.OBV[0] = obv_current
            return

        # 核心修复3：获取历史数据（兜底处理，防止索引越界）
        prev_close = self.data.close[-1] if len(self.data.close) > 1 else current_close
        prev_obv = self.lines.OBV[-1] if len(self.lines.OBV) > 1 else self.p.initial_value

        # OBV核心算法（A股券商标准）
        if current_close > prev_close:
            # 收盘价上涨 → OBV = 前值 + 当日成交量
            obv_current = prev_obv + current_vol
        elif current_close < prev_close:
            # 收盘价下跌 → OBV = 前值 - 当日成交量
            obv_current = prev_obv - current_vol
        else:
            # 收盘价不变 → OBV = 前值
            obv_current = prev_obv

        # 赋值最终OBV值（确保数值有效，非NaN）
        self.lines.OBV[0] = obv_current if not math.isnan(obv_current) else self.p.initial_value


class OBV(_OBVBase):
    """
    A股券商版标准OBV（能量潮）指标
    适配backtrader所有版本，解决空数据/首根K线/成交量为0等报错问题
    使用方式（和KDJ完全一致）：
    1. 放入backtrader/indicators/目录下（命名为obv.py）
    2. 策略中调用：self.obv = bt.indicators.OBV(self.data)
    3. 可选开启均线：self.obv = bt.indicators.OBV(self.data, plot_ma=True, ma_period=5)
    """

    def __init__(self):
        """简单初始化，复用基础类逻辑（和KDJ子类风格一致）"""
        super(OBV, self).__init__()


# ===================== 核心：注册别名 OnBalanceVolume =====================
# 让 bt.indicators.OnBalanceVolume 指向我们实现的 OBV 类，兼容原有调用代码
import backtrader.indicators as btind
# 注册别名，覆盖/补充 indicators 模块的属性
btind.OnBalanceVolume = OBV
btind.onbalancevolume = OBV  # 小写别名（可选，兼容不同命名习惯）