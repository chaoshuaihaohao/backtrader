import backtrader as bt
import talib


# 1. 在类外部定义修复函数，确保类定义时可以访问
def _fix_talib_name(name):
    raw_name = name[3:].lower()
    if raw_name[0].isdigit():
        return f"p{raw_name}"
    return raw_name


class KLineAnalyzer(bt.Indicator):
    """
    全量K线形态分析指示器 (修复 NameError 作用域问题)
    """
    # 基础汇总线
    lines = ('is_bullish', 'is_bearish', 'is_neutral',)

    # 2. 获取所有 TA-Lib 形态名并处理命名
    _cdl_names = [name for name in dir(talib) if name.startswith('CDL')]

    # 使用外部函数生成 lines 数组
    lines = lines + tuple(_fix_talib_name(name) for name in _cdl_names)

    def __init__(self):
        o, h, l, c = self.data.open, self.data.high, self.data.low, self.data.close

        for func_name in self._cdl_names:
            # 同样使用外部函数获取一致的线名
            line_name = _fix_talib_name(func_name)

            # 实例化具体形态指标
            talib_ind = getattr(bt.talib, func_name)(o, h, l, c)

            # 绑定到 self.l (lines)
            setattr(self.l, line_name, talib_ind)

    def next(self):
        # 排除汇总线，获取形态线数据
        # 索引从 3 开始，跳过 is_bullish, is_bearish, is_neutral
        pattern_names = self.lines.getlinealiases()[3:]
        pattern_values = [getattr(self.l, name)[0] for name in pattern_names]

        # 统计
        has_bull = any(v > 0 for v in pattern_values)
        has_bear = any(v < 0 for v in pattern_values)

        self.l.is_bullish[0] = 1.0 if has_bull else 0.0
        self.l.is_bearish[0] = 1.0 if has_bear else 0.0
        self.l.is_neutral[0] = 1.0 if not (has_bull or has_bear) else 0.0

    def get_active_patterns(self, direction='bull'):
        """获取当前触发的具体形态名"""
        active = []
        pattern_names = self.lines.getlinealiases()[3:]
        for name in pattern_names:
            val = getattr(self.l, name)[0]
            if direction == 'bull' and val > 0:
                active.append(name)
            elif direction == 'bear' and val < 0:
                active.append(name)
        return active