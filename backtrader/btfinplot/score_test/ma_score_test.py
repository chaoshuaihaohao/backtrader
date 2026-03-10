import backtrader as bt
import unittest
import math


def _check_ma_divergence(strat, divergence_type):
    """
    独立的MA背离检测函数（完全使用Backtrader原生数据访问）
    :param strat: 策略对象（Backtrader Strategy实例）
    :param divergence_type: 背离类型 '底背离'/'顶背离'
    :return: True/False
    """
    try:
        # 检查数据长度是否足够
        if len(strat.data) < 10:
            return False

        # Backtrader原生索引规则：[0]=当前bar, [-1]=上一个bar, [-5]=5个bar前
        # 获取关键价格数据（原生格式）
        low_0 = strat.data.low[0]  # 当前最低价
        low_5 = strat.data.low[-5]  # 5个bar前最低价
        low_10 = strat.data.low[-10]  # 10个bar前最低价

        high_0 = strat.data.high[0]  # 当前最高价
        high_5 = strat.data.high[-5]  # 5个bar前最高价
        high_10 = strat.data.high[-10]  # 10个bar前最高价

        # 获取MA60数据（原生格式）
        ma60_0 = strat.ma60[0]  # 当前MA60
        ma60_5 = strat.ma60[-5]  # 5个bar前MA60
        ma60_10 = strat.ma60[-10]  # 10个bar前MA60

        # 严格匹配背离规则
        if divergence_type == '底背离':
            # 股价新低，均线不新低
            return (low_0 < low_5) and (low_0 < low_10) and (ma60_0 > ma60_5) and (ma60_0 > ma60_10)
        elif divergence_type == '顶背离':
            # 股价新高，均线不新高
            return (high_0 > high_5) and (high_0 > high_10) and (ma60_0 < ma60_5) and (ma60_0 < ma60_10)
        return False
    except Exception as e:
        print(f"MA背离检测异常: {e}")
        return False

# ===================== 第一步：定义 MA 规则表（Table-Driven）=====================
# 优先级数字越小，优先级越高
MA_RULES = [
    {
        "priority": 1,
        "signal_name": "股价下跌并跌破MA5",
        "signal_type": "趋势反转",
        "weight": 1.0,
        "condition_func": lambda strat: strat.data.close[-1] > strat.data.close[0] and
            strat.data.close[0] < strat.ma5[0],
        "buy_score": 0.0,
        "sell_score": 2.0
    },
    {
        "priority": 1,
        "signal_name": "股价不破MA5",
        "signal_type": "趋势反转",
        "weight": 1.0,
        "condition_func": lambda strat: strat.data.close[-1] > strat.data.close[0] and
            strat.data.close[0] > strat.ma5[0],
        "buy_score": 1.0,
        "sell_score": 1.0
    },
    # ---------------- 优先级 1：趋势反转/背离 (权重最高) ----------------
    # {
    #     "priority": 1,
    #     "signal_name": "底背离",
    #     "signal_type": "趋势反转",
    #     "weight": 2.0,
    #     "condition_func": lambda strat: _check_ma_divergence(strat, '底背离'),
    #     "buy_score": 2.0,
    #     "sell_score": 0.0
    # },
    # {
    #     "priority": 1,
    #     "signal_name": "顶背离",
    #     "signal_type": "趋势反转",
    #     "weight": 2.0,
    #     "condition_func": lambda strat: _check_ma_divergence(strat, '顶背离'),
    #     "buy_score": 0.0,
    #     "sell_score": 2.0
    # },
    #
    # # ---------------- 优先级 2：趋势启动/结束 (放量突破) ----------------
    # {
    #     "priority": 2,
    #     "signal_name": "放量突破长期均线",
    #     "signal_type": "趋势启动",
    #     "weight": 1.8,
    #     "condition_func": lambda strat: (
    #         len(strat.data) >= 2 and
    #         (strat.data.close > strat.ma60) and
    #         (strat.data.close[-1] > strat.ma60[-1]) and
    #         (strat.volume >= 1.5 * strat.vol5)
    #     ),
    #     "buy_score": 1.8,
    #     "sell_score": 0.0
    # },
    # {
    #     "priority": 2,
    #     "signal_name": "放量跌破长期均线",
    #     "signal_type": "趋势结束",
    #     "weight": 1.8,
    #     "condition_func": lambda strat: (
    #         len(strat.data) >= 2 and
    #         (strat.data.close < strat.ma60) and
    #         (strat.data.close[-1] < strat.ma60[-1]) and
    #         (strat.volume > 1.5 * strat.vol5)
    #     ),
    #     "buy_score": 0.0,
    #     "sell_score": 1.8
    # },
    #
    # # ---------------- 优先级 3：金叉/死叉/形态 (动态信号) ----------------
    # {
    #     "priority": 3,
    #     "signal_name": "黄金交叉",
    #     "signal_type": "金叉/死叉",
    #     "weight": 1.5,
    #     "condition_func": lambda strat: (
    #         (strat.ma5 > strat.ma10 and strat.ma5[-1] < strat.ma10[-1]) or
    #         (strat.ma10 > strat.ma60 and strat.ma10[-1] < strat.ma60[-1])
    #     ),
    #     "buy_score": 1.5,
    #     "sell_score": 0.0
    # },
    # {
    #     "priority": 3,
    #     "signal_name": "死亡交叉",
    #     "signal_type": "金叉/死叉",
    #     "weight": 1.5,
    #     "condition_func": lambda strat: (
    #         (strat.ma5 < strat.ma10 and strat.ma5[-1] > strat.ma10[-1]) or
    #         (strat.ma10 < strat.ma60 and strat.ma10[-1] > strat.ma60[-1])
    #     ),
    #     "buy_score": 0.0,
    #     "sell_score": 1.5
    # },
    # {
    #     "priority": 3,
    #     "signal_name": "回踩MA60企稳",
    #     "signal_type": "支撑/压力",
    #     "weight": 1.5,
    #     "condition_func": lambda strat: (
    #         (strat.data.low >= strat.ma60 * 0.99) and
    #         (strat.data.close > strat.data.open) and
    #         (strat.volume < 0.8 * strat.vol5)
    #     ),
    #     "buy_score": 1.5,
    #     "sell_score": 0.0
    # },
    # {
    #     "priority": 3,
    #     "signal_name": "反弹遇MA60回落",
    #     "signal_type": "支撑/压力",
    #     "weight": 1.5,
    #     "condition_func": lambda strat: (
    #         (strat.data.high <= strat.ma60 * 1.01) and
    #         (strat.data.close < strat.data.open) and
    #         (strat.volume > 1.2 * strat.vol5)
    #     ),
    #     "buy_score": 0.0,
    #     "sell_score": 1.5
    # },
    #
    # # ---------------- 优先级 4：排列/位置 (静态趋势) ----------------
    # {
    #     "priority": 4,
    #     "signal_name": "标准多头排列",
    #     "signal_type": "排列",
    #     "weight": 1.2,
    #     "condition_func": lambda strat: (
    #         (strat.ma5 > strat.ma10 > strat.ma20 > strat.ma60) and
    #         (strat.ma60 > strat.ma60[-1])
    #     ),
    #     "buy_score": 1.2,
    #     "sell_score": 0.3
    # },
    # {
    #     "priority": 4,
    #     "signal_name": "标准空头排列",
    #     "signal_type": "排列",
    #     "weight": 1.2,
    #     "condition_func": lambda strat: (
    #         (strat.ma5 < strat.ma10 < strat.ma20 < strat.ma60) and
    #         (strat.ma60 < strat.ma60[-1])
    #     ),
    #     "buy_score": 0.3,
    #     "sell_score": 1.2
    # },
    # {
    #     "priority": 4,
    #     "signal_name": "股价在MA60上方",
    #     "signal_type": "位置",
    #     "weight": 1.0,
    #     "condition_func": lambda strat: (strat.data.close > strat.ma60),
    #     "buy_score": 1.0,
    #     "sell_score": 0.3
    # },
    # {
    #     "priority": 4,
    #     "signal_name": "股价在MA60下方",
    #     "signal_type": "位置",
    #     "weight": 1.0,
    #     "condition_func": lambda strat: (strat.data.close < strat.ma60),
    #     "buy_score": 0.3,
    #     "sell_score": 1.0
    # },
    #
    # # ---------------- 优先级 5：粘合/震荡 (低信号) ----------------
    # {
    #     "priority": 5,
    #     "signal_name": "均线粘合收敛",
    #     "signal_type": "震荡",
    #     "weight": 0.5,
    #     "condition_func": lambda strat: (
    #         abs(strat.ma5 - strat.ma10) < 1 and
    #         abs(strat.ma10 - strat.ma20) < 1 and
    #         abs(strat.ma20 - strat.ma60) < 1
    #     ),
    #     "buy_score": 0.5,
    #     "sell_score": 0.5
    # },
    # {
    #     "priority": 5,
    #     "signal_name": "均线缠绕无方向",
    #     "signal_type": "震荡",
    #     "weight": 0.5,
    #     "condition_func": lambda strat: not (
    #         (strat.ma5 > strat.ma10 > strat.ma20) or
    #         (strat.ma5 < strat.ma10 < strat.ma20)
    #     ),
    #     "buy_score": 0.5,
    #     "sell_score": 0.5
    # }
]


# ===================== 第二步：通用评分计算函数 =====================
def calculate_ma_score(strategy):
    """
    基于规则表的 MA 评分函数
    逻辑：遍历规则表，找到第一个匹配的规则并返回分数
    """
    # 1. 检查数据长度
    if len(strategy.data) < 10:  # 背离检测需要至少10根K线
        return {
            "signal_type": "数据不足",
            "buy_score": 0.0,
            "sell_score": 0.0,
            "net_score": 0.0,
            "triggered_signals": [],
            "raw_buy": 0.0,
            "raw_sell": 0.0
        }

    # 2. 遍历规则表（按优先级顺序）
    for rule in MA_RULES:
        try:
            # 执行条件函数
            if rule["condition_func"](strategy):
                return {
                    "signal_type": rule["signal_type"],
                    "signal_name": rule["signal_name"],
                    "buy_score": rule["buy_score"],
                    "sell_score": rule["sell_score"],
                    "net_score": rule["buy_score"] - rule["sell_score"],
                    "triggered_signals": [f"{rule['signal_name']}({rule['signal_type']})"],
                    "raw_buy": rule["buy_score"],
                    "raw_sell": rule["sell_score"]
                }
        except Exception as e:
            # 防止某个规则出错导致整个策略崩溃
            print(f"MA规则匹配异常 [{rule.get('signal_name')}]: {e}")
            continue

    # 3. 无匹配规则
    return {
        "signal_type": "无信号",
        "buy_score": 0.0,
        "sell_score": 0.0,
        "net_score": 0.0,
        "triggered_signals": [],
        "raw_buy": 0.0,
        "raw_sell": 0.0
    }


# ========== 模拟Backtrader对象（适配背离检测） ==========
class MockParams:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class BacktraderLine:
    """模拟Backtrader的line对象，仅存储数据"""

    def __init__(self, values):
        # values格式：[current, prev_1, prev_2, ..., prev_19]（共20个值）
        self.array = list(values)

    def __len__(self):
        return len(self.array)

    def __getitem__(self, idx):
        """补充索引实现，适配Backtrader的[-n]索引方式"""
        if idx >= 0:
            return self.array[idx] if idx < len(self.array) else 0.0
        else:
            # 修正：Backtrader中[-1]是上一个bar，对应array[1]，[-5]对应array[5]
            pos = abs(idx)
            return self.array[pos] if pos < len(self.array) else 0.0


class MockData:
    """修正后的Mock Data对象，正确实现__len__方法"""

    def __init__(self, high_vals, low_vals, close_vals, open_vals):
        self.high = BacktraderLine(high_vals)
        self.low = BacktraderLine(low_vals)
        self.close = BacktraderLine(close_vals)
        self.open = BacktraderLine(open_vals)

    def __len__(self):
        # 以close的长度作为data的长度
        return len(self.close.array)


class MockStrategy:
    def __init__(self,
                 ma5_vals=None, ma10_vals=None, ma20_vals=None, ma60_vals=None,
                 high_vals=None, low_vals=None, close_vals=None, open_vals=None,
                 volume_vals=None, volume_ma5_vals=None,
                 stock_type="中盘股"):
        default_len = 20
        # 默认值：[当前值, 前1日, 前2日...前19日]
        self.ma5 = BacktraderLine(ma5_vals if ma5_vals else [10.0] * default_len)
        self.ma10 = BacktraderLine(ma10_vals if ma10_vals else [9.8] * default_len)
        self.ma20 = BacktraderLine(ma20_vals if ma20_vals else [9.6] * default_len)
        self.ma60 = BacktraderLine(ma60_vals if ma60_vals else [9.5] * default_len)

        default_close = close_vals if close_vals else [10.0] * default_len
        # 修正：使用独立的MockData类创建data对象，正确实现__len__
        self.data = MockData(
            high_vals=high_vals if high_vals else [x * 1.05 for x in default_close],
            low_vals=low_vals if low_vals else [x * 0.95 for x in default_close],
            close_vals=default_close,
            open_vals=open_vals if open_vals else [x * 0.98 for x in default_close]
        )

        self.volume = BacktraderLine(volume_vals if volume_vals else [10000] * default_len)
        self.volume_ma5 = BacktraderLine(volume_ma5_vals if volume_ma5_vals else [10000] * default_len)

        self.p = MockParams(stock_type=stock_type)


# ========== 完整测试用例（覆盖所有14条规则 + 背离函数单独测试） ==========
class TestMAScoreCalculation(unittest.TestCase):
    def setUp(self):
        print("\n--- 开始测试 ---")

    # ---------------------- 测试最高优先级：背离（规则1-2） ----------------------
    def test_01_bottom_divergence(self):
        """测试1：底背离（规则1）- 独立函数+评分函数双重验证"""
        print("【测试1】底背离（规则1）")
        # 构造数据：严格匹配规则 (low[0]<low[-5]<low[-10] 且 MA60[0]>MA60[-5]>MA60[-10])
        # low数组：[8.0, 8.1, 8.2, 8.3, 8.4, 8.5, 8.6, 8.7, 8.8, 8.9, 9.0, ...]
        low_vals = [8.0 + i * 0.1 for i in range(20)]  # low[0]=8.0, low[-5]=8.5, low[-10]=9.0
        # MA60数组：[10.0, 9.9, 9.8, 9.7, 9.6, 9.5, 9.4, 9.3, 9.2, 9.1, 9.0, ...]
        ma60_vals = [10.0 - i * 0.1 for i in range(20)]  # MA60[0]=10.0, MA60[-5]=9.5, MA60[-10]=9.0

        strat = MockStrategy(
            low_vals=low_vals,
            ma60_vals=ma60_vals,
            close_vals=[10.0] * 20,
            high_vals=[10.0] * 20
        )

        # 1. 单独测试背离检测函数
        self.assertEqual(_check_ma_divergence(strat, '底背离'), True)
        self.assertEqual(_check_ma_divergence(strat, '顶背离'), False)

        # 2. 测试评分函数（仅基础分）
        score = calculate_ma_score(strat)
        print(f"测试1结果: {score}")
        self.assertEqual(score["signal_type"], "最高（反转）-底背离（股价新低，均线不新低）")
        self.assertEqual(score["buy_score"], 2.0)  # 纯基础分，无权重
        self.assertEqual(score["sell_score"], 0.0)
        self.assertEqual(score["net_score"], 2.0)

    def test_02_top_divergence(self):
        """测试2：顶背离（规则2）- 独立函数+评分函数双重验证"""
        print("【测试2】顶背离（规则2）")
        # 构造数据：严格匹配规则 (high[0]>high[-5]>high[-10] 且 MA60[0]<MA60[-5]<MA60[-10])
        # high数组：[12.0, 11.9, 11.8, 11.7, 11.6, 11.5, 11.4, ...]
        high_vals = [12.0 - i * 0.1 for i in range(20)]  # high[0]=12.0, high[-5]=11.5, high[-10]=11.0
        # MA60数组：[8.0, 8.1, 8.2, 8.3, 8.4, 8.5, 8.6, ...]
        ma60_vals = [8.0 + i * 0.1 for i in range(20)]  # MA60[0]=8.0, MA60[-5]=8.5, MA60[-10]=9.0

        strat = MockStrategy(
            high_vals=high_vals,
            ma60_vals=ma60_vals,
            close_vals=[10.0] * 20,
            low_vals=[10.0] * 20
        )

        # 1. 单独测试背离检测函数
        self.assertEqual(_check_ma_divergence(strat, '顶背离'), True)
        self.assertEqual(_check_ma_divergence(strat, '底背离'), False)

        # 2. 测试评分函数
        score = calculate_ma_score(strat)
        print(f"测试2结果: {score}")
        self.assertEqual(score["signal_type"], "最高（反转）-顶背离（股价新高，均线不新高）")
        self.assertEqual(score["sell_score"], 2.0)  # 纯基础分，无权重
        self.assertEqual(score["buy_score"], 0.0)
        self.assertEqual(score["net_score"], -2.0)

    # ---------------------- 测试高优先级：趋势启动/结束（规则3-4） ----------------------
    def test_03_break_ma60_with_volume(self):
        """测试3：放量突破长期均线（规则3）"""
        print("【测试3】放量突破长期均线（规则3）")
        close_vals = [10.5, 10.4] + [10.0] * 18  # close[0]=10.5, close[-1]=10.4
        ma60_vals = [10.0, 9.9] + [9.0] * 18  # MA60[0]=10.0, MA60[-1]=9.9
        volume_vals = [15000] + [10000] * 19  # volume[0]=15000
        volume_ma5_vals = [10000] * 20  # volume_ma5[0]=10000

        strat = MockStrategy(
            close_vals=close_vals,
            ma60_vals=ma60_vals,
            volume_vals=volume_vals,
            volume_ma5_vals=volume_ma5_vals
        )
        score = calculate_ma_score(strat)
        print(f"测试3结果: {score}")
        self.assertEqual(score["signal_type"], "高（趋势启动）-放量突破长期均线")
        self.assertEqual(score["buy_score"], 1.8)  # 纯基础分
        self.assertEqual(score["sell_score"], 0.0)

    def test_04_break_ma60_down_with_volume(self):
        """测试4：放量跌破长期均线（规则4）"""
        print("【测试4】放量跌破长期均线（规则4）")
        close_vals = [9.5, 9.6] + [10.0] * 18  # close[0]=9.5, close[-1]=9.6
        ma60_vals = [10.0, 10.1] + [10.2] * 18  # MA60[0]=10.0, MA60[-1]=10.1
        volume_vals = [16000] + [10000] * 19  # volume[0]=16000
        volume_ma5_vals = [10000] * 20  # volume_ma5[0]=10000

        strat = MockStrategy(
            close_vals=close_vals,
            ma60_vals=ma60_vals,
            volume_vals=volume_vals,
            volume_ma5_vals=volume_ma5_vals
        )
        score = calculate_ma_score(strat)
        print(f"测试4结果: {score}")
        self.assertEqual(score["signal_type"], "高（趋势结束）-放量跌破长期均线")
        self.assertEqual(score["sell_score"], 1.8)  # 纯基础分
        self.assertEqual(score["buy_score"], 0.0)

    # ---------------------- 测试中高优先级：金叉/死叉/回踩/遇压（规则5-8） ----------------------
    def test_05_golden_cross(self):
        """测试5：黄金交叉（规则5）"""
        print("【测试5】黄金交叉（规则5）")
        ma5_vals = [10.1, 9.9] + [9.0] * 18  # MA5[0]=10.1, MA5[-1]=9.9
        ma10_vals = [9.9, 10.0] + [9.0] * 18  # MA10[0]=9.9, MA10[-1]=10.0

        strat = MockStrategy(ma5_vals=ma5_vals, ma10_vals=ma10_vals)
        score = calculate_ma_score(strat)
        print(f"测试5结果: {score}")
        self.assertEqual(score["signal_type"], "中高（金叉）-短期上穿长期（黄金交叉）")
        self.assertEqual(score["buy_score"], 1.5)  # 纯基础分
        self.assertEqual(score["sell_score"], 0.0)

    def test_06_death_cross(self):
        """测试6：死亡交叉（规则6）"""
        print("【测试6】死亡交叉（规则6）")
        ma5_vals = [9.9, 10.1] + [9.0] * 18  # MA5[0]=9.9, MA5[-1]=10.1
        ma10_vals = [10.0, 9.9] + [9.0] * 18  # MA10[0]=10.0, MA10[-1]=9.9

        strat = MockStrategy(ma5_vals=ma5_vals, ma10_vals=ma10_vals)
        score = calculate_ma_score(strat)
        print(f"测试6结果: {score}")
        self.assertEqual(score["signal_type"], "中高（死叉）-短期下穿长期（死亡交叉）")
        self.assertEqual(score["sell_score"], 1.5)  # 纯基础分
        self.assertEqual(score["buy_score"], 0.0)

    def test_07_ma60_support_rebound(self):
        """测试7：回踩MA60企稳反弹（规则7）"""
        print("【测试7】回踩MA60企稳反弹（规则7）")
        low_vals = [9.9] + [10.0] * 19  # low[0]=9.9 (≥10.0*0.99)
        close_vals = [10.2] + [10.0] * 19  # close[0]=10.2 > open[0]=9.8
        open_vals = [9.8] + [10.0] * 19  # open[0]=9.8
        volume_vals = [7500] + [10000] * 19  # volume[0]=7500 < 10000*0.8
        ma60_vals = [10.0] * 20  # MA60[0]=10.0

        strat = MockStrategy(
            low_vals=low_vals,
            close_vals=close_vals,
            open_vals=open_vals,
            volume_vals=volume_vals,
            volume_ma5_vals=[10000] * 20,
            ma60_vals=ma60_vals
        )
        score = calculate_ma_score(strat)
        print(f"测试7结果: {score}")
        self.assertEqual(score["signal_type"], "中高（回踩支撑）-回踩MA60企稳反弹")
        self.assertEqual(score["buy_score"], 1.5)  # 纯基础分
        self.assertEqual(score["sell_score"], 0.0)

    def test_08_ma60_resistance_drop(self):
        """测试8：反弹遇MA60回落（规则8）"""
        print("【测试8】反弹遇MA60回落（规则8）")
        high_vals = [10.1] + [10.0] * 19  # high[0]=10.1 (≤10.0*1.01)
        close_vals = [9.8] + [10.0] * 19  # close[0]=9.8 < open[0]=10.2
        open_vals = [10.2] + [10.0] * 19  # open[0]=10.2
        volume_vals = [12500] + [10000] * 19  # volume[0]=12500 > 10000*1.2
        ma60_vals = [10.0] * 20  # MA60[0]=10.0

        strat = MockStrategy(
            high_vals=high_vals,
            close_vals=close_vals,
            open_vals=open_vals,
            volume_vals=volume_vals,
            volume_ma5_vals=[10000] * 20,
            ma60_vals=ma60_vals
        )
        score = calculate_ma_score(strat)
        print(f"测试8结果: {score}")
        self.assertEqual(score["signal_type"], "中高（反弹遇压）-反弹遇MA60回落")
        self.assertEqual(score["sell_score"], 1.5)  # 纯基础分
        self.assertEqual(score["buy_score"], 0.0)

    # ---------------------- 测试常规优先级：排列/位置（规则9-12） ----------------------
    def test_09_bullish_arrangement(self):
        """测试9：标准多头排列（规则9）"""
        print("【测试9】标准多头排列（规则9）")
        ma5_vals = [10.5] * 20  # MA5[0]=10.5 > MA10[0]=10.3
        ma10_vals = [10.3] * 20  # MA10[0]=10.3 > MA20[0]=10.1
        ma20_vals = [10.1] * 20  # MA20[0]=10.1 > MA60[0]=10.0
        ma60_vals = [10.0, 9.9] + [9.0] * 18  # MA60[0]=10.0 > MA60[-1]=9.9

        strat = MockStrategy(
            ma5_vals=ma5_vals,
            ma10_vals=ma10_vals,
            ma20_vals=ma20_vals,
            ma60_vals=ma60_vals
        )
        score = calculate_ma_score(strat)
        print(f"测试9结果: {score}")
        self.assertEqual(score["signal_type"], "常规（多头排列）-标准多头排列")
        self.assertEqual(score["buy_score"], 1.2)  # 纯基础分
        self.assertEqual(score["sell_score"], 0.3)

    def test_10_bearish_arrangement(self):
        """测试10：标准空头排列（规则10）"""
        print("【测试10】标准空头排列（规则10）")
        ma5_vals = [9.5] * 20  # MA5[0]=9.5 < MA10[0]=9.7
        ma10_vals = [9.7] * 20  # MA10[0]=9.7 < MA20[0]=9.9
        ma20_vals = [9.9] * 20  # MA20[0]=9.9 < MA60[0]=10.0
        ma60_vals = [10.0, 10.1] + [10.2] * 18  # MA60[0]=10.0 < MA60[-1]=10.1

        strat = MockStrategy(
            ma5_vals=ma5_vals,
            ma10_vals=ma10_vals,
            ma20_vals=ma20_vals,
            ma60_vals=ma60_vals
        )
        score = calculate_ma_score(strat)
        print(f"测试10结果: {score}")
        self.assertEqual(score["signal_type"], "常规（空头排列）-标准空头排列")
        self.assertEqual(score["sell_score"], 1.2)  # 纯基础分
        self.assertEqual(score["buy_score"], 0.3)

    def test_11_above_ma60(self):
        """测试11：股价在MA60上方（规则11）"""
        print("【测试11】股价在MA60上方（规则11）")
        close_vals = [10.5] * 20  # close[0]=10.5 > MA60[0]=10.0
        ma60_vals = [10.0] * 20  # MA60[0]=10.0

        strat = MockStrategy(close_vals=close_vals, ma60_vals=ma60_vals)
        score = calculate_ma_score(strat)
        print(f"测试11结果: {score}")
        self.assertEqual(score["signal_type"], "常规（均线之上）-股价在MA60上方")
        self.assertEqual(score["buy_score"], 1.0)  # 纯基础分
        self.assertEqual(score["sell_score"], 0.3)

    def test_12_below_ma60(self):
        """测试12：股价在MA60下方（规则12）"""
        print("【测试12】股价在MA60下方（规则12）")
        close_vals = [9.5] * 20  # close[0]=9.5 < MA60[0]=10.0
        ma60_vals = [10.0] * 20  # MA60[0]=10.0

        strat = MockStrategy(close_vals=close_vals, ma60_vals=ma60_vals)
        score = calculate_ma_score(strat)
        print(f"测试12结果: {score}")
        self.assertEqual(score["signal_type"], "常规（均线之下）-股价在MA60下方")
        self.assertEqual(score["sell_score"], 1.0)  # 纯基础分
        self.assertEqual(score["buy_score"], 0.3)

    # ---------------------- 测试低优先级：粘合/震荡（规则13-14） ----------------------
    def test_13_ma_convergence(self):
        """测试13：多条均线粘合收敛（规则13）"""
        print("【测试13】多条均线粘合收敛（规则13）")
        ma5_vals = [10.0] * 20  # MA5[0]=10.0
        ma10_vals = [10.5] * 20  # MA10[0]=10.5 (差值0.5<1)
        ma20_vals = [10.8] * 20  # MA20[0]=10.8 (差值0.3<1)
        ma60_vals = [10.9] * 20  # MA60[0]=10.9 (差值0.1<1)

        strat = MockStrategy(
            ma5_vals=ma5_vals,
            ma10_vals=ma10_vals,
            ma20_vals=ma20_vals,
            ma60_vals=ma60_vals
        )
        score = calculate_ma_score(strat)
        print(f"测试13结果: {score}")
        self.assertEqual(score["signal_type"], "低（粘合整理）-多条均线粘合收敛")
        self.assertEqual(score["buy_score"], 0.5)  # 纯基础分
        self.assertEqual(score["sell_score"], 0.5)

    def test_14_ma_tangle(self):
        """测试14：均线缠绕无方向（规则14）"""
        print("【测试14】均线缠绕无方向（规则14）")
        ma5_vals = [10.0] * 20  # MA5[0]=10.0
        ma10_vals = [9.8] * 20  # MA10[0]=9.8 < MA5[0]
        ma20_vals = [10.2] * 20  # MA20[0]=10.2 > MA10[0] → 不满足多头/空头排列

        strat = MockStrategy(
            ma5_vals=ma5_vals,
            ma10_vals=ma10_vals,
            ma20_vals=ma20_vals
        )
        score = calculate_ma_score(strat)
        print(f"测试14结果: {score}")
        self.assertEqual(score["signal_type"], "低（横盘震荡）-均线缠绕无方向")
        self.assertEqual(score["buy_score"], 0.5)  # 纯基础分
        self.assertEqual(score["sell_score"], 0.5)

    # ---------------------- 边界测试 ----------------------
    def test_15_no_signal(self):
        """测试15：无信号匹配"""
        print("【测试15】无信号匹配")
        # 构造不满足任何规则的数据
        ma5_vals = [12.0] * 20
        ma10_vals = [11.0] * 20
        ma20_vals = [10.0] * 20
        ma60_vals = [11.5] * 20
        close_vals = [11.5] * 20
        volume_vals = [10000] * 20

        strat = MockStrategy(
            ma5_vals=ma5_vals,
            ma10_vals=ma10_vals,
            ma20_vals=ma20_vals,
            ma60_vals=ma60_vals,
            close_vals=close_vals,
            volume_vals=volume_vals
        )
        score = calculate_ma_score(strat)
        print(f"测试15结果: {score}")
        self.assertEqual(score["signal_type"], "无信号")
        self.assertEqual(score["buy_score"], 0.0)
        self.assertEqual(score["sell_score"], 0.0)

    def test_16_divergence_edge_case(self):
        """测试16：背离边界场景（数据不足）"""
        print("【测试16】背离边界场景（数据不足）")
        # 数据长度不足10天，背离检测应返回False
        strat = MockStrategy(
            low_vals=[8.0] * 5,  # 仅5个值
            ma60_vals=[10.0] * 5,  # 仅5个值
            high_vals=[12.0] * 5,
            close_vals=[10.0] * 5  # 确保data长度为5
        )
        self.assertEqual(_check_ma_divergence(strat, '底背离'), False)
        self.assertEqual(_check_ma_divergence(strat, '顶背离'), False)


if __name__ == '__main__':
    unittest.main(verbosity=2)