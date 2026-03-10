import unittest
import math


# ===================== 第一步：工具函数 =====================
def safe_get_boll(line, idx):
    """安全获取BacktraderLine的数值"""
    try:
        return line[idx] if isinstance(line, (list, BacktraderLine)) else 0.0
    except (IndexError, TypeError):
        return 0.0


def float_equals(a, b, tolerance=1e-9):
    """浮点数相等判断（解决精度问题）"""
    return math.isclose(a, b, abs_tol=tolerance)


# ===================== 第一步：BOLL钝化判断函数 =====================
def _check_boll钝化(strat):
    try:
        if len(strat.boll_upper) < 5 or len(strat.boll_lower) < 5:
            return False

        close_values = []
        upper_values = []
        lower_values = []
        for i in range(-5, 0):
            close = safe_get_boll(strat.data.close, i)
            upper = safe_get_boll(strat.boll_upper, i)
            lower = safe_get_boll(strat.boll_lower, i)
            if upper == 0 or lower == 0:
                return False
            close_values.append(close)
            upper_values.append(upper)
            lower_values.append(lower)

        upper_ratio = [c / u for c, u in zip(close_values, upper_values)]
        lower_ratio = [c / l for c, l in zip(close_values, lower_values)]

        upper_钝化 = (all(ratio > 0.995 for ratio in upper_ratio) and
                      len(close_values) >= 2 and (close_values[-1] - close_values[0]) / close_values[0] > 0.08)
        lower_钝化 = (all(ratio < 1.005 for ratio in lower_ratio) and
                      len(close_values) >= 2 and (close_values[0] - close_values[-1]) / close_values[0] > 0.08)

        return upper_钝化 or lower_钝化
    except Exception:
        return False


def _check_boll_divergence(strat, divergence_type):
    """检查BOLL背离（修复：确保取最新的10个数据）"""
    try:
        if len(strat.boll_mid) < 10 or len(strat.data.close) < 10:
            return False

        # 修复：明确获取最后10个数据
        close_prices = strat.data.close.get(size=10)
        mid_values = strat.boll.mid.get(size=10)

        if len(close_prices) < 10 or len(mid_values) < 10:
            return False

        # 前5个和后5个数据分割
        close_first5 = close_prices[:5]
        close_last5 = close_prices[5:]
        mid_first5 = mid_values[:5]
        mid_last5 = mid_values[5:]

        if divergence_type == '底背离':
            # 收盘价后5天新低，中轨后5天新高
            return (min(close_last5) < min(close_first5) and
                    min(mid_last5) > min(mid_first5))
        elif divergence_type == '顶背离':
            # 收盘价后5天新高，中轨后5天新低
            return (max(close_last5) > max(close_first5) and
                    max(mid_last5) < max(mid_first5))
        return False
    except Exception as e:
        print(f"背离检查异常: {e}")
        return False


# ===================== 第二步：BOLL评分规则表（修复：强制获取数值）======================
BOLL_SCORE_TABLE = [
    # ---------------- 优先级：最高（反转信号）----------------
    {
        "priority": 1,
        "signal_name": "底背离",
        "signal_type": "趋势反转",
        "weight": 2.0,
        "condition_func": lambda strat: _check_boll_divergence(strat, '底背离'),
        "buy_score": 2.0,
        "sell_score": 0.0
    },
    {
        "priority": 1,
        "signal_name": "顶背离",
        "signal_type": "趋势反转",
        "weight": 2.0,
        "condition_func": lambda strat: _check_boll_divergence(strat, '顶背离'),
        "buy_score": 0.0,
        "sell_score": 2.0
    },

    # ---------------- 优先级：高（极端行情）----------------
    {
        "priority": 2,
        "signal_name": "下轨超卖",
        "signal_type": "极端行情",
        "weight": 1.5,
        "condition_func": lambda strat: (
                len(strat.data.close) >= 2 and
                (strat.data.close[-1] < strat.boll.bot[-1]) and
                (strat.data.volume[-1] < 0.8 * strat.vol5[-1])
        ),
        "buy_score": 1.8,
        "sell_score": 0.0
    },
    {
        "priority": 2,
        "signal_name": "上轨超买",
        "signal_type": "极端行情",
        "weight": 1.5,
        "condition_func": lambda strat: (
                len(strat.data.close) >= 2 and
                (strat.data.close[-1] > strat.boll.top[-1]) and
                (strat.data.volume[-1] > 1.2 * strat.vol5[-1]) and
                (strat.data.close[-1] <= strat.data.close[-2])
        ),
        "buy_score": 0.0,
        "sell_score": 1.8
    },

    # ---------------- 优先级：中高（确认信号）----------------
    {
        "priority": 3,
        "signal_name": "下轨反弹",
        "signal_type": "确认信号",
        "weight": 1.2,
        "condition_func": lambda strat: (
                len(strat.data.close) >= 2 and
                (strat.data.close[-2] < strat.boll.bot[-2]) and
                (strat.data.close[-1] >= strat.boll.bot[-1]) and
                (strat.data.close[-1] <= strat.boll.mid[-1])
        ),
        "buy_score": 1.5,
        "sell_score": 0.0
    },
    {
        "priority": 3,
        "signal_name": "上轨回落",
        "signal_type": "确认信号",
        "weight": 1.2,
        "condition_func": lambda strat: (
                len(strat.data.close) >= 2 and
                (strat.data.close[-2] > strat.boll.top[-2]) and
                (strat.data.close[-1] <= strat.boll.top[-1]) and
                (strat.data.close[-1] >= strat.boll.mid[-1])
        ),
        "buy_score": 0.0,
        "sell_score": 1.5
    },

    # ---------------- 优先级：中高（趋势信号）----------------
    {
        "priority": 4,
        "signal_name": "强势多头",
        "signal_type": "趋势信号",
        "weight": 1.2,
        "condition_func": lambda strat: (
                len(strat.boll.mid) >= 4 and
                (strat.boll.mid[-1] > strat.boll.mid[-2] > strat.boll.mid[-3] > strat.boll.mid[-4]) and
                (strat.data.close[-1] > strat.boll.mid[-1])
        ),
        "buy_score": 1.2,
        "sell_score": 0.3
    },
    {
        "priority": 4,
        "signal_name": "强势空头",
        "signal_type": "趋势信号",
        "weight": 1.2,
        "condition_func": lambda strat: (
                len(strat.boll.mid) >= 4 and
                (strat.boll.mid[-1] < strat.boll.mid[-2] < strat.boll.mid[-3] < strat.boll.mid[-4]) and
                (strat.data.close[-1] < strat.boll.mid[-1])
        ),
        "buy_score": 0.3,
        "sell_score": 1.2
    },

    # ---------------- 优先级：低（震荡/方向）----------------
    {
        "priority": 5,
        "signal_name": "收口整理",
        "signal_type": "震荡",
        "weight": 0.5,
        "condition_func": lambda strat: (
                len(strat.boll.top) >= 5 and len(strat.boll.bot) >= 5 and
                # 修复：确保带宽收缩条件严格满足
                ((strat.boll.top[-1] - strat.boll.bot[-1]) < 0.8 * (strat.boll.top[-5] - strat.boll.bot[-5])) and
                (abs(strat.data.close[-1] - strat.boll.mid[-1]) / strat.boll.mid[-1] < 0.02)
        ),
        "buy_score": 0.5,
        "sell_score": 0.5
    },
    {
        "priority": 5,
        "signal_name": "中轨之上",
        "signal_type": "方向",
        "weight": 1.0,
        "condition_func": lambda strat: (
                strat.data.close[-1] > strat.boll.mid[-1]
        ),
        "buy_score": 1.0,
        "sell_score": 0.3
    },
    {
        "priority": 5,
        "signal_name": "中轨之下",
        "signal_type": "方向",
        "weight": 1.0,
        "condition_func": lambda strat: (
                strat.data.close[-1] < strat.boll.mid[-1]
        ),
        "buy_score": 0.3,
        "sell_score": 1.0
    },
]


# ===================== 第三步：核心评分函数（适配新规则）======================
def calculate_boll_score(strategy):
    """
    基于规则表的 BOLL 评分函数
    逻辑：遍历规则表，按优先级顺序匹配，返回第一个匹配的规则分数
    """
    # 1. 检查数据长度和钝化
    if len(strategy.data.close) < 10:
        return {
            "signal_type": "数据不足",
            "signal_name": "",
            "buy_score": 0.0,
            "sell_score": 0.0,
            "net_score": 0.0,
            "triggered_signals": [],
            "extreme_market": False
        }

    is_dull = _check_boll钝化(strategy)

    # 2. 遍历规则表（按优先级顺序）
    for rule in BOLL_SCORE_TABLE:
        try:
            # --- 特殊逻辑：跳过钝化时的规则 (下轨超卖/上轨超买) ---
            if is_dull and rule["signal_name"] in ["下轨超卖", "上轨超买"]:
                continue

            # --- 执行条件函数 ---
            if rule["condition_func"](strategy):
                # 应用权重计算最终分数
                buy_score = round(rule["buy_score"] * rule["weight"], 2)
                sell_score = round(rule["sell_score"] * rule["weight"], 2)

                # 构造完整的signal_type（匹配测试用例预期格式）
                priority_map = {
                    1: "最高（反转）",
                    2: "高（极端）",
                    3: "中高（确认）",
                    4: "中高（趋势）",
                    5: {
                        "震荡": "低（平衡）",
                        "方向": "常规（方向）"
                    }.get(rule["signal_type"], rule["signal_type"])
                }

                priority_desc = priority_map[rule["priority"]]
                full_signal_type = f"{priority_desc}-{rule['signal_name']}"

                return {
                    "signal_type": full_signal_type,
                    "signal_name": rule["signal_name"],
                    "buy_score": buy_score,
                    "sell_score": sell_score,
                    "net_score": round(buy_score - sell_score, 2),
                    "triggered_signals": [f"{rule['signal_name']}({rule['signal_type']})"],
                    "extreme_market": rule["signal_type"] == "极端行情"
                }
        except Exception as e:
            print(f"BOLL规则匹配异常 [{rule.get('signal_name')}]: {e}")
            continue

    # 3. 检查极端单边行情（修复：放宽判断条件，允许微小误差）
    try:
        close_last = strategy.data.close[-1]
        high_last = strategy.data.high[-1]
        low_last = strategy.data.low[-1]
        # 允许浮点数微小误差
        if float_equals(close_last, high_last) and float_equals(close_last, low_last):
            return {
                "signal_type": "极端单边行情-停止操作",
                "signal_name": "极端单边行情",
                "buy_score": 0.0,
                "sell_score": 0.0,
                "net_score": 0.0,
                "triggered_signals": [],
                "extreme_market": True
            }
    except Exception as e:
        print(f"极端单边行情检查异常: {e}")
        pass

    # 4. 无匹配规则
    return {
        "signal_type": "无信号",
        "signal_name": "",
        "buy_score": 0.0,
        "sell_score": 0.0,
        "net_score": 0.0,
        "triggered_signals": [],
        "extreme_market": False
    }


# ===================== 第五步：模拟Backtrader对象 =====================
class MockParams:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class BacktraderLine:
    def __init__(self, values):
        self.array = list(values)

    def __getitem__(self, idx):
        try:
            # 处理负数索引
            if idx < 0:
                return self.array[idx]
            return self.array[idx] if idx < len(self.array) else 0.0
        except (IndexError, TypeError):
            return 0.0

    def get(self, size=None):
        if size is None or size <= 0:
            return self.array.copy()
        return self.array[-size:] if len(self.array) >= size else self.array.copy()

    def __len__(self):
        return len(self.array)


class MockStrategy:
    def __init__(self,
                 boll_mid_vals=None, boll_top_vals=None, boll_bot_vals=None,
                 close_vals=None, high_vals=None, low_vals=None,
                 volume_vals=None, vol5_vals=None,
                 stock_type="中盘股"):
        default_len = 20
        self.boll = type('obj', (object,), {
            'mid': BacktraderLine(boll_mid_vals or [10.0] * default_len),
            'top': BacktraderLine(boll_top_vals or [11.0] * default_len),
            'bot': BacktraderLine(boll_bot_vals or [9.0] * default_len)
        })
        self.boll_upper = self.boll.top
        self.boll_lower = self.boll.bot
        close_vals = close_vals or [10.0] * default_len
        self.data = type('obj', (object,), {
            'close': BacktraderLine(close_vals),
            'high': BacktraderLine(high_vals or [x * 1.05 for x in close_vals]),
            'low': BacktraderLine(low_vals or [x * 0.95 for x in close_vals]),
            'volume': BacktraderLine(volume_vals or [10000] * default_len)
        })
        self.vol5 = BacktraderLine(vol5_vals or [10000] * default_len)
        self.p = MockParams(stock_type=stock_type)


# ===================== 第六步：完整测试用例 =====================
class TestBOLLScoreCalculation(unittest.TestCase):
    def test_01_bottom_divergence(self):
        print("【测试1】底背离")
        # 构造严格的底背离数据：
        # 前5天收盘价范围：7.6-9.0，最小值8.0
        # 后5天收盘价范围：6.5-6.9，最小值6.5 < 8.0
        # 前5天中轨范围：10.0-10.4，最小值10.0
        # 后5天中轨范围：10.5-10.9，最小值10.5 > 10.0
        close_vals = [9.0, 8.5, 8.0, 8.5, 9.0, 7.6, 7.5, 7.0, 6.9, 6.8,
                      6.7, 6.6, 6.55, 6.52, 6.51, 6.5, 6.5, 6.5, 6.5, 6.5]
        boll_mid_vals = [10.0, 10.1, 10.2, 10.3, 10.4, 10.5, 10.6, 10.7, 10.8, 10.9,
                         11.0, 11.1, 11.2, 11.3, 11.4, 11.5, 11.6, 11.7, 11.8, 11.9]
        strat = MockStrategy(
            boll_mid_vals=boll_mid_vals,
            close_vals=close_vals
        )
        score = calculate_boll_score(strat)
        print(f"测试1结果: {score}")
        self.assertEqual(score["signal_type"], "最高（反转）-底背离")
        self.assertEqual(score["buy_score"], 4.0)
        self.assertEqual(score["sell_score"], 0.0)

    def test_02_top_divergence(self):
        """测试2：最高优先级 - 顶背离"""
        print("【测试2】顶背离")
        # 构造严格的顶背离数据：
        # 前5天收盘价范围：11.0-12.0，最大值12.0
        # 后5天收盘价范围：12.1-12.5，最大值12.5 > 12.0
        # 前5天中轨范围：9.6-10.0，最大值10.0
        # 后5天中轨范围：9.1-9.5，最大值9.5 < 10.0
        close_vals = [11.0, 11.5, 12.0, 11.5, 11.0, 12.1, 12.2, 12.3, 12.4, 12.5,
                      12.6, 12.7, 12.8, 12.9, 13.0, 13.1, 13.2, 13.3, 13.4, 13.5]
        boll_mid_vals = [10.0, 9.9, 9.8, 9.7, 9.6, 9.5, 9.4, 9.3, 9.2, 9.1,
                         9.0, 8.9, 8.8, 8.7, 8.6, 8.5, 8.4, 8.3, 8.2, 8.1]
        strat = MockStrategy(
            boll_mid_vals=boll_mid_vals,
            close_vals=close_vals
        )
        score = calculate_boll_score(strat)
        print(f"测试2结果: {score}")
        self.assertEqual(score["signal_type"], "最高（反转）-顶背离")
        self.assertEqual(score["sell_score"], 4.0)
        self.assertEqual(score["buy_score"], 0.0)

    def test_03_lower_band_oversell(self):
        """测试3：高优先级 - 下轨超卖"""
        print("【测试3】下轨超卖")
        close_vals = [8.8] * 20
        volume_vals = [7000] * 20
        strat = MockStrategy(
            close_vals=close_vals,
            volume_vals=volume_vals,
            boll_bot_vals=[9.0] * 20,
            vol5_vals=[10000] * 20
        )
        score = calculate_boll_score(strat)
        print(f"测试3结果: {score}")
        self.assertEqual(score["signal_type"], "高（极端）-下轨超卖")
        self.assertEqual(score["buy_score"], 2.7)
        self.assertEqual(score["sell_score"], 0.0)

    def test_04_upper_band_overbuy(self):
        """测试4：高优先级 - 上轨超买"""
        print("【测试4】上轨超买")
        close_vals = [11.2, 11.3, 11.2] + [11.2] * 17
        volume_vals = [13000] * 20
        strat = MockStrategy(
            close_vals=close_vals,
            volume_vals=volume_vals,
            boll_top_vals=[11.0] * 20,
            vol5_vals=[10000] * 20
        )
        score = calculate_boll_score(strat)
        print(f"测试4结果: {score}")
        self.assertEqual(score["signal_type"], "高（极端）-上轨超买")
        self.assertEqual(score["sell_score"], 2.7)
        self.assertEqual(score["buy_score"], 0.0)

    def test_05_lower_band_rebound(self):
        """测试5：中高优先级 - 下轨反弹"""
        print("【测试5】下轨反弹")
        close_vals = [9.0] * 18 + [8.8, 9.2]
        strat = MockStrategy(
            close_vals=close_vals,
            boll_bot_vals=[9.0] * 20,
            boll_mid_vals=[10.0] * 20
        )
        score = calculate_boll_score(strat)
        print(f"测试5结果: {score}")
        self.assertEqual(score["signal_type"], "中高（确认）-下轨反弹")
        # 使用浮点数相等判断
        self.assertTrue(float_equals(score["buy_score"], 1.8))
        self.assertEqual(score["sell_score"], 0.0)

    def test_06_upper_band_retracement(self):
        """测试6：中高优先级 - 上轨回落"""
        print("【测试6】上轨回落")
        close_vals = [10.0] * 18 + [11.2, 10.8]
        strat = MockStrategy(
            close_vals=close_vals,
            boll_top_vals=[11.0] * 20,
            boll_mid_vals=[10.0] * 20
        )
        score = calculate_boll_score(strat)
        print(f"测试6结果: {score}")
        self.assertEqual(score["signal_type"], "中高（确认）-上轨回落")
        # 使用浮点数相等判断
        self.assertTrue(float_equals(score["sell_score"], 1.8))
        self.assertEqual(score["buy_score"], 0.0)

    def test_07_strong_bull_trend(self):
        """测试7：中高优先级 - 强势多头"""
        print("【测试7】强势多头")
        boll_mid_vals = [10.0] * 16 + [9.9, 10.0, 10.1, 10.2]
        close_vals = [10.3] * 20
        strat = MockStrategy(
            boll_mid_vals=boll_mid_vals,
            close_vals=close_vals
        )
        score = calculate_boll_score(strat)
        print(f"测试7结果: {score}")
        self.assertEqual(score["signal_type"], "中高（趋势）-强势多头")
        self.assertEqual(score["buy_score"], 1.44)
        self.assertEqual(score["sell_score"], 0.36)

    def test_08_strong_bear_trend(self):
        """测试8：中高优先级 - 强势空头"""
        print("【测试8】强势空头")
        boll_mid_vals = [10.0] * 16 + [10.2, 10.1, 10.0, 9.9]
        close_vals = [9.8] * 20
        strat = MockStrategy(
            boll_mid_vals=boll_mid_vals,
            close_vals=close_vals
        )
        score = calculate_boll_score(strat)
        print(f"测试8结果: {score}")
        self.assertEqual(score["signal_type"], "中高（趋势）-强势空头")
        self.assertEqual(score["sell_score"], 1.44)
        self.assertEqual(score["buy_score"], 0.36)

    def test_09_above_mid_band(self):
        """测试9：常规优先级 - 中轨之上"""
        print("【测试9】中轨之上")
        close_vals = [10.2] * 20
        strat = MockStrategy(
            close_vals=close_vals,
            boll_mid_vals=[10.0] * 20
        )
        score = calculate_boll_score(strat)
        print(f"测试9结果: {score}")
        self.assertEqual(score["signal_type"], "常规（方向）-中轨之上")
        self.assertEqual(score["buy_score"], 1.0)
        self.assertEqual(score["sell_score"], 0.3)

    def test_10_below_mid_band(self):
        """测试10：常规优先级 - 中轨之下"""
        print("【测试10】中轨之下")
        close_vals = [9.8] * 20
        strat = MockStrategy(
            close_vals=close_vals,
            boll_mid_vals=[10.0] * 20
        )
        score = calculate_boll_score(strat)
        print(f"测试10结果: {score}")
        self.assertEqual(score["signal_type"], "常规（方向）-中轨之下")
        self.assertEqual(score["sell_score"], 1.0)
        self.assertEqual(score["buy_score"], 0.3)

    def test_11_contract_consolidation(self):
        """测试11：低优先级 - 收口整理（修复数据）"""
        print("【测试11】收口整理")
        # 修复：确保收口整理规则优先触发
        # 5天前带宽：12.0 - 8.0 = 4.0
        # 最新带宽：10.1 - 9.9 = 0.2 < 0.8*4.0=3.2
        # 收盘价10.0，中轨10.0，偏离0% < 2%
        boll_top_vals = [12.0] * 15 + [10.1] * 5
        boll_bot_vals = [8.0] * 15 + [9.9] * 5
        close_vals = [10.0] * 20  # 刚好在中轨，避免触发中轨之上
        strat = MockStrategy(
            boll_top_vals=boll_top_vals,
            boll_bot_vals=boll_bot_vals,
            boll_mid_vals=[10.0] * 20,
            close_vals=close_vals
        )
        score = calculate_boll_score(strat)
        print(f"测试11结果: {score}")
        self.assertEqual(score["signal_type"], "低（平衡）-收口整理")
        self.assertEqual(score["buy_score"], 0.25)
        self.assertEqual(score["sell_score"], 0.25)

    def test_12_extreme_single_side(self):
        """测试12：无优先级 - 极端单边行情（修复数据）"""
        print("【测试12】极端单边行情")
        # 确保收盘价=最高价=最低价
        close_vals = [10.0] * 19 + [11.0]
        high_vals = [10.5] * 19 + [11.0]
        low_vals = [9.5] * 19 + [11.0]
        strat = MockStrategy(
            close_vals=close_vals,
            high_vals=high_vals,
            low_vals=low_vals,
            # 确保不触发中轨之上规则
            boll_mid_vals=[11.0] * 20
        )
        score = calculate_boll_score(strat)
        print(f"测试12结果: {score}")
        self.assertEqual(score["signal_type"], "极端单边行情-停止操作")
        self.assertEqual(score["buy_score"], 0.0)
        self.assertEqual(score["sell_score"], 0.0)

    def test_13_no_signal(self):
        """测试13：无信号匹配"""
        print("【测试13】无信号匹配")
        global BOLL_SCORE_TABLE
        original_table = BOLL_SCORE_TABLE.copy()
        BOLL_SCORE_TABLE = [rule for rule in BOLL_SCORE_TABLE
                            if rule["signal_name"] not in ["中轨之上", "中轨之下", "收口整理"]]

        close_vals = [10.0] * 20
        strat = MockStrategy(close_vals=close_vals)
        score = calculate_boll_score(strat)
        print(f"测试13结果: {score}")
        self.assertEqual(score["signal_type"], "无信号")
        self.assertEqual(score["buy_score"], 0.0)
        self.assertEqual(score["sell_score"], 0.0)

        BOLL_SCORE_TABLE = original_table

    def test_14_dull_condition(self):
        """测试14：钝化情况（规则3/4被跳过，规则10生效）"""
        print("【测试14】钝化情况（规则3/4被跳过，规则10生效）")
        close_vals = [10.0] * 15 + [9.0, 8.9, 8.8, 8.7, 8.0]
        boll_bot_vals = [9.0] * 20
        boll_mid_vals = [10.0] * 20

        strat = MockStrategy(
            close_vals=close_vals,
            boll_bot_vals=boll_bot_vals,
            boll_mid_vals=boll_mid_vals
        )
        score = calculate_boll_score(strat)
        print(f"测试14结果: {score}")
        self.assertEqual(score["signal_type"], "常规（方向）-中轨之下")
        self.assertEqual(score["sell_score"], 1.0)
        self.assertEqual(score["buy_score"], 0.3)


if __name__ == '__main__':
    unittest.main(verbosity=2)