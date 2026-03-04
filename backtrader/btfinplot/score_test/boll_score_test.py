import unittest


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
    """检查BOLL背离"""
    try:
        if len(strat.boll_mid) < 10 or len(strat.data.close) < 10:
            return False
        close_prices = list(strat.data.close.get(size=10)) if len(strat.data.close) >= 10 else []
        mid_values = list(strat.boll_mid.get(size=10)) if len(strat.boll_mid) >= 10 else []

        if len(close_prices) < 10 or len(mid_values) < 10:
            return False

        if divergence_type == '底背离':
            return min(close_prices[-5:]) < min(close_prices[:5]) and min(mid_values[-5:]) > min(mid_values[:5])
        elif divergence_type == '顶背离':
            return max(close_prices[-5:]) > max(close_prices[:5]) and max(mid_values[-5:]) < max(mid_values[:5])
        return False
    except Exception:
        return False

# ===================== 第二步：BOLL评分规则表（修复后） =====================
# 关键修复：将"收口整理"移到"中轨之上/之下"之前
BOLL_SCORE_TABLE = [
    {"优先级": "无（实操必看）", "序号": 12, "信号类型": "极端单边行情",
     "条件函数": lambda v: (v['close[0]'] == v['high[0]'] == v['low[0]']) and
                           ((v['close[0]'] / v['close[-1]'] >= 1.1) or (v['close[0]'] / v['close[-1]'] <= 0.9)),
     "买入得分": 0.0, "卖出得分": 0.0, "备注": "原书重点：此时本表所有得分失效，立即停止操作"},

    {"优先级": "最高（反转）", "序号": 1, "信号类型": "底背离",
     "条件函数": lambda v: (v['low[0]'] < v['low[-5]']) and (v['low[0]'] < v['low[-10]']) and
                           (v['boll_mid[0]'] > v['boll_mid[-5]']) and (v['boll_mid[0]'] > v['boll_mid[-10]']),
     "买入得分": 2.0, "卖出得分": 0.0, "备注": "所有行情通用，最可靠的反转信号"},
    {"优先级": "最高（反转）", "序号": 2, "信号类型": "顶背离",
     "条件函数": lambda v: (v['high[0]'] > v['high[-5]']) and (v['high[0]'] > v['high[-10]']) and
                           (v['boll_mid[0]'] < v['boll_mid[-5]']) and (v['boll_mid[0]'] < v['boll_mid[-10]']),
     "买入得分": 0.0, "卖出得分": 2.0, "备注": "所有行情通用，最可靠的反转信号"},

    {"优先级": "高（极端）", "序号": 3, "信号类型": "下轨超卖",
     "条件函数": lambda v: (v['close[0]'] < v['boll_low[0]']) and (v['volume[0]'] < 0.8 * v['volume_ma5[0]']),
     "买入得分": 1.8, "卖出得分": 0.0, "备注": "震荡市有效，单边市易钝化"},
    {"优先级": "高（极端）", "序号": 4, "信号类型": "上轨超买",
     "条件函数": lambda v: (v['close[0]'] > v['boll_up[0]']) and (v['volume[0]'] > 1.2 * v['volume_ma5[0]']) and
                           (v['close[0]'] <= v['close[-1]']),
     "买入得分": 0.0, "卖出得分": 1.8, "备注": "震荡市有效，单边市易钝化"},

    {"优先级": "中高（确认）", "序号": 5, "信号类型": "下轨反弹",
     "条件函数": lambda v: (v['close[-1]'] < v['boll_low[-1]']) and (v['close[0]'] >= v['boll_low[0]']) and
                           (v['close[0]'] <= v['boll_mid[0]']),
     "买入得分": 1.5, "卖出得分": 0.0, "备注": "确认超卖后的反弹信号"},
    {"优先级": "中高（确认）", "序号": 6, "信号类型": "上轨回落",
     "条件函数": lambda v: (v['close[-1]'] > v['boll_up[-1]']) and (v['close[0]'] <= v['boll_up[0]']) and
                           (v['close[0]'] >= v['boll_mid[0]']),
     "买入得分": 0.0, "卖出得分": 1.5, "备注": "确认超买后的回调信号"},

    {"优先级": "中高（趋势）", "序号": 7, "信号类型": "强势多头",
     "条件函数": lambda v: (v['boll_mid[0]'] > v['boll_mid[-1]']) and (v['boll_mid[-1]'] > v['boll_mid[-2]']) and
                           (v['boll_mid[-2]'] > v['boll_mid[-3]']) and (v['close[0]'] > v['boll_mid[0]']),
     "买入得分": 1.2, "卖出得分": 0.3, "备注": "趋势市中，以持有为主"},
    {"优先级": "中高（趋势）", "序号": 8, "信号类型": "强势空头",
     "条件函数": lambda v: (v['boll_mid[0]'] < v['boll_mid[-1]']) and (v['boll_mid[-1]'] < v['boll_mid[-2]']) and
                           (v['boll_mid[-2]'] < v['boll_mid[-3]']) and (v['close[0]'] < v['boll_mid[0]']),
     "买入得分": 0.3, "卖出得分": 1.2, "备注": "趋势市中，以观望为主"},

    # 关键修复：将"收口整理"移到"中轨之上/之下"之前
    {"优先级": "低（平衡）", "序号": 11, "信号类型": "收口整理",
     "条件函数": lambda v: ((v['boll_up[0]'] - v['boll_low[0]']) < 0.8 * (v['boll_up[-5]'] - v['boll_low[-5]'])) and
                           (abs(v['close[0]'] - v['boll_mid[0]']) / v['boll_mid[0]'] < 0.02),
     "买入得分": 0.5, "卖出得分": 0.5, "备注": "无明确方向，建议观望"},

    {"优先级": "常规（方向）", "序号": 9, "信号类型": "中轨之上",
     "条件函数": lambda v: v['close[0]'] > v['boll_mid[0]'],
     "买入得分": 1.0, "卖出得分": 0.3, "备注": "原书多空划分核心标准"},
    {"优先级": "常规（方向）", "序号": 10, "信号类型": "中轨之下",
     "条件函数": lambda v: v['close[0]'] < v['boll_mid[0]'],
     "买入得分": 0.3, "卖出得分": 1.0, "备注": "原书多空划分核心标准"}
]


# ===================== 第三步：安全取值函数 =====================
def safe_get_boll(arr, idx):
    try:
        return float(arr[idx])
    except (IndexError, TypeError, ValueError):
        return 0.0


# ===================== 第四步：核心评分函数（已修复索引问题） =====================
def calculate_boll_score(strategy):
    boll_mid = strategy.boll.mid
    boll_up = strategy.boll.top
    boll_low = strategy.boll.bot
    high = strategy.data.high
    low = strategy.data.low
    close = strategy.data.close
    volume = strategy.data.volume
    volume_ma5 = strategy.volume_ma5

    # 关键修复：索引转换为相对当前的正确位置
    var_dict = {
        # 当前值（最新） -> 用 -1
        'boll_mid[0]': safe_get_boll(boll_mid, -1),
        'boll_mid[-1]': safe_get_boll(boll_mid, -2),
        'boll_mid[-2]': safe_get_boll(boll_mid, -3),
        'boll_mid[-3]': safe_get_boll(boll_mid, -4),
        'boll_mid[-5]': safe_get_boll(boll_mid, -6),  # 5天前
        'boll_mid[-10]': safe_get_boll(boll_mid, -11),  # 10天前

        'boll_up[0]': safe_get_boll(boll_up, -1),
        'boll_up[-1]': safe_get_boll(boll_up, -2),
        'boll_up[-5]': safe_get_boll(boll_up, -6),

        'boll_low[0]': safe_get_boll(boll_low, -1),
        'boll_low[-1]': safe_get_boll(boll_low, -2),
        'boll_low[-5]': safe_get_boll(boll_low, -6),

        'high[0]': safe_get_boll(high, -1),
        'high[-5]': safe_get_boll(high, -6),
        'high[-10]': safe_get_boll(high, -11),

        'low[0]': safe_get_boll(low, -1),
        'low[-5]': safe_get_boll(low, -6),  # 5天前
        'low[-10]': safe_get_boll(low, -11),  # 10天前

        'close[0]': safe_get_boll(close, -1),
        'close[-1]': safe_get_boll(close, -2),

        'volume[0]': safe_get_boll(volume, -1),
        'volume_ma5[0]': safe_get_boll(volume_ma5, -1),

        'abs': abs, 'max': max, 'min': min
    }

    # 新增：检查BOLL钝化状态
    is_dull = _check_boll钝化(strategy)

    priority_weight_map = {
        '最高': 2.0,
        '高': 1.5,
        '中高': 1.2,
        '常规': 1.0,
        '低': 0.5
    }

    for rule in BOLL_SCORE_TABLE:
        try:
            # 跳过钝化时的规则3和4（下轨超卖/上轨超买）
            if is_dull and rule["序号"] in [3, 4]:
                continue

            condition_met = rule["条件函数"](var_dict)
            if condition_met:
                if rule.get("序号") == 12:
                    return {
                        "signal_type": "极端单边行情-停止操作",
                        "buy_score": 0.0,
                        "sell_score": 0.0,
                        "net_score": 0.0,
                        "extreme_market": True,
                        "triggered_signals": ["极端单边行情"]
                    }
                raw_buy = float(rule["买入得分"])
                raw_sell = float(rule["卖出得分"])
                priority_key = rule["优先级"].split("（")[0].strip()
                # weight = priority_weight_map.get(priority_key, 1.0)
                weight = 1
                final_buy = round(raw_buy * weight, 2)
                final_sell = round(raw_sell * weight, 2)
                signal_name = f"{rule['优先级']}-{rule['信号类型']}"
                return {
                    "signal_type": signal_name,
                    "buy_score": final_buy,
                    "sell_score": final_sell,
                    "net_score": round(final_buy - final_sell, 2),
                    "extreme_market": False,
                    "triggered_signals": [signal_name],
                    "weight_used": weight
                }
        except Exception as e:
            print(f"规则{rule['序号']}执行异常: {e}")
            continue

    return {
        "signal_type": "无信号",
        "buy_score": 0.0,
        "sell_score": 0.0,
        "net_score": 0.0,
        "extreme_market": False,
        "triggered_signals": []
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
            return self.array[idx]
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
                 volume_vals=None, volume_ma5_vals=None,
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
        self.volume_ma5 = BacktraderLine(volume_ma5_vals or [10000] * default_len)
        self.p = MockParams(stock_type=stock_type)


# ===================== 第六步：完整测试用例 =====================
class TestBOLLScoreCalculation(unittest.TestCase):
    def test_01_bottom_divergence(self):
        print("【测试1】底背离")
        low_vals = [9.0, 8.5, 8.0, 8.5, 9.0, 8.0] * 3 + [7.5, 7.5]
        boll_mid_vals = [10.0, 10.1, 10.2, 10.3, 10.4, 10.5] * 3 + [10.6, 10.6]
        close_vals = [8.0] * 19 + [7.5]
        strat = MockStrategy(
            boll_mid_vals=boll_mid_vals,
            low_vals=low_vals,
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
        high_vals = [11.0, 11.5, 12.0, 11.5, 11.0, 12.0] * 3 + [12.5]
        boll_mid_vals = [10.0, 9.9, 9.8, 9.7, 9.6, 9.5] * 3 + [9.4]
        close_vals = [11.5] * 19 + [12.5]
        strat = MockStrategy(
            boll_mid_vals=boll_mid_vals,
            high_vals=high_vals,
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
            boll_bot_vals=[9.0] * 20
        )
        score = calculate_boll_score(strat)
        print(f"测试3结果: {score}")
        self.assertEqual(score["signal_type"], "高（极端）-下轨超卖")
        self.assertEqual(score["buy_score"], 2.7)
        self.assertEqual(score["sell_score"], 0.0)

    def test_04_upper_band_overbuy(self):
        """测试4：高优先级 - 上轨超买"""
        print("【测试4】上轨超买")
        close_vals = [11.2] * 20
        volume_vals = [13000] * 20
        strat = MockStrategy(
            close_vals=close_vals,
            volume_vals=volume_vals,
            boll_top_vals=[11.0] * 20
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
        self.assertEqual(score["buy_score"], 1.8)
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
        self.assertEqual(score["sell_score"], 1.8)
        self.assertEqual(score["buy_score"], 0.0)

    def test_07_strong_bull_trend(self):
        """测试7：中高优先级 - 强势多头（修复数据构造）"""
        print("【测试7】强势多头")
        # 修复：确保中轨连续3日向上（9.9 < 10.0 < 10.1 < 10.2）
        boll_mid_vals = [10.0] * 16 + [9.9, 10.0, 10.1, 10.2]
        close_vals = [10.3] * 20  # 10.3 > 10.2
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
        """测试8：中高优先级 - 强势空头（修复数据构造）"""
        print("【测试8】强势空头")
        # 修复：确保中轨连续3日向下（10.2 > 10.1 > 10.0 > 9.9）
        boll_mid_vals = [10.0] * 16 + [10.2, 10.1, 10.0, 9.9]
        close_vals = [9.8] * 20  # 9.8 < 9.9
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
        """测试11：低优先级 - 收口整理（修复数据构造）"""
        print("【测试11】收口整理")
        # 修复：前15天保持带宽4.0（12.0-8.0），最后5天带宽缩窄到1.6（10.8-9.2）
        boll_top_vals = [12.0] * 15 + [10.8] * 5
        boll_bot_vals = [8.0] * 15 + [9.2] * 5
        close_vals = [10.1] * 20
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
        """测试12：无优先级 - 极端单边行情"""
        print("【测试12】极端单边行情")
        close_vals = [10.0] * 19 + [11.0]
        high_vals = [10.5] * 19 + [11.0]
        low_vals = [9.5] * 19 + [11.0]
        strat = MockStrategy(
            close_vals=close_vals,
            high_vals=high_vals,
            low_vals=low_vals
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
        BOLL_SCORE_TABLE = [rule for rule in BOLL_SCORE_TABLE if rule["序号"] not in [9, 10, 11]]

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
        # 构造钝化条件：5天前close=9.0，当前close=8.0，下跌11.1% > 8%
        close_vals = [10.0] * 15 + [9.0, 8.9, 8.8, 8.7, 8.0]
        boll_bot_vals = [9.0] * 20  # 修正：使用正确的参数名boll_bot_vals
        boll_mid_vals = [10.0] * 20  # 中轨固定10.0

        strat = MockStrategy(
            close_vals=close_vals,
            boll_bot_vals=boll_bot_vals,  # 修正：使用正确的参数名
            boll_mid_vals=boll_mid_vals
        )
        score = calculate_boll_score(strat)
        print(f"测试14结果: {score}")

        # 钝化时规则3（下轨超卖）被跳过，应触发规则10（中轨之下）
        self.assertEqual(score["signal_type"], "常规（方向）-中轨之下")
        self.assertEqual(score["sell_score"], 1.0)
        self.assertEqual(score["buy_score"], 0.3)


if __name__ == '__main__':
    unittest.main(verbosity=2)