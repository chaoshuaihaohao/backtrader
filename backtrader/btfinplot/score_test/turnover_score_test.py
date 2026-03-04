import unittest

# ========== 1. 业务规则配置（简化：直接定义得分，无需单独权重字典）==========
TURNOVER_RULES = {
    "大盘股": {
        "极低位_E": 1.0,  # <1% (E区间)
        "低位_D": (1.0, 2.0),  # 1%-2% (D区间)
        "健康区_C": (1.0, 3.0),  # 1%-3% (C区间)
        "高位_B": (3.0, 5.0),  # 3%-5% (B区间)
        "极端高位_A": 5.0,  # >5% (A区间)
        "放量突破阈值": 3.0 * 1.5  # 4.5%
    },
    "中盘股": {
        "极低位_E": 1.0,  # <1% (E区间)
        "低位_D": (1.0, 2.0),  # 1%-2% (D区间)
        "健康区_C": (2.0, 5.0),  # 2%-5% (C区间)
        "高位_B": (5.0, 8.0),  # 5%-8% (B区间)
        "极端高位_A": 8.0,  # >8% (A区间)
        "放量突破阈值": 5.0 * 1.5  # 7.5%
    },
    "小盘股": {
        "极低位_E": 1.0,  # <1% (E区间)
        "低位_D": (1.0, 3.0),  # 1%-3% (D区间)
        "健康区_C": (3.0, 8.0),  # 3%-8% (C区间)
        "高位_B": (8.0, 15.0),  # 8%-15% (B区间)
        "极端高位_A": 15.0,  # >15% (A区间)
        "放量突破阈值": 8.0 * 1.5  # 12%
    }
}

# 移除：SIGNAL_WEIGHTS 本身就是业务规则的得分，无需单独定义
# SIGNAL_WEIGHTS = {
#     "趋势反转": 2.0,
#     "连续极值": 1.5,
#     "动态确认": 1.2,
#     "静态结构": 1.0,
#     "平衡观望": 0.5
# }


# ========== 2. 安全取值函数（保持不变）==========
def safe_get(line_obj, idx, default=0.0):
    """
    Backtrader专用安全取值函数
    - idx=0: 当前/最新Bar（列表最后一个元素）
    - idx=-1: 上一个Bar（列表倒数第二个元素）
    - idx=-2: 上上个Bar（列表倒数第三个元素）
    """
    try:
        data_list = line_obj.array if hasattr(line_obj, 'array') else []
        if not data_list:
            return default
        if idx > 0:
            return default

        last_element_idx = len(data_list) - 1
        target_pos = last_element_idx + idx
        if target_pos < 0 or target_pos >= len(data_list):
            return default

        return data_list[target_pos]
    except (IndexError, TypeError, AttributeError):
        return default


# ========== 3. 核心评分函数（修复：去掉重复的权重乘法）==========
def calculate_turnover_score(strat):
    """
    换手率评分函数（对齐业务手册：得分直接使用规则定义值，无额外权重）
    返回值：净得分 = 买入得分 - 卖出得分（正向为看多，负向为看空）
    """
    if not hasattr(strat, "turnover") or len(strat.turnover.array) < 3:
        return 0.0

    stock_type = getattr(strat.p, "stock_type", "中盘股")
    cfg = TURNOVER_RULES.get(stock_type, TURNOVER_RULES["中盘股"])

    # 提取核心数据
    to_0 = safe_get(strat.turnover, 0)  # 当前换手率
    to_1 = safe_get(strat.turnover, -1)  # 前1日换手率
    to_2 = safe_get(strat.turnover, -2)  # 前2日换手率
    close_0 = safe_get(strat.data.close, 0)  # 当前收盘价
    close_1 = safe_get(strat.data.close, -1)  # 前1日收盘价
    ma60_0 = safe_get(strat.ma60, 0, default=close_0)  # 当前MA60

    # 计算20日最高价
    close_20d = []
    for i in range(0, -20, -1):
        close_20d.append(safe_get(strat.data.close, i))
    max_20d_close = max(close_20d) if close_20d else close_0

    # 判断换手率所属区间
    turnover_range = ""
    if to_0 > cfg["极端高位_A"]:
        turnover_range = "A"  # 极端高位
    elif cfg["高位_B"][0] <= to_0 <= cfg["高位_B"][1]:
        turnover_range = "B"  # 高位
    elif cfg["健康区_C"][0] <= to_0 <= cfg["健康区_C"][1]:
        turnover_range = "C"  # 健康区间
    elif cfg["低位_D"][0] <= to_0 <= cfg["低位_D"][1]:
        turnover_range = "D"  # 低位
    elif to_0 < cfg["极低位_E"]:
        turnover_range = "E"  # 极低位
    else:
        turnover_range = "UNKNOWN"

    # 初始化得分
    buy_score = 0.0
    sell_score = 0.0

    # ========== 区间A：极端高位（>极端阈值）==========
    if turnover_range == "A":
        # 规则1：放量突破确认（趋势反转，买入得分2）→ 直接用2.0，无额外乘法
        if (to_0 >= cfg["放量突破阈值"]) and (close_0 > ma60_0) and (close_0 > close_1):
            buy_score = 2.0
        # 规则2：天量见顶（趋势反转，卖出得分2）
        elif (to_1 > cfg["极端高位_A"]) and (close_1 == max_20d_close) and (to_0 < to_1 * 0.8):
            sell_score = 2.0
        # 规则3：放量杀跌（趋势反转，卖出得分1.8）
        elif (to_0 > cfg["极端高位_A"]) and (close_0 < ma60_0) and (close_0 < close_1):
            sell_score = 1.8
        # 补充规则：极端高换手横盘（偏空，卖出得分0.5）
        elif abs(close_0 - close_1) / close_1 < 0.02:
            sell_score = 0.5
        # 其他极端高换手（偏空，卖出得分1.5）
        else:
            sell_score = 1.5

    # ========== 区间B：高位（高换手区间）==========
    elif turnover_range == "B":
        # 规则：连续高换手（连续极值，卖出得分1.5）
        if (to_2 >= cfg["高位_B"][0]) and (to_1 >= cfg["高位_B"][0]) and (to_0 >= cfg["高位_B"][0]) and (
                close_0 <= close_1):
            sell_score = 1.5

    # ========== 区间C：健康区间 ==========
    elif turnover_range == "C":
        # 规则1：健康量价（动态确认，买入得分1.5）
        if close_0 > close_1:
            buy_score = 1.5
        # 规则2：量价背离（动态确认，卖出得分1.5）
        elif (close_0 > close_1) and (to_0 < cfg["健康区_C"][0]):
            sell_score = 1.5
        # 规则3：温和放量（静态结构，买入0.8/卖出0.3）
        else:
            buy_score = 0.8
            sell_score = 0.3

    # ========== 区间D：低位 ==========
    elif turnover_range == "D":
        # 规则1：洗盘换手（动态确认，买入得分1.2）
        if (to_0 < cfg["健康区_C"][0]) and (close_0 > ma60_0) and (close_0 < close_1):
            buy_score = 1.2
        # 规则2：连续地量（连续极值，买入得分1.5）
        elif (to_2 <= cfg["极低位_E"]) and (to_1 <= cfg["极低位_E"]) and (to_0 <= cfg["极低位_E"]):
            buy_score = 1.5

    # ========== 区间E：极低位 ==========
    elif turnover_range == "E":
        # 规则1：地量筑底（趋势反转，买入得分1.8）
        if (to_2 <= cfg["极低位_E"]) and (to_1 <= cfg["极低位_E"]) and (to_0 <= cfg["极低位_E"]) and (
                close_0 > close_1):
            buy_score = 1.8
        # 规则2：温和缩量（静态结构，买入0.5/卖出0.5）
        elif abs(close_0 - close_1) / close_1 < 0.02:
            buy_score = 0.5
            sell_score = 0.5
        # 规则3：无量横盘（平衡观望，买入0.5/卖出0.5）
        else:
            buy_score = 0.5
            sell_score = 0.5

    # 计算净得分
    net_score = round(buy_score - sell_score, 2)
    return net_score


# ========== 4. 模拟Backtrader对象（保持不变）==========
class MockParams:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class BacktraderLine:
    def __init__(self, values):
        self.array = list(values)

    def __getitem__(self, idx):
        try:
            if idx >= 0:
                return self.array[idx] if idx < len(self.array) else 0.0
            else:
                return self.array[idx] if len(self.array) >= abs(idx) else 0.0
        except (IndexError, TypeError):
            return 0.0

    def get(self, size=None):
        if size is None or size <= 0:
            return self.array.copy()
        return self.array[-size:] if len(self.array) >= size else self.array.copy()

    def __len__(self):
        return len(self.array)


class MockStrategy:
    def __init__(self, turnover_vals, close_vals=None, ma60_vals=None, stock_type="中盘股"):
        self.turnover = BacktraderLine(turnover_vals)
        default_close = [10.0] * len(turnover_vals) if close_vals is None else close_vals
        self.data = type('obj', (object,), {
            'close': BacktraderLine(default_close),
            'high': BacktraderLine([x * 1.05 for x in default_close]),
            'open': BacktraderLine([x * 0.98 for x in default_close])
        })
        self.p = MockParams(stock_type=stock_type)
        default_ma60 = [9.5] * len(turnover_vals) if ma60_vals is None else ma60_vals
        self.ma60 = BacktraderLine(default_ma60)


# ========== 5. 测试用例（修正预期值：对齐无重复权重的得分）==========
class TestTurnoverScoreCalculation(unittest.TestCase):
    def setUp(self):
        print("\n--- Setting up Test ---")

    def test_01_low_turnover_dapan_stock(self):
        """测试1：大盘股低换手率 (0.74%) - 无量横盘"""
        print("【测试1】大盘股低换手率")
        turnover_data = [0.1] * 20 + [0.74, 0.74, 0.74]
        close_data = [10.0] * 20 + [10.0, 10.0, 10.0]
        strat = MockStrategy(turnover_data, close_vals=close_data, stock_type="大盘股")
        score = calculate_turnover_score(strat)
        print(f"测试1结果: 换手率0.74% -> 评分 {score}")
        self.assertEqual(score, 0.0, "大盘股极低位横盘应返回0分")

    def test_02_high_turnover_xiaopan_stock(self):
        """测试2：小盘股高换手率 (26.59%) - 放量杀跌"""
        print("【测试2】小盘股高换手率")
        turnover_data = [10.0] * 10 + [27.0, 26.59]
        close_data = [15.0] * 10 + [15.0, 14.0]
        ma60_data = [14.5] * 12
        strat = MockStrategy(turnover_data, close_vals=close_data, ma60_vals=ma60_data, stock_type="小盘股")
        score = calculate_turnover_score(strat)
        print(f"测试2结果: 换手率26.59% -> 评分 {score}")
        self.assertEqual(score, -1.8, "小盘股放量杀跌应返回-1.8分")  # 原-3.6 → 修正为-1.8

    def test_03_mid_turnover_zhongpan_stock(self):
        """测试3：中盘股中等换手率 (3.0%) - 健康量价"""
        print("【测试3】中盘股中等换手率")
        turnover_data = [1.0] * 15 + [3.0]
        close_data = [10.0] * 15 + [10.1]
        strat = MockStrategy(turnover_data, close_vals=close_data, stock_type="中盘股")
        score = calculate_turnover_score(strat)
        print(f"测试3结果: 换手率3.0% -> 评分 {score}")
        self.assertEqual(score, 1.5, "中盘股健康量价应返回1.5分")  # 原1.8 → 修正为1.5

    def test_04_no_data(self):
        """测试4：无数据情况"""
        print("【测试4】无数据")
        strat = MockStrategy([], stock_type="大盘股")
        score = calculate_turnover_score(strat)
        print(f"测试4结果: 无数据 -> 评分 {score}")
        self.assertEqual(score, 0, "无数据时应返回0分")

    def test_05_extreme_high_turnover_breakout(self):
        """测试5：小盘股极端高换手突破（18%）- 放量突破确认"""
        print("【测试5】小盘股极端高换手突破 (18%)")
        turnover_data = [5.0] * 10 + [17.0, 18.0]
        close_data = [20.0] * 10 + [20.0, 20.5]
        ma60_data = [20.0] * 12
        strat = MockStrategy(turnover_data, close_vals=close_data, ma60_vals=ma60_data, stock_type="小盘股")
        score = calculate_turnover_score(strat)
        print(f"测试5结果: 换手率18% (小盘股) -> 评分 {score}")
        self.assertEqual(score, 2.0, "小盘股放量突破应返回2.0分")  # 原4.0 → 修正为2.0

    def test_06_continuous_low_volume_bottom(self):
        """测试6：大盘股连续地量筑底（0.5%）"""
        print("【测试6】连续地量筑底 (连续3日 0.5%)")
        turnover_data = [10.0] * 5 + [0.5, 0.5, 0.5]
        close_data = [9.0] * 5 + [9.0, 9.0, 9.1]
        strat = MockStrategy(turnover_data, close_vals=close_data, stock_type="大盘股")
        score = calculate_turnover_score(strat)
        print(f"测试6结果: 连续地量 -> 评分 {score}")
        self.assertEqual(score, 1.8, "连续地量筑底应返回1.8分")  # 原3.6 → 修正为1.8

    def test_07_continuous_high_volume_stagnation(self):
        """测试7：小盘股连续高换手滞涨（14%）"""
        print("【测试7】小盘股连续高换手滞涨 (14%)")
        turnover_data = [5.0] * 5 + [14.0, 14.0, 14.0]
        close_data = [15.0] * 5 + [15.0, 15.0, 15.0]
        strat = MockStrategy(turnover_data, close_vals=close_data, stock_type="小盘股")
        score = calculate_turnover_score(strat)
        print(f"测试7结果: 连续高换手 -> 评分 {score}")
        self.assertEqual(score, -1.5, "连续高换手滞涨应返回-1.5分")  # 原-2.25 → 修正为-1.5

    def test_08_stock_type_thresholds(self):
        """测试8：不同盘口阈值差异（同为 4% 换手率）"""
        print("【测试8】不同盘口阈值差异 (同为 4% 换手率)")
        turnover_data = [1.0] * 10 + [4.0]
        close_data = [10.0] * 10 + [10.1]

        strat_dapan = MockStrategy(turnover_data, close_vals=close_data, stock_type="大盘股")
        score_dapan = calculate_turnover_score(strat_dapan)

        strat_zhongpan = MockStrategy(turnover_data, close_vals=close_data, stock_type="中盘股")
        score_zhongpan = calculate_turnover_score(strat_zhongpan)

        strat_xiaopan = MockStrategy(turnover_data, close_vals=close_data, stock_type="小盘股")
        score_xiaopan = calculate_turnover_score(strat_xiaopan)

        print(f"测试8结果 - 大盘股(4%): {score_dapan}, 中盘股(4%): {score_zhongpan}, 小盘股(4%): {score_xiaopan}")
        self.assertEqual(score_dapan, 0.0, "大盘股4%应返回0分")
        self.assertEqual(score_zhongpan, 1.5, "中盘股4%应返回1.5分")  # 原1.8 → 修正为1.5
        self.assertEqual(score_xiaopan, 1.5, "小盘股4%应返回1.5分")  # 原1.8 → 修正为1.5

    def test_09_volume_shrink_during_callback(self):
        """测试9：中盘股回调洗盘换手（1.5%）"""
        print("【测试9】回调洗盘换手 (1.5%)")
        turnover_data = [3.0] * 5 + [1.5]
        close_data = [10.0] * 5 + [9.9]
        ma60_data = [9.8] * 6
        strat = MockStrategy(turnover_data, close_vals=close_data, ma60_vals=ma60_data, stock_type="中盘股")
        score = calculate_turnover_score(strat)
        print(f"测试9结果: 洗盘换手 -> 评分 {score}")
        self.assertEqual(score, 1.2, "回调洗盘缩量应返回1.2分")  # 原1.44 → 修正为1.2

    def test_10_extreme_high_turnover_sideways(self):
        """测试10：小盘股极端高换手横盘（16%）"""
        print("【测试10】小盘股极端高换手横盘")
        turnover_data = [5.0] * 10 + [16.0]
        close_data = [20.0] * 10 + [20.1]
        ma60_data = [20.2] * 11
        strat = MockStrategy(turnover_data, close_vals=close_data, ma60_vals=ma60_data, stock_type="小盘股")
        score = calculate_turnover_score(strat)
        print(f"测试10结果: 换手率16%（横盘）-> 评分 {score}")
        self.assertEqual(score, -0.5, "极端高换手横盘应返回-0.5分")


if __name__ == '__main__':
    unittest.main(verbosity=2)