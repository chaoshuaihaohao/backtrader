import unittest
import backtrader as bt
import math

# ===================== 核心常量 =====================
PRICE_POSITION_KDJ_COEFF = {
    "低位": {"超卖": 1.2, "超买": 1.0},
    "中位": {"超卖": 1.0, "超买": 1.0},
    "高位": {"超卖": 1.0, "超买": 1.2}
}


def _get_kdj_rule_table(strategy):
    """
    动态生成KDJ评分规则表（绑定策略对象用于条件判断）
    策略对象(strategy)需包含的核心属性：
    - k/d/j: KDJ指标序列，[0]为当前值，[-1]前1日，[-2]前2日
    - price: 价格序列，[0]当前价格
    - ma5/ma10/ma20/ma60: 均线序列，[0]当前值，[-1]前1日
    - vol: 成交量序列，[0]当前值；vol_5ma: 5日均量序列，[0]当前值
    - rsi6: 6日RSI序列，[0]当前值
    - boll_price/boll_lb/boll_ub/boll_mid/boll_bandwidth: BOLL带相关，[0]当前值
    - bias6: 6日BIAS序列，[0]当前值；bias_over_sell/bias_over_buy: BIAS超卖/超买阈值
    - macd_diff/macd_dea/macd_bar: MACD相关序列，[0]当前值，[-1]前1日
    - turnover: 换手率序列，[0]当前值；turnover_low_threshold: 低换手阈值
    """
    return [
        # ===================== 最高（趋势反转）- 权重2.0 =====================
        # {"priority": "最高（趋势反转）",
        #  "signal_type": "dynamic",
        #  "description": "低位反转金叉",
        #  "code_condition": lambda: (strategy.k[0] > strategy.d[0] and strategy.k[-1] < strategy.d[-1] and  # kdj金叉
        #                             strategy.k[-1] < 30 and strategy.d[-1] < 30),
        #  "confirm_condition": lambda: (
        #      # 核心必满足：MA60向上 + 放量
        #          (strategy.price[0] >= strategy.ma60[0] and strategy.ma60[0] >= strategy.ma60[-1]) and
        #          (strategy.vol[0] >= 1.5 * strategy.vol5[0]) and
        #          # 辅助至少1项：RSI6超卖 / BIAS超卖 / MACD DIFF≥DEA
        #          (strategy.rsi6[0] < 30 or strategy.bias6[0] < -5 or strategy.macd_diff[0] >=
        #           strategy.macd_dea[0])),
        #  "buy_score": 2.0, "sell_score": 0.0, "signal_type_weight": 2.0, "is_extreme": False
        #  },
        {"priority": "最高（趋势反转）",
         "signal_type": "dynamic",
         "description": "低位反转金叉",
         "code_condition": lambda: (
                 strategy.k[0] > strategy.d[0] and strategy.k[-1] < strategy.d[-1] and  # KDJ金叉
                 strategy.k[-1] < 30 and strategy.d[-1] < 30 and  # 金叉前处于超卖区
                 strategy.k[0] - strategy.k[-1] >= 3  # K线从低位快速回升（排除粘合金叉）
         ),
         "confirm_condition": lambda: (
             # 核心1：趋势不弱（MA60走平/向上，股价不跌破MA60太多）
                 (strategy.price[0] >= strategy.ma60[0] * 0.98 and strategy.ma60[0] >= strategy.ma60[-1] * 0.995) and
                 # 核心2：量能合理（温和放量，排除天量诱多）
                 (1.2 * strategy.vol_5ma[0] <= strategy.vol[0] <= 3.0 * strategy.vol_5ma[0]) and
                 # 核心3：辅助至少1项（兼容低位反转的指标特征）
                 (
                         strategy.rsi6[0] < 35 or  # RSI轻微超卖（放宽一点，避免漏信号）
                         strategy.bias6[0] < -4 or  # BIAS超卖（比-5放宽，适配不同股票）
                         (strategy.macd_diff[0] >= strategy.macd_dea[0] - 0.3 and strategy.macd_bar[0] >
                          strategy.macd_bar[-1])  # MACD绿柱缩短/即将金叉
                 )
         ),
         "buy_score": 2.0, "sell_score": 0.0, "signal_type_weight": 2.0, "is_extreme": False
         },
        {"priority": "最高（趋势反转）",
         "signal_type": "dynamic",
         "description": "高位反转死叉",
         "code_condition": lambda: (strategy.k[0] < strategy.d[0] and strategy.k[-1] > strategy.d[-1] and  # kdj金叉
                                    strategy.k[-1] > 80 and strategy.d[-1] > 80),
         "confirm_condition": lambda: (
             # 核心必满足：MA60向上 + 放量
                 (strategy.vol[0] >= 1.5 * strategy.vol_5ma[0]) and
                 # 辅助至少1项：RSI6超卖 / BIAS超卖 / MACD DIFF≥DEA
                 (strategy.rsi6[0] > 70 or strategy.bias6[0] > 5 or strategy.macd_diff[0] >=
                  strategy.macd_dea[0])
         ),
         "buy_score": 0.0, "sell_score": 2.0, "signal_type_weight": 2.0, "is_extreme": False},
    ]
# 优先级映射（用于排序）
PRIORITY_ORDER = {
    "最高（趋势反转）": 5,
    "高（连续极值）": 4,
    "中高（动态金叉分级）": 3,
    "中高（动态死叉分级）": 3,
    "中高（动态拐头）": 3,
    "中高（动态企稳）": 3,
    "中高（动态后效）": 3,
    "常规（静态排列）": 2,
    "低（粘合平衡）": 1
}


def kdj_get_price_position(strat):
    """
    价格位置计算 - 修复索引越界问题，严格遵循Backtrader索引规则
    :param strat: backtrader策略实例
    :return: 低位/中位/高位
    """
    try:
        # Backtrader原生长度判断
        if len(strat.data.close) < 20 or len(strat.ma60) < 1:
            return '中位'

        # 直接访问backtrader指标的当前值
        close = float(strat.data.close[0])
        ma60 = float(strat.ma60[0])

        # 修复：正确读取最近20个收盘价（使用负数索引）
        recent_closes = []
        # 读取最近20个bar：0=当前, -1=前1, -2=前2...-19=前19
        for i in range(20):
            idx = -i if i > 0 else 0  # i=0→0, i=1→-1, i=2→-2...
            try:
                # 检查索引是否有效
                if abs(idx) < len(strat.data.close):
                    recent_closes.append(float(strat.data.close[idx]))
                else:
                    break  # 数据不足时停止
            except Exception:
                break

        if len(recent_closes) < 10:
            return '中位'

        # 价格位置判断逻辑
        if close <= ma60:
            return '低位'
        elif close >= max(recent_closes):  # 修正：>= 更合理
            return '高位'
        else:
            return '中位'
    except Exception as e:
        # 详细打印错误信息（便于调试）
        data_len = len(strat.data.close) if (hasattr(strat, 'data') and hasattr(strat.data, 'close')) else 'N/A'
        print(f"获取价格位置出错: {e} | 数据长度: {data_len}")
        return '中位'

def calculate_kdj_score(strategy):
    """
    KDJ评分计算核心函数 - 重构为使用lambda条件的动态规则表
    :param strategy: backtrader策略实例
    :return: 包含各类得分的字典
    """
    # 1. 校验必要属性
    required_attrs = ['k', 'd', 'j', 'data', 'ma60']
    for attr in required_attrs:
        if not hasattr(strategy, attr):
            raise ValueError(f"策略实例缺少必要属性：{attr}")
    if not hasattr(strategy.data, 'close'):
        raise ValueError("strategy.data 缺少 close 属性")

    # 2. 动态生成规则表（绑定当前strategy对象）
    KDJ_RULE_TABLE = _get_kdj_rule_table(strategy)

    # 3. 匹配评分规则（直接调用lambda函数判断条件）
    matched_rules = []
    for rule in KDJ_RULE_TABLE:
        try:
            # 直接执行lambda条件函数
            if rule["code_condition"]():
                matched_rules.append(rule)
        except Exception as e:
            print(f"匹配规则{rule['description']}出错: {e}")
            continue

    # 4. 无匹配规则的情况
    if not matched_rules:
        price_pos = kdj_get_price_position(strategy)
        return {
            "signal_type": "无信号",
            "raw_buy": 0.0,
            "raw_sell": 0.0,
            "signal_type_weight": 1.0,
            "price_position": price_pos,
            "buy_score": 0.0,
            "sell_score": 0.0,
            "net_score": 0.0
        }

    # 5. 排序获取最优规则（优化排序逻辑，确保优先级准确）
    def get_sort_key(rule):
        priority_level = PRIORITY_ORDER.get(rule['priority'], 0)
        # 修复：更准确的条件复杂度计算
        condition_code = rule['code_condition'].__code__.co_code
        condition_complexity = len(condition_code) if condition_code else 0
        sticky_bonus = 1 if "粘合" in rule['description'] else 0

        return (
            -priority_level,  # 优先级降序（最高优先级先选）
            -rule["signal_type_weight"],  # 权重降序
            sticky_bonus,  # 粘合规则优先
            -condition_complexity,  # 条件复杂度降序（更复杂的规则更具体）
            KDJ_RULE_TABLE.index(rule)  # 原始顺序升序（保证相同优先级的规则顺序）
        )

    # 按排序键排序，取第一个（最优）规则
    matched_rules_sorted = sorted(matched_rules, key=get_sort_key)
    best_rule = matched_rules_sorted[0]

    # 6. 辅助：获取当前KDJ值（用于返回结果，不影响核心逻辑）
    def get_kdj_val(indicator, idx):
        """内联极简版指标值读取"""
        try:
            if len(indicator) > abs(idx):
                val = float(indicator[idx])
                return max(0.0, min(100.0, val))
            return 50.0
        except:
            return 50.0

    # 7. 计算得分（逻辑不变）
    raw_buy = round(best_rule["buy_score"], 4)
    raw_sell = round(best_rule["sell_score"], 4)
    weight = round(best_rule["signal_type_weight"], 4)

    # 基础得分计算
    weighted_buy = round(raw_buy * weight, 4)
    weighted_sell = round(raw_sell * weight, 4)

    # 8. 价格位置修正（仅对极值信号生效）
    price_pos = kdj_get_price_position(strategy)
    if best_rule.get("is_extreme", False):
        if "超卖" in best_rule["description"]:
            weighted_buy = round(weighted_buy * PRICE_POSITION_KDJ_COEFF[price_pos]['超卖'], 4)
        elif "超买" in best_rule["description"]:
            weighted_sell = round(weighted_sell * PRICE_POSITION_KDJ_COEFF[price_pos]['超买'], 4)

    # 9. 计算净得分并返回（补充KDJ值用于调试）
    return {
        "signal_type": f"{best_rule['priority']}-{best_rule['description']}",
        "raw_buy": raw_buy,
        "raw_sell": raw_sell,
        "signal_type_weight": weight,
        "price_position": price_pos,
        "buy_score": weighted_buy,
        "sell_score": weighted_sell,
        "net_score": round(weighted_buy - weighted_sell, 2),
        "kdj_current": {"K": get_kdj_val(strategy.k, 0), "D": get_kdj_val(strategy.d, 0), "J": get_kdj_val(strategy.j, 0)},
        "kdj_prev": {"K": get_kdj_val(strategy.k, -1), "D": get_kdj_val(strategy.d, -1), "J": get_kdj_val(strategy.j, -1)}
    }


# ===================== 模拟KDJ策略对象 =====================
class BacktraderLine:
    """修复后的BacktraderLine模拟类，严格匹配Backtrader的索引行为"""

    def __init__(self, values):
        # values格式：[当前值(0), 前1值(-1), 前2值(-2), ...]
        # Backtrader中：line[0] = 当前值，line[-1] = 前1值，line[1] 会报错（索引越界）
        self.array = list(values)

    def __getitem__(self, idx):
        try:
            if idx == 0:
                # 0 = 当前值
                return self.array[0] if len(self.array) > 0 else 0.0
            elif idx < 0:
                # 负索引：-1 = 前1值（array[1]），-2 = 前2值（array[2]）...
                pos = abs(idx)
                return self.array[pos] if pos < len(self.array) else 0.0
            else:
                # 正索引（除0外）在Backtrader中会报错，这里返回0.0
                return 0.0
        except Exception as e:
            print(f"BacktraderLine索引{idx}出错: {e}")
            return 0.0

    def __len__(self):
        return len(self.array)


class MockData:
    """模拟Backtrader的DataFeed对象"""

    def __init__(self, close_vals):
        self.close = BacktraderLine(close_vals)

    def __len__(self):
        return len(self.close)


class MockKDJStrategy:
    """模拟KDJ策略对象"""

    def __init__(self, k_vals, d_vals, j_vals, close_vals=None, ma60_vals=None):
        # k_vals格式：[当前值(0), 前1值(-1), 前2值(-2), ...]
        self.k = BacktraderLine(k_vals)
        self.d = BacktraderLine(d_vals)
        self.j = BacktraderLine(j_vals)

        # 默认收盘价（至少20个数据，避免索引越界）
        default_len = max(len(k_vals), len(d_vals), len(j_vals), 20)
        default_close = [10.0] * default_len if close_vals is None else close_vals
        self.data = MockData(default_close)

        # 默认MA60
        default_ma60 = [9.5] * len(default_close) if ma60_vals is None else ma60_vals
        self.ma60 = BacktraderLine(default_ma60)


# ===================== 测试用例 =====================
class TestKDJScoreCalculation(unittest.TestCase):
    def setUp(self):
        print("\n--- 开始测试 ---")

    # ===================== 1. 最高（趋势反转）=====================
    def test_01_highest_bullish_crossover_low(self):
        """测试1：最高（趋势反转）- 低位反转金叉"""
        print("【测试1】低位反转金叉")
        # K[0]=29(当前), K[-1]=25(前1); D[0]=27, D[-1]=28
        k_vals = [29, 25]
        d_vals = [27, 28]
        j_vals = [32, 30]
        strat = MockKDJStrategy(k_vals, d_vals, j_vals)
        result = calculate_kdj_score(strat)
        print(f"测试结果: {result['signal_type']} | 净得分: {result['net_score']}")
        self.assertEqual(result['signal_type'], "最高（趋势反转）-低位反转金叉")
        self.assertEqual(result['net_score'], 4.0)

    def test_02_highest_bearish_crossunder_high(self):
        """测试2：最高（趋势反转）- 高位反转死叉"""
        print("【测试2】高位反转死叉")
        # K[0]=71, K[-1]=75; D[0]=73, D[-1]=72
        k_vals = [71, 75]
        d_vals = [73, 72]
        j_vals = [78, 80]
        strat = MockKDJStrategy(k_vals, d_vals, j_vals)
        result = calculate_kdj_score(strat)
        print(f"测试结果: {result['signal_type']} | 净得分: {result['net_score']}")
        self.assertEqual(result['signal_type'], "最高（趋势反转）-高位反转死叉")
        self.assertEqual(result['net_score'], -4.0)

    # ===================== 2. 高（连续极值）=====================
    def test_03_high_extreme_oversold(self):
        """测试3：高（连续极值）- 极端超卖"""
        print("【测试3】极端超卖")
        # K[0]=8, K[-1]=9, K[-2]=25（排除连续超卖）
        k_vals = [8, 9, 25]
        d_vals = [11, 10, 28]
        j_vals = [13, 12, 30]
        # 构造收盘价避免触发连续超卖（前2值=25 > 20）
        close_vals = [10.0] * 20
        close_vals[2] = 25  # K[-2] = 25
        strat = MockKDJStrategy(k_vals, d_vals, j_vals, close_vals=close_vals)
        result = calculate_kdj_score(strat)
        print(f"测试结果: {result['signal_type']} | 净得分: {result['net_score']}")
        self.assertEqual(result['signal_type'], "高（连续极值）-极端超卖")
        self.assertEqual(result['net_score'], 3.0)

    def test_04_high_extreme_overbought(self):
        """测试4：高（连续极值）- 极端超买（高位，修正系数1.2）"""
        print("【测试4】极端超买（高位修正）")
        k_vals = [91, 92, 85]
        d_vals = [89, 88, 82]
        j_vals = [94, 95, 88]
        close_vals = [10.2] + [10.0] * 19
        ma60_vals = [10.0] * 20
        strat = MockKDJStrategy(k_vals, d_vals, j_vals, close_vals=close_vals, ma60_vals=ma60_vals)
        result = calculate_kdj_score(strat)
        print(
            f"测试结果: {result['signal_type']} | 净得分: {result['net_score']} | 价格位置: {result['price_position']}")
        self.assertEqual(result['signal_type'], "高（连续极值）-极端超买")
        self.assertEqual(result['net_score'], -3.6)

    def test_05_high_continuous_oversold(self):
        """测试5：高（连续极值）- 连续超卖"""
        print("【测试5】连续超卖")
        # K[0]=18, K[-1]=19, K[-2]=17; D[0]=28
        k_vals = [18, 19, 17]
        d_vals = [28, 27, 25]
        j_vals = [21, 22, 20]
        strat = MockKDJStrategy(k_vals, d_vals, j_vals)
        result = calculate_kdj_score(strat)
        print(f"测试结果: {result['signal_type']} | 净得分: {result['net_score']}")
        self.assertEqual(result['signal_type'], "高（连续极值）-连续超卖")
        self.assertEqual(result['net_score'], 3.0)

    def test_06_high_continuous_overbought(self):
        """测试6：高（连续极值）- 连续超买（中位，无修正）"""
        print("【测试6】连续超买")
        # K[0]=82, K[-1]=81, K[-2]=83; D[0]=72
        k_vals = [82, 81, 83]
        d_vals = [72, 73, 75]
        j_vals = [86, 84, 85]
        # 构造收盘价为中位（避免价格位置修正）
        close_vals = [10.0] * 20
        ma60_vals = [10.0] * 20
        strat = MockKDJStrategy(k_vals, d_vals, j_vals, close_vals=close_vals, ma60_vals=ma60_vals)
        result = calculate_kdj_score(strat)
        print(f"测试结果: {result['signal_type']} | 净得分: {result['net_score']}")
        self.assertEqual(result['signal_type'], "高（连续极值）-连续超买")
        self.assertEqual(result['net_score'], -3.0)

    # ===================== 3. 中高（动态金叉分级）=====================
    def test_07_midhigh_oversold_crossover(self):
        """测试7：中高（动态金叉分级）- 超卖金叉"""
        print("【测试7】超卖金叉")
        # K[0]=29, K[-1]=28; D[0]=28, D[-1]=30 (排除趋势反转条件：D[-1]=30 ≥30)
        k_vals = [29, 28]
        d_vals = [28, 30]
        j_vals = [31, 32]
        strat = MockKDJStrategy(k_vals, d_vals, j_vals)
        result = calculate_kdj_score(strat)
        print(f"测试结果: {result['signal_type']} | 净得分: {result['net_score']}")
        self.assertEqual(result['signal_type'], "中高（动态金叉分级）-超卖金叉")
        self.assertEqual(result['net_score'], 2.16)

    def test_08_midhigh_oscillate_crossover(self):
        """测试8：中高（动态金叉分级）- 震荡金叉"""
        print("【测试8】震荡金叉")
        # K[0]=41, K[-1]=40; D[0]=40, D[-1]=42
        k_vals = [41, 40]
        d_vals = [40, 42]
        j_vals = [44, 45]
        strat = MockKDJStrategy(k_vals, d_vals, j_vals)
        result = calculate_kdj_score(strat)
        print(f"测试结果: {result['signal_type']} | 净得分: {result['net_score']}")
        self.assertEqual(result['signal_type'], "中高（动态金叉分级）-震荡金叉")
        self.assertEqual(result['net_score'], 1.8)

    def test_09_midhigh_high_crossover(self):
        """测试9：中高（动态金叉分级）- 高位金叉"""
        print("【测试9】高位金叉")
        # K[0]=73, K[-1]=72; D[0]=72, D[-1]=75
        k_vals = [73, 72]
        d_vals = [72, 75]
        j_vals = [77, 78]
        strat = MockKDJStrategy(k_vals, d_vals, j_vals)
        result = calculate_kdj_score(strat)
        print(f"测试结果: {result['signal_type']} | 净得分: {result['net_score']}")
        self.assertEqual(result['signal_type'], "中高（动态金叉分级）-高位金叉")
        self.assertEqual(result['net_score'], -1.2)

    def test_10_midhigh_overbought_crossunder(self):
        """测试10：中高（动态死叉分级）- 超买死叉"""
        print("【测试10】超买死叉")
        # K[0]=73, K[-1]=75; D[0]=74, D[-1]=71 (排除趋势反转条件：D[-1]=71 <70 不成立)
        k_vals = [73, 75]
        d_vals = [74, 71]
        j_vals = [78, 80]
        strat = MockKDJStrategy(k_vals, d_vals, j_vals)
        result = calculate_kdj_score(strat)
        print(f"测试结果: {result['signal_type']} | 净得分: {result['net_score']}")
        self.assertEqual(result['signal_type'], "中高（动态死叉分级）-超买死叉")
        self.assertEqual(result['net_score'], -2.16)

    def test_11_midhigh_oscillate_crossunder(self):
        """测试11：中高（动态死叉分级）- 震荡死叉"""
        print("【测试11】震荡死叉")
        # K[0]=59, K[-1]=60; D[0]=60, D[-1]=58
        k_vals = [59, 60]
        d_vals = [60, 58]
        j_vals = [63, 65]
        strat = MockKDJStrategy(k_vals, d_vals, j_vals)
        result = calculate_kdj_score(strat)
        print(f"测试结果: {result['signal_type']} | 净得分: {result['net_score']}")
        self.assertEqual(result['signal_type'], "中高（动态死叉分级）-震荡死叉")
        self.assertEqual(result['net_score'], -1.8)

    # ===================== 4. 中高（动态拐头）=====================
    def test_12_midhigh_low_turnaround(self):
        """测试12：中高（动态拐头）- 低位拐头"""
        print("【测试12】低位拐头")
        # K[0]=35, K[-1]=28；确保不触发金叉和企稳条件
        k_vals = [35, 28]
        d_vals = [38, 30]  # K[0] < D[0] 避免金叉
        j_vals = [38, 32]
        strat = MockKDJStrategy(k_vals, d_vals, j_vals)
        result = calculate_kdj_score(strat)
        print(f"测试结果: {result['signal_type']} | 净得分: {result['net_score']}")
        self.assertEqual(result['signal_type'], "中高（动态拐头）-低位拐头")
        self.assertEqual(result['net_score'], 2.16)

    def test_13_midhigh_high_turnaround(self):
        """测试13：中高（动态拐头）- 高位拐头"""
        print("【测试13】高位拐头")
        # K[0]=65, K[-1]=75；确保不触发死叉条件
        k_vals = [65, 75]
        d_vals = [60, 72]  # K[0] > D[0] 避免死叉
        j_vals = [70, 80]
        strat = MockKDJStrategy(k_vals, d_vals, j_vals)
        result = calculate_kdj_score(strat)
        print(f"测试结果: {result['signal_type']} | 净得分: {result['net_score']}")
        self.assertEqual(result['signal_type'], "中高（动态拐头）-高位拐头")
        self.assertEqual(result['net_score'], -2.16)

    # ===================== 5. 中高（动态企稳）=====================
    def test_14_midhigh_oversold_stabilize(self):
        """测试14：中高（动态企稳）- 超卖企稳"""
        print("【测试14】超卖企稳")
        # K[0]=34, K[-1]=28, K[-2]=27；K[0]-K[-1]=6 ≥5，不触发金叉
        k_vals = [34, 28, 27]
        d_vals = [38, 31, 30]  # K[0] < D[0] 避免金叉
        j_vals = [38, 33, 32]
        strat = MockKDJStrategy(k_vals, d_vals, j_vals)
        result = calculate_kdj_score(strat)
        print(f"测试结果: {result['signal_type']} | 净得分: {result['net_score']}")
        self.assertEqual(result['signal_type'], "中高（动态企稳）-超卖企稳")
        self.assertEqual(result['net_score'], 1.8)

    def test_15_midhigh_overbought_fall(self):
        """测试15：中高（动态企稳）- 超买回落"""
        print("【测试15】超买回落")
        # K[0]=66, K[-1]=72, K[-2]=73；K[-1]-K[0]=6 ≥5，不触发死叉
        k_vals = [66, 72, 73]
        d_vals = [60, 71, 70]  # K[0] > D[0] 避免死叉
        j_vals = [69, 74, 75]
        strat = MockKDJStrategy(k_vals, d_vals, j_vals)
        result = calculate_kdj_score(strat)
        print(f"测试结果: {result['signal_type']} | 净得分: {result['net_score']}")
        self.assertEqual(result['signal_type'], "中高（动态企稳）-超买回落")
        self.assertEqual(result['net_score'], -1.8)

    # ===================== 6. 中高（动态后效）=====================
    def test_16_midhigh_crossover_after_overbought(self):
        """测试16：中高（动态后效）- 金叉后超买"""
        print("【测试16】金叉后超买")
        # K[0]=75, K[-1]=72, K[-2]=68; D[0]=73, D[-1]=70, D[-2]=70
        k_vals = [75, 72, 68]
        d_vals = [73, 70, 70]
        j_vals = [79, 76, 78]
        strat = MockKDJStrategy(k_vals, d_vals, j_vals)
        result = calculate_kdj_score(strat)
        print(f"测试结果: {result['signal_type']} | 净得分: {result['net_score']}")
        self.assertEqual(result['signal_type'], "中高（动态后效）-金叉后超买")
        self.assertEqual(result['net_score'], -2.16)

    def test_17_midhigh_crossunder_after_oversold(self):
        """测试17：中高（动态后效）- 死叉后超卖"""
        print("【测试17】死叉后超卖")
        # K[0]=28, K[-1]=30, K[-2]=32; D[0]=30, D[-1]=31, D[-2]=30
        k_vals = [28, 30, 32]
        d_vals = [30, 31, 30]
        j_vals = [31, 33, 35]
        strat = MockKDJStrategy(k_vals, d_vals, j_vals)
        result = calculate_kdj_score(strat)
        print(f"测试结果: {result['signal_type']} | 净得分: {result['net_score']}")
        self.assertEqual(result['signal_type'], "中高（动态后效）-死叉后超卖")
        self.assertEqual(result['net_score'], 1.2)

    # ===================== 7. 常规（静态排列）- 标准多头 =====================
    def test_18_normal_standard_bullish_low(self):
        """测试18：常规（静态排列）- 标准多头（K<20）"""
        print("【测试18】标准多头（K<20）")
        # 仅匹配标准多头规则（避免触发连续超卖）
        k_vals = [18]
        d_vals = [15]
        j_vals = [25]
        # 构造K[-1] = 25 >20，避免连续超卖
        strat = MockKDJStrategy([18, 25], [15, 18], [25, 22])
        result = calculate_kdj_score(strat)
        print(f"测试结果: {result['signal_type']} | 净得分: {result['net_score']}")
        self.assertEqual(result['signal_type'], "常规（静态排列）-标准多头")
        self.assertEqual(result['net_score'], 2.0)

    def test_19_normal_standard_bullish_mid1(self):
        """测试19：常规（静态排列）- 标准多头（20<=K<=50）"""
        print("【测试19】标准多头（20<=K<=50）")
        k_vals = [35]
        d_vals = [30]
        j_vals = [45]
        strat = MockKDJStrategy(k_vals, d_vals, j_vals)
        result = calculate_kdj_score(strat)
        print(f"测试结果: {result['signal_type']} | 净得分: {result['net_score']}")
        self.assertEqual(result['signal_type'], "常规（静态排列）-标准多头")
        self.assertEqual(result['net_score'], 1.5)

    def test_20_normal_standard_bullish_mid2(self):
        """测试20：常规（静态排列）- 标准多头（50<=K<=70）"""
        print("【测试20】标准多头（50<=K<=70）")
        k_vals = [60]
        d_vals = [55]
        j_vals = [65]
        strat = MockKDJStrategy(k_vals, d_vals, j_vals)
        result = calculate_kdj_score(strat)
        print(f"测试结果: {result['signal_type']} | 净得分: {result['net_score']}")
        self.assertEqual(result['signal_type'], "常规（静态排列）-标准多头")
        self.assertEqual(result['net_score'], 0.5)

    def test_21_normal_standard_bullish_high(self):
        """测试21：常规（静态排列）- 标准多头（K>70）"""
        print("【测试21】标准多头（K>70）")
        k_vals = [75]
        d_vals = [70]
        j_vals = [85]
        strat = MockKDJStrategy(k_vals, d_vals, j_vals)
        result = calculate_kdj_score(strat)
        print(f"测试结果: {result['signal_type']} | 净得分: {result['net_score']}")
        self.assertEqual(result['signal_type'], "常规（静态排列）-标准多头")
        self.assertEqual(result['net_score'], -2.0)

    # ===================== 8. 常规（静态排列）- 偏多结构 =====================
    def test_22_normal_bullish_bias_low(self):
        """测试22：常规（静态排列）- 偏多结构（K<30）"""
        print("【测试22】偏多结构（K<30）")
        k_vals = [25]
        d_vals = [28]
        j_vals = [30]
        strat = MockKDJStrategy(k_vals, d_vals, j_vals)
        result = calculate_kdj_score(strat)
        print(f"测试结果: {result['signal_type']} | 净得分: {result['net_score']}")
        self.assertEqual(result['signal_type'], "常规（静态排列）-偏多结构")
        self.assertEqual(result['net_score'], 1.5)

    def test_23_normal_bullish_bias_mid(self):
        """测试23：常规（静态排列）- 偏多结构（30<=K<=70）"""
        print("【测试23】偏多结构（30<=K<=70）")
        k_vals = [50]
        d_vals = [52]
        j_vals = [55]
        strat = MockKDJStrategy(k_vals, d_vals, j_vals)
        result = calculate_kdj_score(strat)
        print(f"测试结果: {result['signal_type']} | 净得分: {result['net_score']}")
        self.assertEqual(result['signal_type'], "常规（静态排列）-偏多结构")
        self.assertEqual(result['net_score'], 0.0)

    def test_24_normal_bullish_bias_high(self):
        """测试24：常规（静态排列）- 偏多结构（K>70）"""
        print("【测试24】偏多结构（K>70）")
        k_vals = [75]
        d_vals = [78]
        j_vals = [80]
        strat = MockKDJStrategy(k_vals, d_vals, j_vals)
        result = calculate_kdj_score(strat)
        print(f"测试结果: {result['signal_type']} | 净得分: {result['net_score']}")
        self.assertEqual(result['signal_type'], "常规（静态排列）-偏多结构")
        self.assertEqual(result['net_score'], -1.8)

    # ===================== 9. 常规（静态排列）- 偏空结构 =====================
    def test_25_normal_bearish_bias_low(self):
        """测试25：常规（静态排列）- 偏空结构（K<30）"""
        print("【测试25】偏空结构（K<30）")
        k_vals = [25]
        d_vals = [22]
        j_vals = [20]
        strat = MockKDJStrategy(k_vals, d_vals, j_vals)
        result = calculate_kdj_score(strat)
        print(f"测试结果: {result['signal_type']} | 净得分: {result['net_score']}")
        self.assertEqual(result['signal_type'], "常规（静态排列）-偏空结构")
        self.assertEqual(result['net_score'], 1.2)

    def test_26_normal_bearish_bias_mid(self):
        """测试26：常规（静态排列）- 偏空结构（30<=K<=70）"""
        print("【测试26】偏空结构（30<=K<=70）")
        k_vals = [50]
        d_vals = [48]
        j_vals = [45]
        strat = MockKDJStrategy(k_vals, d_vals, j_vals)
        result = calculate_kdj_score(strat)
        print(f"测试结果: {result['signal_type']} | 净得分: {result['net_score']}")
        self.assertEqual(result['signal_type'], "常规（静态排列）-偏空结构")
        self.assertEqual(result['net_score'], -0.5)

    def test_27_normal_bearish_bias_high(self):
        """测试27：常规（静态排列）- 偏空结构（K>70）"""
        print("【测试27】偏空结构（K>70）")
        k_vals = [75]
        d_vals = [72]
        j_vals = [70]
        strat = MockKDJStrategy(k_vals, d_vals, j_vals)
        result = calculate_kdj_score(strat)
        print(f"测试结果: {result['signal_type']} | 净得分: {result['net_score']}")
        self.assertEqual(result['signal_type'], "常规（静态排列）-偏空结构")
        self.assertEqual(result['net_score'], -1.5)

    # ===================== 10. 常规（静态排列）- 标准空头 =====================
    def test_28_normal_standard_bearish_low(self):
        """测试28：常规（静态排列）- 标准空头（K<20）"""
        print("【测试28】标准空头（K<20）")
        k_vals = [18, 25]
        d_vals = [20, 18]
        j_vals = [15, 12]
        strat = MockKDJStrategy(k_vals, d_vals, j_vals)
        result = calculate_kdj_score(strat)
        print(f"测试结果: {result['signal_type']} | 净得分: {result['net_score']}")
        self.assertEqual(result['signal_type'], "常规（静态排列）-标准空头")
        self.assertEqual(result['net_score'], 1.5)

    def test_29_normal_standard_bearish_mid1(self):
        """测试29：常规（静态排列）- 标准空头（20<=K<=50）"""
        print("【测试29】标准空头（20<=K<=50）")
        k_vals = [35]
        d_vals = [38]
        j_vals = [30]
        strat = MockKDJStrategy(k_vals, d_vals, j_vals)
        result = calculate_kdj_score(strat)
        print(f"测试结果: {result['signal_type']} | 净得分: {result['net_score']}")
        self.assertEqual(result['signal_type'], "常规（静态排列）-标准空头")
        self.assertEqual(result['net_score'], -1.5)

    def test_30_normal_standard_bearish_mid2(self):
        """测试30：常规（静态排列）- 标准空头（50<=K<=70）"""
        print("【测试30】标准空头（50<=K<=70）")
        k_vals = [60]
        d_vals = [65]
        j_vals = [55]
        strat = MockKDJStrategy(k_vals, d_vals, j_vals)
        result = calculate_kdj_score(strat)
        print(f"测试结果: {result['signal_type']} | 净得分: {result['net_score']}")
        self.assertEqual(result['signal_type'], "常规（静态排列）-标准空头")
        self.assertEqual(result['net_score'], -2.0)

    def test_31_normal_standard_bearish_high(self):
        """测试31：常规（静态排列）- 标准空头（K>70）"""
        print("【测试31】标准空头（K>70）")
        k_vals = [75]
        d_vals = [80]
        j_vals = [70]
        strat = MockKDJStrategy(k_vals, d_vals, j_vals)
        result = calculate_kdj_score(strat)
        print(f"测试结果: {result['signal_type']} | 净得分: {result['net_score']}")
        self.assertEqual(result['signal_type'], "常规（静态排列）-标准空头")
        self.assertEqual(result['net_score'], -2.0)

    # ===================== 11. 低（粘合平衡）=====================
    def test_32_low_three_line_sticky_low(self):
        """测试32：低（粘合平衡）- 三线粘合（K<50）"""
        print("【测试32】三线粘合（K<50）")
        k_vals = [45]
        d_vals = [46]
        j_vals = [45]
        strat = MockKDJStrategy(k_vals, d_vals, j_vals)
        result = calculate_kdj_score(strat)
        print(f"测试结果: {result['signal_type']} | 净得分: {result['net_score']}")
        self.assertEqual(result['signal_type'], "低（粘合平衡）-三线粘合")
        self.assertEqual(result['net_score'], 0.25)

    def test_33_low_three_line_sticky_high(self):
        """测试33：低（粘合平衡）- 三线粘合（K>50）"""
        print("【测试33】三线粘合（K>50）")
        k_vals = [55]
        d_vals = [56]
        j_vals = [55]
        strat = MockKDJStrategy(k_vals, d_vals, j_vals)
        result = calculate_kdj_score(strat)
        print(f"测试结果: {result['signal_type']} | 净得分: {result['net_score']}")
        self.assertEqual(result['signal_type'], "低（粘合平衡）-三线粘合")
        self.assertEqual(result['net_score'], -0.25)

    def test_34_low_bullish_sticky_low(self):
        """测试34：低（粘合平衡）- 偏多粘合（K<30）"""
        print("【测试34】偏多粘合（K<30）")
        k_vals = [25]
        d_vals = [26]
        j_vals = [24]
        strat = MockKDJStrategy(k_vals, d_vals, j_vals)
        result = calculate_kdj_score(strat)
        print(f"测试结果: {result['signal_type']} | 净得分: {result['net_score']}")
        self.assertEqual(result['signal_type'], "低（粘合平衡）-偏多粘合")
        self.assertEqual(result['net_score'], 0.75)

    def test_35_low_bullish_sticky_high(self):
        """测试35：低（粘合平衡）- 偏多粘合（K>70）"""
        print("【测试35】偏多粘合（K>70）")
        k_vals = [75]
        d_vals = [76]
        j_vals = [74]
        strat = MockKDJStrategy(k_vals, d_vals, j_vals)
        result = calculate_kdj_score(strat)
        print(f"测试结果: {result['signal_type']} | 净得分: {result['net_score']}")
        self.assertEqual(result['signal_type'], "低（粘合平衡）-偏多粘合")
        self.assertEqual(result['net_score'], -0.75)

    def test_36_low_bearish_sticky_low(self):
        """测试36：低（粘合平衡）- 偏空粘合（K<30）"""
        print("【测试36】偏空粘合（K<30）")
        k_vals = [25]
        d_vals = [26]
        j_vals = [27]
        strat = MockKDJStrategy(k_vals, d_vals, j_vals)
        result = calculate_kdj_score(strat)
        print(f"测试结果: {result['signal_type']} | 净得分: {result['net_score']}")
        self.assertEqual(result['signal_type'], "低（粘合平衡）-偏空粘合")
        self.assertEqual(result['net_score'], 0.6)

    def test_37_low_bearish_sticky_high(self):
        """测试37：低（粘合平衡）- 偏空粘合（K>70）"""
        print("【测试37】偏空粘合（K>70）")
        k_vals = [75]
        d_vals = [76]
        j_vals = [77]
        strat = MockKDJStrategy(k_vals, d_vals, j_vals)
        result = calculate_kdj_score(strat)
        print(f"测试结果: {result['signal_type']} | 净得分: {result['net_score']}")
        self.assertEqual(result['signal_type'], "低（粘合平衡）-偏空粘合")
        self.assertEqual(result['net_score'], -0.9)


if __name__ == '__main__':
    unittest.main(verbosity=2)