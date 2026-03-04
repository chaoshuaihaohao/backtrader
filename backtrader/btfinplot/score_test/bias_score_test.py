import unittest
from typing import Dict, List, Optional, Tuple

# ===================== 第一步：定义股票类型阈值配置 =====================
STOCK_TYPE_BIAS_THRESHOLDS = {
    'large_cap': {  # 大盘股
        'oversell_6': -5.0, 'oversell_12': -5.5, 'oversell_24': -8.0,
        'overbuy_6': 5.0, 'overbuy_12': 6.0, 'overbuy_24': 9.0
    },
    'mid_cap': {  # 中盘股
        'oversell_6': -7.0, 'oversell_12': -7.5, 'oversell_24': -10.0,
        'overbuy_6': 7.0, 'overbuy_12': 8.0, 'overbuy_24': 11.0
    },
    'small_cap': {  # 小盘股
        'oversell_6': -10.0, 'oversell_12': -10.5, 'oversell_24': -13.0,
        'overbuy_6': 10.0, 'overbuy_12': 11.0, 'overbuy_24': 14.0
    }
}


# ===================== 第二步：辅助函数：判断BIAS背离 =====================
def _check_bias_divergence(close, bias6, divergence_type: str) -> bool:
    """
    严谨判断BIAS顶底背离
    :param close: Backtrader Line对象（收盘价）
    :param bias6: Backtrader Line对象（BIAS6）
    :param divergence_type: '底背离' 或 '顶背离'
    :return: 是否触发背离
    """
    try:
        # 取最近10个数据点，分为前后两段比较
        period = 10
        half_period = period // 2
        if len(close) < period or len(bias6) < period:
            return False

        # 获取价格和BIAS6的历史值（0=当日，-1=前1日...）
        # 注意：这里从远到近排列，即 [10日前, 9日前, ..., 前1日, 当日]
        closes = [close[-i] for i in range(period - 1, -1, -1)]
        biases = [bias6[-i] for i in range(period - 1, -1, -1)]

        prev_closes = closes[:half_period]
        curr_closes = closes[half_period:]
        prev_biases = biases[:half_period]
        curr_biases = biases[half_period:]

        if divergence_type == '底背离':
            # 价格创新低，BIAS6未创新低
            price_prev_low = min(prev_closes)
            price_curr_low = min(curr_closes)
            bias_prev_low = min(prev_biases)
            bias_curr_low = min(curr_biases)
            return price_curr_low < price_prev_low and bias_curr_low > bias_prev_low

        elif divergence_type == '顶背离':
            # 价格创新高，BIAS6未创新高
            price_prev_high = max(prev_closes)
            price_curr_high = max(curr_closes)
            bias_prev_high = max(prev_biases)
            bias_curr_high = max(curr_biases)
            return price_curr_high > price_prev_high and bias_curr_high < bias_prev_high

        return False
    except Exception:
        return False


# ===================== 第三步：辅助函数：判断股价位置（修复版） =====================
def _get_price_position(close, ma60) -> Tuple[str, float, float]:
    """
    判断股价位置，返回位置类型和对应的超买/超卖修正系数
    :param close: Backtrader Line对象（收盘价）
    :param ma60: Backtrader Line对象（MA60）
    :return: (位置类型, 超买修正系数, 超卖修正系数)
    """
    try:
        if len(close) < 1 or len(ma60) < 1:
            return '中位', 0.5, 1.5

        curr_close = close[0]
        curr_ma60 = ma60[0]

        # 低位：MA60下方，或刚站上MA60不超过3%
        if curr_close < curr_ma60 or (curr_close >= curr_ma60 and curr_close / curr_ma60 <= 1.03):
            return '低位', 0.0, 2.0

        # 高位：远离MA60（>10%）且创20日新高（如果数据足够）
        is_high = False
        if len(close) >= 20:
            recent_20_high = max([close[-i] for i in range(20)])
            if curr_close / curr_ma60 > 1.1 and curr_close >= recent_20_high:
                is_high = True
        else:
            # 数据不足20天时，只要远离MA60（>10%）也视为高位
            if curr_close / curr_ma60 > 1.1:
                is_high = True

        if is_high:
            return '高位', 2.0, 0.5

        # 中位
        return '中位', 0.5, 1.5
    except Exception:
        return '中位', 0.5, 1.5


# ===================== 第四步：定义完整的BIAS评分规则表 =====================
def _get_bias_score_table(strategy):
    """动态生成BIAS评分规则表（绑定策略对象用于背离判断）"""
    return [
        # 【最高优先级】极端单边行情钝化：直接归零
        {
            "优先级": "无（实操必看）", "信号类型权重": 0.0,
            "信号类型": "极端单边行情-钝化",
            "条件函数": lambda: (
                # 连续2日涨停/跌停（简化版：当日一字板且涨跌幅达标）
                    len(strategy.data.close) >= 2 and
                    strategy.data.close[0] == strategy.data.high[0] == strategy.data.low[0] and
                    (
                            (strategy.data.close[0] / strategy.data.close[-1] >= 1.098) or  # 涨停
                            (strategy.data.close[0] / strategy.data.close[-1] <= 0.902)  # 跌停
                    )
            ),
            "买入得分": 0.0, "卖出得分": 0.0, "是否超买超卖": False,
            "备注": "连续涨跌停钝化，得分归零"
        },

        # 【最高】趋势反转：底背离
        {
            "优先级": "最高（趋势反转）", "信号类型权重": 2.0,
            "信号类型": "底背离",
            "条件函数": lambda: _check_bias_divergence(strategy.data.close, strategy.bias6, '底背离'),
            "买入得分": 2.0, "卖出得分": 0.0, "是否超买超卖": False,
            "备注": "股价新低，BIAS6未新低"
        },
        # 【最高】趋势反转：顶背离
        {
            "优先级": "最高（趋势反转）", "信号类型权重": 2.0,
            "信号类型": "顶背离",
            "条件函数": lambda: _check_bias_divergence(strategy.data.close, strategy.bias6, '顶背离'),
            "买入得分": 0.0, "卖出得分": 2.0, "是否超买超卖": False,
            "备注": "股价新高，BIAS6未新高"
        },

        # 【高】连续极值：极端超卖共振
        {
            "优先级": "高（连续极值）", "信号类型权重": 1.5,
            "信号类型": "极端超卖共振",
            "条件函数": lambda: (
                    strategy.bias6[0] < strategy._bias_thresholds['oversell_6'] and
                    strategy.bias12[0] < strategy._bias_thresholds['oversell_12'] and
                    strategy.bias24[0] < strategy._bias_thresholds['oversell_24']
            ),
            "买入得分": 2.0, "卖出得分": 0.0, "是否超买超卖": True, "超卖类型": "超卖",
            "备注": "BIAS6/12/24同步超卖"
        },
        # 【高】连续极值：极端超买共振
        {
            "优先级": "高（连续极值）", "信号类型权重": 1.5,
            "信号类型": "极端超买共振",
            "条件函数": lambda: (
                    strategy.bias6[0] > strategy._bias_thresholds['overbuy_6'] and
                    strategy.bias12[0] > strategy._bias_thresholds['overbuy_12'] and
                    strategy.bias24[0] > strategy._bias_thresholds['overbuy_24']
            ),
            "买入得分": 0.0, "卖出得分": 2.0, "是否超买超卖": True, "超卖类型": "超买",
            "备注": "BIAS6/12/24同步超买"
        },
        # 【高】连续极值：6日极端超卖
        {
            "优先级": "高（连续极值）", "信号类型权重": 1.5,
            "信号类型": "6日极端超卖",
            "条件函数": lambda: strategy.bias6[0] < strategy._bias_thresholds['oversell_6'],
            "买入得分": 1.8, "卖出得分": 0.0, "是否超买超卖": True, "超卖类型": "超卖",
            "备注": "BIAS6单独超卖"
        },
        # 【高】连续极值：6日极端超买
        {
            "优先级": "高（连续极值）", "信号类型权重": 1.5,
            "信号类型": "6日极端超买",
            "条件函数": lambda: strategy.bias6[0] > strategy._bias_thresholds['overbuy_6'],
            "买入得分": 0.0, "卖出得分": 1.8, "是否超买超卖": True, "超卖类型": "超买",
            "备注": "BIAS6单独超买"
        },

        # 【中高】动态信号：强势多头排列
        {
            "优先级": "中高（动态）", "信号类型权重": 1.2,
            "信号类型": "强势多头",
            "条件函数": lambda: strategy.bias6[0] > strategy.bias12[0] > strategy.bias24[0],
            "买入得分": 1.5, "卖出得分": 0.3, "是否超买超卖": False,
            "备注": "BIAS6>BIAS12>BIAS24"
        },
        # 【中高】动态信号：强势空头排列
        {
            "优先级": "中高（动态）", "信号类型权重": 1.2,
            "信号类型": "强势空头",
            "条件函数": lambda: strategy.bias6[0] < strategy.bias12[0] < strategy.bias24[0],
            "买入得分": 0.3, "卖出得分": 1.5, "是否超买超卖": False,
            "备注": "BIAS6<BIAS12<BIAS24"
        },
        # 【中高】动态信号：超卖回归
        {
            "优先级": "中高（动态）", "信号类型权重": 1.2,
            "信号类型": "超卖回归",
            "条件函数": lambda: (
                    len(strategy.bias6) >= 2 and
                    strategy.bias6[-1] < strategy._bias_thresholds['oversell_6'] and
                    strategy._bias_thresholds['oversell_6'] <= strategy.bias6[0] <= 0
            ),
            "买入得分": 1.5, "卖出得分": 0.0, "是否超买超卖": False,
            "备注": "从超卖区间回到平衡区间"
        },
        # 【中高】动态信号：超买回落
        {
            "优先级": "中高（动态）", "信号类型权重": 1.2,
            "信号类型": "超买回落",
            "条件函数": lambda: (
                    len(strategy.bias6) >= 2 and
                    strategy.bias6[-1] > strategy._bias_thresholds['overbuy_6'] and
                    0 <= strategy.bias6[0] <= strategy._bias_thresholds['overbuy_6']
            ),
            "买入得分": 0.0, "卖出得分": 1.5, "是否超买超卖": False,
            "备注": "从超买区间回到平衡区间"
        },

        # 【低】粘合平衡：零轴附近（优先于多空区域判断）
        {
            "优先级": "低（粘合平衡）", "信号类型权重": 0.5,
            "信号类型": "零轴附近",
            "条件函数": lambda: -1.0 <= strategy.bias6[0] <= 1.0,
            "买入得分": 0.5, "卖出得分": 0.5, "是否超买超卖": False,
            "备注": "BIAS6在-1%~1%之间"
        },

        # 【常规】静态排列：多头区域
        {
            "优先级": "常规（静态）", "信号类型权重": 1.0,
            "信号类型": "多头区域",
            "条件函数": lambda: strategy.bias6[0] > 0,
            "买入得分": 1.0, "卖出得分": 0.3, "是否超买超卖": False,
            "备注": "BIAS6>0"
        },
        # 【常规】静态排列：空头区域
        {
            "优先级": "常规（静态）", "信号类型权重": 1.0,
            "信号类型": "空头区域",
            "条件函数": lambda: strategy.bias6[0] < 0,
            "买入得分": 0.3, "卖出得分": 1.0, "是否超买超卖": False,
            "备注": "BIAS6<0"
        },
    ]


# ===================== 第五步：核心BIAS评分计算函数 =====================
def calculate_bias_score(strategy) -> Dict:
    """
    严格按照评分手册计算BIAS净得分
    :param strategy: Backtrader策略实例（需包含bias6, bias12, bias24, data.close, ma60, p.stock_type）
    :return: 评分结果字典
    """
    # 1. 初始化策略阈值（缓存到策略对象避免重复计算）
    if not hasattr(strategy, '_bias_thresholds'):
        stock_type = getattr(strategy.p, 'stock_type', 'mid_cap')
        strategy._bias_thresholds = STOCK_TYPE_BIAS_THRESHOLDS.get(stock_type, STOCK_TYPE_BIAS_THRESHOLDS['mid_cap'])

    # 2. 获取股价位置和修正系数
    position, overbuy_corr, oversell_corr = _get_price_position(strategy.data.close, strategy.ma60)

    # 3. 遍历规则表（按优先级从高到低）
    score_table = _get_bias_score_table(strategy)
    for rule in score_table:
        try:
            # 检查条件是否触发
            if not rule["条件函数"]():
                continue

            # 极端钝化特殊处理
            if rule["信号类型"] == "极端单边行情-钝化":
                return {
                    "signal_type": rule["信号类型"],
                    "raw_buy": 0.0, "raw_sell": 0.0,
                    "signal_weight": 0.0,
                    "position": position,
                    "position_corr_overbuy": 1.0, "position_corr_oversell": 1.0,
                    "final_buy": 0.0, "final_sell": 0.0,
                    "net_score": 0.0,
                    "extreme_passivation": True,
                    "triggered_rule": rule
                }

            # 正常信号处理
            raw_buy = float(rule["买入得分"])
            raw_sell = float(rule["卖出得分"])
            signal_weight = float(rule["信号类型权重"])

            # 基础加权
            weighted_buy = raw_buy * signal_weight
            weighted_sell = raw_sell * signal_weight

            # 股价位置修正（仅超买/超卖类信号）
            final_buy = weighted_buy
            final_sell = weighted_sell
            if rule["是否超买超卖"]:
                if rule["超卖类型"] == "超卖":
                    final_buy = weighted_buy * oversell_corr
                elif rule["超卖类型"] == "超买":
                    final_sell = weighted_sell * overbuy_corr

            # 计算净得分
            net_score = round(final_buy - final_sell, 4)

            return {
                "signal_type": f"{rule['优先级']}-{rule['信号类型']}",
                "raw_buy": raw_buy, "raw_sell": raw_sell,
                "signal_weight": signal_weight,
                "position": position,
                "position_corr_overbuy": overbuy_corr, "position_corr_oversell": oversell_corr,
                "final_buy": round(final_buy, 4), "final_sell": round(final_sell, 4),
                "net_score": net_score,
                "extreme_passivation": False,
                "triggered_rule": rule
            }

        except Exception as e:
            print(f"规则执行异常 [{rule['信号类型']}]: {e}")
            continue

    # 无信号匹配
    return {
        "signal_type": "无信号",
        "raw_buy": 0.0, "raw_sell": 0.0,
        "signal_weight": 0.0,
        "position": "未知",
        "position_corr_overbuy": 1.0, "position_corr_oversell": 1.0,
        "final_buy": 0.0, "final_sell": 0.0,
        "net_score": 0.0,
        "extreme_passivation": False,
        "triggered_rule": None
    }


# ===================== 第六步：模拟Backtrader环境（修复版：自动补全数据） =====================
class MockParams:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class BacktraderLine:
    """
    模拟Backtrader的Line对象
    索引规则：
    - line[0] 或 line[0] = 当日
    - line[-1] = 前1日
    - line[-2] = 前2日
    内部存储：self._values = [当日, 前1日, 前2日, ...]
    """

    def __init__(self, values: List[float]):
        # 确保至少有一个数据
        if not values:
            values = [0.0]
        # values存储顺序：[当日, 前1日, 前2日, ...]
        self._values = list(values)

    def __getitem__(self, idx: int) -> float:
        try:
            if idx >= 0:
                # 正索引：0=当日，1=前1日（兼容部分用法）
                return self._values[idx] if idx < len(self._values) else 0.0
            else:
                # 负索引：-1=前1日，-2=前2日（标准Backtrader用法）
                # 转换：-1 -> 1, -2 -> 2...
                pos_idx = abs(idx)
                return self._values[pos_idx] if pos_idx < len(self._values) else 0.0
        except (IndexError, TypeError):
            return 0.0

    def __len__(self) -> int:
        return len(self._values)


def _pad_data(data: List[float], min_len: int = 20) -> List[float]:
    """
    自动补全数据到最小长度
    向前补全（用最早的一个值填充），因为data是[当日, 前1日, 前2日...]
    """
    if len(data) >= min_len:
        return data[:min_len]
    # 用最早的一个值（列表最后一个元素）向前填充
    pad_value = data[-1] if data else 0.0
    padding = [pad_value] * (min_len - len(data))
    return data + padding


class MockStrategy:
    def __init__(self,
                 bias6: List[float], bias12: List[float], bias24: List[float],
                 close: List[float], high: Optional[List[float]] = None,
                 low: Optional[List[float]] = None, ma60: Optional[List[float]] = None,
                 stock_type: str = "mid_cap"):

        min_data_len = 20

        # 补全所有数据到20天
        padded_close = _pad_data(close, min_data_len)
        padded_bias6 = _pad_data(bias6, min_data_len)
        padded_bias12 = _pad_data(bias12, min_data_len)
        padded_bias24 = _pad_data(bias24, min_data_len)

        # 初始化指标Line
        self.bias6 = BacktraderLine(padded_bias6)
        self.bias12 = BacktraderLine(padded_bias12)
        self.bias24 = BacktraderLine(padded_bias24)

        # 初始化价格数据
        padded_high = _pad_data(high or [c * 1.05 for c in close], min_data_len)
        padded_low = _pad_data(low or [c * 0.95 for c in close], min_data_len)

        self.data = type('obj', (object,), {
            'close': BacktraderLine(padded_close),
            'high': BacktraderLine(padded_high),
            'low': BacktraderLine(padded_low)
        })

        # 初始化MA60（默认比收盘价高1%作为中位，可自定义）
        if ma60:
            padded_ma60 = _pad_data(ma60, min_data_len)
        else:
            padded_ma60 = [c * 1.01 for c in padded_close]

        self.ma60 = BacktraderLine(padded_ma60)

        # 股票类型参数
        self.p = MockParams(stock_type=stock_type)
        # 清空阈值缓存
        if hasattr(self, '_bias_thresholds'):
            delattr(self, '_bias_thresholds')


# ===================== 第七步：单元测试（修复版） =====================
class TestBIASScoreCalculation(unittest.TestCase):
    def setUp(self):
        print("\n" + "=" * 50)

    def test_01_extreme_passivation(self):
        """测试1：最高优先级 - 连续涨跌停钝化"""
        print("【测试1】极端单边行情钝化")
        # 连续2日一字涨停
        close = [11.0, 10.0, 9.0]
        high = [11.0, 10.0, 9.0]
        low = [11.0, 10.0, 9.0]
        # 即使有其他信号（如超买），也应被钝化覆盖
        bias6 = [12.0, 10.0, 8.0]
        bias12 = [10.0, 8.0, 6.0]
        bias24 = [8.0, 6.0, 4.0]

        strat = MockStrategy(bias6, bias12, bias24, close, high, low)
        result = calculate_bias_score(strat)

        print(f"结果: {result}")
        self.assertTrue(result["extreme_passivation"])
        self.assertEqual(result["net_score"], 0.0)

    def test_02_bottom_divergence(self):
        """测试2：最高优先级 - 底背离（无位置修正）"""
        print("【测试2】底背离")
        # 构造10天数据：前5天价格高，后5天价格低（创新低）
        # 前5天（远）：10.0, 10.1, 10.2, 10.3, 10.4
        # 后5天（近）：9.8, 9.7, 9.6, 9.5, 9.4 (创新低)
        close = [9.4, 9.5, 9.6, 9.7, 9.8, 10.0, 10.1, 10.2, 10.3, 10.4]

        # BIAS6：前5天低（-8），后5天高（-6）（未创新低）
        bias6 = [-5.8, -6.0, -6.2, -6.4, -6.6, -7.8, -8.0, -8.2, -8.4, -8.6]
        bias12 = [-4.0] * 10
        bias24 = [-2.0] * 10
        ma60 = [10.0] * 10  # 低位（价格<MA60）

        strat = MockStrategy(bias6, bias12, bias24, close, ma60=ma60)
        result = calculate_bias_score(strat)

        print(f"结果: {result}")
        self.assertIn("底背离", result["signal_type"])
        self.assertEqual(result["signal_weight"], 2.0)
        self.assertEqual(result["final_buy"], 4.0)  # 2.0 * 2.0
        self.assertEqual(result["net_score"], 4.0)

    def test_03_extreme_oversell_low_position(self):
        """测试3：高优先级 - 低位极端超卖（位置修正翻倍）"""
        print("【测试3】低位极端超卖共振")
        # 构造数据：价格9.0 < MA60 10.0（低位）
        bias6 = [-11.0, -10.5, -10.0]
        bias12 = [-11.0, -10.5, -10.0]
        bias24 = [-14.0, -13.5, -13.0]
        close = [9.0, 9.2, 9.4]
        ma60 = [10.0, 10.0, 10.0]  # 低位

        strat = MockStrategy(bias6, bias12, bias24, close, ma60=ma60, stock_type='small_cap')
        result = calculate_bias_score(strat)

        print(f"结果: {result}")
        self.assertIn("极端超卖共振", result["signal_type"])
        self.assertEqual(result["position"], "低位")
        self.assertEqual(result["position_corr_oversell"], 2.0)
        # 计算：2.0(raw) * 1.5(信号权重) * 2.0(位置修正) = 6.0
        self.assertEqual(result["final_buy"], 6.0)
        self.assertEqual(result["net_score"], 6.0)

    def test_04_extreme_overbuy_high_position(self):
        """测试4：高优先级 - 高位极端超买（位置修正翻倍）"""
        print("【测试4】高位极端超买共振")
        bias6 = [12.0, 11.5, 11.0]
        bias12 = [12.0, 11.5, 11.0]
        bias24 = [15.0, 14.5, 14.0]
        # 构造20天数据，当日创20日新高
        close = [12.0] + [11.0 - i * 0.1 for i in range(19)]
        ma60 = [10.0] * 20  # 高位（12.0 / 10.0 = 1.2 > 1.1）

        strat = MockStrategy(bias6, bias12, bias24, close, ma60=ma60, stock_type='small_cap')
        result = calculate_bias_score(strat)

        print(f"结果: {result}")
        self.assertIn("极端超买共振", result["signal_type"])
        self.assertEqual(result["position"], "高位")
        self.assertEqual(result["position_corr_overbuy"], 2.0)
        # 计算：2.0(raw) * 1.5(信号权重) * 2.0(位置修正) = 6.0
        self.assertEqual(result["final_sell"], 6.0)
        self.assertEqual(result["net_score"], -6.0)

    def test_05_strong_bull(self):
        """测试5：中高优先级 - 强势多头排列"""
        print("【测试5】强势多头")
        bias6 = [5.0, 4.5, 4.0]
        bias12 = [3.0, 2.5, 2.0]
        bias24 = [1.0, 0.5, 0.0]
        close = [10.5, 10.4, 10.3]
        ma60 = [10.0, 10.0, 10.0]  # 中位

        strat = MockStrategy(bias6, bias12, bias24, close, ma60=ma60)
        result = calculate_bias_score(strat)

        print(f"结果: {result}")
        self.assertIn("强势多头", result["signal_type"])
        self.assertEqual(result["signal_weight"], 1.2)
        self.assertEqual(result["final_buy"], 1.8)  # 1.5 * 1.2
        self.assertEqual(result["final_sell"], 0.36)  # 0.3 * 1.2
        self.assertEqual(result["net_score"], 1.44)

    def test_06_zero_axis(self):
        """测试6：低优先级 - 零轴附近"""
        print("【测试6】零轴附近")
        # 构造数据：BIAS6在0.5（零轴附近）
        # 且不满足强势多头（BIAS6 > BIAS12 但 BIAS12 < BIAS24）
        bias6 = [0.5, 0.3, 0.1]
        bias12 = [0.2, 0.1, 0.0]
        bias24 = [0.4, 0.2, 0.1]  # BIAS12(0.2) < BIAS24(0.4)
        close = [10.1, 10.0, 9.9]

        strat = MockStrategy(bias6, bias12, bias24, close)
        result = calculate_bias_score(strat)

        print(f"结果: {result}")
        self.assertIn("零轴附近", result["signal_type"])
        self.assertEqual(result["signal_weight"], 0.5)
        self.assertEqual(result["net_score"], 0.0)  # 0.25 - 0.25

    def test_07_bull_zone(self):
        """测试7：常规优先级 - 多头区域"""
        print("【测试7】多头区域")
        # 构造数据：BIAS6=2.5>0（多头区域）
        # 且不满足强势多头排列（BIAS12 < BIAS24）
        bias6 = [2.5, 2.0, 1.5]
        bias12 = [1.5, 1.0, 0.5]
        bias24 = [2.0, 1.5, 1.0]  # BIAS12(1.5) < BIAS24(2.0)
        close = [10.3, 10.2, 10.1]

        strat = MockStrategy(bias6, bias12, bias24, close)
        result = calculate_bias_score(strat)

        print(f"结果: {result}")
        self.assertIn("多头区域", result["signal_type"])
        self.assertEqual(result["net_score"], 0.7)  # 1.0 - 0.3


if __name__ == '__main__':
    unittest.main(verbosity=2)