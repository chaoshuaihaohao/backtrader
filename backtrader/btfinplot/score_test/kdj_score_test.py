import unittest
import backtrader as bt
import math
import backtrader.btfinplot as btfinplot
# ===================== 核心常量 =====================
PRICE_POSITION_KDJ_COEFF = {
    "低位": {"超卖": 1.2, "超买": 1.0},
    "中位": {"超卖": 1.0, "超买": 1.0},
    "高位": {"超卖": 1.0, "超买": 1.2}
}


def _get_kdj_rule_table(strat):
    """
    动态生成KDJ评分规则表（绑定策略对象用于条件判断）
    策略对象(strat)需包含的核心属性：
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
        {
            "priority": "最高（趋势反转）",
            "signal_type": "dynamic",
            "description": "KDJ金叉 or 即将金叉",
            # === 修复点：将条件放入 conditions 列表中 ===
            "conditions": [
                {
                    "name": "KDJ金叉或即将金叉",
                    # 注意：这里保留了你原来的复杂逻辑作为一个子条件
                    "code_condition": lambda: (
                        (strat.kdj_gold_cross[0] > 0 or  # KDJ金叉
                         kdj_about_to_gold_cross(strat))
                    ),
                    "buy_score": 2.0,
                    "sell_score": 0.0
                },
                {
                    "name": "日【MACD死叉、死叉中下降】尽量不买",
                    # 注意：这里保留了你原来的复杂逻辑作为一个子条件
                    "code_condition": lambda: (
                        (btfinplot.get_macd_status(strat).get('cross_status') in ['死叉' ,'死叉中下降'])
                    ),
                    "buy_score": 0.0,
                    "sell_score": 2.0
                },
                {
                    "name": "KDJ金叉或即将金叉 并且【放量】 买入",
                    # 注意：这里保留了你原来的复杂逻辑作为一个子条件
                    "code_condition": lambda: (
                        strat.kdj_gold_cross[0] > 0 or  # KDJ金叉
                        kdj_about_to_gold_cross(strat) and
                        strat.data0.close[-1] < strat.data0.close[0] and strat.vol[0] > strat.vol[-1] * 1.10 and
                        strat.turnover[0] > strat.turnover[-1] * 1.0
                    ),
                    "buy_score": 2.0,
                    "sell_score": 0.0
                },
                # {
                #     "name": "【周MACD死叉、死叉中下降】回测发现:买入时不要看这个",
                #     # 注意：这里保留了你原来的复杂逻辑作为一个子条件
                #     "code_condition": lambda: (
                #         (btfinplot.get_week_macd_status(strat).get('cross_status') in ['死叉' ,'死叉中下降'])
                #     ),
                #     "buy_score": 0.0,
                #     "sell_score": 2.0
                # },
                # {
                #     "name": "【周KDJ金叉、即将金叉、或死叉上升】 回测发现:买入时不要看这个",
                #     "code_condition": lambda: (
                #         (week_get_kdj_status(strat) == '金叉' or
                #          week_get_kdj_status(strat) == '即将金叉' or
                #          week_get_kdj_status(strat) == '死叉中上升')
                #     ),
                #     "buy_score": 2.0,
                #     "sell_score": 0.0
                # },
                # {
                #     "name": "【周KDJ死叉、即将死叉、金叉中下降】 买入时不需要看这个",
                #     # 注意：这里保留了你原来的复杂逻辑作为一个子条件
                #     "code_condition": lambda: (
                #         (week_get_kdj_status(strat) == '死叉' or
                #          week_get_kdj_status(strat) == '即将死叉' or
                #          week_get_kdj_status(strat) == '金叉中下降')
                #     ),
                #     "buy_score": 0.0,
                #     "sell_score":0.0
                # }
            ],
            "confirm_condition": lambda: (
                # (strat.obv[-1] < strat.obv and strat.ad[-1] < strat.ad)
                True
            ),
            "signal_type_weight": 2.0,
            "is_extreme": False
        },
        {
            "priority": "最高（趋势反转）",
            "signal_type": "dynamic",
            "description": "kdj死叉 or 即将死叉",
            "conditions": [
                {
                    "name": "KDJ死叉或即将死叉",
                    "code_condition": lambda: (
                        (strat.kdj_gold_cross[0] < 0 or kdj_about_to_dead_cross(strat))
                    ),
                    "buy_score": 0.0,
                    "sell_score": 2.0
                },
                # {
                #     "name": "【周KDJ金叉、即将金叉、或死叉上升】回测发现:买入时不要看这个",
                #     # 注意：这里保留了你原来的复杂逻辑作为一个子条件
                #     "code_condition": lambda: (
                #         (week_get_kdj_status(strat) == '金叉' or
                #          week_get_kdj_status(strat) == '即将金叉' or
                #          week_get_kdj_status(strat) == '死叉中上升')
                #     ),
                #     "buy_score": 2.0,
                #     "sell_score": 0.0
                # },
                # {
                #     "name": "周KDJ死叉、即将死叉、金叉中下降",
                #     # 注意：这里保留了你原来的复杂逻辑作为一个子条件
                #     "code_condition": lambda: (
                #         (week_get_kdj_status(strat) == '死叉' or
                #          week_get_kdj_status(strat) == '即将死叉' or
                #          week_get_kdj_status(strat) == '金叉中下降')
                #     ),
                #     "buy_score": 0.0,
                #     "sell_score":2.0
                # }
            ],
            "confirm_condition": lambda: (True), # 简化示例，保留原逻辑
            "signal_type_weight": 1.0,
            "is_extreme": False
        },
        # {
        #     "priority": "",
        #     "signal_type": "dynamic",
        #     "description": "KDJ向下，注意，是J不是K，且放量大跌",
        #     "conditions": [
        #         {
        #             "name": "KDJ.J向下",
        #             "code_condition": lambda: (
        #                 strat.j[-1] > strat.j[0]
        #             ),
        #             "buy_score": 0.0,
        #             "sell_score": 2.0
        #         },
        #         {
        #             "name": "放量大跌 and 换手率超高",
        #             "code_condition": lambda: (
        #                 strat.data0.close[-1] > strat.data0.close[0] and strat.vol[0] > strat.vol[-1] * 1.5 and strat.turnover[0] > strat.turnover[-1] * 1.5
        #             ),
        #             "buy_score": 0.0,
        #             "sell_score": 2.0
        #         },
        #     ],
        #     "confirm_condition": lambda: (True),  # 简化示例，保留原逻辑
        #     "signal_type_weight": 1.0,
        #     "is_extreme": False
        # },
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


def get_kdj_status(strat):
    """
    判断KDJ金叉死叉状态，包含：死叉、金叉、即将死叉、即将金叉、死叉中上升/下降、金叉中上升/下降
    :param strat: 策略实例（包含k/d/kdj_gold_cross等属性）
    :return: KDJ状态字符串
    """
    # ===== 1. 基础校验：确保必要属性存在 =====
    has_k = hasattr(strat, 'k') and len(strat.k) > 1
    has_d = hasattr(strat, 'd') and len(strat.d) > 1
    has_cross = hasattr(strat, 'kdj_gold_cross') and len(strat.kdj_gold_cross) >= 1

    # ===== 2. 核心状态判断 =====
    # 2.1 死叉（K下穿D的信号）
    if has_cross and strat.kdj_gold_cross[0] < 0:
        return "死叉"
    # 2.2 金叉（K上穿D的信号）
    elif has_cross and strat.kdj_gold_cross[0] > 0:
        return "金叉"
    # 2.3 即将死叉（K在D上、向下拐头、距离极近）
    elif has_k and has_d:
        if (strat.k[0] > strat.d[0] and  # K在D上方（未死叉）
                strat.k[0] < strat.k[-1] and  # K线向下拐头
                kdj_about_to_dead_cross(strat)):  # 距离近，即将死叉
            return "即将死叉"
    # 2.4 即将金叉（K在D下、向上拐头、距离极近）
    elif has_k and has_d:
        if (strat.k[0] < strat.d[0] and  # K在D下方（未金叉）
                strat.k[0] > strat.k[-1] and  # K线向上拐头
                kdj_about_to_gold_cross(strat)):  # 距离近，即将金叉
            return "即将金叉"

    # ===== 3. 死叉中/金叉中 + 趋势细分 =====
    if has_k and has_d:
        # 3.1 死叉中（K < D）：区分上升/下降
        if strat.k[0] < strat.d[0]:
            # 死叉中上升：K线自身向上（最新K > 前一根K）
            if strat.k[0] > strat.k[-1]:
                return "死叉中上升"
            # 死叉中下降：K线自身向下（最新K <= 前一根K）
            else:
                return "死叉中下降"
        # 3.2 金叉中（K > D）：区分上升/下降
        elif strat.k[0] > strat.d[0]:
            # 金叉中上升：K线自身向上（最新K > 前一根K）
            if strat.k[0] > strat.k[-1]:
                return "金叉中上升"
            # 金叉中下降：K线自身向下（最新K <= 前一根K）
            else:
                return "金叉中下降"

    # ===== 4. 兜底：无有效数据/未匹配任何状态 =====
    return "其他"

def week_get_kdj_status(strat):
    """
    判断周KDJ金叉死叉状态，包含：死叉、金叉、即将死叉、即将金叉、死叉中上升/下降、金叉中上升/下降
    :param strat: 策略实例（需包含week_k/week_d/week_kdj_gold_cross等属性）
    :return: 周KDJ状态字符串（如："死叉"、"金叉中上升"、"即将金叉"等）
    """
    # ===== 前置校验：确保属性存在且长度足够 =====
    # 校验金叉死叉信号列表
    has_week_cross = hasattr(strat, 'week_kdj_gold_cross') and len(strat.week_kdj_gold_cross) >= 1
    # 校验周K/周D线数据（至少2根才能判断趋势）
    has_week_k = hasattr(strat, 'week_k') and len(strat.week_k) > 1
    has_week_d = hasattr(strat, 'week_d') and len(strat.week_d) > 1

    # ===== 1. 核心状态：死叉/金叉（明确的交叉信号） =====
    if has_week_cross:
        # 死叉：K下穿D的信号（week_kdj_gold_cross[0] < 0）
        if strat.week_kdj_gold_cross[0] < 0:
            return "死叉"
        # 金叉：K上穿D的信号（week_kdj_gold_cross[0] > 0）
        elif strat.week_kdj_gold_cross[0] > 0:
            return "金叉"

    # ===== 2. 预警状态：即将死叉/即将金叉（未交叉但临近） =====
    if has_week_k and has_week_d:
        # 即将死叉：K在D上方 + K向下拐头 + 距离极近
        if (strat.week_k[0] > strat.week_d[0]  # K在D上方（还没死叉）
                and strat.week_k[0] < strat.week_k[-1]  # K线向下拐头（最新K < 前一根K）
                and week_kdj_about_to_dead_cross(strat)):  # 距离近，即将死叉
            return "即将死叉"

        # 即将金叉：K在D下方 + K向上拐头 + 距离极近
        if (strat.week_k[0] < strat.week_d[0]  # K在D下方（还没金叉）
                and strat.week_k[0] > strat.week_k[-1]  # K线向上拐头（最新K > 前一根K）
                and week_kdj_about_to_gold_cross(strat)):  # 距离近，即将金叉
            return "即将金叉"

    # ===== 3. 持续状态：死叉中/金叉中 + 趋势细分（上升/下降） =====
    if has_week_k and has_week_d:
        # 死叉中（K < D）：区分上升/下降（基于K线自身趋势）
        if strat.week_k[0] < strat.week_d[0]:
            # 死叉中上升：死叉状态下，K线自身向上（最新K > 前一根K）
            if strat.week_k[0] > strat.week_k[-1]:
                return "死叉中上升"
            # 死叉中下降：死叉状态下，K线自身向下（最新K <= 前一根K）
            else:
                return "死叉中下降"

        # 金叉中（K > D）：区分上升/下降（基于K线自身趋势）
        elif strat.week_k[0] > strat.week_d[0]:
            # 金叉中上升：金叉状态下，K线自身向上（最新K > 前一根K）
            if strat.week_k[0] > strat.week_k[-1]:
                return "金叉中上升"
            # 金叉中下降：金叉状态下，K线自身向下（最新K <= 前一根K）
            else:
                return "金叉中下降"

    # ===== 4. 兜底状态：无有效数据/未匹配任何状态 =====
    return "其他"

def kdj_about_to_gold_cross(strat, window=5):
    """
    基于 KDJ 历史波动率的动态阈值 (纯 Backtrader 实现)

    逻辑：利用 Backtrader 内置的统计函数计算过去N天K-D距离的标准差，
          动态调整“即将金叉”的判定阈值。
    """
    # 1. 安全检查：确保数据长度足够
    if len(strat.k) <= window:
        return False

    # 2. 获取历史数据 (使用 Backtrader 的切片语法 [-window:] 获取最近window个值)
    # 注意：backtrader 的索引 0 是当前，-1 是上一根，切片遵循 Python 列表习惯
    k_hist = strat.k.get(size=window)  # 返回一个包含 window 个数值的列表
    d_hist = strat.d.get(size=window)

    # 3. 计算历史距离的标准差 (使用 Python 内置的 statistics 模块或手动计算)
    # 方案 A: 使用 Python 内置 statistics (推荐，无需 numpy)
    import statistics

    # 计算过去 window 天 K 和 D 的绝对距离列表
    dist_history = [abs(k - d) for k, d in zip(k_hist, d_hist)]

    # 计算标准差
    try:
        historical_std = statistics.stdev(dist_history) if len(dist_history) > 1 else dist_history[0]
    except statistics.StatisticsError:
        # 如果数据全是相同的导致无法计算标准差，使用均值作为替代
        historical_std = statistics.mean(dist_history)

    # 4. 计算动态阈值
    dynamic_threshold = historical_std  # 取标准差的一半
    dynamic_threshold = max(dynamic_threshold, 1.0)  # 保底值
    # print(f"dynamic_threshold {dynamic_threshold}")
    # 5. 当前状态判断 (这部分逻辑不变)
    current_k = strat.k[0]
    current_d = strat.d[0]
    prev_k = strat.k[-1]
    prev_d = strat.d[-1]

    condition_not_crossed = current_k <= current_d  # 未金叉
    condition_k_rising = current_k > prev_k  # K线向上
    condition_distance_close = abs(current_k - current_d) < dynamic_threshold  # 距离够近
    condition_d_stable = (current_d - prev_d) >= -1.5  # D线未垂直暴跌
    # 5. 最近两天不能有死叉
    condition_not_dead = not (strat.kdj_gold_cross[-2] < 0 or strat.kdj_gold_cross[-1] < 0)
    # 6. 综合判断
    if (condition_not_crossed and
            condition_k_rising and
            condition_distance_close and
            condition_d_stable and
            condition_not_dead):
        return True

    return False


def kdj_about_to_dead_cross(strat, window=5):
    """
    基于 KDJ 历史波动率的动态阈值 - 判定“即将死叉”

    逻辑：K值在D值上方，K线开始向下拐头，且K与D的距离小于历史波动率阈值。
    """
    # 1. 安全检查：确保数据长度足够
    if len(strat.k) <= window:
        return False

    # 2. 获取历史数据
    k_hist = strat.k.get(size=window)
    d_hist = strat.d.get(size=window)

    # 3. 计算历史距离的标准差
    import statistics
    dist_history = [abs(k - d) for k, d in zip(k_hist, d_hist)]

    try:
        historical_std = statistics.stdev(dist_history) if len(dist_history) > 1 else dist_history
    except statistics.StatisticsError:
        historical_std = statistics.mean(dist_history)

    # 4. 计算动态阈值 (复用你的逻辑)
    dynamic_threshold = historical_std
    dynamic_threshold = max(dynamic_threshold, 1.0)  # 保底值

    # 5. 当前状态判断 (核心逻辑反转)
    current_k = strat.k
    current_d = strat.d
    prev_k = strat.k[-1]
    prev_d = strat.d[-1]

    # --- 即将死叉的条件 ---
    # 1. K在D上方 (还没死叉，但快了)
    condition_not_crossed = current_k >= current_d
    # 2. K线开始向下拐头 (死叉的动力)
    condition_k_falling = current_k < prev_k
    # 3. 距离够近 (在波动率阈值内)
    condition_distance_close = abs(current_k - current_d) < dynamic_threshold
    # 4. D线未垂直暴涨 (排除D线突然拉升导致的假死叉)
    condition_d_stable = (current_d - prev_d) <= 1.5

    # 6. 综合判断
    if (condition_not_crossed and
            condition_k_falling and
            condition_distance_close and
            condition_d_stable):
        return True

    return False

def week_kdj_about_to_gold_cross(strat, window=5):
    """
    基于 KDJ 历史波动率的动态阈值 (纯 Backtrader 实现)

    逻辑：利用 Backtrader 内置的统计函数计算过去N天K-D距离的标准差，
          动态调整“即将金叉”的判定阈值。
    """
    # 1. 安全检查：确保数据长度足够
    if len(strat.week_k) <= window:
        return False

    # 2. 获取历史数据 (使用 Backtrader 的切片语法 [-window:] 获取最近window个值)
    # 注意：backtrader 的索引 0 是当前，-1 是上一根，切片遵循 Python 列表习惯
    k_hist = strat.week_k.get(size=window)  # 返回一个包含 window 个数值的列表
    d_hist = strat.week_d.get(size=window)

    # 3. 计算历史距离的标准差 (使用 Python 内置的 statistics 模块或手动计算)
    # 方案 A: 使用 Python 内置 statistics (推荐，无需 numpy)
    import statistics

    # 计算过去 window 天 K 和 D 的绝对距离列表
    dist_history = [abs(week_k - week_d) for week_k, week_d in zip(k_hist, d_hist)]

    # 计算标准差
    try:
        historical_std = statistics.stdev(dist_history) if len(dist_history) > 1 else dist_history[0]
    except statistics.StatisticsError:
        # 如果数据全是相同的导致无法计算标准差，使用均值作为替代
        historical_std = statistics.mean(dist_history)

    # 4. 计算动态阈值
    dynamic_threshold = historical_std  # 取标准差的一半
    dynamic_threshold = max(dynamic_threshold, 1.0)  # 保底值
    # print(f"dynamic_threshold {dynamic_threshold}")
    # 5. 当前状态判断 (这部分逻辑不变)
    current_k = strat.week_k[0]
    current_d = strat.week_d[0]
    prev_k = strat.week_k[-1]
    prev_d = strat.week_d[-1]

    condition_not_crossed = current_k <= current_d  # 未金叉
    condition_k_rising = current_k > prev_k  # K线向上
    condition_distance_close = abs(current_k - current_d) < dynamic_threshold  # 距离够近
    condition_d_stable = (current_d - prev_d) >= -1.5  # D线未垂直暴跌
    # 5. 最近两天不能有死叉
    condition_not_dead = not (strat.week_kdj_gold_cross[-2] < 0 or strat.week_kdj_gold_cross[-1] < 0)

    # 6. 综合判断
    if (condition_not_crossed and
            condition_k_rising and
            condition_distance_close and
            condition_d_stable and
            condition_not_dead):
        return True

    return False


def week_kdj_about_to_dead_cross(strat, window=5):
    """
    基于 KDJ 历史波动率的动态阈值 - 判定“即将死叉”

    逻辑：K值在D值上方，K线开始向下拐头，且K与D的距离小于历史波动率阈值。
    """
    # 1. 安全检查：确保数据长度足够
    if len(strat.week_k) <= window:
        return False

    # 2. 获取历史数据
    k_hist = strat.week_k.get(size=window)
    d_hist = strat.week_d.get(size=window)

    # 3. 计算历史距离的标准差
    import statistics
    dist_history = [abs(week_k - week_d) for week_k, week_d in zip(k_hist, d_hist)]

    try:
        historical_std = statistics.stdev(dist_history) if len(dist_history) > 1 else dist_history
    except statistics.StatisticsError:
        historical_std = statistics.mean(dist_history)

    # 4. 计算动态阈值 (复用你的逻辑)
    dynamic_threshold = historical_std
    dynamic_threshold = max(dynamic_threshold, 1.0)  # 保底值

    # 5. 当前状态判断 (核心逻辑反转)
    current_k = strat.week_k
    current_d = strat.week_d
    prev_k = strat.week_k[-1]
    prev_d = strat.week_d[-1]

    # --- 即将死叉的条件 ---
    # 1. K在D上方 (还没死叉，但快了)
    condition_not_crossed = current_k >= current_d
    # 2. K线开始向下拐头 (死叉的动力)
    condition_k_falling = current_k < prev_k
    # 3. 距离够近 (在波动率阈值内)
    condition_distance_close = abs(current_k - current_d) < dynamic_threshold
    # 4. D线未垂直暴涨 (排除D线突然拉升导致的假死叉)
    condition_d_stable = (current_d - prev_d) <= 1.5

    # 6. 综合判断
    if (condition_not_crossed and
            condition_k_falling and
            condition_distance_close and
            condition_d_stable):
        return True

    return False

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


def calculate_kdj_score(strat):
    """
    KDJ评分计算核心函数 - 支持规则内叠加，规则间互斥（多匹配报错）
    """
    # 1. 校验必要属性
    required_attrs = ['k', 'd', 'j', 'data', 'ma60']
    for attr in required_attrs:
        if not hasattr(strat, attr):
            raise ValueError(f"策略实例缺少必要属性：{attr}")
    if not hasattr(strat.data, 'close'):
        raise ValueError("strat.data 缺少 close 属性")

    # 2. 动态生成规则表
    KDJ_RULE_TABLE = _get_kdj_rule_table(strat)

    # 3. 初始化变量
    matched_rule_result = None  # 存储最终匹配的那一条规则的计算结果
    matched_rule_count = 0  # 计数器：记录匹配了多少条规则

    # 4. 遍历所有规则 (外层循环：寻找匹配的规则)
    for rule in KDJ_RULE_TABLE:
        try:
            # --- 步骤 A：检查整条规则的确认条件 ---
            confirm_ok = True
            if "confirm_condition" in rule and callable(rule["confirm_condition"]):
                confirm_ok = rule["confirm_condition"]()

            if not confirm_ok:
                continue  # 如果确认条件不通过，跳过这条规则

            # --- 步骤 B：遍历这条规则下的所有子条件 (内层循环：累加分数) ---
            # 初始化该规则的得分
            rule_raw_buy = 0.0
            rule_raw_sell = 0.0
            rule_matched_any_sub = False  # 标记这条规则是否至少有一个子条件命中

            conditions = rule.get("conditions", [])
            for sub_rule in conditions:
                try:
                    if "code_condition" in sub_rule and callable(sub_rule["code_condition"]):
                        if sub_rule["code_condition"]():
                            rule_raw_buy += sub_rule.get("buy_score", 0.0)
                            rule_raw_sell += sub_rule.get("sell_score", 0.0)
                            rule_matched_any_sub = True
                            print(f"命中子条件: {sub_rule.get('name', 'Unknown')} "
                                  f"(Buy+{sub_rule.get('buy_score', 0)}, Sell+{sub_rule.get('sell_score', 0)})")
                except Exception as e:
                    print(f"子条件 {sub_rule.get('name', 'Unknown')} 计算出错: {e}")
                    continue

            # --- 步骤 C：判断该规则是否整体命中 ---
            # 如果该规则下有任何一个子条件命中，则视为该规则命中
            if rule_matched_any_sub:
                # === 关键逻辑：互斥检查 ===
                if matched_rule_count >= 1:
                    # 如果已经匹配过至少一条规则，现在又匹配了一条，报错！
                    raise ValueError(f"严重错误：一组数据匹配了多条规则。"
                                     f"已匹配规则数: {matched_rule_count + 1}。"
                                     f"请检查规则逻辑避免重叠。")

                # 记录匹配结果
                matched_rule_count += 1
                matched_rule_result = {
                    "rule": rule,
                    "raw_buy": rule_raw_buy,
                    "raw_sell": rule_raw_sell
                }

        except Exception as e:
            # 捕获规则级错误（包括上面的互斥报错）
            if "严重错误" in str(e):
                raise e  # 重新抛出互斥错误
            print(f"规则 {rule.get('description', 'Unknown')} 执行出错: {e}")
            continue

    # 5. 处理匹配结果
    # 情况 A: 没有匹配任何规则
    if matched_rule_result is None:
        price_pos = kdj_get_price_position(strat)
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

    # 情况 B: 匹配了一条规则 (matched_rule_count == 1)
    # (如果大于1，上面的循环里已经报错并抛出了)

    best_rule = matched_rule_result["rule"]
    raw_buy = matched_rule_result["raw_buy"]
    raw_sell = matched_rule_result["raw_sell"]

    # 6. 计算权重和修正
    weight = best_rule.get("signal_type_weight", 1.0)
    weighted_buy = round(raw_buy * weight, 4)
    weighted_sell = round(raw_sell * weight, 4)

    # 7. 价格位置修正
    price_pos = kdj_get_price_position(strat)
    if best_rule.get("is_extreme", False):
        if "超卖" in best_rule["description"]:
            weighted_buy = round(weighted_buy * PRICE_POSITION_KDJ_COEFF[price_pos]['超卖'], 4)
        elif "超买" in best_rule["description"]:
            weighted_sell = round(weighted_sell * PRICE_POSITION_KDJ_COEFF[price_pos]['超买'], 4)

    # 8. 返回结果
    return {
        "signal_type": f"{best_rule['priority']}-{best_rule['description']}",
        "raw_buy": raw_buy,
        "raw_sell": raw_sell,
        "signal_type_weight": weight,
        "price_position": price_pos,
        "buy_score": weighted_buy,
        "sell_score": weighted_sell,
        "net_score": round(weighted_buy - weighted_sell, 2)
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


class MockKDJstrat:
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
        strat = MockKDJstrat(k_vals, d_vals, j_vals)
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
        strat = MockKDJstrat(k_vals, d_vals, j_vals)
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
        strat = MockKDJstrat(k_vals, d_vals, j_vals, close_vals=close_vals)
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
        strat = MockKDJstrat(k_vals, d_vals, j_vals, close_vals=close_vals, ma60_vals=ma60_vals)
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
        strat = MockKDJstrat(k_vals, d_vals, j_vals)
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
        strat = MockKDJstrat(k_vals, d_vals, j_vals, close_vals=close_vals, ma60_vals=ma60_vals)
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
        strat = MockKDJstrat(k_vals, d_vals, j_vals)
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
        strat = MockKDJstrat(k_vals, d_vals, j_vals)
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
        strat = MockKDJstrat(k_vals, d_vals, j_vals)
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
        strat = MockKDJstrat(k_vals, d_vals, j_vals)
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
        strat = MockKDJstrat(k_vals, d_vals, j_vals)
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
        strat = MockKDJstrat(k_vals, d_vals, j_vals)
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
        strat = MockKDJstrat(k_vals, d_vals, j_vals)
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
        strat = MockKDJstrat(k_vals, d_vals, j_vals)
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
        strat = MockKDJstrat(k_vals, d_vals, j_vals)
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
        strat = MockKDJstrat(k_vals, d_vals, j_vals)
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
        strat = MockKDJstrat(k_vals, d_vals, j_vals)
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
        strat = MockKDJstrat([18, 25], [15, 18], [25, 22])
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
        strat = MockKDJstrat(k_vals, d_vals, j_vals)
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
        strat = MockKDJstrat(k_vals, d_vals, j_vals)
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
        strat = MockKDJstrat(k_vals, d_vals, j_vals)
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
        strat = MockKDJstrat(k_vals, d_vals, j_vals)
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
        strat = MockKDJstrat(k_vals, d_vals, j_vals)
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
        strat = MockKDJstrat(k_vals, d_vals, j_vals)
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
        strat = MockKDJstrat(k_vals, d_vals, j_vals)
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
        strat = MockKDJstrat(k_vals, d_vals, j_vals)
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
        strat = MockKDJstrat(k_vals, d_vals, j_vals)
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
        strat = MockKDJstrat(k_vals, d_vals, j_vals)
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
        strat = MockKDJstrat(k_vals, d_vals, j_vals)
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
        strat = MockKDJstrat(k_vals, d_vals, j_vals)
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
        strat = MockKDJstrat(k_vals, d_vals, j_vals)
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
        strat = MockKDJstrat(k_vals, d_vals, j_vals)
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
        strat = MockKDJstrat(k_vals, d_vals, j_vals)
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
        strat = MockKDJstrat(k_vals, d_vals, j_vals)
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
        strat = MockKDJstrat(k_vals, d_vals, j_vals)
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
        strat = MockKDJstrat(k_vals, d_vals, j_vals)
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
        strat = MockKDJstrat(k_vals, d_vals, j_vals)
        result = calculate_kdj_score(strat)
        print(f"测试结果: {result['signal_type']} | 净得分: {result['net_score']}")
        self.assertEqual(result['signal_type'], "低（粘合平衡）-偏空粘合")
        self.assertEqual(result['net_score'], -0.9)


if __name__ == '__main__':
    unittest.main(verbosity=2)