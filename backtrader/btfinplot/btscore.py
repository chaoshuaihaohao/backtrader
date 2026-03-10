#!/usr/bin/env python3

import numpy as np
import math
from . import score_test


# 3. 指标最终权重（手册原版）
INDICATOR_FINAL_WEIGHT = {
    'MACD': 1.0,
    'RSI': 1.0,
    'KDJ': 1.0,
    'BIAS': 1.0,
    'BOLL': 1.0,
    '均线': 1.0,
    '成交量': 1.0,
    '换手率': 1.0
}

# ========== 修复：全局阈值初始化为字典（而非列表） ==========
DECISION_THRESHOLDS = {
    '强力买入': 0.0,
    '积极买入': 0.0,
    '谨慎买入': 0.0,
    '观望': 0.0,
    '谨慎卖出': 0.0,
    '强力卖出': 0.0
}

# ========== 新增：默认指标开关 ==========
DEFAULT_INDICATOR_SWITCH = {
    'MACD': False,
    'RSI': False,
    'KDJ': True,
    'BIAS': False,
    'BOLL': False,
    '均线': False,
    '成交量': False,
    '换手率': False
}

# 单指标基础得分范围（-2 ~ +2）
SINGLE_INDICATOR_SCORE_RANGE = (-2, 2)

def get_ma_trend_global(ma_line, window=10, slope_threshold=0.0005):
    """
    计算均线趋势（斜率和趋势方向）
    :param ma_line: backtrader均线指标线
    :param window: 计算斜率的窗口
    :param slope_threshold: 趋势判断的斜率阈值
    :return: (slope, trend) slope=斜率值，trend=趋势字符串（上涨/横盘/下跌）
    """
    if len(ma_line) < window:
        return 0.0, "未知"

    # 获取最近window个均线值
    ma_values = [ma_line[i] for i in range(-window + 1, 1)]
    x = np.arange(window)
    y = np.array(ma_values)

    # 计算线性回归斜率
    slope = np.polyfit(x, y, 1)[0]

    # 判断趋势
    if slope > slope_threshold:
        trend = "上涨"
    elif slope < -slope_threshold:
        trend = "下跌"
    else:
        trend = "横盘"

    return slope, trend

def calculate_dynamic_decision_thresholds(indicator_switch):
    """
    根据开启的指标动态计算决策阈值
    :param indicator_switch: 指标开关字典
    :return: 动态生成的DECISION_THRESHOLDS（字典类型）
    """
    # 1. 计算开启指标的总权重
    total_weight = 0.0
    for indicator, is_enabled in indicator_switch.items():
        if is_enabled and indicator in INDICATOR_FINAL_WEIGHT:
            total_weight += INDICATOR_FINAL_WEIGHT[indicator]

    # 2. 单指标基础分极值（-2 ~ +2）
    min_single_score, max_single_score = SINGLE_INDICATOR_SCORE_RANGE

    # 3. 计算总得分极值（总权重 × 单指标极值）
    max_total_score = total_weight * max_single_score
    min_total_score = total_weight * min_single_score

    # 4. 按比例分配各决策阈值（保持原比例关系）
    # 原比例：强力买入=6, 积极买入=4.5, 谨慎买入=4, 观望=0, 谨慎卖出=-4.5, 强力卖出=-6
    # 比例系数：强力买入=1.0, 积极买入=0.75, 谨慎买入=0.6667, 谨慎卖出=-0.75, 强力卖出=-1.0
    thresholds = {
        '强力买入': round(max_total_score * 1.0, 2),
        '积极买入': round(max_total_score * 0.75, 2),
        '谨慎买入': round(max_total_score * 0.6667, 2),
        '观望': 0.0,
        # '谨慎卖出': round(min_total_score * 0.75, 2),
        '谨慎卖出': round(min_total_score * 0.75, 2),
        '强力卖出': round(min_total_score * 1.0, 2)
    }

    return thresholds

def update_global_decision_thresholds(indicator_switch=None):
    """
    更新全局决策阈值（供外部策略调用）
    :param indicator_switch: 指标开关字典，默认使用DEFAULT_INDICATOR_SWITCH
    """
    global DECISION_THRESHOLDS  # 声明使用全局变量
    if indicator_switch is None:
        indicator_switch = DEFAULT_INDICATOR_SWITCH.copy()
    DECISION_THRESHOLDS = calculate_dynamic_decision_thresholds(indicator_switch)

def calculate_total_score(strat, indicator_switch=None):
    """
    严格匹配手册的总分计算函数（新增指标开关功能）
    :param strat: 策略实例
    :param indicator_switch: 指标开关字典，如 {'成交量': False, 'MACD': True}，默认全部开启
    :return: total_score, decision, details
    """
    # 初始化指标开关（兼容默认值）
    if indicator_switch is None:
        indicator_switch = DEFAULT_INDICATOR_SWITCH.copy()
    # 校验开关字典的键（防止传错键导致报错）
    for key in indicator_switch.keys():
        if key not in INDICATOR_FINAL_WEIGHT:
            raise ValueError(f"无效的指标开关键：{key}，可选键：{list(INDICATOR_FINAL_WEIGHT.keys())}")

    # ========== 核心修改：动态计算决策阈值并更新全局 ==========
    dynamic_thresholds = calculate_dynamic_decision_thresholds(indicator_switch)

    # 初始化各指标得分
    indicator_scores = {
        'MACD': 0.0,
        'RSI': 0.0,
        'KDJ': 0.0,
        'BIAS': 0.0,
        'BOLL': 0.0,
        '均线': 0.0,
        '成交量': 0.0,
        '换手率': 0.0
    }

    # 1. 计算MACD得分
    try:
        indicator_scores['MACD'] = score_test.calculate_macd_score(strat)
    except Exception:
        indicator_scores['MACD'] = 0.0

    # 2. 计算RSI得分
    try:
        rsi_result = score_test.calculate_rsi_score(strat) if len(strat.data) > 20 else {"net_score": 0.0}
        indicator_scores['RSI'] = rsi_result["net_score"]
        # print(f"RSI SCORE {indicator_scores['RSI']}")
    except Exception:
        print("RSI SCORE Exception")
        indicator_scores['RSI'] = 0.0

    # 3. 计算KDJ得分
    try:
        kdj_result = score_test.calculate_kdj_score(strat) if len(strat.data) > 20 else {"net_score": 0.0}
        indicator_scores['KDJ'] = kdj_result["net_score"]
    except Exception:
        indicator_scores['KDJ'] = 0.0

    # 4. 计算BIAS得分（极端行情屏蔽：连续涨跌停钝化时归零）
    try:
        bias_result = score_test.calculate_bias_score(strat) if len(strat.data) > 20 else {"net_score": 0.0}
        indicator_scores['BIAS'] = bias_result["net_score"]
    except Exception:
        indicator_scores['BIAS'] = 0.0

    # 5. 计算BOLL得分（极端行情屏蔽：连续涨跌停钝化时归零）
    try:
        boll_result = score_test.calculate_boll_score(strat) if len(strat.data) > 20 else {"net_score": 0.0}
        indicator_scores['BOLL'] = boll_result["net_score"]
    except Exception:
        indicator_scores['BOLL'] = 0.0

    # 6. 计算均线得分
    try:
        ma_result = score_test.calculate_ma_score(strat) if len(strat.data) > 20 else {"net_score": 0.0}
        indicator_scores['均线'] = ma_result["net_score"]
    except Exception:
        indicator_scores['均线'] = 0.0

    # 7. 计算成交量得分
    try:
        volume_result = score_test.calculate_volume_score(strat) if len(strat.data) > 20 else {"net_score": 0.0}
        indicator_scores['成交量'] = volume_result["net_score"]
    except Exception:
        indicator_scores['成交量'] = 0.0

    # 8. 计算换手率得分
    try:
        indicator_scores['换手率'] = score_test.calculate_turnover_score(strat) if len(strat.data) > 20 else 0.0
    except Exception:
        indicator_scores['换手率'] = 0.0

    # ========== 核心修改：根据开关计算总分 ==========
    total_score = 0.0
    for indicator, score in indicator_scores.items():
        # 仅累加开关为True的指标得分×权重
        if indicator_switch.get(indicator, True):
            total_score += score * INDICATOR_FINAL_WEIGHT[indicator]

    # ========== 严格按动态阈值判断交易决策 ==========
    if total_score >= dynamic_thresholds['强力买入']:
        decision = '强力买入/重仓'
    elif dynamic_thresholds['积极买入'] <= total_score < dynamic_thresholds['强力买入']:
        decision = '积极买入/加仓'
    elif dynamic_thresholds['谨慎买入'] <= total_score < dynamic_thresholds['积极买入']:
        decision = '谨慎买入/低吸'
    elif dynamic_thresholds['谨慎卖出'] < total_score < dynamic_thresholds['谨慎买入']:
        decision = '观望/震荡'
    elif dynamic_thresholds['强力卖出'] < total_score <= dynamic_thresholds['谨慎卖出']:
        decision = '谨慎卖出/减仓'
    elif total_score <= dynamic_thresholds['强力卖出']:
        decision = '强力卖出/空仓'
    else:
        decision = '观望/震荡'

    # 整理详细得分（新增开关状态和动态阈值，便于排查）
    details = {
        'MACD得分': indicator_scores['MACD'],
        'RSI得分': indicator_scores['RSI'],
        'KDJ得分': indicator_scores['KDJ'],
        'BIAS得分': indicator_scores['BIAS'],
        'BOLL得分': indicator_scores['BOLL'],
        '均线得分': indicator_scores['均线'],
        '成交量得分': indicator_scores['成交量'],
        '换手率得分': indicator_scores['换手率'],
        '指标开关状态': indicator_switch,
        '动态决策阈值': dynamic_thresholds,  # 新增：记录动态阈值
        '总得分': round(total_score, 2),
        '决策': decision
    }

    total_score = round(total_score, 2)
    current_date = strat.data.datetime.date(0) if len(strat.data) > 0 else "未知日期"
    # print(f"{current_date} total_score: {details}")
    return total_score, decision, details

# ========== 初始化全局阈值（程序启动时执行） ==========
update_global_decision_thresholds(DEFAULT_INDICATOR_SWITCH)