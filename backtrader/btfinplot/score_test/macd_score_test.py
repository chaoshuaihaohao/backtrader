import unittest
import numpy as np
from scipy.signal import find_peaks
from datetime import datetime, timedelta

# ========== 全局函数定义 ==========

# 第一步：修复规则表优先级（背离规则移到最顶部）
MACD_SCORE_TABLE = [
    # 通用强信号规则（修复放量死叉越界问题）
    {
        "description": "放量死叉",
        "code_condition": lambda strat: (
                len(strat.mcross) > 0 and strat.mcross[0] < 0 and
                len(strat.vol) >= 6 and strat.vol[0] > strat.vol5[0]
        ),
        "买入评分": 0.0,
        "卖出评分": 10.0
    },
    # 零轴上方场景（优先级：强→弱）
    {
        "description": "零轴上方|金叉或即将金叉（绿柱缩短且 DEA-DIFF < 动态阈值）",
        "code_condition": lambda strat: (
                is_about_to_gold_cross(strat) or is_just_gold_cross(strat)
        ),
        "买入评分": 2.0,
        "卖出评分": 0.0
    },
    {
        "description": "零轴上方|红柱缩减",
        "code_condition": lambda strat: (
                strat.macd.macd[0] > 0 and
                strat.macd.histo[-2] > strat.macd.histo[-1] > strat.macd.histo[0] > 0
        ),
        "买入评分": 1.0,
        "卖出评分": 1.2
    },
    {
        "description": "零轴上方|金叉多日",
        "code_condition": lambda strat: (
            strat.macd.macd[0] > 0 and
            (strat.macd_hist[-2] > 0 and strat.macd_hist[-1] > 0 and strat.macd_hist[0] > 0)
        ),
        "买入评分": 2.0,
        "卖出评分": 0.0
    },
    {
        "description": "零轴下方|红柱缩减",
        "code_condition": lambda strat: (
            strat.macd.macd[0] < 0 and
            strat.macd.histo[-2] > strat.macd.histo[-1] > strat.macd.histo[0] > 0
        ),
        "买入评分": 0.8,
        "卖出评分": 1.2
    },
    {
        "description": "绿柱放大",
        "code_condition": lambda strat: (
            0 > strat.macd.histo[-2] > strat.macd.histo[-1] > strat.macd.histo[0]
        ),
        "买入评分": 0.0,
        "卖出评分": 2.0
    },
    {
        "description": "绿柱缩减",
        "code_condition": lambda strat: (
                strat.macd.histo[-2] < strat.macd.histo[-1] < strat.macd.histo[0] < 0
        ),
        "买入评分": 1.5,
        "卖出评分": 0.0
    },
    # {
    #     "description": "零轴上方|金叉多日，红柱放大|无背离",
    #     "code_condition": lambda strat: (
    #             strat.macd.macd[0] > 0 and
    #             is_gold_cross_days_red_up(strat) and
    #             get_macd_divergence(strat) == "无背离"
    #     ),
    #     "买入评分": 1.0,
    #     "卖出评分": 1.0
    # },
    # {
    #     "description": "零轴上方|金叉多日，红柱缩减|无背离",
    #     "code_condition": lambda strat: (
    #             strat.macd.macd[0] > 0 and
    #             is_gold_cross_days_red_down(strat) and
    #             get_macd_divergence(strat) == "无背离"
    #     ),
    #     "买入评分": 1.0,
    #     "卖出评分": 1.0
    # },
    # {
    #     "description": "零轴上方|即将死叉（红柱缩短且 DIFF-DEA < 动态阈值）|无背离",
    #     "code_condition": lambda strat: (
    #             strat.macd.macd[0] > 0 and
    #             is_about_to_death_cross(strat) and
    #             get_macd_divergence(strat) == "无背离"
    #     ),
    #     "买入评分": 0.3,
    #     "卖出评分": 1.2
    # },
    # {
    #     "description": "零轴上方|刚死叉|无背离",
    #     "code_condition": lambda strat: (
    #             strat.macd.macd[0] > 0 and
    #             is_just_death_cross(strat) and
    #             get_macd_divergence(strat) == "无背离"
    #     ),
    #     "买入评分": 0.3,
    #     "卖出评分": 1.5
    # },
    # {
    #     "description": "零轴上方|死叉多日，绿柱缩减|无背离",
    #     "code_condition": lambda strat: (
    #             strat.macd.macd[0] > 0 and
    #             is_death_cross_days_green_down(strat) and
    #             get_macd_divergence(strat) == "无背离"
    #     ),
    #     "买入评分": 0.8,
    #     "卖出评分": 0.6
    # },
    # {
    #     "description": "零轴上方|死叉多日，绿柱放大|无背离",
    #     "code_condition": lambda strat: (
    #             strat.macd.macd[0] > 0 and
    #             is_death_cross_days_green_up(strat) and
    #             get_macd_divergence(strat) == "无背离"
    #     ),
    #     "买入评分": 0.0,
    #     "卖出评分": 2.0
    # },
    # # 零轴下方场景（优先级：强→弱）
    # {
    #     "description": "零轴下方|即将金叉（绿柱缩短且 DEA-DIFF < 动态阈值）|无背离",
    #     "code_condition": lambda strat: (
    #             strat.macd.macd[0] < 0 and
    #             is_about_to_gold_cross(strat) and
    #             get_macd_divergence(strat) == "无背离"
    #     ),
    #     "买入评分": 1.2,
    #     "卖出评分": 0.2
    # },
    # {
    #     "description": "零轴下方|刚金叉|无背离",
    #     "code_condition": lambda strat: (
    #             strat.macd.macd[0] < 0 and
    #             is_just_gold_cross(strat) and
    #             get_macd_divergence(strat) == "无背离"
    #     ),
    #     "买入评分": 1.5,
    #     "卖出评分": 0.3
    # },
    # {
    #     "description": "零轴下方|金叉多日，红柱放大|无背离",
    #     "code_condition": lambda strat: (
    #             strat.macd.macd[0] < 0 and
    #             is_gold_cross_days_red_up(strat) and
    #             get_macd_divergence(strat) == "无背离"
    #     ),
    #     "买入评分": 1.2,
    #     "卖出评分": 0.2
    # },
    # {
    #     "description": "零轴下方|金叉多日，红柱缩减|无背离",
    #     "code_condition": lambda strat: (
    #             strat.macd.macd[0] < 0 and
    #             is_gold_cross_days_red_down(strat) and
    #             get_macd_divergence(strat) == "无背离"
    #     ),
    #     "买入评分": 0.5,
    #     "卖出评分": 0.4
    # },
    # {
    #     "description": "零轴下方|即将死叉（红柱缩短且 DIFF-DEA < 动态阈值）|无背离",
    #     "code_condition": lambda strat: (
    #             strat.macd.macd[0] < 0 and
    #             is_about_to_death_cross(strat) and
    #             get_macd_divergence(strat) == "无背离"
    #     ),
    #     "买入评分": 0.3,
    #     "卖出评分": 0.8
    # },
    # {
    #     "description": "零轴下方|刚死叉|无背离",
    #     "code_condition": lambda strat: (
    #             strat.macd.macd[0] < 0 and
    #             is_just_death_cross(strat) and
    #             get_macd_divergence(strat) == "无背离"
    #     ),
    #     "买入评分": 0.3,
    #     "卖出评分": 1.2
    # },
    # {
    #     "description": "零轴下方|死叉多日，绿柱缩减|无背离",
    #     "code_condition": lambda strat: (
    #             strat.macd.macd[0] < 0 and
    #             is_death_cross_days_green_down(strat) and
    #             get_macd_divergence(strat) == "无背离"
    #     ),
    #     "买入评分": 0.8,
    #     "卖出评分": 0.4
    # },
    # {
    #     "description": "零轴下方|死叉多日，绿柱放大|无背离",
    #     "code_condition": lambda strat: (
    #             strat.macd.macd[0] < 0 and
    #             is_death_cross_days_green_up(strat) and
    #             get_macd_divergence(strat) == "无背离"
    #     ),
    #     "买入评分": 0.2,
    #     "卖出评分": 1.5
    # }
]

# 兜底评分表（匹配不到精准规则时使用）
FALLBACK_SCORE_TABLE = {
    ("零轴上方", "顶背离"): {"buy": 0.3, "sell": 0.7},
    ("零轴上方", "无背离"): {"buy": 0.6, "sell": 0.4},
    ("零轴上方", "底背离"): {"buy": 0.8, "sell": 0.2},
    ("零轴下方", "顶背离"): {"buy": 0.1, "sell": 0.9},
    ("零轴下方", "无背离"): {"buy": 0.4, "sell": 0.6},
    ("零轴下方", "底背离"): {"buy": 0.7, "sell": 0.3},
}


def get_macd_status(strat, max_lookback=60):
    """
    获取MACD状态的完整描述（精简版）

    参数:
        strat: 策略实例 (包含 macd, macd_gold_cross 等指标)
        max_lookback: 最大回溯天数，防止数据过长

    返回:
        dict: 包含零轴位置、交叉状态、距离天数等信息
    """
    result = {
        'zero_position': '未知',  # 零轴上方 / 零轴下方
        'cross_status': '未知',  # 金叉 / 死叉 / 金叉中上升 / 金叉中下降 / 死叉中上升 / 死叉中下降
        'last_gold_day': -1,  # 距离最近一次金叉的天数 (0表示今天，-1表示未找到)
        'last_dead_day': -1,  # 距离最近一次死叉的天数 (0表示今天，-1表示未找到)
    }

    # 1. 检查数据长度
    if len(strat.macd.macd) < 1:
        return result

    # ========== 2. 判断零轴位置 ==========
    result['zero_position'] = "零轴上方" if strat.macd.macd[0] > 0 else "零轴下方"

    # ========== 3. 回溯查找最近的金叉和死叉 ==========
    last_gold_idx = -1
    last_dead_idx = -1

    data_len = min(len(strat.mcross), max_lookback)

    for i in range(data_len):
        try:
            cross_val = strat.mcross[0] if i == 0 else strat.mcross[-i]
        except:
            break

        if cross_val > 0 and last_gold_idx == -1:
            last_gold_idx = i
        elif cross_val < 0 and last_dead_idx == -1:
            last_dead_idx = i

        if last_gold_idx != -1 and last_dead_idx != -1:
            break

    result['last_gold_day'] = last_gold_idx
    result['last_dead_day'] = last_dead_idx

    # ========== 4. 判断状态 ==========
    # 情况A：今天是金叉或死叉
    if last_gold_idx == 0:
        result['cross_status'] = '金叉'
    elif last_dead_idx == 0:
        result['cross_status'] = '死叉'
    else:
        # 情况B：非交叉日，判断趋势
        # 比较谁更近：金叉更近 还是 死叉更近？
        is_gold_nearer = (last_gold_idx != -1) and (last_dead_idx == -1 or last_gold_idx < last_dead_idx)
        is_dead_nearer = (last_dead_idx != -1) and (last_gold_idx == -1 or last_dead_idx < last_gold_idx)

        has_hist_data = len(strat.macd_hist) >= 2
        hist_now = strat.macd_hist[0] if has_hist_data else 0
        hist_prev = strat.macd_hist[-1] if has_hist_data else 0

        if is_gold_nearer:
            # 最近一次是金叉
            if has_hist_data:
                result['cross_status'] = '金叉中上升' if hist_now > hist_prev else '金叉中下降'
            else:
                result['cross_status'] = '金叉中'
        elif is_dead_nearer:
            # 最近一次是死叉
            if has_hist_data:
                result['cross_status'] = '死叉中上升' if hist_now > hist_prev else '死叉中下降'
            else:
                result['cross_status'] = '死叉中'
        else:
            result['cross_status'] = '震荡/无明确交叉'

    return result


def get_week_macd_status(strat, max_lookback=12):
    """
    获取MACD状态的完整描述（精简版）

    参数:
        strat: 策略实例 (包含 macd, macd_gold_cross 等指标)
        max_lookback: 最大回溯天数，防止数据过长

    返回:
        dict: 包含零轴位置、交叉状态、距离天数等信息
    """
    result = {
        'zero_position': '未知',  # 零轴上方 / 零轴下方
        'cross_status': '未知',  # 金叉 / 死叉 / 金叉中上升 / 金叉中下降 / 死叉中上升 / 死叉中下降
        'last_gold_day': -1,  # 距离最近一次金叉的天数 (0表示今天，-1表示未找到)
        'last_dead_day': -1,  # 距离最近一次死叉的天数 (0表示今天，-1表示未找到)
    }

    # 1. 检查数据长度
    if len(strat.week_macd.macd) < 1:
        return result

    # ========== 2. 判断零轴位置 ==========
    result['zero_position'] = "零轴上方" if strat.week_macd.macd[0] > 0 else "零轴下方"

    # ========== 3. 回溯查找最近的金叉和死叉 ==========
    last_gold_idx = -1
    last_dead_idx = -1

    data_len = min(len(strat.week_mcross), max_lookback)

    for i in range(data_len):
        try:
            cross_val = strat.week_mcross[0] if i == 0 else strat.week_mcross[-i]
        except:
            break

        if cross_val > 0 and last_gold_idx == -1:
            last_gold_idx = i
        elif cross_val < 0 and last_dead_idx == -1:
            last_dead_idx = i

        if last_gold_idx != -1 and last_dead_idx != -1:
            break

    result['last_gold_day'] = last_gold_idx
    result['last_dead_day'] = last_dead_idx

    # ========== 4. 判断状态 ==========
    # 情况A：今天是金叉或死叉
    if last_gold_idx == 0:
        result['cross_status'] = '金叉'
    elif last_dead_idx == 0:
        result['cross_status'] = '死叉'
    else:
        # 情况B：非交叉日，判断趋势
        # 比较谁更近：金叉更近 还是 死叉更近？
        is_gold_nearer = (last_gold_idx != -1) and (last_dead_idx == -1 or last_gold_idx < last_dead_idx)
        is_dead_nearer = (last_dead_idx != -1) and (last_gold_idx == -1 or last_dead_idx < last_gold_idx)

        has_hist_data = len(strat.week_macd_hist) >= 2
        hist_now = strat.week_macd_hist[0] if has_hist_data else 0
        hist_prev = strat.week_macd_hist[-1] if has_hist_data else 0

        if is_gold_nearer:
            # 最近一次是金叉
            if has_hist_data:
                result['cross_status'] = '金叉中上升' if hist_now > hist_prev else '金叉中下降'
            else:
                result['cross_status'] = '金叉中'
        elif is_dead_nearer:
            # 最近一次是死叉
            if has_hist_data:
                result['cross_status'] = '死叉中上升' if hist_now > hist_prev else '死叉中下降'
            else:
                result['cross_status'] = '死叉中'
        else:
            result['cross_status'] = '震荡/无明确交叉'

    return result



# 辅助函数：获取动态阈值（简化逻辑，适配测试）
def get_dynamic_threshold(strat):
    """
    计算动态临界阈值（适配标的自身波动特性）：
    核心公式：临界阈值 = max( min( 近20日MACD波动幅度 × 10% , 3 ), 0.1 )
    """
    try:
        # 获取近20日MACD（DIF）数据
        macd_20d = strat.macd.macd.get(size=20)
        if len(macd_20d) < 20:
            return 0.2  # 数据不足时返回默认值

        # 计算波动幅度
        macd_max = max(macd_20d)
        macd_min = min(macd_20d)
        volatility = macd_max - macd_min

        # 按公式计算动态阈值
        threshold = volatility * 0.1  # 波动幅度 × 10%
        threshold = min(threshold, 3)  # 上限兜底：最大3
        threshold = max(threshold, 0.1)  # 下限兜底：最小0.1

        return threshold
    except Exception as e:
        print(f"计算动态阈值异常: {e}")
        return 0.2  # 异常时返回默认值


# 辅助函数：判断即将金叉（简化条件，适配测试）
def is_about_to_gold_cross(strat):
    """判断是否即将金叉（绿柱缩短且 DEA-DIFF < 动态阈值）"""
    try:
        hist_list = list(strat.macd_hist.get(size=5)) if len(strat.macd_hist) >= 5 else []
        macd_list = list(strat.macd.macd.get(size=5)) if len(strat.macd.macd) >= 5 else []
        signal_list = list(strat.macd.signal.get(size=5)) if len(strat.macd.signal) >= 5 else []

        if len(hist_list) < 3:
            return False

        # 修正索引：Backtrader Line对象 [0]是最新值，[-1]是次新
        hist_now = hist_list[0]  # 当日
        hist_prev = hist_list[1]  # 前1日
        macd_now = macd_list[0]  # 当日DIF
        signal_now = signal_list[0]  # 当日DEA

        # 简化条件：绿柱 + 绿柱缩短 + DEA-DIFF < 动态阈值
        dynamic_threshold = get_dynamic_threshold(strat)
        is_shortening = hist_now > hist_prev  # 当日绿柱比前一日短（绝对值变小）
        gap_small = (signal_now - macd_now) < dynamic_threshold

        return hist_now < 0 and is_shortening and gap_small
    except Exception as e:
        print(f"判断即将金叉异常: {e}")
        return False


# 辅助函数：判断刚金叉
def is_just_gold_cross(strat):
    """判断是否刚金叉（当日MACD上穿DEA）"""
    try:
        if hasattr(strat, 'mcross') and len(strat.mcross) > 0:
            latest_mcross = strat.mcross[0]
            return latest_mcross == 1
        return False
    except Exception as e:
        print(f"判断刚金叉异常: {e}")
        return False


# 辅助函数：修复金叉多日红柱放大逻辑
def is_gold_cross_days_red_up(strat):
    """判断是否金叉多日，红柱放大（红柱且最新值>前值）"""
    try:
        if len(strat.macd_hist) < 3:
            return False

        # 红柱判断：最新3期都是红柱
        hist_0 = strat.macd_hist[0]  # 当日
        hist_1 = strat.macd_hist[-1]  # 前1日
        hist_2 = strat.macd_hist[-2]  # 前2日

        # 金叉多日+红柱放大
        is_red = hist_0 > 0 and hist_1 > 0 and hist_2 > 0
        is_up = hist_0 > hist_1 > hist_2  # 红柱持续放大

        # 确认金叉状态（mcross有1且持续多日）
        is_gold_cross = len(strat.mcross) > 3 and any([x == 1 for x in strat.mcross.get(size=3)])

        return is_red and is_up and is_gold_cross
    except Exception as e:
        print(f"判断金叉多日红柱放大异常: {e}")
        return False


# 辅助函数：修复金叉多日红柱缩减逻辑
def is_gold_cross_days_red_down(strat):
    """判断是否金叉多日，红柱缩减（红柱且最新值<前值）"""
    try:
        if len(strat.macd_hist) < 3:
            return False

        # 红柱判断：最新3期都是红柱
        hist_0 = strat.macd_hist[0]  # 当日
        hist_1 = strat.macd_hist[-1]  # 前1日
        hist_2 = strat.macd_hist[-2]  # 前2日

        # 金叉多日+红柱缩减
        is_red = hist_0 > 0 and hist_1 > 0 and hist_2 > 0
        is_down = hist_0 < hist_1 < hist_2  # 红柱持续缩减

        # 确认金叉状态
        is_gold_cross = len(strat.mcross) > 3 and any([x == 1 for x in strat.mcross.get(size=3)])

        return is_red and is_down and is_gold_cross
    except Exception as e:
        print(f"判断金叉多日红柱缩减异常: {e}")
        return False


# 辅助函数：判断即将死叉（简化条件）
def is_about_to_death_cross(strat):
    """判断是否即将死叉（红柱缩短且 DIFF-DEA < 动态阈值）"""
    try:
        hist_list = list(strat.macd_hist.get(size=5)) if len(strat.macd_hist) >= 5 else []
        macd_list = list(strat.macd.macd.get(size=5)) if len(strat.macd.macd) >= 5 else []
        signal_list = list(strat.macd.signal.get(size=5)) if len(strat.macd.signal) >= 5 else []

        if len(hist_list) < 3:
            return False

        # 修正索引：Backtrader Line对象 [0]是最新值
        hist_now = hist_list[0]  # 当日
        hist_prev = hist_list[1]  # 前1日
        macd_now = macd_list[0]  # 当日DIF
        signal_now = signal_list[0]  # 当日DEA

        # 简化条件：红柱 + 红柱缩短 + DIFF-DEA < 动态阈值
        dynamic_threshold = get_dynamic_threshold(strat)
        is_shortening = hist_now < hist_prev  # 当日红柱比前一日短
        gap_small = (macd_now - signal_now) < dynamic_threshold

        return hist_now > 0 and is_shortening and gap_small
    except Exception as e:
        print(f"判断即将死叉异常: {e}")
        return False


# 辅助函数：判断刚死叉
def is_just_death_cross(strat):
    """判断是否刚死叉（当日MACD下穿DEA）"""
    try:
        if hasattr(strat, 'mcross') and len(strat.mcross) > 0:
            latest_mcross = strat.mcross[0]
            return latest_mcross < 0
        return False
    except Exception as e:
        print(f"判断刚死叉异常: {e}")
        return False


# 辅助函数：修复死叉多日绿柱放大逻辑
def is_death_cross_days_green_up(strat):
    """判断是否死叉多日，绿柱放大（绿柱且绝对值变大）"""
    try:
        if len(strat.macd_hist) < 3:
            return False

        # 绿柱判断：最新3期都是绿柱
        hist_0 = strat.macd_hist[0]  # 当日
        hist_1 = strat.macd_hist[-1]  # 前1日
        hist_2 = strat.macd_hist[-2]  # 前2日

        # 死叉多日+绿柱放大（绝对值变大）
        is_green = hist_0 < 0 and hist_1 < 0 and hist_2 < 0
        is_up = hist_0 < hist_1 < hist_2  # 绿柱持续放大

        return is_green and is_up
    except Exception as e:
        print(f"判断死叉多日绿柱放大异常: {e}")
        return False


# 辅助函数：修复死叉多日绿柱缩减逻辑
def is_death_cross_days_green_down(strat):
    """判断是否死叉多日，绿柱缩减（绿柱且绝对值变小）"""
    try:
        if len(strat.macd_hist) < 3:
            return False

        # 绿柱判断：最新3期都是绿柱
        hist_0 = strat.macd_hist[0]  # 当日
        hist_1 = strat.macd_hist[-1]  # 前1日
        hist_2 = strat.macd_hist[-2]  # 前2日

        # 死叉多日+绿柱缩减（绝对值变小）
        is_green = hist_0 < 0 and hist_1 < 0 and hist_2 < 0
        is_down = hist_0 > hist_1 > hist_2  # 绿柱持续缩减

        # 确认死叉状态
        is_death_cross = len(strat.mcross) > 3 and any([x == -1 for x in strat.mcross.get(size=3)])

        return is_green and is_down and is_death_cross
    except Exception as e:
        print(f"判断死叉多日绿柱缩减异常: {e}")
        return False

def get_macd_divergence(
        strat,
        month_window=20,  # 一个月交易日数
        price_threshold=0.005,  # 价格最小波动阈值（0.5%）
        macd_threshold=0.01,  # MACD最小波动阈值（1%）
        peak_valley_window=3,  # 波峰波谷识别窗口（3日极值）
        min_bar_interval=8  # 新增：两个极值的最小间隔（低于这个直接无视）
):
    """
    Backtrader策略专用MACD背离判断函数（极简版：直接用价格峰谷对应DIFF值）
    :param strat: Backtrader策略实例（self）
    :param month_window: 计算背离的时间窗口（交易日数），默认20
    :param price_threshold: 价格最小波动阈值（相对值），默认0.005（0.5%）
    :param macd_threshold: MACD最小波动阈值（相对值），默认0.01（1%）
    :param peak_valley_window: 波峰波谷识别窗口（3日极值）
    :param min_bar_interval: 新增：两个价格极值的最小间隔（默认8根K线，低于则无视）
    :return: "顶背离"/"底背离"/"无背离"
    """

    # ===================== 1. 安全提取Backtrader数据 =====================
    def _safe_extract_line_data(line_obj, size):
        """安全提取Line对象数据，返回时间正序列表 [最早, ..., 最新]"""
        try:
            if len(line_obj) < size:
                return []
            # Backtrader的get(size=N)返回最近N个值，按时间正序排列
            return list(line_obj.get(size=size))
        except Exception as e:
            print(f"提取数据异常: {e}")
            return []

    # 提取指定窗口数据：顶背离用high，底背离用low，MACD用DIF（核心）
    high_data = _safe_extract_line_data(strat.data.high, month_window)   # 最高价（顶背离用）
    low_data = _safe_extract_line_data(strat.data.low, month_window)     # 最低价（底背离用）
    macd_diff_data = _safe_extract_line_data(strat.macd.macd, month_window)  # MACD-DIFF（直接用，不找峰谷）

    # 数据不足直接返回无背离
    if len(high_data) < month_window or len(low_data) < month_window or len(macd_diff_data) < month_window:
        return "无背离"

    # ===================== 2. 仅识别价格的波峰波谷（MACD直接用对应值） =====================
    def _get_price_extremes(values, is_peak=True):
        """
        仅识别价格的波峰/波谷（MACD不找峰谷），不足2个则补充边界值兜底
        :param values: 价格序列 [最早, ..., 最新]
        :param is_peak: True=波峰（顶背离），False=波谷（底背离）
        :return: 价格极值列表 [(索引, 数值), ...]（按时间排序，取最后2个）
        """
        if is_peak:
            # 找价格波峰（顶背离用high）
            peak_idxs, _ = find_peaks(values, distance=peak_valley_window)
            extremes = [(idx, values[idx]) for idx in peak_idxs]
        else:
            # 找价格波谷（底背离用low）
            valley_idxs, _ = find_peaks(-np.array(values), distance=peak_valley_window)
            extremes = [(idx, values[idx]) for idx in valley_idxs]

        # 边界值兜底：不足2个极值时，补充起始点和终点（避免无数据可对比）
        if len(extremes) < 2:
            extremes.insert(0, (0, values[0]))  # 补充最早点
            if len(extremes) < 2:
                extremes.append((len(values) - 1, values[-1]))  # 补充最新点

        # 按时间排序，只取最后2个极值（最新的两个峰/谷）
        extremes.sort(key=lambda x: x[0])
        return extremes[-2:] if len(extremes) >= 2 else []

    # ===================== 3. 顶背离判断（核心：价格峰 → 对应DIFF值） =====================
    def _is_top_divergence():
        """顶背离：价格后峰>前峰，但对应DIFF后值<=前值"""
        # 1. 取价格的最后两个波峰
        price_peaks = _get_price_extremes(high_data, is_peak=True)
        if len(price_peaks) < 2:
            return False
        prev_peak_idx, prev_peak_price = price_peaks[0]  # 前一个价格峰（时间早）
        curr_peak_idx, curr_peak_price = price_peaks[1]  # 当前价格峰（时间晚）

        # 新增：无视短间隔 → 间隔小于min_bar_interval直接返回False
        if curr_peak_idx - prev_peak_idx < min_bar_interval:
            return False

        # 2. 直接取对应时间点的DIFF值（核心修改：不用找MACD峰谷）
        prev_diff = macd_diff_data[prev_peak_idx]
        curr_diff = macd_diff_data[curr_peak_idx]

        # 3. 阈值判断（相对值，避免小波动误判）
        # 价格条件：后峰 ≥ 前峰 × (1 + 阈值) → 价格真的创新高
        price_cond = curr_peak_price >= prev_peak_price * (1 + price_threshold)
        # MACD条件：后值 ≤ 前值 × (1 - 阈值) → DIFF未创新高（甚至下跌）
        macd_cond = curr_diff <= prev_diff * (1 - macd_threshold)

        return price_cond and macd_cond

    # ===================== 4. 底背离判断（核心：价格谷 → 对应DIFF值） =====================
    def _is_bottom_divergence():
        """底背离：价格后谷<前谷，但对应DIFF后值>=前值"""
        # 1. 取价格的最后两个波谷
        price_valleys = _get_price_extremes(low_data, is_peak=False)
        if len(price_valleys) < 2:
            return False
        prev_valley_idx, prev_valley_price = price_valleys[0]  # 前一个价格谷
        curr_valley_idx, curr_valley_price = price_valleys[1]  # 当前价格谷

        # 新增：无视短间隔 → 间隔小于min_bar_interval直接返回False
        if curr_valley_idx - prev_valley_idx < min_bar_interval:
            return False

        # 2. 直接取对应时间点的DIFF值（核心修改：不用找MACD谷值）
        prev_diff = macd_diff_data[prev_valley_idx]
        curr_diff = macd_diff_data[curr_valley_idx]

        # 3. 阈值判断（相对值）
        # 价格条件：后谷 ≤ 前谷 × (1 - 阈值) → 价格真的创新低
        price_cond = curr_valley_price <= prev_valley_price * (1 - price_threshold)
        # MACD条件：后值 ≥ 前值 × (1 + 阈值) → DIFF未创新低（甚至上涨）
        macd_cond = curr_diff >= prev_diff * (1 + macd_threshold)

        return price_cond and macd_cond

    # ===================== 5. 最终判定 =====================
    if _is_top_divergence():
        return "顶背离"
    elif _is_bottom_divergence():
        return "底背离"
    else:
        return "无背离"


# 核心评分计算函数（修复所有错误+补充兜底分）
def calculate_macd_score(strat):
    # 数据不足时返回兜底分（中性）
    if len(strat.macd.macd) < 20 or len(strat.macd_hist) < 20 or len(strat.data.close) < 20:
        return 0.5 - 0.5  # 中性分（0）

    try:
        # 提前获取日期（修复作用域问题）
        current_date = "未知日期"
        try:
            current_date = strat.data.datetime.date(0).strftime('%Y-%m-%d')
        except Exception:
            current_date = "2025-01-01"  # 模拟数据兜底

        # 遍历规则表，找到匹配的规则
        buy_score = 0.0
        sell_score = 0.0
        matched_rule = None

        for rule in MACD_SCORE_TABLE:
            try:
                # 执行lambda条件判断
                if rule["code_condition"](strat):
                    buy_score = rule["买入评分"]
                    sell_score = rule["卖出评分"]
                    matched_rule = rule["description"]
                    break  # 匹配到第一条规则后退出（优先级最高）
            except Exception as e:
                # 单个规则判断失败，继续下一个
                print(f"规则 {rule['description']} 判断失败: {e}")
                continue

        # print(f"{strat.macd_hist[-2]} {strat.macd_hist[-1]} {strat.macd_hist[0]}")
        # 打印匹配结果（可选）
        if matched_rule:
            divergence_type = get_macd_divergence(strat)
            # print(f"{current_date} 匹配规则：{matched_rule} {divergence_type}| 买入得分: {buy_score}, 卖出得分: {sell_score}")
        else:
            # 匹配不到精准规则，按零轴+背离给兜底分
            divergence_type = get_macd_divergence(strat)
            macd_pos = "零轴上方" if strat.macd.macd[0] > 0 else "零轴下方"
            fallback_score = FALLBACK_SCORE_TABLE.get((macd_pos, divergence_type), {"buy": 0.5, "sell": 0.5})
            buy_score = fallback_score["buy"]
            sell_score = fallback_score["sell"]
            # print(f"{current_date} 未匹配到精准规则，使用兜底分 | {macd_pos} {divergence_type} | 买入得分: {buy_score}, 卖出得分: {sell_score}")

        # 返回最终净得分
        net_score = buy_score - sell_score
        return net_score
    except Exception as e:
        print(f"评分计算异常: {e}")
        import traceback
        traceback.print_exc()
        return 0.0


# ========== 模拟backtrader的Line对象 ==========
class BacktraderLine:
    """完全复刻backtrader的Line对象行为"""

    def __init__(self, values):
        self._history = list(values)  # 时间正序：[最早, ..., 最新]

    def __len__(self):
        return len(self._history)

    def __getitem__(self, idx):
        # 核心修复：统一处理 idx <= 0 的情况（Backtrader 风格）
        if idx <= 0:
            list_idx = (len(self._history) - 1) + idx
            if 0 <= list_idx < len(self._history):
                return self._history[list_idx]
            else:
                return None
        else:
            if idx < len(self._history):
                return self._history[idx]
            else:
                return None

    def get(self, size=None):
        if size is None:
            return self._history.copy()
        return self._history[-size:] if size <= len(self._history) else self._history.copy()


# ========== 模拟backtrader的DataFeed对象 ==========
class BacktraderDataFeed:
    def __init__(self, close_prices, high_prices=None, low_prices=None, vol_prices=None, dates=None):
        self.close = BacktraderLine(close_prices)
        self.high = BacktraderLine(high_prices if high_prices else close_prices)
        self.low = BacktraderLine(low_prices if low_prices else close_prices)
        self.vol = BacktraderLine(vol_prices if vol_prices else [1000] * len(close_prices))  # 成交量默认值

        # 补充datetime属性（模拟日期）
        if dates is None:
            base_date = datetime(2025, 1, 1)
            dates = [base_date + timedelta(days=i) for i in range(len(close_prices))]
        self.datetime = BacktraderLine([d.timestamp() for d in dates])

    def date(self, idx):
        """模拟Backtrader的datetime.date()方法"""
        ts = self.datetime[idx]
        if ts:
            return datetime.fromtimestamp(ts).date()
        return datetime(2025, 1, 1).date()


# ========== 模拟MACD指标对象 ==========
class MACDIndicator:
    def __init__(self, macd_values, signal_values):
        self.macd = BacktraderLine(macd_values)  # DIF
        self.signal = BacktraderLine(signal_values)  # DEA


# ========== 模拟策略对象 ==========
class MockStrategy:
    def __init__(self, macd_vals, signal_vals, hist_vals, close_prices, high_prices=None, low_prices=None,
                 mcross_vals=None, vol_vals=None, dates=None):
        self.data = BacktraderDataFeed(close_prices, high_prices, low_prices, vol_vals, dates)
        self.macd = MACDIndicator(macd_vals, signal_vals)
        self.macd_hist = BacktraderLine(hist_vals)  # MACD柱状线
        self.mcross = BacktraderLine(mcross_vals or [])  # 金叉死叉标记
        self.vol = self.data.vol  # 成交量引用


# ========== MACD辅助计算函数 ==========
def calculate_macd_hist(macd_vals, signal_vals):
    """自动计算MACD柱状线（Hist = DIF - DEA）"""
    return [round(m - s, 4) for m, s in zip(macd_vals, signal_vals)]


def calculate_mcross(macd_vals, signal_vals):
    """
    自动计算金叉标记（mcross）
    - mcross=1：当日MACD上穿SIGNAL（前一日MACD<signal，当日MACD>=signal）
    - mcross=-1：当日MACD下穿SIGNAL（前一日MACD>signal，当日MACD<=signal）
    - mcross=0：无交叉
    """
    mcross_vals = [0] * len(macd_vals)
    for i in range(1, len(macd_vals)):
        prev_macd = macd_vals[i - 1]
        curr_macd = macd_vals[i]
        prev_signal = signal_vals[i - 1]
        curr_signal = signal_vals[i]

        # 金叉判定
        if prev_macd < prev_signal and curr_macd >= curr_signal:
            mcross_vals[i] = 1
        # 死叉判定
        elif prev_macd > prev_signal and curr_macd <= curr_signal:
            mcross_vals[i] = -1
        else:
            mcross_vals[i] = 0
    return mcross_vals


# ========== 完整的测试用例 ==========
class TestMACDScoreCalculation(unittest.TestCase):
    def setUp(self):
        """通用初始化：创建基础数据模板"""
        self.base_20d = list(range(20))  # 20日基础索引
        # 生成模拟日期
        base_date = datetime(2025, 1, 1)
        self.dates = [base_date + timedelta(days=i) for i in range(20)]

    # ========== 基础验证用例 ==========
    def test_data_index_access(self):
        """验证模拟对象的索引行为"""
        close_prices = [10.0, 10.1, 10.2, 10.3, 10.4]
        data = BacktraderDataFeed(close_prices)
        self.assertEqual(data.close[0], 10.4)  # 当日
        self.assertEqual(data.close[-1], 10.3)  # 前1日
        self.assertEqual(data.close[-2], 10.2)  # 前2日
        self.assertEqual(data.close[-5], None)  # 越界

    # ========== 零轴上方场景 ==========
    def test_rule_zero_above_soon_gold_no_divergence(self):
        """规则：零轴上方+即将金叉+无背离 → 买入2.0分"""
        # 构造精准匹配的数据
        close_prices = [
            58.20, 58.90, 59.50, 60.10, 60.80,
            61.40, 62.00, 62.50, 62.90, 63.20,
            62.80, 62.30, 61.90, 61.50, 61.20,
            61.80, 62.40, 63.00, 63.50, 63.80
        ]
        high_prices = [x + 0.5 for x in close_prices]
        low_prices = [x - 0.5 for x in close_prices]

        # DIF：最后5期回升，接近DEA（零轴上方）
        macd_vals = [
            0.72, 0.75, 0.78, 0.81, 0.83,
            0.85, 0.84, 0.82, 0.80, 0.79,
            0.76, 0.73, 0.70, 0.68, 0.50,
            0.55, 0.60, 0.65, 0.70, 0.75
        ]
        # DEA：稳定在0.80
        signal_vals = [0.80] * 20
        # 计算柱状线（绿柱且缩短）
        hist_vals = calculate_macd_hist(macd_vals, signal_vals)
        # 金叉标记：无交叉
        mcross_vals = [0] * 20

        # 构造策略对象（传入high/low/dates）
        strat = MockStrategy(
            macd_vals, signal_vals, hist_vals,
            close_prices, high_prices, low_prices,
            mcross_vals=mcross_vals, dates=self.dates
        )
        # 计算得分
        score = calculate_macd_score(strat)
        self.assertAlmostEqual(score, 2.0, places=1)

    def test_rule_zero_above_just_death_top_divergence(self):
        """规则：零轴上方+刚死叉+顶背离 → 卖出3.0分（净得分-3.0）"""
        # 构造顶背离数据：价格新高，DIF未新高
        close_prices = [
            10.0, 10.5, 11.0, 11.5, 12.0,
            12.5, 13.0, 13.5, 14.0, 14.5,
            15.0, 14.8, 14.6, 14.4, 14.2,
            14.5, 14.8, 15.1, 15.4, 15.7  # 价格新高
        ]
        high_prices = [x + 0.3 for x in close_prices]  # 顶背离用high
        low_prices = [x - 0.3 for x in close_prices]

        macd_vals = [
            0.5, 0.6, 0.7, 0.8, 0.9,
            1.0, 1.1, 1.2, 1.3, 1.4,
            1.3, 1.2, 1.1, 1.0, 0.9,
            0.8, 0.7, 0.6, 0.5, 0.4  # DIF未新高
        ]
        signal_vals = [
            0.4, 0.5, 0.6, 0.7, 0.8,
            0.9, 1.0, 1.1, 1.2, 1.3,
            1.25, 1.2, 1.15, 1.1, 1.05,
            1.0, 0.95, 0.9, 0.85, 0.8  # DEA
        ]
        # 计算柱状线
        hist_vals = calculate_macd_hist(macd_vals, signal_vals)
        # 刚死叉：最后1期mcross=-1
        mcross_vals = [0] * 20
        mcross_vals[-1] = -1

        # 构造策略对象（传入high/low/dates）
        strat = MockStrategy(
            macd_vals, signal_vals, hist_vals,
            close_prices, high_prices, low_prices,
            mcross_vals=mcross_vals, dates=self.dates
        )
        # 计算得分
        score = calculate_macd_score(strat)
        self.assertAlmostEqual(score, -3.0, places=1)

    # ========== 零轴下方场景 ==========
    def test_rule_zero_below_just_gold_bottom_divergence(self):
        """规则：零轴下方+刚金叉+底背离 → 买入3.0分（净得分3.0）"""
        # 构造底背离数据：价格新低，DIF未新低
        close_prices = [
            20.0, 19.5, 19.0, 18.5, 18.0,
            17.5, 17.0, 16.5, 16.0, 15.5,
            15.0, 15.2, 15.4, 15.6, 15.8,
            15.5, 15.2, 14.9, 14.6, 14.3  # 价格新低
        ]
        high_prices = [x + 0.3 for x in close_prices]
        low_prices = [x - 0.3 for x in close_prices]  # 底背离用low

        macd_vals = [
            -1.0, -0.9, -0.8, -0.7, -0.6,
            -0.5, -0.4, -0.3, -0.2, -0.1,
            0.0, -0.1, -0.2, -0.3, -0.4,
            -0.3, -0.2, -0.1, 0.0, 0.1  # DIF未新低
        ]
        signal_vals = [
            -0.9, -0.8, -0.7, -0.6, -0.5,
            -0.4, -0.3, -0.2, -0.1, 0.0,
            -0.05, -0.1, -0.15, -0.2, -0.25,
            -0.2, -0.15, -0.1, -0.05, 0.0  # DEA
        ]
        # 计算柱状线
        hist_vals = calculate_macd_hist(macd_vals, signal_vals)
        # 刚金叉：最后1期mcross=1
        mcross_vals = [0] * 20
        mcross_vals[-1] = 1

        # 构造策略对象（传入high/low/dates）
        strat = MockStrategy(
            macd_vals, signal_vals, hist_vals,
            close_prices, high_prices, low_prices,
            mcross_vals=mcross_vals, dates=self.dates
        )
        # 计算得分
        score = calculate_macd_score(strat)
        self.assertAlmostEqual(score, 3.0, places=1)

    def test_rule_zero_below_death_days_green_up(self):
        """规则：零轴下方+死叉多日+绿柱放大 → 卖出1.5分（净得分-1.3）"""
        # 构造精准匹配的数据
        close_prices = [10.0 - i * 0.1 for i in range(20)]  # 持续下跌
        high_prices = [x + 0.2 for x in close_prices]
        low_prices = [x - 0.2 for x in close_prices]

        macd_vals = [-0.5 - i * 0.05 for i in range(20)]  # DIF持续下跌（零轴下方）
        signal_vals = [-0.4 - i * 0.04 for i in range(20)]  # DEA持续下跌
        hist_vals = calculate_macd_hist(macd_vals, signal_vals)  # 绿柱持续放大
        # 死叉标记：最后5期死叉
        mcross_vals = [0] * 15 + [-1] * 5

        # 构造策略对象（传入dates）
        strat = MockStrategy(
            macd_vals, signal_vals, hist_vals,
            close_prices, high_prices, low_prices,
            mcross_vals=mcross_vals, dates=self.dates
        )
        # 计算得分
        score = calculate_macd_score(strat)
        self.assertAlmostEqual(score, -1.3, places=1)

    def test_fallback_score(self):
        """测试兜底分：零轴下方+无背离 → 买入0.4，卖出0.6，净得分-0.2"""
        # 构造无精准规则匹配的数据
        close_prices = [10.0 + i * 0.01 for i in range(20)]  # 横盘
        high_prices = [x + 0.1 for x in close_prices]
        low_prices = [x - 0.1 for x in close_prices]

        macd_vals = [-0.2 - i * 0.01 for i in range(20)]  # 零轴下方横盘
        signal_vals = [-0.1 - i * 0.01 for i in range(20)]
        hist_vals = calculate_macd_hist(macd_vals, signal_vals)
        mcross_vals = [0] * 20  # 无金叉死叉

        strat = MockStrategy(
            macd_vals, signal_vals, hist_vals,
            close_prices, high_prices, low_prices,
            mcross_vals=mcross_vals, dates=self.dates
        )
        score = calculate_macd_score(strat)
        self.assertAlmostEqual(score, -0.2, places=1)


if __name__ == '__main__':
    unittest.main(verbosity=2)