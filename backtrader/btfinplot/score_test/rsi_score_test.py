import unittest
import math

# ===================== 第一步：定义核心配置（和手册完全对齐） =====================
# 1. 信号类型权重配置（核心：权重数值决定优先级）
RSI_SIGNAL_WEIGHTS = {
    '趋势反转': 2.0,
    '连续极值': 1.5,
    '动态金叉/死叉/拐头/企稳': 1.2,
    '静态排列': 1.0,
    '粘合平衡': 0.5
}
RSI_RULES = [
    # ---------------- 优先级1：趋势反转 (权重2.0) ----------------
    {
        "priority": 1,
        "signal_name": "突破金叉",
        "signal_type": "趋势反转",
        "condition_func": lambda d: (d['rsi6_0'] > d['rsi12_0']) and (d['rsi6_-1'] < d['rsi12_-1']) and
                                    (d['rsi6_-1'] < 50) and (d['rsi6_0'] > 50),
        "buy_score": 2.0,
        "sell_score": 0.0
    },
    {
        "priority": 1,
        "signal_name": "破位死叉",
        "signal_type": "趋势反转",
        "condition_func": lambda d: (d['rsi6_0'] < d['rsi12_0']) and (d['rsi6_-1'] > d['rsi12_-1']) and
                                    (d['rsi6_-1'] > 50) and (d['rsi6_0'] < 50),
        "buy_score": 0.0,
        "sell_score": 2.0
    },

    # ---------------- 优先级2：连续极值 (权重1.5) ----------------
    {
        "priority": 2,
        "signal_name": "连续超卖",
        "signal_type": "连续极值",
        "condition_func": lambda d: (d['rsi6_0'] < 20) and (d['rsi6_-1'] < 20) and (d['rsi6_-2'] < 20) and
                                    (d['rsi12_0'] < 30),
        "buy_score": 2.0,
        "sell_score": 0.0
    },
    {
        "priority": 2,
        "signal_name": "连续超买",
        "signal_type": "连续极值",
        "condition_func": lambda d: (d['rsi6_0'] > 80) and (d['rsi6_-1'] > 80) and (d['rsi6_-2'] > 80) and
                                    (d['rsi12_0'] > 70),
        "buy_score": 0.0,
        "sell_score": 2.0
    },
    {
        "priority": 2,
        "signal_name": "极端超卖",
        "signal_type": "连续极值",
        "condition_func": lambda d: (d['rsi6_0'] < 10) and (d['rsi6_-1'] < 10),
        "buy_score": 2.0,
        "sell_score": 0.0
    },
    {
        "priority": 2,
        "signal_name": "极端超买",
        "signal_type": "连续极值",
        "condition_func": lambda d: (d['rsi6_0'] > 90) and (d['rsi6_-1'] > 90),
        "buy_score": 0.0,
        "sell_score": 2.0
    },

    # ---------------- 优先级3：动态金叉/死叉/拐头/企稳/延续 (权重1.2) ----------------
    {
        "priority": 3,
        "signal_name": "超卖金叉",
        "signal_type": "动态金叉/死叉/拐头/企稳",
        "condition_func": lambda d: (d['rsi6_0'] > d['rsi12_0']) and (d['rsi6_-1'] < d['rsi12_-1']) and
                                    (d['rsi6_0'] < 30),
        "buy_score": 1.8,
        "sell_score": 0.0
    },
    {
        "priority": 3,
        "signal_name": "震荡金叉",
        "signal_type": "动态金叉/死叉/拐头/企稳",
        "condition_func": lambda d: (d['rsi6_0'] > d['rsi12_0']) and (d['rsi6_-1'] < d['rsi12_-1']) and
                                    (30 <= d['rsi6_0'] <= 50),
        "buy_score": 1.5,
        "sell_score": 0.0
    },
    {
        "priority": 3,
        "signal_name": "高位金叉",
        "signal_type": "动态金叉/死叉/拐头/企稳",
        "condition_func": lambda d: (d['rsi6_0'] > d['rsi12_0']) and (d['rsi6_-1'] < d['rsi12_-1']) and
                                    (d['rsi6_0'] > 70),
        "buy_score": 0.0,
        "sell_score": 1.0
    },
    {
        "priority": 3,
        "signal_name": "超买死叉",
        "signal_type": "动态金叉/死叉/拐头/企稳",
        "condition_func": lambda d: (d['rsi6_0'] < d['rsi12_0']) and (d['rsi6_-1'] > d['rsi12_-1']) and
                                    (d['rsi6_0'] > 70),
        "buy_score": 0.0,
        "sell_score": 1.8
    },
    {
        "priority": 3,
        "signal_name": "震荡死叉",
        "signal_type": "动态金叉/死叉/拐头/企稳",
        "condition_func": lambda d: (d['rsi6_0'] < d['rsi12_0']) and (d['rsi6_-1'] > d['rsi12_-1']) and
                                    (50 <= d['rsi6_0'] <= 70),
        "buy_score": 0.0,
        "sell_score": 1.5
    },
    {
        "priority": 3,
        "signal_name": "低位死叉",
        "signal_type": "动态金叉/死叉/拐头/企稳",
        "condition_func": lambda d: (d['rsi6_0'] < d['rsi12_0']) and (d['rsi6_-1'] > d['rsi12_-1']) and
                                    (d['rsi6_0'] < 30),
        "buy_score": 1.0,
        "sell_score": 0.0
    },
    {
        "priority": 3,
        "signal_name": "低位拐头",
        "signal_type": "动态金叉/死叉/拐头/企稳",
        "condition_func": lambda d: (d['rsi6_-1'] < 30) and (30 <= d['rsi6_0'] <= 50),
        "buy_score": 1.8,
        "sell_score": 0.0
    },
    {
        "priority": 3,
        "signal_name": "高位拐头",
        "signal_type": "动态金叉/死叉/拐头/企稳",
        "condition_func": lambda d: (d['rsi6_-1'] > 70) and (50 <= d['rsi6_0'] <= 70),
        "buy_score": 0.0,
        "sell_score": 1.8
    },
    {
        "priority": 3,
        "signal_name": "超卖企稳",
        "signal_type": "动态金叉/死叉/拐头/企稳",
        "condition_func": lambda d: (d['rsi6_-1'] < 30) and (d['rsi6_-2'] < 30) and (d['rsi6_0'] - d['rsi6_-1'] >= 5) and
                                    (d['rsi6_0'] < 30),  # 新增：确保不触发低位拐头
        "buy_score": 1.5,
        "sell_score": 0.0
    },
    {
        "priority": 3,
        "signal_name": "超买回落",
        "signal_type": "动态金叉/死叉/拐头/企稳",
        "condition_func": lambda d: (d['rsi6_-1'] > 70) and (d['rsi6_-2'] > 70) and (d['rsi6_-1'] - d['rsi6_0'] >= 5) and
                                    (d['rsi6_0'] > 70),  # 新增：确保不触发高位拐头
        "buy_score": 0.0,
        "sell_score": 1.5
    },
    {
        "priority": 3,
        "signal_name": "金叉延续2日",
        "signal_type": "动态金叉/死叉/拐头/企稳",
        "condition_func": lambda d: (d['rsi6_-2'] > d['rsi12_-2'] and d['rsi6_-3'] < d['rsi12_-3']) and
                                    (d['rsi6_-1'] > d['rsi12_-1']) and (d['rsi6_0'] > d['rsi12_0']) and
                                    (30 <= d['rsi6_0'] <= 50) and not (d['rsi6_-3'] > d['rsi12_-3'] and d['rsi6_-4'] < d['rsi12_-4']),  # 排除3日延续
        "buy_score": 1.5,
        "sell_score": 0.0
    },
    {
        "priority": 3,
        "signal_name": "金叉延续3日",
        "signal_type": "动态金叉/死叉/拐头/企稳",
        "condition_func": lambda d: (d['rsi6_-3'] > d['rsi12_-3'] and d['rsi6_-4'] < d['rsi12_-4']) and
                                    (d['rsi6_-2'] > d['rsi12_-2']) and (d['rsi6_-1'] > d['rsi12_-1']) and
                                    (d['rsi6_0'] > d['rsi12_0']) and (30 <= d['rsi6_0'] <= 50),
        "buy_score": 1.8,
        "sell_score": 0.0
    },
    {
        "priority": 3,
        "signal_name": "金叉后超买",
        "signal_type": "动态金叉/死叉/拐头/企稳",
        "condition_func": lambda d: ((d['rsi6_-2'] > d['rsi12_-2'] and d['rsi6_-3'] < d['rsi12_-3']) or
                                     (d['rsi6_-3'] > d['rsi12_-3'] and d['rsi6_-4'] < d['rsi12_-4'])) and
                                    (d['rsi6_-1'] > d['rsi12_-1']) and (d['rsi6_0'] > d['rsi12_0']) and
                                    (d['rsi6_0'] > 70),
        "buy_score": 0.0,
        "sell_score": 1.8
    },
    {
        "priority": 3,
        "signal_name": "死叉延续2日",
        "signal_type": "动态金叉/死叉/拐头/企稳",
        "condition_func": lambda d: (d['rsi6_-2'] < d['rsi12_-2'] and d['rsi6_-3'] > d['rsi12_-3']) and
                                    (d['rsi6_-1'] < d['rsi12_-1']) and (d['rsi6_0'] < d['rsi12_0']) and
                                    (50 <= d['rsi6_0'] <= 70) and not (d['rsi6_-3'] < d['rsi12_-3'] and d['rsi6_-4'] > d['rsi12_-4']),  # 排除3日延续
        "buy_score": 0.0,
        "sell_score": 1.5
    },
    {
        "priority": 3,
        "signal_name": "死叉延续3日",
        "signal_type": "动态金叉/死叉/拐头/企稳",
        "condition_func": lambda d: (d['rsi6_-3'] < d['rsi12_-3'] and d['rsi6_-4'] > d['rsi12_-4']) and
                                    (d['rsi6_-2'] < d['rsi12_-2']) and (d['rsi6_-1'] < d['rsi12_-1']) and
                                    (d['rsi6_0'] < d['rsi12_0']) and (50 <= d['rsi6_0'] <= 70),
        "buy_score": 0.0,
        "sell_score": 1.8
    },
    {
        "priority": 3,
        "signal_name": "死叉后超卖",
        "signal_type": "动态金叉/死叉/拐头/企稳",
        "condition_func": lambda d: ((d['rsi6_-2'] < d['rsi12_-2'] and d['rsi6_-3'] > d['rsi12_-3']) or
                                     (d['rsi6_-3'] < d['rsi12_-3'] and d['rsi6_-4'] > d['rsi12_-4'])) and
                                    (d['rsi6_-1'] < d['rsi12_-1']) and (d['rsi6_0'] < d['rsi12_0']) and
                                    (d['rsi6_0'] < 30) and not (d['rsi6_0'] < d['rsi12_0'] and d['rsi6_-1'] > d['rsi12_-1']),  # 排除低位死叉
        "buy_score": 1.0,
        "sell_score": 0.0
    },

    # ---------------- 优先级4：静态排列 (权重1.0) ----------------
    {
        "priority": 4,
        "signal_name": "最强多头(RSI6<20)",
        "signal_type": "静态排列",
        "condition_func": lambda d: (d['rsi6_0'] > d['rsi12_0']) and (d['rsi12_0'] > d['rsi24_0']) and
                                    (d['rsi6_0'] < 20),
        "buy_score": 2.0,
        "sell_score": 0.0
    },
    {
        "priority": 4,
        "signal_name": "最强多头(20<=RSI6<=50)",
        "signal_type": "静态排列",
        "condition_func": lambda d: (d['rsi6_0'] > d['rsi12_0']) and (d['rsi12_0'] > d['rsi24_0']) and
                                    (20 <= d['rsi6_0'] <= 50),
        "buy_score": 1.5,
        "sell_score": 0.0
    },
    {
        "priority": 4,
        "signal_name": "最强多头(50<=RSI6<=70)",
        "signal_type": "静态排列",
        "condition_func": lambda d: (d['rsi6_0'] > d['rsi12_0']) and (d['rsi12_0'] > d['rsi24_0']) and
                                    (50 <= d['rsi6_0'] <= 70),
        "buy_score": 1.0,
        "sell_score": 0.5
    },
    {
        "priority": 4,
        "signal_name": "最强多头(RSI6>70)",
        "signal_type": "静态排列",
        "condition_func": lambda d: (d['rsi6_0'] > d['rsi12_0']) and (d['rsi12_0'] > d['rsi24_0']) and
                                    (d['rsi6_0'] > 70),
        "buy_score": 0.0,
        "sell_score": 2.0
    },
    {
        "priority": 4,
        "signal_name": "短多长空(RSI6<30)",
        "signal_type": "静态排列",
        "condition_func": lambda d: (d['rsi6_0'] > d['rsi12_0']) and (d['rsi12_0'] < d['rsi24_0']) and
                                    (d['rsi6_0'] < 30),
        "buy_score": 1.5,
        "sell_score": 0.0
    },
    {
        "priority": 4,
        "signal_name": "短多长空(30<=RSI6<=70)",
        "signal_type": "静态排列",
        "condition_func": lambda d: (d['rsi6_0'] > d['rsi12_0']) and (d['rsi12_0'] < d['rsi24_0']) and
                                    (30 <= d['rsi6_0'] <= 70),
        "buy_score": 0.8,
        "sell_score": 0.8
    },
    {
        "priority": 4,
        "signal_name": "短多长空(RSI6>70)",
        "signal_type": "静态排列",
        "condition_func": lambda d: (d['rsi6_0'] > d['rsi12_0']) and (d['rsi12_0'] < d['rsi24_0']) and
                                    (d['rsi6_0'] > 70),
        "buy_score": 0.0,
        "sell_score": 1.8
    },
    {
        "priority": 4,
        "signal_name": "短空多长(RSI6<30)",
        "signal_type": "静态排列",
        "condition_func": lambda d: (d['rsi6_0'] < d['rsi12_0']) and (d['rsi12_0'] > d['rsi24_0']) and
                                    (d['rsi6_0'] < 30),
        "buy_score": 1.2,
        "sell_score": 0.0
    },
    {
        "priority": 4,
        "signal_name": "短空多长(30<=RSI6<=70)",
        "signal_type": "静态排列",
        "condition_func": lambda d: (d['rsi6_0'] < d['rsi12_0']) and (d['rsi12_0'] > d['rsi24_0']) and
                                    (30 <= d['rsi6_0'] <= 70),
        "buy_score": 0.5,
        "sell_score": 1.0
    },
    {
        "priority": 4,
        "signal_name": "短空多长(RSI6>70)",
        "signal_type": "静态排列",
        "condition_func": lambda d: (d['rsi6_0'] < d['rsi12_0']) and (d['rsi12_0'] > d['rsi24_0']) and
                                    (d['rsi6_0'] > 70),
        "buy_score": 0.0,
        "sell_score": 1.5
    },
    {
        "priority": 4,
        "signal_name": "最强空头(RSI6<20)",
        "signal_type": "静态排列",
        "condition_func": lambda d: (d['rsi6_0'] < d['rsi12_0']) and (d['rsi12_0'] < d['rsi24_0']) and
                                    (d['rsi6_0'] < 20),
        "buy_score": 1.5,
        "sell_score": 0.0
    },
    {
        "priority": 4,
        "signal_name": "最强空头(20<=RSI6<=50)",
        "signal_type": "静态排列",
        "condition_func": lambda d: (d['rsi6_0'] < d['rsi12_0']) and (d['rsi12_0'] < d['rsi24_0']) and
                                    (20 <= d['rsi6_0'] <= 50),
        "buy_score": 0.0,
        "sell_score": 1.5
    },
    {
        "priority": 4,
        "signal_name": "最强空头(50<=RSI6<=70)",
        "signal_type": "静态排列",
        "condition_func": lambda d: (d['rsi6_0'] < d['rsi12_0']) and (d['rsi12_0'] < d['rsi24_0']) and
                                    (50 <= d['rsi6_0'] <= 70),
        "buy_score": 0.0,
        "sell_score": 2.0
    },
    {
        "priority": 4,
        "signal_name": "最强空头(RSI6>70)",
        "signal_type": "静态排列",
        "condition_func": lambda d: (d['rsi6_0'] < d['rsi12_0']) and (d['rsi12_0'] < d['rsi24_0']) and
                                    (d['rsi6_0'] > 70),
        "buy_score": 0.0,
        "sell_score": 2.0
    },

    # ---------------- 优先级5：粘合平衡 (权重0.5) ----------------
    {
        "priority": 5,
        "signal_name": "完全粘合(RSI6<50)",
        "signal_type": "粘合平衡",
        "condition_func": lambda d: (abs(d['rsi6_0'] - d['rsi12_0']) < 2) and
                                    (abs(d['rsi12_0'] - d['rsi24_0']) < 2) and (d['rsi6_0'] < 50) and
                                    # 排除静态排列信号
                                    not ((d['rsi6_0'] > d['rsi12_0'] and d['rsi12_0'] > d['rsi24_0']) or
                                         (d['rsi6_0'] > d['rsi12_0'] and d['rsi12_0'] < d['rsi24_0']) or
                                         (d['rsi6_0'] < d['rsi12_0'] and d['rsi12_0'] > d['rsi24_0']) or
                                         (d['rsi6_0'] < d['rsi12_0'] and d['rsi12_0'] < d['rsi24_0'])),
        "buy_score": 0.5,
        "sell_score": 0.0
    },
    {
        "priority": 5,
        "signal_name": "完全粘合(RSI6>50)",
        "signal_type": "粘合平衡",
        "condition_func": lambda d: (abs(d['rsi6_0'] - d['rsi12_0']) < 2) and
                                    (abs(d['rsi12_0'] - d['rsi24_0']) < 2) and (d['rsi6_0'] > 50) and
                                    # 排除静态排列信号
                                    not ((d['rsi6_0'] > d['rsi12_0'] and d['rsi12_0'] > d['rsi24_0']) or
                                         (d['rsi6_0'] > d['rsi12_0'] and d['rsi12_0'] < d['rsi24_0']) or
                                         (d['rsi6_0'] < d['rsi12_0'] and d['rsi12_0'] > d['rsi24_0']) or
                                         (d['rsi6_0'] < d['rsi12_0'] and d['rsi12_0'] < d['rsi24_0'])),
        "buy_score": 0.0,
        "sell_score": 0.5
    },
    {
        "priority": 5,
        "signal_name": "偏多粘合(RSI6<30)",
        "signal_type": "粘合平衡",
        "condition_func": lambda d: (abs(d['rsi6_0'] - d['rsi12_0']) < 2) and
                                    (d['rsi12_0'] > d['rsi24_0']) and (d['rsi6_0'] < 30) and
                                    not (d['rsi6_0'] < d['rsi12_0'] and d['rsi12_0'] > d['rsi24_0']),  # 排除短空多长
        "buy_score": 1.5,
        "sell_score": 0.0
    },
    {
        "priority": 5,
        "signal_name": "偏多粘合(RSI6>70)",
        "signal_type": "粘合平衡",
        "condition_func": lambda d: (abs(d['rsi6_0'] - d['rsi12_0']) < 2) and
                                    (d['rsi12_0'] > d['rsi24_0']) and (d['rsi6_0'] > 70) and
                                    not (d['rsi6_0'] < d['rsi12_0'] and d['rsi12_0'] > d['rsi24_0']),  # 排除短空多长
        "buy_score": 0.0,
        "sell_score": 1.5
    },
    {
        "priority": 5,
        "signal_name": "偏空粘合(RSI6<30)",
        "signal_type": "粘合平衡",
        "condition_func": lambda d: (abs(d['rsi6_0'] - d['rsi12_0']) < 2) and
                                    (d['rsi12_0'] < d['rsi24_0']) and (d['rsi6_0'] < 30) and
                                    not (d['rsi6_0'] < d['rsi12_0'] and d['rsi12_0'] < d['rsi24_0']),  # 排除最强空头
        "buy_score": 1.2,
        "sell_score": 0.0
    },
    # 找到RSI_RULES中「偏空粘合(RSI6>70)」的规则，修改condition_func：
    {
        "priority": 5,
        "signal_name": "偏空粘合(RSI6>70)",
        "signal_type": "粘合平衡",
        "condition_func": lambda d: (abs(d['rsi6_0'] - d['rsi12_0']) < 2) and
                                    (d['rsi12_0'] < d['rsi24_0']) and (d['rsi6_0'] > 70) and
                                    # 新增：同时排除最强空头 + 短多长空
                                    not (d['rsi6_0'] < d['rsi12_0'] and d['rsi12_0'] < d['rsi24_0']) and  # 排除最强空头
                                    not (d['rsi6_0'] > d['rsi12_0'] and d['rsi12_0'] < d['rsi24_0']),  # 排除短多长空
        "buy_score": 0.0,
        "sell_score": 1.8
    },
]


# ===================== 第二步：工具函数（增强鲁棒性） =====================
def safe_get_rsi(arr, idx):
    """安全获取RSI值，处理越界/异常，返回0-100之间的值"""
    try:
        # idx为负数时，-1表示最后一个元素（当日），-2表示倒数第二个（前一日），以此类推
        if idx >= 0:
            val = float(arr[idx])
        else:
            # 负数索引直接取倒数
            val = float(arr[idx]) if abs(idx) <= len(arr) else 50.0
        return max(0.0, min(100.0, val))  # 限制RSI在0-100合理范围
    except (ValueError, IndexError, TypeError):
        return 50.0


def prepare_rsi_data(strategy):
    """整理RSI数据为字典，补充所有需要的历史字段"""
    # 获取RSI数组（兼容Backtrader Line对象和普通列表）
    rsi6_arr = strategy.rsi6.array if hasattr(strategy.rsi6, 'array') else strategy.rsi6
    rsi12_arr = strategy.rsi12.array if hasattr(strategy.rsi12, 'array') else strategy.rsi12
    rsi24_arr = strategy.rsi24.array if hasattr(strategy.rsi24, 'array') else strategy.rsi24

    # 整理所有规则需要的字段，使用安全取值函数
    rsi_data = {
        # RSI6 当日/历史值（-1=当日，-2=前1日，-3=前2日，-4=前3日，-5=前4日）
        'rsi6_0': safe_get_rsi(rsi6_arr, -1),  # 最新值（最后一个元素）
        'rsi6_-1': safe_get_rsi(rsi6_arr, -2),  # 前1日
        'rsi6_-2': safe_get_rsi(rsi6_arr, -3),  # 前2日
        'rsi6_-3': safe_get_rsi(rsi6_arr, -4),  # 前3日
        'rsi6_-4': safe_get_rsi(rsi6_arr, -5),  # 前4日
        # RSI12 当日/历史值
        'rsi12_0': safe_get_rsi(rsi12_arr, -1),
        'rsi12_-1': safe_get_rsi(rsi12_arr, -2),
        'rsi12_-2': safe_get_rsi(rsi12_arr, -3),
        'rsi12_-3': safe_get_rsi(rsi12_arr, -4),
        'rsi12_-4': safe_get_rsi(rsi12_arr, -5),
        # RSI24 当日值（仅需当日）
        'rsi24_0': safe_get_rsi(rsi24_arr, -1),
        # 辅助函数
        'abs': abs
    }
    return rsi_data


# ===================== 第三步：核心评分函数（优化后逻辑） =====================
def calculate_rsi_score(strategy):
    """
    优化后的RSI评分逻辑：
    1. 先匹配所有符合条件的信号
    2. 从匹配结果中筛选出权重最高的信号（核心优化点）
    3. 计算最终得分，完全贴合「只保留权重最高信号」的规则
    """
    # 1. 整理RSI数据
    rsi_data = prepare_rsi_data(strategy)

    # 2. 初始化结果
    default_result = {
        "signal_name": "无匹配信号",
        "signal_type": "静态排列",
        "raw_buy": 0.0,
        "raw_sell": 0.0,
        "weight": 1.0,
        "weighted_buy": 0.0,
        "weighted_sell": 0.0,
        "net_score": 0.0
    }

    # 3. 第一步：全量匹配所有符合条件的信号
    matched_signals = []
    for rule_idx, rule in enumerate(RSI_RULES):
        try:
            if rule['condition_func'](rsi_data):
                # 匹配成功，记录信号+对应的权重+规则索引（保证排序稳定性）
                signal_weight = RSI_SIGNAL_WEIGHTS.get(rule['signal_type'], 1.0)
                matched_signals.append({
                    "rule": rule,
                    "weight": signal_weight,
                    "priority": rule['priority'],
                    "rule_idx": rule_idx  # 规则在列表中的原始位置，用于稳定排序
                })
        except Exception as e:
            # 单个规则匹配失败不影响整体，仅打印调试信息
            print(f"【RSI调试】规则[{rule['signal_name']}]匹配失败: {str(e)[:50]}")
            continue

    # 4. 第二步：无匹配信号，返回默认值
    if not matched_signals:
        return default_result

    # 5. 第三步：筛选权重最高的信号（核心逻辑）
    # 排序规则：
    # 1. 权重降序（weight越大越优先）
    # 2. 业务优先级升序（priority越小越优先）
    # 3. 规则原始索引升序（保证同权重同优先级时按定义顺序）
    matched_signals.sort(key=lambda x: (-x['weight'], x['priority'], x['rule_idx']))
    highest_signal = matched_signals[0]['rule']
    highest_weight = matched_signals[0]['weight']

    # 6. 计算最终得分
    final_result = {
        "signal_name": highest_signal["signal_name"],
        "signal_type": highest_signal["signal_type"],
        "raw_buy": highest_signal["buy_score"],
        "raw_sell": highest_signal["sell_score"],
        "weight": highest_weight,
        "weighted_buy": highest_signal["buy_score"] * highest_weight,
        "weighted_sell": highest_signal["sell_score"] * highest_weight,
        "net_score": round((highest_signal["buy_score"] - highest_signal["sell_score"]) * highest_weight, 2)
    }

    return final_result


# ========== 第四步：模拟Backtrader对象（修复默认值污染问题）==========
class MockRSIStrategy:
    """专门用于RSI测试的模拟策略对象（修复默认值污染问题）"""

    def __init__(self, rsi6_vals, rsi12_vals=None, rsi24_vals=None):
        # 不自动补充默认值，仅使用传入的数值，避免历史数据污染
        self.rsi6 = BacktraderLine(rsi6_vals)
        # 若未传入rsi12/rsi24，使用和rsi6相同的长度，值为50（中性值）
        if rsi12_vals is None:
            rsi12_vals = [50.0] * len(rsi6_vals)
        self.rsi12 = BacktraderLine(rsi12_vals)

        if rsi24_vals is None:
            rsi24_vals = [50.0] * len(rsi6_vals)
        self.rsi24 = BacktraderLine(rsi24_vals)


class BacktraderLine:
    """模拟Backtrader的Line对象"""

    def __init__(self, values):
        # 确保所有值都是浮点数，空列表则初始化为[50.0]
        if not values:
            self.array = [50.0]
        else:
            self.array = [float(v) for v in values]

    def __getitem__(self, idx):
        try:
            return self.array[idx]
        except IndexError:
            return 50.0

    def get(self, size=None):
        if size is None or size <= 0:
            return self.array.copy()
        return self.array[-size:] if len(self.array) >= size else self.array.copy()

    def __len__(self):
        return len(self.array)


# ========== 第五步：完整的RSI测试用例（适配修复后的Mock类） ==========
class TestRSIScoreCalculation(unittest.TestCase):
    def setUp(self):
        """每个测试用例执行前的初始化"""
        print("\n--- 开始测试 ---")

    # ---------------- 测试优先级1：趋势反转 (权重2.0) ----------------
    def test_01_breakthrough_gold_cross(self):
        """测试1：突破金叉（趋势反转）"""
        print("【测试1】突破金叉")
        # RSI6: 前一日45（-2），当日55（-1）；RSI12: 前一日48（-2），当日52（-1）
        # 传入长度为2的列表，索引-1=55（当日），-2=45（前一日）
        strat = MockRSIStrategy(rsi6_vals=[45, 55], rsi12_vals=[48, 52])
        result = calculate_rsi_score(strat)
        print(f"测试结果: {result['signal_name']} | 净得分: {result['net_score']}")
        self.assertEqual(result['signal_name'], "突破金叉")
        self.assertEqual(result['net_score'], 4.0)  # (2-0)*2=4

    def test_02_breakdown_death_cross(self):
        """测试2：破位死叉（趋势反转）"""
        print("【测试2】破位死叉")
        # RSI6: 前一日55（-2），当日45（-1）；RSI12: 前一日52（-2），当日48（-1）
        strat = MockRSIStrategy(rsi6_vals=[55, 45], rsi12_vals=[52, 48])
        result = calculate_rsi_score(strat)
        print(f"测试结果: {result['signal_name']} | 净得分: {result['net_score']}")
        self.assertEqual(result['signal_name'], "破位死叉")
        self.assertEqual(result['net_score'], -4.0)  # (0-2)*2=-4

    # ---------------- 测试优先级2：连续极值 (权重1.5) ----------------
    def test_03_continuous_oversold(self):
        """测试3：连续超卖（连续极值）"""
        print("【测试3】连续超卖")
        # RSI6连续3日：18(-3),19(-2),17(-1)；RSI12当日：28(-1)
        strat = MockRSIStrategy(rsi6_vals=[18, 19, 17], rsi12_vals=[29, 28, 28])
        result = calculate_rsi_score(strat)
        print(f"测试结果: {result['signal_name']} | 净得分: {result['net_score']}")
        self.assertEqual(result['signal_name'], "连续超卖")
        self.assertEqual(result['net_score'], 3.0)  # (2-0)*1.5=3

    def test_04_continuous_overbought(self):
        """测试4：连续超买（连续极值）"""
        print("【测试4】连续超买")
        # RSI6连续3日：82(-3),81(-2),83(-1)；RSI12当日：72(-1)
        strat = MockRSIStrategy(rsi6_vals=[82, 81, 83], rsi12_vals=[71, 72, 72])
        result = calculate_rsi_score(strat)
        print(f"测试结果: {result['signal_name']} | 净得分: {result['net_score']}")
        self.assertEqual(result['signal_name'], "连续超买")
        self.assertEqual(result['net_score'], -3.0)  # (0-2)*1.5=-3

    def test_05_extreme_oversold(self):
        """测试5：极端超卖（连续极值）"""
        print("【测试5】极端超卖")
        # RSI6连续2日：9(-2),8(-1)
        strat = MockRSIStrategy(rsi6_vals=[9, 8])
        result = calculate_rsi_score(strat)
        print(f"测试结果: {result['signal_name']} | 净得分: {result['net_score']}")
        self.assertEqual(result['signal_name'], "极端超卖")
        self.assertEqual(result['net_score'], 3.0)  # (2-0)*1.5=3

    def test_06_extreme_overbought(self):
        """测试6：极端超买（连续极值）"""
        print("【测试6】极端超买")
        # RSI6连续2日：91(-2),92(-1)
        strat = MockRSIStrategy(rsi6_vals=[91, 92])
        result = calculate_rsi_score(strat)
        print(f"测试结果: {result['signal_name']} | 净得分: {result['net_score']}")
        self.assertEqual(result['signal_name'], "极端超买")
        self.assertEqual(result['net_score'], -3.0)  # (0-2)*1.5=-3

    # ---------------- 测试优先级3：动态金叉/死叉/拐头/企稳 (权重1.2) ----------------
    def test_07_oversold_gold_cross(self):
        """测试7：超卖金叉（动态）"""
        print("【测试7】超卖金叉")
        # RSI6: 前一日25(-2)，当日28(-1)；RSI12: 前一日27(-2)，当日27(-1)
        strat = MockRSIStrategy(rsi6_vals=[25, 28], rsi12_vals=[27, 27])
        result = calculate_rsi_score(strat)
        print(f"测试结果: {result['signal_name']} | 净得分: {result['net_score']}")
        self.assertEqual(result['signal_name'], "超卖金叉")
        self.assertEqual(result['net_score'], 2.16)  # (1.8-0)*1.2=2.16

    def test_08_shock_gold_cross(self):
        """测试8：震荡金叉（动态）"""
        print("【测试8】震荡金叉")
        # RSI6: 前一日35(-2)，当日45(-1)；RSI12: 前一日38(-2)，当日43(-1)
        strat = MockRSIStrategy(rsi6_vals=[35, 45], rsi12_vals=[38, 43])
        result = calculate_rsi_score(strat)
        print(f"测试结果: {result['signal_name']} | 净得分: {result['net_score']}")
        self.assertEqual(result['signal_name'], "震荡金叉")
        self.assertEqual(result['net_score'], 1.8)  # (1.5-0)*1.2=1.8

    def test_09_high_gold_cross(self):
        """测试9：高位金叉（动态）"""
        print("【测试9】高位金叉")
        # RSI6: 前一日72(-2)，当日75(-1)；RSI12: 前一日73(-2)，当日74(-1)
        strat = MockRSIStrategy(rsi6_vals=[72, 75], rsi12_vals=[73, 74])
        result = calculate_rsi_score(strat)
        print(f"测试结果: {result['signal_name']} | 净得分: {result['net_score']}")
        self.assertEqual(result['signal_name'], "高位金叉")
        self.assertEqual(result['net_score'], -1.2)  # (0-1)*1.2=-1.2

    def test_10_overbought_death_cross(self):
        """测试10：超买死叉（动态）"""
        print("【测试10】超买死叉")
        # RSI6: 前一日78(-2)，当日75(-1)；RSI12: 前一日77(-2)，当日76(-1)
        strat = MockRSIStrategy(rsi6_vals=[78, 75], rsi12_vals=[77, 76])
        result = calculate_rsi_score(strat)
        print(f"测试结果: {result['signal_name']} | 净得分: {result['net_score']}")
        self.assertEqual(result['signal_name'], "超买死叉")
        self.assertEqual(result['net_score'], -2.16)  # (0-1.8)*1.2=-2.16

    def test_11_shock_death_cross(self):
        """测试11：震荡死叉（动态）"""
        print("【测试11】震荡死叉")
        # RSI6: 前一日65(-2)，当日60(-1)；RSI12: 前一日63(-2)，当日62(-1)
        strat = MockRSIStrategy(rsi6_vals=[65, 60], rsi12_vals=[63, 62])
        result = calculate_rsi_score(strat)
        print(f"测试结果: {result['signal_name']} | 净得分: {result['net_score']}")
        self.assertEqual(result['signal_name'], "震荡死叉")
        self.assertEqual(result['net_score'], -1.8)  # (0-1.5)*1.2=-1.8

    def test_12_low_death_cross(self):
        """测试12：低位死叉（动态）"""
        print("【测试12】低位死叉")
        # RSI6: 前一日28(-2)，当日25(-1)；RSI12: 前一日27(-2)，当日26(-1)
        strat = MockRSIStrategy(rsi6_vals=[28, 25], rsi12_vals=[27, 26])
        result = calculate_rsi_score(strat)
        print(f"测试结果: {result['signal_name']} | 净得分: {result['net_score']}")
        self.assertEqual(result['signal_name'], "低位死叉")
        self.assertEqual(result['net_score'], 1.2)  # (1-0)*1.2=1.2

    def test_13_low_turnaround(self):
        """测试13：低位拐头（动态）"""
        print("【测试13】低位拐头")
        # RSI6: 前一日28(-2)，当日40(-1)
        strat = MockRSIStrategy(rsi6_vals=[28, 40])
        result = calculate_rsi_score(strat)
        print(f"测试结果: {result['signal_name']} | 净得分: {result['net_score']}")
        self.assertEqual(result['signal_name'], "低位拐头")
        self.assertEqual(result['net_score'], 2.16)  # (1.8-0)*1.2=2.16

    def test_14_high_turnaround(self):
        """测试14：高位拐头（动态）"""
        print("【测试14】高位拐头")
        # RSI6: 前一日75(-2)，当日65(-1)
        strat = MockRSIStrategy(rsi6_vals=[75, 65])
        result = calculate_rsi_score(strat)
        print(f"测试结果: {result['signal_name']} | 净得分: {result['net_score']}")
        self.assertEqual(result['signal_name'], "高位拐头")
        self.assertEqual(result['net_score'], -2.16)  # (0-1.8)*1.2=-2.16

    def test_15_oversold_stabilization(self):
        """测试15：超卖企稳（动态）"""
        print("【测试15】超卖企稳")
        # RSI6: 前两日28(-3),25(-2)，当日29(-1)（上涨4点？不，25→29是上涨4？修改为25→30，但<30）
        strat = MockRSIStrategy(rsi6_vals=[28, 25, 29])
        result = calculate_rsi_score(strat)
        print(f"测试结果: {result['signal_name']} | 净得分: {result['net_score']}")
        self.assertEqual(result['signal_name'], "超卖企稳")
        self.assertEqual(result['net_score'], 1.8)  # (1.5-0)*1.2=1.8

    def test_16_overbought_pullback(self):
        """测试16：超买回落（动态）"""
        print("【测试16】超买回落")
        # RSI6: 前两日75(-3),78(-2)，当日72(-1)（下跌6点）
        strat = MockRSIStrategy(rsi6_vals=[75, 78, 72])
        result = calculate_rsi_score(strat)
        print(f"测试结果: {result['signal_name']} | 净得分: {result['net_score']}")
        self.assertEqual(result['signal_name'], "超买回落")
        self.assertEqual(result['net_score'], -1.8)  # (0-1.5)*1.2=-1.8

    def test_17_gold_cross_continue_2d(self):
        """测试17：金叉延续2日（动态）"""
        print("【测试17】金叉延续2日")
        # 前3日死叉（-4=40 < -3=42），前2日金叉（-3=42 > -2=41），前1日金叉（-2=41 > -1=40），当日金叉（-1=40 > 0=39）；RSI6当日40
        strat = MockRSIStrategy(
            rsi6_vals=[40, 42, 41, 40],
            rsi12_vals=[42, 41, 40, 39]
        )
        result = calculate_rsi_score(strat)
        print(f"测试结果: {result['signal_name']} | 净得分: {result['net_score']}")
        self.assertEqual(result['signal_name'], "金叉延续2日")
        self.assertEqual(result['net_score'], 1.8)  # (1.5-0)*1.2=1.8

    def test_18_gold_cross_continue_3d(self):
        """测试18：金叉延续3日（动态）"""
        print("【测试18】金叉延续3日")
        # 前4日死叉（-5=45 < -4=47），前3日金叉（-4=43 > -3=45？修正：-4=47 > -3=45），前2日金叉，前1日金叉，当日金叉；RSI6当日45
        strat = MockRSIStrategy(
            rsi6_vals=[45, 47, 46, 45, 44, 45],
            rsi12_vals=[47, 45, 44, 43, 42, 44]
        )
        result = calculate_rsi_score(strat)
        print(f"测试结果: {result['signal_name']} | 净得分: {result['net_score']}")
        self.assertEqual(result['signal_name'], "金叉延续3日")
        self.assertEqual(result['net_score'], 2.16)  # (1.8-0)*1.2=2.16

    def test_19_gold_cross_after_overbought(self):
        """测试19：金叉后超买（动态）"""
        print("【测试19】金叉后超买")
        # 前3日金叉，当日金叉；RSI6当日75
        strat = MockRSIStrategy(
            rsi6_vals=[70, 72, 75, 73, 75],
            rsi12_vals=[72, 71, 74, 72, 74]
        )
        result = calculate_rsi_score(strat)
        print(f"测试结果: {result['signal_name']} | 净得分: {result['net_score']}")
        self.assertEqual(result['signal_name'], "金叉后超买")
        self.assertEqual(result['net_score'], -2.16)  # (0-1.8)*1.2=-2.16

    def test_20_death_cross_continue_2d(self):
        """测试20：死叉延续2日（动态）"""
        print("【测试20】死叉延续2日")
        # 前3日金叉（-4=60 > -3=58），前2日死叉（-3=62 < -2=60），前1日死叉（-2=58 < -1=59），当日死叉（-1=59 < 0=60）；RSI6当日60
        strat = MockRSIStrategy(
            rsi6_vals=[60, 62, 58, 59, 60],
            rsi12_vals=[58, 60, 59, 60, 61]
        )
        result = calculate_rsi_score(strat)
        print(f"测试结果: {result['signal_name']} | 净得分: {result['net_score']}")
        self.assertEqual(result['signal_name'], "死叉延续2日")
        self.assertEqual(result['net_score'], -1.8)  # (0-1.5)*1.2=-1.8

    def test_21_death_cross_continue_3d(self):
        """测试21：死叉延续3日（动态）"""
        print("【测试21】死叉延续3日")
        # 前4日金叉，前3日死叉，前2日死叉，前1日死叉，当日死叉；RSI6当日65
        strat = MockRSIStrategy(
            rsi6_vals=[65, 67, 64, 65, 66, 65],
            rsi12_vals=[63, 65, 65, 66, 67, 66]
        )
        result = calculate_rsi_score(strat)
        print(f"测试结果: {result['signal_name']} | 净得分: {result['net_score']}")
        self.assertEqual(result['signal_name'], "死叉延续3日")
        self.assertEqual(result['net_score'], -2.16)  # (0-1.8)*1.2=-2.16

    def test_22_death_cross_after_oversold(self):
        """测试22：死叉后超卖（动态）"""
        print("【测试22】死叉后超卖")
        # 前3日死叉，当日死叉；RSI6当日25
        strat = MockRSIStrategy(
            rsi6_vals=[28, 26, 25, 27, 25],
            rsi12_vals=[27, 25, 26, 26, 26]
        )
        result = calculate_rsi_score(strat)
        print(f"测试结果: {result['signal_name']} | 净得分: {result['net_score']}")
        self.assertEqual(result['signal_name'], "死叉后超卖")
        self.assertEqual(result['net_score'], 1.2)  # (1-0)*1.2=1.2

    # ---------------- 测试优先级4：静态排列 (权重1.0) ----------------
    def test_23_strong_bull_rsi6_below_20(self):
        """测试23：最强多头(RSI6<20)（静态排列）"""
        print("【测试23】最强多头(RSI6<20)")
        # RSI6:18, RSI12:15, RSI24:10
        strat = MockRSIStrategy(rsi6_vals=[18], rsi12_vals=[15], rsi24_vals=[10])
        result = calculate_rsi_score(strat)
        print(f"测试结果: {result['signal_name']} | 净得分: {result['net_score']}")
        self.assertEqual(result['signal_name'], "最强多头(RSI6<20)")
        self.assertEqual(result['net_score'], 2.0)  # (2-0)*1=2

    def test_24_strong_bull_rsi6_20_50(self):
        """测试24：最强多头(20<=RSI6<=50)（静态排列）"""
        print("【测试24】最强多头(20<=RSI6<=50)")
        # RSI6:35, RSI12:30, RSI24:25
        strat = MockRSIStrategy(rsi6_vals=[35], rsi12_vals=[30], rsi24_vals=[25])
        result = calculate_rsi_score(strat)
        print(f"测试结果: {result['signal_name']} | 净得分: {result['net_score']}")
        self.assertEqual(result['signal_name'], "最强多头(20<=RSI6<=50)")
        self.assertEqual(result['net_score'], 1.5)  # (1.5-0)*1=1.5

    def test_25_strong_bull_rsi6_50_70(self):
        """测试25：最强多头(50<=RSI6<=70)（静态排列）"""
        print("【测试25】最强多头(50<=RSI6<=70)")
        # RSI6:60, RSI12:55, RSI24:50
        strat = MockRSIStrategy(rsi6_vals=[60], rsi12_vals=[55], rsi24_vals=[50])
        result = calculate_rsi_score(strat)
        print(f"测试结果: {result['signal_name']} | 净得分: {result['net_score']}")
        self.assertEqual(result['signal_name'], "最强多头(50<=RSI6<=70)")
        self.assertEqual(result['net_score'], 0.5)  # (1-0.5)*1=0.5

    def test_26_strong_bull_rsi6_above_70(self):
        """测试26：最强多头(RSI6>70)（静态排列）"""
        print("【测试26】最强多头(RSI6>70)")
        # RSI6:75, RSI12:70, RSI24:65
        strat = MockRSIStrategy(rsi6_vals=[75], rsi12_vals=[70], rsi24_vals=[65])
        result = calculate_rsi_score(strat)
        print(f"测试结果: {result['signal_name']} | 净得分: {result['net_score']}")
        self.assertEqual(result['signal_name'], "最强多头(RSI6>70)")
        self.assertEqual(result['net_score'], -2.0)  # (0-2)*1=-2

    def test_27_short_long_bear_rsi6_below_30(self):
        """测试27：短多长空(RSI6<30)（静态排列）"""
        print("【测试27】短多长空(RSI6<30)")
        # RSI6:28, RSI12:25, RSI24:30
        strat = MockRSIStrategy(rsi6_vals=[28], rsi12_vals=[25], rsi24_vals=[30])
        result = calculate_rsi_score(strat)
        print(f"测试结果: {result['signal_name']} | 净得分: {result['net_score']}")
        self.assertEqual(result['signal_name'], "短多长空(RSI6<30)")
        self.assertEqual(result['net_score'], 1.5)  # (1.5-0)*1=1.5

    def test_28_short_long_bear_rsi6_30_70(self):
        """测试28：短多长空(30<=RSI6<=70)（静态排列）"""
        print("【测试28】短多长空(30<=RSI6<=70)")
        # RSI6:50, RSI12:45, RSI24:55
        strat = MockRSIStrategy(rsi6_vals=[50], rsi12_vals=[45], rsi24_vals=[55])
        result = calculate_rsi_score(strat)
        print(f"测试结果: {result['signal_name']} | 净得分: {result['net_score']}")
        self.assertEqual(result['signal_name'], "短多长空(30<=RSI6<=70)")
        self.assertEqual(result['net_score'], 0.0)  # (0.8-0.8)*1=0

    def test_29_short_long_bear_rsi6_above_70(self):
        """测试29：短多长空(RSI6>70)（静态排列）"""
        print("【测试29】短多长空(RSI6>70)")
        # RSI6:75, RSI12:70, RSI24:80
        strat = MockRSIStrategy(rsi6_vals=[75], rsi12_vals=[70], rsi24_vals=[80])
        result = calculate_rsi_score(strat)
        print(f"测试结果: {result['signal_name']} | 净得分: {result['net_score']}")
        self.assertEqual(result['signal_name'], "短多长空(RSI6>70)")
        self.assertEqual(result['net_score'], -1.8)  # (0-1.8)*1=-1.8

    def test_30_short_bear_long_rsi6_below_30(self):
        """测试30：短空多长(RSI6<30)（静态排列）"""
        print("【测试30】短空多长(RSI6<30)")
        # RSI6:28, RSI12:30, RSI24:25
        strat = MockRSIStrategy(rsi6_vals=[28], rsi12_vals=[30], rsi24_vals=[25])
        result = calculate_rsi_score(strat)
        print(f"测试结果: {result['signal_name']} | 净得分: {result['net_score']}")
        self.assertEqual(result['signal_name'], "短空多长(RSI6<30)")
        self.assertEqual(result['net_score'], 1.2)  # (1.2-0)*1=1.2

    def test_31_short_bear_long_rsi6_30_70(self):
        """测试31：短空多长(30<=RSI6<=70)（静态排列）"""
        print("【测试31】短空多长(30<=RSI6<=70)")
        # RSI6:50, RSI12:55, RSI24:50
        strat = MockRSIStrategy(rsi6_vals=[50], rsi12_vals=[55], rsi24_vals=[50])
        result = calculate_rsi_score(strat)
        print(f"测试结果: {result['signal_name']} | 净得分: {result['net_score']}")
        self.assertEqual(result['signal_name'], "短空多长(30<=RSI6<=70)")
        self.assertEqual(result['net_score'], -0.5)  # (0.5-1)*1=-0.5

    def test_32_short_bear_long_rsi6_above_70(self):
        """测试32：短空多长(RSI6>70)（静态排列）"""
        print("【测试32】短空多长(RSI6>70)")
        # RSI6:75, RSI12:80, RSI24:75
        strat = MockRSIStrategy(rsi6_vals=[75], rsi12_vals=[80], rsi24_vals=[75])
        result = calculate_rsi_score(strat)
        print(f"测试结果: {result['signal_name']} | 净得分: {result['net_score']}")
        self.assertEqual(result['signal_name'], "短空多长(RSI6>70)")
        self.assertEqual(result['net_score'], -1.5)  # (0-1.5)*1=-1.5

    def test_33_strong_bear_rsi6_below_20(self):
        """测试33：最强空头(RSI6<20)（静态排列）"""
        print("【测试33】最强空头(RSI6<20)")
        # RSI6:18, RSI12:20, RSI24:25
        strat = MockRSIStrategy(rsi6_vals=[18], rsi12_vals=[20], rsi24_vals=[25])
        result = calculate_rsi_score(strat)
        print(f"测试结果: {result['signal_name']} | 净得分: {result['net_score']}")
        self.assertEqual(result['signal_name'], "最强空头(RSI6<20)")
        self.assertEqual(result['net_score'], 1.5)  # (1.5-0)*1=1.5

    def test_34_strong_bear_rsi6_20_50(self):
        """测试34：最强空头(20<=RSI6<=50)（静态排列）"""
        print("【测试34】最强空头(20<=RSI6<=50)")
        # RSI6:35, RSI12:40, RSI24:45
        strat = MockRSIStrategy(rsi6_vals=[35], rsi12_vals=[40], rsi24_vals=[45])
        result = calculate_rsi_score(strat)
        print(f"测试结果: {result['signal_name']} | 净得分: {result['net_score']}")
        self.assertEqual(result['signal_name'], "最强空头(20<=RSI6<=50)")
        self.assertEqual(result['net_score'], -1.5)  # (0-1.5)*1=-1.5

    def test_35_strong_bear_rsi6_50_70(self):
        """测试35：最强空头(50<=RSI6<=70)（静态排列）"""
        print("【测试35】最强空头(50<=RSI6<=70)")
        # RSI6:60, RSI12:65, RSI24:70
        strat = MockRSIStrategy(rsi6_vals=[60], rsi12_vals=[65], rsi24_vals=[70])
        result = calculate_rsi_score(strat)
        print(f"测试结果: {result['signal_name']} | 净得分: {result['net_score']}")
        self.assertEqual(result['signal_name'], "最强空头(50<=RSI6<=70)")
        self.assertEqual(result['net_score'], -2.0)  # (0-2)*1=-2

    def test_36_strong_bear_rsi6_above_70(self):
        """测试36：最强空头(RSI6>70)（静态排列）"""
        print("【测试36】最强空头(RSI6>70)")
        # RSI6:75, RSI12:80, RSI24:85
        strat = MockRSIStrategy(rsi6_vals=[75], rsi12_vals=[80], rsi24_vals=[85])
        result = calculate_rsi_score(strat)
        print(f"测试结果: {result['signal_name']} | 净得分: {result['net_score']}")
        self.assertEqual(result['signal_name'], "最强空头(RSI6>70)")
        self.assertEqual(result['net_score'], -2.0)  # (0-2)*1=-2

    # ---------------- 测试优先级5：粘合平衡 (权重0.5) ----------------
    def test_37_fully_sticky_rsi6_below_50(self):
        """测试37：完全粘合(RSI6<50)（粘合平衡）"""
        print("【测试37】完全粘合(RSI6<50)")
        # RSI6:45, RSI12:46, RSI24:45（差值都<2）
        strat = MockRSIStrategy(rsi6_vals=[45], rsi12_vals=[46], rsi24_vals=[45])
        result = calculate_rsi_score(strat)
        print(f"测试结果: {result['signal_name']} | 净得分: {result['net_score']}")
        self.assertEqual(result['signal_name'], "完全粘合(RSI6<50)")
        self.assertEqual(result['net_score'], 0.25)  # (0.5-0)*0.5=0.25

    def test_38_fully_sticky_rsi6_above_50(self):
        """测试38：完全粘合(RSI6>50)（粘合平衡）"""
        print("【测试38】完全粘合(RSI6>50)")
        # RSI6:55, RSI12:56, RSI24:55（差值都<2）
        strat = MockRSIStrategy(rsi6_vals=[55], rsi12_vals=[56], rsi24_vals=[55])
        result = calculate_rsi_score(strat)
        print(f"测试结果: {result['signal_name']} | 净得分: {result['net_score']}")
        self.assertEqual(result['signal_name'], "完全粘合(RSI6>50)")
        self.assertEqual(result['net_score'], -0.25)  # (0-0.5)*0.5=-0.25

    def test_39_bullish_sticky_rsi6_below_30(self):
        """测试39：偏多粘合(RSI6<30)（粘合平衡）"""
        print("【测试39】偏多粘合(RSI6<30)")
        # RSI6:28, RSI12:29, RSI24:27（RSI6与RSI12差值<2，RSI12>RSI24）
        strat = MockRSIStrategy(rsi6_vals=[28], rsi12_vals=[29], rsi24_vals=[27])
        result = calculate_rsi_score(strat)
        print(f"测试结果: {result['signal_name']} | 净得分: {result['net_score']}")
        self.assertEqual(result['signal_name'], "偏多粘合(RSI6<30)")
        self.assertEqual(result['net_score'], 0.75)  # (1.5-0)*0.5=0.75

    def test_40_bullish_sticky_rsi6_above_70(self):
        """测试40：偏多粘合(RSI6>70)（粘合平衡）"""
        print("【测试40】偏多粘合(RSI6>70)")
        # RSI6:75, RSI12:76, RSI24:74（RSI6与RSI12差值<2，RSI12>RSI24）
        strat = MockRSIStrategy(rsi6_vals=[75], rsi12_vals=[76], rsi24_vals=[74])
        result = calculate_rsi_score(strat)
        print(f"测试结果: {result['signal_name']} | 净得分: {result['net_score']}")
        self.assertEqual(result['signal_name'], "偏多粘合(RSI6>70)")
        self.assertEqual(result['net_score'], -0.75)  # (0-1.5)*0.5=-0.75

    def test_41_bearish_sticky_rsi6_below_30(self):
        """测试41：偏空粘合(RSI6<30)（粘合平衡）"""
        print("【测试41】偏空粘合(RSI6<30)")
        # RSI6:28, RSI12:29, RSI24:30（差值<2，RSI12 < RSI24）
        strat = MockRSIStrategy(rsi6_vals=[28], rsi12_vals=[29], rsi24_vals=[30])
        result = calculate_rsi_score(strat)
        print(f"测试结果: {result['signal_name']} | 净得分: {result['net_score']}")
        self.assertEqual(result['signal_name'], "偏空粘合(RSI6<30)")
        self.assertEqual(result['net_score'], 0.6)  # (1.2-0)*0.5=0.6

    def test_42_bearish_sticky_rsi6_above_70(self):
        """测试42：偏空粘合(RSI6>70)（粘合平衡）"""
        print("【测试42】偏空粘合(RSI6>70)")
        # 最终版测试数据：避开所有静态排列，仅触发偏空粘合
        # RSI6:75, RSI12:76（差值=1<2，RSI6<RSI12），RSI24:77（RSI12<RSI24）
        strat = MockRSIStrategy(rsi6_vals=[75], rsi12_vals=[76], rsi24_vals=[77])
        result = calculate_rsi_score(strat)
        print(f"测试结果: {result['signal_name']} | 净得分: {result['net_score']}")
        self.assertEqual(result['signal_name'], "偏空粘合(RSI6>70)")
        self.assertEqual(result['net_score'], -0.9)  # (0-1.8)*0.5=-0.9

    def test_99_no_signal(self):
        """测试99：无任何信号匹配"""
        print("【测试99】无信号")
        # 中性数据，不触发任何规则
        strat = MockRSIStrategy(rsi6_vals=[50], rsi12_vals=[50], rsi24_vals=[50])
        result = calculate_rsi_score(strat)
        print(f"测试结果: {result['signal_name']} | 净得分: {result['net_score']}")
        self.assertEqual(result['signal_name'], "无匹配信号")
        self.assertEqual(result['net_score'], 0.0)

if __name__ == '__main__':
    unittest.main(verbosity=2)