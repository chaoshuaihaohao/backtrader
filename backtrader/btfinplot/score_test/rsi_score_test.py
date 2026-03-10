import unittest
import math

# ===================== 第一步：定义核心配置（和手册完全对齐） =====================
RSI_RULES = [
    # ---------------- 优先级1：趋势反转 (权重2.0) ----------------
    {
        "priority": 1,
        "signal_name": "突破金叉",
        "signal_type": "趋势反转",
        "weight": 2.0,  # 👈 权重直接写在这里
        "condition_func": lambda strat: (strat.rsi6[0] - strat.rsi6[-1] >= 14),
        "buy_score": 2.0,
        "sell_score": 0.0
    },
    {
        "priority": 1,
        "signal_name": "突破金叉",
        "signal_type": "趋势反转",
        "weight": 1.0, # 👈 权重直接写在这里
        "condition_func": lambda strat: (5 < strat.rsi6[0] - strat.rsi6[-1] < 14),
        "buy_score": 1.5,
        "sell_score": 0.0
    },
    {
        "priority": 1,
        "signal_name": "突破金叉",
        "signal_type": "趋势反转",
        "weight": 1.0,  # 👈 权重直接写在这里
        "condition_func": lambda strat: (strat.rsi6[0] > strat.rsi6[-1]),
        "buy_score": 1.0,
        "sell_score": 0.0
    },
    {
        "priority": 1,
        "signal_name": "破位死叉",
        "signal_type": "趋势反转",
        "weight": 1.0,  # 👈 权重直接写在这里
        "condition_func": lambda strat: (strat.rsi6[-1] - strat.rsi6[0] >= 15),
        "buy_score": 0.0,
        "sell_score": 2.0
    },
    {
        "priority": 1,
        "signal_name": "破位死叉",
        "signal_type": "趋势反转",
        "weight": 1.0,  # 👈 权重直接写在这里
        "condition_func": lambda strat: (5 < strat.rsi6[-1] - strat.rsi6[0] < 15),
        "buy_score": 0.0,
        "sell_score": 1.5
    },
    {
        "priority": 1,
        "signal_name": "破位死叉",
        "signal_type": "趋势反转",
        "weight": 1.0, # 👈 权重直接写在这里
        "condition_func": lambda strat: (strat.rsi6[-1] > strat.rsi6[0]),
        "buy_score": 0.0,
        "sell_score": 1.0
    },
    #
    # # ---------------- 优先级2：连续极值 (权重1.5) ----------------
    # {
    #     "priority": 2,
    #     "signal_name": "连续超卖",
    #     "signal_type": "连续极值",
    #     "condition_func": lambda d: (d['rsi6_0'] < 20) and (d['rsi6_-1'] < 20) and (d['rsi6_-2'] < 20) and
    #                                 (d['rsi12_0'] < 30),
    #     "buy_score": 2.0,
    #     "sell_score": 0.0
    # },
    # {
    #     "priority": 2,
    #     "signal_name": "连续超买",
    #     "signal_type": "连续极值",
    #     "condition_func": lambda d: (d['rsi6_0'] > 80) and (d['rsi6_-1'] > 80) and (d['rsi6_-2'] > 80) and
    #                                 (d['rsi12_0'] > 70),
    #     "buy_score": 0.0,
    #     "sell_score": 2.0
    # },
    # {
    #     "priority": 2,
    #     "signal_name": "极端超卖",
    #     "signal_type": "连续极值",
    #     "condition_func": lambda d: (d['rsi6_0'] < 10) and (d['rsi6_-1'] < 10),
    #     "buy_score": 2.0,
    #     "sell_score": 0.0
    # },
    # {
    #     "priority": 2,
    #     "signal_name": "极端超买",
    #     "signal_type": "连续极值",
    #     "condition_func": lambda d: (d['rsi6_0'] > 90) and (d['rsi6_-1'] > 90),
    #     "buy_score": 0.0,
    #     "sell_score": 2.0
    # },
    #
    # # ---------------- 优先级3：动态金叉/死叉/拐头/企稳/延续 (权重1.2) ----------------
    # {
    #     "priority": 3,
    #     "signal_name": "超卖金叉",
    #     "signal_type": "动态金叉/死叉/拐头/企稳",
    #     "condition_func": lambda d: (d['rsi6_0'] > d['rsi12_0']) and (d['rsi6_-1'] < d['rsi12_-1']) and
    #                                 (d['rsi6_0'] < 30),
    #     "buy_score": 1.8,
    #     "sell_score": 0.0
    # },
    # {
    #     "priority": 3,
    #     "signal_name": "震荡金叉",
    #     "signal_type": "动态金叉/死叉/拐头/企稳",
    #     "condition_func": lambda d: (d['rsi6_0'] > d['rsi12_0']) and (d['rsi6_-1'] < d['rsi12_-1']) and
    #                                 (30 <= d['rsi6_0'] <= 50),
    #     "buy_score": 1.5,
    #     "sell_score": 0.0
    # },
    # {
    #     "priority": 3,
    #     "signal_name": "高位金叉",
    #     "signal_type": "动态金叉/死叉/拐头/企稳",
    #     "condition_func": lambda d: (d['rsi6_0'] > d['rsi12_0']) and (d['rsi6_-1'] < d['rsi12_-1']) and
    #                                 (d['rsi6_0'] > 70),
    #     "buy_score": 0.0,
    #     "sell_score": 1.0
    # },
    # {
    #     "priority": 3,
    #     "signal_name": "超买死叉",
    #     "signal_type": "动态金叉/死叉/拐头/企稳",
    #     "condition_func": lambda d: (d['rsi6_0'] < d['rsi12_0']) and (d['rsi6_-1'] > d['rsi12_-1']) and
    #                                 (d['rsi6_0'] > 70),
    #     "buy_score": 0.0,
    #     "sell_score": 1.8
    # },
    # {
    #     "priority": 3,
    #     "signal_name": "震荡死叉",
    #     "signal_type": "动态金叉/死叉/拐头/企稳",
    #     "condition_func": lambda d: (d['rsi6_0'] < d['rsi12_0']) and (d['rsi6_-1'] > d['rsi12_-1']) and
    #                                 (50 <= d['rsi6_0'] <= 70),
    #     "buy_score": 0.0,
    #     "sell_score": 1.5
    # },
    # {
    #     "priority": 3,
    #     "signal_name": "低位死叉",
    #     "signal_type": "动态金叉/死叉/拐头/企稳",
    #     "condition_func": lambda d: (d['rsi6_0'] < d['rsi12_0']) and (d['rsi6_-1'] > d['rsi12_-1']) and
    #                                 (d['rsi6_0'] < 30),
    #     "buy_score": 1.0,
    #     "sell_score": 0.0
    # },
    # {
    #     "priority": 3,
    #     "signal_name": "低位拐头",
    #     "signal_type": "动态金叉/死叉/拐头/企稳",
    #     "condition_func": lambda d: (d['rsi6_-1'] < 30) and (30 <= d['rsi6_0'] <= 50),
    #     "buy_score": 1.8,
    #     "sell_score": 0.0
    # },
    # {
    #     "priority": 3,
    #     "signal_name": "高位拐头",
    #     "signal_type": "动态金叉/死叉/拐头/企稳",
    #     "condition_func": lambda d: (d['rsi6_-1'] > 70) and (50 <= d['rsi6_0'] <= 70),
    #     "buy_score": 0.0,
    #     "sell_score": 1.8
    # },
    # {
    #     "priority": 3,
    #     "signal_name": "超卖企稳",
    #     "signal_type": "动态金叉/死叉/拐头/企稳",
    #     "condition_func": lambda d: (d['rsi6_-1'] < 30) and (d['rsi6_-2'] < 30) and (d['rsi6_0'] - d['rsi6_-1'] >= 5) and
    #                                 (d['rsi6_0'] < 30),  # 新增：确保不触发低位拐头
    #     "buy_score": 1.5,
    #     "sell_score": 0.0
    # },
    # {
    #     "priority": 3,
    #     "signal_name": "超买回落",
    #     "signal_type": "动态金叉/死叉/拐头/企稳",
    #     "condition_func": lambda d: (d['rsi6_-1'] > 70) and (d['rsi6_-2'] > 70) and (d['rsi6_-1'] - d['rsi6_0'] >= 5) and
    #                                 (d['rsi6_0'] > 70),  # 新增：确保不触发高位拐头
    #     "buy_score": 0.0,
    #     "sell_score": 1.5
    # },
    # {
    #     "priority": 3,
    #     "signal_name": "金叉延续2日",
    #     "signal_type": "动态金叉/死叉/拐头/企稳",
    #     "condition_func": lambda d: (d['rsi6_-2'] > d['rsi12_-2'] and d['rsi6_-3'] < d['rsi12_-3']) and
    #                                 (d['rsi6_-1'] > d['rsi12_-1']) and (d['rsi6_0'] > d['rsi12_0']) and
    #                                 (30 <= d['rsi6_0'] <= 50) and not (d['rsi6_-3'] > d['rsi12_-3'] and d['rsi6_-4'] < d['rsi12_-4']),  # 排除3日延续
    #     "buy_score": 1.5,
    #     "sell_score": 0.0
    # },
    # {
    #     "priority": 3,
    #     "signal_name": "金叉延续3日",
    #     "signal_type": "动态金叉/死叉/拐头/企稳",
    #     "condition_func": lambda d: (d['rsi6_-3'] > d['rsi12_-3'] and d['rsi6_-4'] < d['rsi12_-4']) and
    #                                 (d['rsi6_-2'] > d['rsi12_-2']) and (d['rsi6_-1'] > d['rsi12_-1']) and
    #                                 (d['rsi6_0'] > d['rsi12_0']) and (30 <= d['rsi6_0'] <= 50),
    #     "buy_score": 1.8,
    #     "sell_score": 0.0
    # },
    # {
    #     "priority": 3,
    #     "signal_name": "金叉后超买",
    #     "signal_type": "动态金叉/死叉/拐头/企稳",
    #     "condition_func": lambda d: ((d['rsi6_-2'] > d['rsi12_-2'] and d['rsi6_-3'] < d['rsi12_-3']) or
    #                                  (d['rsi6_-3'] > d['rsi12_-3'] and d['rsi6_-4'] < d['rsi12_-4'])) and
    #                                 (d['rsi6_-1'] > d['rsi12_-1']) and (d['rsi6_0'] > d['rsi12_0']) and
    #                                 (d['rsi6_0'] > 70),
    #     "buy_score": 0.0,
    #     "sell_score": 1.8
    # },
    # {
    #     "priority": 3,
    #     "signal_name": "死叉延续2日",
    #     "signal_type": "动态金叉/死叉/拐头/企稳",
    #     "condition_func": lambda d: (d['rsi6_-2'] < d['rsi12_-2'] and d['rsi6_-3'] > d['rsi12_-3']) and
    #                                 (d['rsi6_-1'] < d['rsi12_-1']) and (d['rsi6_0'] < d['rsi12_0']) and
    #                                 (50 <= d['rsi6_0'] <= 70) and not (d['rsi6_-3'] < d['rsi12_-3'] and d['rsi6_-4'] > d['rsi12_-4']),  # 排除3日延续
    #     "buy_score": 0.0,
    #     "sell_score": 1.5
    # },
    # {
    #     "priority": 3,
    #     "signal_name": "死叉延续3日",
    #     "signal_type": "动态金叉/死叉/拐头/企稳",
    #     "condition_func": lambda d: (d['rsi6_-3'] < d['rsi12_-3'] and d['rsi6_-4'] > d['rsi12_-4']) and
    #                                 (d['rsi6_-2'] < d['rsi12_-2']) and (d['rsi6_-1'] < d['rsi12_-1']) and
    #                                 (d['rsi6_0'] < d['rsi12_0']) and (50 <= d['rsi6_0'] <= 70),
    #     "buy_score": 0.0,
    #     "sell_score": 1.8
    # },
    # {
    #     "priority": 3,
    #     "signal_name": "死叉后超卖",
    #     "signal_type": "动态金叉/死叉/拐头/企稳",
    #     "condition_func": lambda d: ((d['rsi6_-2'] < d['rsi12_-2'] and d['rsi6_-3'] > d['rsi12_-3']) or
    #                                  (d['rsi6_-3'] < d['rsi12_-3'] and d['rsi6_-4'] > d['rsi12_-4'])) and
    #                                 (d['rsi6_-1'] < d['rsi12_-1']) and (d['rsi6_0'] < d['rsi12_0']) and
    #                                 (d['rsi6_0'] < 30) and not (d['rsi6_0'] < d['rsi12_0'] and d['rsi6_-1'] > d['rsi12_-1']),  # 排除低位死叉
    #     "buy_score": 1.0,
    #     "sell_score": 0.0
    # },

    # ---------------- 优先级4：静态排列 (权重1.0) ----------------
    # {
    #     "priority": 4,
    #     "signal_name": "最强多头(RSI6<20)",
    #     "signal_type": "静态排列",
    #     "condition_func": lambda d: (d['rsi6_0'] > d['rsi12_0']) and (d['rsi12_0'] > d['rsi24_0']) and
    #                                 (d['rsi6_0'] < 20),
    #     "buy_score": 2.0,
    #     "sell_score": 0.0
    # },
    # {
    #     "priority": 4,
    #     "signal_name": "最强多头(20<=RSI6<=50)",
    #     "signal_type": "静态排列",
    #     "condition_func": lambda d: (d['rsi6_0'] > d['rsi12_0']) and (d['rsi12_0'] > d['rsi24_0']) and
    #                                 (20 <= d['rsi6_0'] <= 50),
    #     "buy_score": 1.5,
    #     "sell_score": 0.0
    # },
    # {
    #     "priority": 4,
    #     "signal_name": "最强多头(50<=RSI6<=70)",
    #     "signal_type": "静态排列",
    #     "condition_func": lambda d: (d['rsi6_0'] > d['rsi12_0']) and (d['rsi12_0'] > d['rsi24_0']) and
    #                                 (50 <= d['rsi6_0'] <= 70),
    #     "buy_score": 1.0,
    #     "sell_score": 0.5
    # },
    # {
    #     "priority": 4,
    #     "signal_name": "最强多头(RSI6>70)",
    #     "signal_type": "静态排列",
    #     "condition_func": lambda d: (d['rsi6_0'] > d['rsi12_0']) and (d['rsi12_0'] > d['rsi24_0']) and
    #                                 (d['rsi6_0'] > 70),
    #     "buy_score": 0.0,
    #     "sell_score": 2.0
    # },
    # {
    #     "priority": 4,
    #     "signal_name": "短多长空(RSI6<30)",
    #     "signal_type": "静态排列",
    #     "condition_func": lambda d: (d['rsi6_0'] > d['rsi12_0']) and (d['rsi12_0'] < d['rsi24_0']) and
    #                                 (d['rsi6_0'] < 30),
    #     "buy_score": 1.5,
    #     "sell_score": 0.0
    # },
    # {
    #     "priority": 4,
    #     "signal_name": "短多长空(30<=RSI6<=70)",
    #     "signal_type": "静态排列",
    #     "condition_func": lambda d: (d['rsi6_0'] > d['rsi12_0']) and (d['rsi12_0'] < d['rsi24_0']) and
    #                                 (30 <= d['rsi6_0'] <= 70),
    #     "buy_score": 0.8,
    #     "sell_score": 0.8
    # },
    # {
    #     "priority": 4,
    #     "signal_name": "短多长空(RSI6>70)",
    #     "signal_type": "静态排列",
    #     "condition_func": lambda d: (d['rsi6_0'] > d['rsi12_0']) and (d['rsi12_0'] < d['rsi24_0']) and
    #                                 (d['rsi6_0'] > 70),
    #     "buy_score": 0.0,
    #     "sell_score": 1.8
    # },
    # {
    #     "priority": 4,
    #     "signal_name": "短空多长(RSI6<30)",
    #     "signal_type": "静态排列",
    #     "condition_func": lambda d: (d['rsi6_0'] < d['rsi12_0']) and (d['rsi12_0'] > d['rsi24_0']) and
    #                                 (d['rsi6_0'] < 30),
    #     "buy_score": 1.2,
    #     "sell_score": 0.0
    # },
    # {
    #     "priority": 4,
    #     "signal_name": "短空多长(30<=RSI6<=70)",
    #     "signal_type": "静态排列",
    #     "condition_func": lambda d: (d['rsi6_0'] < d['rsi12_0']) and (d['rsi12_0'] > d['rsi24_0']) and
    #                                 (30 <= d['rsi6_0'] <= 70),
    #     "buy_score": 0.5,
    #     "sell_score": 1.0
    # },
    # {
    #     "priority": 4,
    #     "signal_name": "短空多长(RSI6>70)",
    #     "signal_type": "静态排列",
    #     "condition_func": lambda d: (d['rsi6_0'] < d['rsi12_0']) and (d['rsi12_0'] > d['rsi24_0']) and
    #                                 (d['rsi6_0'] > 70),
    #     "buy_score": 0.0,
    #     "sell_score": 1.5
    # },
    # {
    #     "priority": 4,
    #     "signal_name": "最强空头(RSI6<20)",
    #     "signal_type": "静态排列",
    #     "condition_func": lambda d: (d['rsi6_0'] < d['rsi12_0']) and (d['rsi12_0'] < d['rsi24_0']) and
    #                                 (d['rsi6_0'] < 20),
    #     "buy_score": 1.5,
    #     "sell_score": 0.0
    # },
    # {
    #     "priority": 4,
    #     "signal_name": "最强空头(20<=RSI6<=50)",
    #     "signal_type": "静态排列",
    #     "condition_func": lambda d: (d['rsi6_0'] < d['rsi12_0']) and (d['rsi12_0'] < d['rsi24_0']) and
    #                                 (20 <= d['rsi6_0'] <= 50),
    #     "buy_score": 0.0,
    #     "sell_score": 1.5
    # },
    # {
    #     "priority": 4,
    #     "signal_name": "最强空头(50<=RSI6<=70)",
    #     "signal_type": "静态排列",
    #     "condition_func": lambda d: (d['rsi6_0'] < d['rsi12_0']) and (d['rsi12_0'] < d['rsi24_0']) and
    #                                 (50 <= d['rsi6_0'] <= 70),
    #     "buy_score": 0.0,
    #     "sell_score": 2.0
    # },
    # {
    #     "priority": 4,
    #     "signal_name": "最强空头(RSI6>70)",
    #     "signal_type": "静态排列",
    #     "condition_func": lambda d: (d['rsi6_0'] < d['rsi12_0']) and (d['rsi12_0'] < d['rsi24_0']) and
    #                                 (d['rsi6_0'] > 70),
    #     "buy_score": 0.0,
    #     "sell_score": 2.0
    # },

    # ---------------- 优先级5：粘合平衡 (权重0.5) ----------------
    # {
    #     "priority": 5,
    #     "signal_name": "完全粘合(RSI6<50)",
    #     "signal_type": "粘合平衡",
    #     "condition_func": lambda d: (abs(d['rsi6_0'] - d['rsi12_0']) < 2) and
    #                                 (abs(d['rsi12_0'] - d['rsi24_0']) < 2) and (d['rsi6_0'] < 50) and
    #                                 # 排除静态排列信号
    #                                 not ((d['rsi6_0'] > d['rsi12_0'] and d['rsi12_0'] > d['rsi24_0']) or
    #                                      (d['rsi6_0'] > d['rsi12_0'] and d['rsi12_0'] < d['rsi24_0']) or
    #                                      (d['rsi6_0'] < d['rsi12_0'] and d['rsi12_0'] > d['rsi24_0']) or
    #                                      (d['rsi6_0'] < d['rsi12_0'] and d['rsi12_0'] < d['rsi24_0'])),
    #     "buy_score": 0.5,
    #     "sell_score": 0.0
    # },
    # {
    #     "priority": 5,
    #     "signal_name": "完全粘合(RSI6>50)",
    #     "signal_type": "粘合平衡",
    #     "condition_func": lambda d: (abs(d['rsi6_0'] - d['rsi12_0']) < 2) and
    #                                 (abs(d['rsi12_0'] - d['rsi24_0']) < 2) and (d['rsi6_0'] > 50) and
    #                                 # 排除静态排列信号
    #                                 not ((d['rsi6_0'] > d['rsi12_0'] and d['rsi12_0'] > d['rsi24_0']) or
    #                                      (d['rsi6_0'] > d['rsi12_0'] and d['rsi12_0'] < d['rsi24_0']) or
    #                                      (d['rsi6_0'] < d['rsi12_0'] and d['rsi12_0'] > d['rsi24_0']) or
    #                                      (d['rsi6_0'] < d['rsi12_0'] and d['rsi12_0'] < d['rsi24_0'])),
    #     "buy_score": 0.0,
    #     "sell_score": 0.5
    # },
    # {
    #     "priority": 5,
    #     "signal_name": "偏多粘合(RSI6<30)",
    #     "signal_type": "粘合平衡",
    #     "condition_func": lambda d: (abs(d['rsi6_0'] - d['rsi12_0']) < 2) and
    #                                 (d['rsi12_0'] > d['rsi24_0']) and (d['rsi6_0'] < 30) and
    #                                 not (d['rsi6_0'] < d['rsi12_0'] and d['rsi12_0'] > d['rsi24_0']),  # 排除短空多长
    #     "buy_score": 1.5,
    #     "sell_score": 0.0
    # },
    # {
    #     "priority": 5,
    #     "signal_name": "偏多粘合(RSI6>70)",
    #     "signal_type": "粘合平衡",
    #     "condition_func": lambda d: (abs(d['rsi6_0'] - d['rsi12_0']) < 2) and
    #                                 (d['rsi12_0'] > d['rsi24_0']) and (d['rsi6_0'] > 70) and
    #                                 not (d['rsi6_0'] < d['rsi12_0'] and d['rsi12_0'] > d['rsi24_0']),  # 排除短空多长
    #     "buy_score": 0.0,
    #     "sell_score": 1.5
    # },
    # {
    #     "priority": 5,
    #     "signal_name": "偏空粘合(RSI6<30)",
    #     "signal_type": "粘合平衡",
    #     "condition_func": lambda d: (abs(d['rsi6_0'] - d['rsi12_0']) < 2) and
    #                                 (d['rsi12_0'] < d['rsi24_0']) and (d['rsi6_0'] < 30) and
    #                                 not (d['rsi6_0'] < d['rsi12_0'] and d['rsi12_0'] < d['rsi24_0']),  # 排除最强空头
    #     "buy_score": 1.2,
    #     "sell_score": 0.0
    # },
    # # 找到RSI_RULES中「偏空粘合(RSI6>70)」的规则，修改condition_func：
    # {
    #     "priority": 5,
    #     "signal_name": "偏空粘合(RSI6>70)",
    #     "signal_type": "粘合平衡",
    #     "condition_func": lambda d: (abs(d['rsi6_0'] - d['rsi12_0']) < 2) and
    #                                 (d['rsi12_0'] < d['rsi24_0']) and (d['rsi6_0'] > 70) and
    #                                 # 新增：同时排除最强空头 + 短多长空
    #                                 not (d['rsi6_0'] < d['rsi12_0'] and d['rsi12_0'] < d['rsi24_0']) and  # 排除最强空头
    #                                 not (d['rsi6_0'] > d['rsi12_0'] and d['rsi12_0'] < d['rsi24_0']),  # 排除短多长空
    #     "buy_score": 0.0,
    #     "sell_score": 1.8
    # },
]

def get_rsi_status(strat, rsi_periods=(6, 12, 24), max_lookback=60, stock_type='中盘股'):
    """
    获取RSI（6,12,24）状态的完整描述（A股实战终极版+盘型参数化）
    核心优化：盘型作为参数传递（默认中盘），无需依赖流通市值数据

    参数:
        strat: 策略实例 (需包含 rsi6/rsi12/rsi24 指标，data包含close/volume字段)
        rsi_periods: RSI周期，默认(6,12,24)
        max_lookback: 最大回溯天数（用于背离/波动率计算）
        stock_type: 股票盘型（可选：大盘股/中盘股/小盘股/微盘股），默认中盘股

    返回:
        dict: 包含盘型、分盘阈值、全量RSI状态的实战信息
    """
    # 校验盘型参数合法性
    valid_stock_types = ['大盘股', '中盘股', '小盘股', '微盘股']
    if stock_type not in valid_stock_types:
        print(f"警告：盘型参数错误（{stock_type}），自动切换为中盘股")
        stock_type = '中盘股'

    # 初始化结果字典
    result = {
        # 盘型相关（参数传入）
        'stock_type': stock_type,  # 大盘/中盘/小盘/微盘
        'rsi_thresholds': {  # 分盘型RSI阈值（超买/超卖/涨幅）
            'over_buy': 70,
            'over_sell': 30,
            'min_rise': 0.5
        },
        # 基础状态
        'over_buy_sell': '未知',  # 超买/超卖/正常区间
        'trend_status': '未知',  # 上涨/下跌/震荡
        'cross_status': '未知',  # 金叉/死叉/无交叉
        'divergence': '未知',  # 顶背离/底背离/无背离
        # 原始数值
        'rsi_6': -1,  # 6日RSI数值
        'rsi_12': -1,  # 12日RSI数值
        'rsi_24': -1,  # 24日RSI数值
        # 核心实战字段
        'rsi6_rise': 0.0,  # RSI6单日涨幅（带小数，精准）
        'avg_volatility': 0.5,  # 近20天RSI6平均波动率（适配不同标的）
        'is_effective_gold': False,  # 有效金叉（A股实战级）
        'is_effective_dead': False,  # 有效死叉（A股实战级）
        'multi_period_resonance': False,  # 多周期共振（6/12/24同步趋势）
        'signal_strength': 0.0  # 信号强度（0-1，越高越可靠）
    }

    # ===================== 1. 分盘型阈值配置（参数化核心） =====================
    # 按传入的盘型配置专属阈值（A股实战标准）
    if stock_type == '大盘股':
        result['rsi_thresholds'] = {'over_buy': 65, 'over_sell': 35, 'min_rise': 0.3}
    elif stock_type == '中盘股':  # 默认
        result['rsi_thresholds'] = {'over_buy': 70, 'over_sell': 30, 'min_rise': 0.5}
    elif stock_type == '小盘股':
        result['rsi_thresholds'] = {'over_buy': 75, 'over_sell': 25, 'min_rise': 0.8}
    elif stock_type == '微盘股':
        result['rsi_thresholds'] = {'over_buy': 80, 'over_sell': 20, 'min_rise': 1.2}

    # ===================== 2. 基础数据校验与异常处理 =====================
    try:
        rsi_data = {
            6: getattr(strat, 'rsi6', []),
            12: getattr(strat, 'rsi12', []),
            24: getattr(strat, 'rsi24', [])
        }
        for period in rsi_periods:
            if len(rsi_data[period]) < 2:
                print(f"警告：{period}日RSI数据不足（仅{len(rsi_data[period])}个），返回默认值")
                return result

        # 提取核心数据
        rsi6 = rsi_data[6][0]
        rsi6_prev = rsi_data[6][-1]
        rsi12 = rsi_data[12][0]
        rsi12_prev = rsi_data[12][-1]
        rsi24 = rsi_data[24][0]

        # 价格/成交量校验（回测/实盘兼容）
        has_price = hasattr(strat.data, 'close') and len(strat.data.close) >= max_lookback
        has_volume = hasattr(strat.data, 'volume') and len(strat.data.volume) >= 2

        # 异常值过滤（RSI值应在0-100之间）
        if not (0 <= rsi6 <= 100 and 0 <= rsi12 <= 100 and 0 <= rsi24 <= 100):
            print(f"警告：RSI值异常（6:{rsi6},12:{rsi12},24:{rsi24}），返回默认值")
            return result

    except Exception as e:
        print(f"数据校验失败：{str(e)}")
        return result

    # ===================== 3. 核心计算逻辑（分盘型适配） =====================
    # 3.1 记录原始RSI数值（保留1位小数）
    result['rsi_6'] = round(rsi6, 1)
    result['rsi_12'] = round(rsi12, 1)
    result['rsi_24'] = round(rsi24, 1)

    # 3.2 计算RSI6单日涨幅（精准到0.1）
    result['rsi6_rise'] = round(rsi6 - rsi6_prev, 1)

    # 3.3 计算近20天RSI6平均波动率（动态适配标的）
    lookback_len = min(20, len(rsi_data[6]) - 1)
    if lookback_len >= 5:  # 至少5个数据点才计算波动率
        vol_list = [abs(rsi_data[6][-i] - rsi_data[6][-(i + 1)]) for i in range(1, lookback_len + 1)]
        result['avg_volatility'] = round(np.mean(vol_list), 1)  # numpy计算更高效

    # ===================== 4. 状态判断逻辑（完全分盘型） =====================
    # 4.1 超买超卖判断（用分盘阈值）
    over_buy = result['rsi_thresholds']['over_buy']
    over_sell = result['rsi_thresholds']['over_sell']
    if rsi6 > over_buy:
        result['over_buy_sell'] = '超买'
    elif rsi6 < over_sell:
        result['over_buy_sell'] = '超卖'
    else:
        result['over_buy_sell'] = '正常区间'

    # 4.2 趋势判断（动态波动率 + 分盘最小涨幅）
    min_rise = result['rsi_thresholds']['min_rise']
    if abs(result['rsi6_rise']) < max(result['avg_volatility'], min_rise):
        result['trend_status'] = '震荡'
    elif result['rsi6_rise'] > 0:
        result['trend_status'] = '上涨'
    else:
        result['trend_status'] = '下跌'

    # 4.3 交叉判断（6日VS12日，核心交叉）
    if rsi6_prev < rsi12_prev and rsi6 > rsi12:
        result['cross_status'] = '金叉'
    elif rsi6_prev > rsi12_prev and rsi6 < rsi12:
        result['cross_status'] = '死叉'
    else:
        result['cross_status'] = '无交叉'

    # 4.4 多周期共振判断（分盘型涨幅要求）
    rsi12_rise = round(rsi12 - rsi12_prev, 1) if len(rsi_data[12]) >= 2 else 0
    rsi24_rise = round(rsi24 - rsi_data[24][-1], 1) if len(rsi_data[24]) >= 2 else 0

    # 分盘共振涨幅要求：大盘股宽松，小盘股严格
    if stock_type == '大盘股':
        reso_6 = 0.4 * result['avg_volatility']
        reso_12 = 0.3 * result['avg_volatility']
        reso_24 = 0.2 * result['avg_volatility']
    elif stock_type == '中盘股':
        reso_6 = 0.5 * result['avg_volatility']
        reso_12 = 0.4 * result['avg_volatility']
        reso_24 = 0.3 * result['avg_volatility']
    else:  # 小盘/微盘
        reso_6 = 0.6 * result['avg_volatility']
        reso_12 = 0.5 * result['avg_volatility']
        reso_24 = 0.4 * result['avg_volatility']

    if (np.sign(result['rsi6_rise']) == np.sign(rsi12_rise) == np.sign(rsi24_rise) and
            abs(result['rsi6_rise']) >= reso_6 and
            abs(rsi12_rise) >= reso_12 and
            abs(rsi24_rise) >= reso_24):
        result['multi_period_resonance'] = True

    # 4.5 有效金叉判断（分盘型权重评分）
    if result['cross_status'] == '金叉':
        score = 0
        # 分盘型权重调整：大盘股侧重长期趋势，小盘股侧重量能
        if stock_type == '大盘股':
            # 大盘股：长期趋势权重更高（25分），量能权重更低（15分）
            if result['over_buy_sell'] != '超买': score += 30  # 位置
            if abs(result['rsi6_rise']) >= result['avg_volatility'] * 0.7: score += 25  # 涨幅
            if has_volume and strat.data.volume[0] >= strat.data.volume[-1] * 1.15: score += 15  # 量能
            if rsi24 > over_sell + 5: score += 25  # 长期趋势
            if result['multi_period_resonance']: score += 5  # 共振
            result['is_effective_gold'] = score >= 70  # 大盘股阈值更低
        elif stock_type in ['小盘股', '微盘股']:
            # 小盘股：量能权重更高（25分），共振权重更高（15分）
            if result['over_buy_sell'] != '超买': score += 25  # 位置
            if abs(result['rsi6_rise']) >= result['avg_volatility'] * 0.9: score += 25  # 涨幅
            if has_volume and strat.data.volume[0] >= strat.data.volume[-1] * 1.3: score += 25  # 量能
            if rsi24 > over_sell: score += 10  # 长期趋势
            if result['multi_period_resonance']: score += 15  # 共振
            result['is_effective_gold'] = score >= 85  # 小盘股阈值更高
        else:  # 中盘股（默认）
            if result['over_buy_sell'] != '超买': score += 30
            if abs(result['rsi6_rise']) >= result['avg_volatility'] * 0.8: score += 25
            if has_volume and strat.data.volume[0] >= strat.data.volume[-1] * 1.2: score += 20
            if rsi24 > 30: score += 15
            if result['multi_period_resonance']: score += 10
            result['is_effective_gold'] = score >= 80  # 中盘股默认阈值
        result['signal_strength'] = round(score / 100, 2)  # 转换为0-1的强度值

    # 4.6 有效死叉判断（对称分盘型）
    elif result['cross_status'] == '死叉':
        score = 0
        if stock_type == '大盘股':
            if result['over_buy_sell'] != '超卖': score += 30
            if abs(result['rsi6_rise']) >= result['avg_volatility'] * 0.7: score += 25
            if has_volume and strat.data.volume[0] >= strat.data.volume[-1] * 1.15: score += 15
            if rsi24 < over_buy - 5: score += 25
            if result['multi_period_resonance']: score += 5
            result['is_effective_dead'] = score >= 70
        elif stock_type in ['小盘股', '微盘股']:
            if result['over_buy_sell'] != '超卖': score += 25
            if abs(result['rsi6_rise']) >= result['avg_volatility'] * 0.9: score += 25
            if has_volume and strat.data.volume[0] >= strat.data.volume[-1] * 1.3: score += 25
            if rsi24 < over_buy: score += 10
            if result['multi_period_resonance']: score += 15
            result['is_effective_dead'] = score >= 85
        else:  # 中盘股
            if result['over_buy_sell'] != '超卖': score += 30
            if abs(result['rsi6_rise']) >= result['avg_volatility'] * 0.8: score += 25
            if has_volume and strat.data.volume[0] >= strat.data.volume[-1] * 1.2: score += 20
            if rsi24 < 70: score += 15
            if result['multi_period_resonance']: score += 10
            result['is_effective_dead'] = score >= 80
        result['signal_strength'] = round(score / 100, 2)

    # 4.7 背离判断（分盘型阈值）
    try:
        if has_price:
            lookback_actual = min(max_lookback, len(strat.data.close), len(rsi_data[6]))
            close_arr = np.array([strat.data.close[-i] for i in range(lookback_actual)])
            rsi6_arr = np.array([rsi_data[6][-i] for i in range(lookback_actual)])

            close_max = close_arr.max()
            rsi6_max = rsi6_arr.max()
            # 分盘型背离差值要求：大盘股≥0.5点，小盘股≥2点
            divergence_diff = 0.5 if stock_type == '大盘股' else 2.0

            if (strat.data.close[0] == close_max and
                    rsi6 < rsi6_max - divergence_diff
                    and rsi6 > over_buy):
                result['divergence'] = '顶背离'
            elif (strat.data.close[0] == close_arr.min() and
                  rsi6 > rsi6_arr.min() + divergence_diff
                  and rsi6 < over_sell):
                result['divergence'] = '底背离'
            else:
                result['divergence'] = '无背离'
        else:
            result['divergence'] = '无背离'
    except Exception as e:
        print(f"背离判断失败：{str(e)}")
        result['divergence'] = '无背离'

    return result

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
            if rule['condition_func'](strategy):
                matched_signals.append({
                    "rule": rule,
                    "weight": rule['weight'],
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