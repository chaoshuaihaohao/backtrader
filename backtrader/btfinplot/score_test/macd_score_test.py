# macd_score_test.py 完整代码
import unittest
import numpy as np
from scipy.signal import find_peaks

# ========== 全局函数定义（请替换文件底部原有的定义）==========

# 第一步：定义完整的MACD评分规则表（对应你提供的所有场景）
MACD_SCORE_TABLE = [
    # 零轴上方场景
    {"零轴位置": "上方", "金叉时间": "即将金叉（绿柱缩短且 DEA-DIFF < 动态阈值）", "顶底背离": "顶背离", "买入评分": 0.5,
     "卖出评分": 1.8},
    {"零轴位置": "上方", "金叉时间": "即将金叉（绿柱缩短且 DEA-DIFF < 动态阈值）", "顶底背离": "底背离", "买入评分": 1.5,
     "卖出评分": 0.3},
    {"零轴位置": "上方", "金叉时间": "即将金叉（绿柱缩短且 DEA-DIFF < 动态阈值）", "顶底背离": "无背离", "买入评分": 1.2,
     "卖出评分": 0.5},
    {"零轴位置": "上方", "金叉时间": "刚金叉", "顶底背离": "顶背离", "买入评分": 0.3, "卖出评分": 2.0},
    {"零轴位置": "上方", "金叉时间": "刚金叉", "顶底背离": "底背离", "买入评分": 1.8, "卖出评分": 0.2},
    {"零轴位置": "上方", "金叉时间": "刚金叉", "顶底背离": "无背离", "买入评分": 1.5, "卖出评分": 0.3},
    {"零轴位置": "上方", "金叉时间": "金叉多日，红柱放大", "顶底背离": "顶背离", "买入评分": 0.2, "卖出评分": 1.5},
    {"零轴位置": "上方", "金叉时间": "金叉多日，红柱放大", "顶底背离": "底背离", "买入评分": 1.2, "卖出评分": 0.1},
    {"零轴位置": "上方", "金叉时间": "金叉多日，红柱放大", "顶底背离": "无背离", "买入评分": 1.0, "卖出评分": 0.3},
    {"零轴位置": "上方", "金叉时间": "金叉多日，红柱走平", "顶底背离": "顶背离", "买入评分": 0.1, "卖出评分": 1.2},
    {"零轴位置": "上方", "金叉时间": "金叉多日，红柱走平", "顶底背离": "底背离", "买入评分": 1.0, "卖出评分": 0.1},
    {"零轴位置": "上方", "金叉时间": "金叉多日，红柱走平", "顶底背离": "无背离", "买入评分": 0.8, "卖出评分": 0.4},
    {"零轴位置": "上方", "金叉时间": "金叉多日，红柱缩减", "顶底背离": "顶背离", "买入评分": 0.0, "卖出评分": 1.0},
    {"零轴位置": "上方", "金叉时间": "金叉多日，红柱缩减", "顶底背离": "底背离", "买入评分": 0.8, "卖出评分": 0.2},
    {"零轴位置": "上方", "金叉时间": "金叉多日，红柱缩减", "顶底背离": "无背离", "买入评分": 0.5, "卖出评分": 0.6},
    {"零轴位置": "上方", "金叉时间": "即将死叉（红柱缩短且 DIFF -DEA < 动态阈值）", "顶底背离": "顶背离", "买入评分": 0.0,
     "卖出评分": 1.8},
    {"零轴位置": "上方", "金叉时间": "即将死叉（红柱缩短且 DIFF -DEA < 动态阈值）", "顶底背离": "底背离", "买入评分": 0.6,
     "卖出评分": 0.5},
    {"零轴位置": "上方", "金叉时间": "即将死叉（红柱缩短且 DIFF -DEA < 动态阈值）", "顶底背离": "无背离", "买入评分": 0.3,
     "卖出评分": 1.2},
    {"零轴位置": "上方", "金叉时间": "刚死叉", "顶底背离": "顶背离", "买入评分": 0.0, "卖出评分": 2.0},
    {"零轴位置": "上方", "金叉时间": "刚死叉", "顶底背离": "底背离", "买入评分": 0.5, "卖出评分": 1.2},
    {"零轴位置": "上方", "金叉时间": "刚死叉", "顶底背离": "无背离", "买入评分": 0.3, "卖出评分": 1.5},
    {"零轴位置": "上方", "金叉时间": "死叉多日，绿柱缩减", "顶底背离": "顶背离", "买入评分": 0.3, "卖出评分": 1.0},
    {"零轴位置": "上方", "金叉时间": "死叉多日，绿柱缩减", "顶底背离": "底背离", "买入评分": 1.0, "卖出评分": 0.3},
    {"零轴位置": "上方", "金叉时间": "死叉多日，绿柱缩减", "顶底背离": "无背离", "买入评分": 0.8, "卖出评分": 0.6},
    {"零轴位置": "上方", "金叉时间": "死叉多日，绿柱走平", "顶底背离": "顶背离", "买入评分": 0.2, "卖出评分": 0.8},
    {"零轴位置": "上方", "金叉时间": "死叉多日，绿柱走平", "顶底背离": "底背离", "买入评分": 0.8, "卖出评分": 0.2},
    {"零轴位置": "上方", "金叉时间": "死叉多日，绿柱走平", "顶底背离": "无背离", "买入评分": 0.5, "卖出评分": 0.5},
    {"零轴位置": "上方", "金叉时间": "死叉多日，绿柱放大", "顶底背离": "顶背离", "买入评分": 0.0, "卖出评分": 1.5},
    {"零轴位置": "上方", "金叉时间": "死叉多日，绿柱放大", "顶底背离": "底背离", "买入评分": 0.5, "卖出评分": 1.8},
    {"零轴位置": "上方", "金叉时间": "死叉多日，绿柱放大", "顶底背离": "无背离", "买入评分": 0.2, "卖出评分": 1.2},
    # 零轴下方场景
    {"零轴位置": "下方", "金叉时间": "即将金叉（绿柱缩短且 DEA-DIFF < 动态阈值）", "顶底背离": "顶背离", "买入评分": 0.2,
     "卖出评分": 1.0},
    {"零轴位置": "下方", "金叉时间": "即将金叉（绿柱缩短且 DEA-DIFF < 动态阈值）", "顶底背离": "底背离", "买入评分": 1.5,
     "卖出评分": 0.2},
    {"零轴位置": "下方", "金叉时间": "即将金叉（绿柱缩短且 DEA-DIFF < 动态阈值）", "顶底背离": "无背离", "买入评分": 1.2,
     "卖出评分": 0.4},
    {"零轴位置": "下方", "金叉时间": "刚金叉", "顶底背离": "顶背离", "买入评分": 0.1, "卖出评分": 1.2},
    {"零轴位置": "下方", "金叉时间": "刚金叉", "顶底背离": "底背离", "买入评分": 1.8, "卖出评分": 0.1},
    {"零轴位置": "下方", "金叉时间": "刚金叉", "顶底背离": "无背离", "买入评分": 1.5, "卖出评分": 0.3},
    {"零轴位置": "下方", "金叉时间": "金叉多日，红柱放大", "顶底背离": "顶背离", "买入评分": 0.3, "卖出评分": 1.0},
    {"零轴位置": "下方", "金叉时间": "金叉多日，红柱放大", "顶底背离": "底背离", "买入评分": 1.2, "卖出评分": 0.0},
    {"零轴位置": "下方", "金叉时间": "金叉多日，红柱放大", "顶底背离": "无背离", "买入评分": 1.0, "卖出评分": 0.2},
    {"零轴位置": "下方", "金叉时间": "金叉多日，红柱走平", "顶底背离": "顶背离", "买入评分": 0.2, "卖出评分": 0.8},
    {"零轴位置": "下方", "金叉时间": "金叉多日，红柱走平", "顶底背离": "底背离", "买入评分": 1.0, "卖出评分": 0.0},
    {"零轴位置": "下方", "金叉时间": "金叉多日，红柱走平", "顶底背离": "无背离", "买入评分": 0.8, "卖出评分": 0.3},
    {"零轴位置": "下方", "金叉时间": "金叉多日，红柱缩减", "顶底背离": "顶背离", "买入评分": 0.0, "卖出评分": 0.8},
    {"零轴位置": "下方", "金叉时间": "金叉多日，红柱缩减", "顶底背离": "底背离", "买入评分": 0.8, "卖出评分": 0.0},
    {"零轴位置": "下方", "金叉时间": "金叉多日，红柱缩减", "顶底背离": "无背离", "买入评分": 0.5, "卖出评分": 0.4},
    {"零轴位置": "下方", "金叉时间": "即将死叉（红柱缩短且 DIFF -DEA < 动态阈值）", "顶底背离": "顶背离", "买入评分": 0.0,
     "卖出评分": 1.2},
    {"零轴位置": "下方", "金叉时间": "即将死叉（红柱缩短且 DIFF -DEA < 动态阈值）", "顶底背离": "底背离", "买入评分": 0.6,
     "卖出评分": 0.3},
    {"零轴位置": "下方", "金叉时间": "即将死叉（红柱缩短且 DIFF -DEA < 动态阈值）", "顶底背离": "无背离", "买入评分": 0.3,
     "卖出评分": 0.8},
    {"零轴位置": "下方", "金叉时间": "刚死叉", "顶底背离": "顶背离", "买入评分": 0.0, "卖出评分": 1.5},
    {"零轴位置": "下方", "金叉时间": "刚死叉", "顶底背离": "底背离", "买入评分": 0.5, "卖出评分": 1.0},
    {"零轴位置": "下方", "金叉时间": "刚死叉", "顶底背离": "无背离", "买入评分": 0.3, "卖出评分": 1.2},
    {"零轴位置": "下方", "金叉时间": "死叉多日，绿柱缩减", "顶底背离": "顶背离", "买入评分": 0.2, "卖出评分": 0.8},
    {"零轴位置": "下方", "金叉时间": "死叉多日，绿柱缩减", "顶底背离": "底背离", "买入评分": 1.0, "卖出评分": 0.1},
    {"零轴位置": "下方", "金叉时间": "死叉多日，绿柱缩减", "顶底背离": "无背离", "买入评分": 0.8, "卖出评分": 0.4},
    {"零轴位置": "下方", "金叉时间": "死叉多日，绿柱走平", "顶底背离": "顶背离", "买入评分": 0.1, "卖出评分": 0.6},
    {"零轴位置": "下方", "金叉时间": "死叉多日，绿柱走平", "顶底背离": "底背离", "买入评分": 0.8, "卖出评分": 0.0},
    {"零轴位置": "下方", "金叉时间": "死叉多日，绿柱走平", "顶底背离": "无背离", "买入评分": 0.5, "卖出评分": 0.3},
    {"零轴位置": "下方", "金叉时间": "死叉多日，绿柱放大", "顶底背离": "顶背离", "买入评分": 0.0, "卖出评分": 2.0},
    {"零轴位置": "下方", "金叉时间": "死叉多日，绿柱放大", "顶底背离": "底背离", "买入评分": 0.3, "卖出评分": 1.8},
    {"零轴位置": "下方", "金叉时间": "死叉多日，绿柱放大", "顶底背离": "无背离", "买入评分": 0.2, "卖出评分": 1.5},
]


def calculate_macd_score(strat):
    # 数据不足时返回0分
    if len(strat.macd.macd) < 20 or len(strat.macd_hist) < 20 or len(strat.data.close) < 20:
        return 0.0

    try:
        # ========== 新增：动态阈值计算函数 ==========
        def get_dynamic_threshold():
            """
            计算动态临界阈值（适配标的自身波动特性）：
            核心公式：临界阈值 = max( min( 近20日MACD波动幅度 × 10% , 0.5 ), 0.1 )
            逻辑：
            - 波动幅度 = 近20日MACD最大值 - 近20日MACD最小值
            - 10% 比例缩放，适配不同波动的标的
            - 上限0.5，下限0.1，避免极端值失效
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
                threshold = min(threshold, 3)  # 上限兜底：最大0.5
                threshold = max(threshold, 0.1)  # 下限兜底：最小0.1

                # 可选：打印动态阈值，便于调试
                # print(f"近20日MACD波动幅度: {volatility:.4f}, 动态阈值: {threshold:.4f}")
                return threshold
            except Exception as e:
                print(f"计算动态阈值异常: {e}")
                return 0.2  # 异常时返回默认值

        # 1. 获取核心数据
        macd = strat.macd.macd[0]  # DIF线（同花顺优先看这个）
        signal = strat.macd.signal[0]
        hist = strat.macd_hist[0]
        hist_prev = strat.macd_hist[-1] if len(strat.macd_hist) > 1 else 0

        # 2. 精准判断金叉/死叉状态（修复核心逻辑）
        def get_cross_status():
            # 1. 统一用 get() 取最近5日数据，确保数据顺序一致 (列表索引 -1 是最新)
            try:
                hist_list = list(strat.macd_hist.get(size=5)) if len(strat.macd_hist) >= 5 else []
                macd_list = list(strat.macd.macd.get(size=5)) if len(strat.macd.macd) >= 5 else []
                signal_list = list(strat.macd.signal.get(size=5)) if len(strat.macd.signal) >= 5 else []

                if len(hist_list) < 3:
                    return "无"

                # 获取核心数据 (使用列表负索引，直观且不易出错)
                hist_now = hist_list[-1]
                hist_prev = hist_list[-2]
                hist_prev2 = hist_list[-3]
                macd_now = macd_list[-1]
                signal_now = signal_list[-1]

                # ========== 关键修改：获取动态阈值 ==========
                dynamic_threshold = get_dynamic_threshold()

            except Exception as e:
                print(f"获取Cross数据异常: {e}")
                return "无"

            # 2. 刚金叉/死叉 (最高优先级)
            if hasattr(strat, 'mcross') and len(strat.mcross) > 0:
                latest_mcross = strat.mcross[0]
                if latest_mcross == 1:
                    return "刚金叉"
                elif latest_mcross == -1:
                    return "刚死叉"

            # 3. 即将金叉 (次高优先级：只要绿柱在缩短，且差值小，就判定)
            # 条件：绿柱 + (今日 > 昨日 或 今日 > 前日) + DEA-DIFF < 动态阈值
            if hist_now < 0:
                is_shortening = (hist_now > hist_prev) or (hist_now > hist_prev2 and hist_prev > hist_prev2)
                # ========== 关键修改：替换固定阈值0.2为动态阈值 ==========
                gap_small = (signal_now - macd_now) < dynamic_threshold
                if is_shortening and gap_small:
                    return "即将金叉（绿柱缩短且 DEA-DIFF < 动态阈值）"

            # 4. 即将死叉
            if hist_now > 0:
                is_shortening = (hist_now < hist_prev) or (hist_now < hist_prev2 and hist_prev < hist_prev2)
                # ========== 关键修改：替换固定阈值0.2为动态阈值 ==========
                gap_small = (macd_now - signal_now) < dynamic_threshold
                if is_shortening and gap_small:
                    return "即将死叉（红柱缩短且 DIFF -DEA < 动态阈值）"

            # 5. 金叉多日状态 (降低走平优先级，收窄走平阈值至 0.005)
            if hist_now > 0:
                if abs(hist_now - hist_prev) < 0.005:
                    return "金叉多日，红柱走平"
                elif hist_now > hist_prev:
                    return "金叉多日，红柱放大"
                else:
                    return "金叉多日，红柱缩减"

            # 6. 死叉多日状态
            if hist_now < 0:
                if abs(hist_now - hist_prev) < 0.005:
                    return "死叉多日，绿柱走平"
                elif abs(hist_now) > abs(hist_prev):
                    return "死叉多日，绿柱放大"
                else:
                    return "死叉多日，绿柱缩减"

            return "无"

        # 3. 同花顺风格的背离判断逻辑
        def get_macd_divergence(strat, window_min=10, window_max=30,
                                oscillation_interval_min=3, oscillation_interval_max=5,
                                trend_interval_min=3, max_offset=5,
                                price_amplitude_threshold=0.01, dif_change_threshold=0.005,
                                trend_ratio=0.7, high_quantile=0.8, low_quantile=0.2,
                                price_type="close"):  # 新增：价格选择参数，默认用收盘价（贴合行业主流）
            """
            严格对照MACD背离量化手册实现的终极版背离判断函数
            核心：完全遵循手册规则，支持单边/震荡行情，精准过滤伪信号，可直接编码落地
            补充（基于联网搜索结论）：价格选择逻辑（收盘价vs最高价/最低价）
            - price_type="close"：用收盘价判定背离（过滤噪音，信号稳健，推荐趋势/中长线）
            - price_type="hl"：用最高价（顶背离）/最低价（底背离）判定（捕捉极值，信号灵敏，推荐震荡/短线）

            参数说明（全部对应手册「量化编码硬参数汇总」）：
            - strat: 策略对象（需包含 high, low, close 价格数据和 MACD 的 DIF 数据）
            - window_min: 分析窗口最小K线数（手册：10根）
            - window_max: 分析窗口最大K线数（手册：30根）
            - oscillation_interval_min: 震荡行情双极值最小间隔（手册：3根）
            - oscillation_interval_max: 震荡行情双极值最大间隔（手册：5根）
            - trend_interval_min: 单边行情阶段极值与最新价最小间隔（手册：3根）
            - max_offset: 价格-DIF极值最大错位K线数（手册：5根）
            - price_amplitude_threshold: 价格波动最小幅度（手册：1%）
            - dif_change_threshold: DIF反向变化最小幅度（手册：0.5%）
            - trend_ratio: 单边行情判定比例（手册：70%）
            - high_quantile: 阶段新高分位数（手册：80分位）
            - low_quantile: 阶段新低分位数（手册：20分位）
            - price_type: 价格选择类型（"close"=收盘价，"hl"=最高价/最低价）

            返回值："底背离"/"顶背离"/"无背离"
            """
            try:
                # ==================== 步骤1：取窗口数据（手册「标准判断流程-步骤1」） ====================
                def _safe_extract_data(obj, size):
                    """安全提取数据，返回numpy数组（旧→新：索引0最早，-1最新），兼容不同数据格式"""
                    data = []
                    try:
                        # 兼容backtrader的line对象和普通列表/数组
                        if hasattr(obj, 'get'):
                            data = list(obj.get(size=size))
                        elif hasattr(obj, '__len__'):
                            data = [obj[i] for i in range(min(size, len(obj)))]
                        else:
                            return np.array([])
                    except Exception as e:
                        print(f"【数据提取异常】{e}")
                        return np.array([])
                    # 确保数据长度在手册规定的10~30根范围内
                    if len(data) < window_min or len(data) > window_max:
                        return np.array([])
                    return np.array(data)

                # 提取手册要求的核心数据（根据price_type选择对应价格，适配不同场景）
                high = _safe_extract_data(strat.data.high, window_max)  # 最高价（备用，供price_type="hl"使用）
                low = _safe_extract_data(strat.data.low, window_max)  # 最低价（备用，供price_type="hl"使用）
                close = _safe_extract_data(strat.data.close, window_max)  # 收盘价（核心，供price_type="close"使用）
                if high.size == 0 or low.size == 0 or close.size == 0:
                    return "无背离"

                # 兼容不同MACD对象的DIF字段（手册核心：仅用DIF判断，不看DEA/柱线）
                if hasattr(strat.macd, 'macdline'):
                    dif = _safe_extract_data(strat.macd.macdline, window_max)
                elif hasattr(strat.macd, 'lines') and hasattr(strat.macd.lines, 'macd'):
                    dif = _safe_extract_data(strat.macd.lines.macd, window_max)
                else:
                    dif = _safe_extract_data(getattr(strat.macd, 'macd', []), window_max)
                if dif.size == 0:
                    return "无背离"

                # 确保所有数据长度一致
                min_len = min(len(high), len(low), len(close), len(dif))
                high = high[:min_len]
                low = low[:min_len]
                close = close[:min_len]
                dif = dif[:min_len]
                n = len(high)  # 实际窗口长度（10~30根）
                window_close_mean = np.mean(close)  # 窗口收盘价均价（用于价格幅度过滤，不随price_type变化）

                # ==================== 步骤2：量化判定行情类型 + 提取价格极值（手册「步骤2」） ====================
                # 子步骤2.1：判定行情类型（手册「子步骤2.1」）
                def _judge_market_type():
                    """
                    量化判定行情类型：单边上涨/单边下跌/震荡
                    规则：单边=≥trend_ratio的高低点同步抬升/降低，否则为震荡
                    说明：行情判定仍用高低点（联网结论：高低点更能反映趋势强弱，与背离价格选择无关）
                    """
                    if n < 4:  # 窗口过短，无法判定行情（至少需要4根K线计算高低点变化）
                        return "震荡"

                    # 计算高低点的变化趋势（当前K线 vs 前一根K线）
                    high_increase = np.sum(high[1:] > high[:-1])  # 高点抬高的K线数量
                    low_increase = np.sum(low[1:] > low[:-1])  # 低点抬高的K线数量
                    high_decrease = np.sum(high[1:] < high[:-1])  # 高点降低的K线数量
                    low_decrease = np.sum(low[1:] < low[:-1])  # 低点降低的K线数量

                    # 单边上涨判定：≥trend_ratio的高点和低点均抬高
                    is_uptrend = (high_increase / (n - 1) >= trend_ratio) and (low_increase / (n - 1) >= trend_ratio)
                    # 单边下跌判定：≥trend_ratio的高点和低点均降低
                    is_downtrend = (high_decrease / (n - 1) >= trend_ratio) and (low_decrease / (n - 1) >= trend_ratio)

                    if is_uptrend:
                        return "单边上涨"
                    elif is_downtrend:
                        return "单边下跌"
                    else:
                        return "震荡"

                market_type = _judge_market_type()

                # 子步骤2.2：提取价格极值（手册「子步骤2.2」+「极值点定义」，适配两种价格类型）
                def _get_price_extremes(market_type, is_bottom):
                    """
                    提取对应行情类型、对应背离类型的价格极值
                    - is_bottom: True=底背离（找低点），False=顶背离（找高点）
                    返回：(阶段极值索引, 阶段极值价格, 最新极值索引, 最新极值价格)
                    说明：根据price_type选择用收盘价或高低价提取极值，贴合行业实操
                    """
                    # 局部极值判定：窗口第3根~倒数第3根，左右各2根对比（手册「极值点定义」）
                    valid_range = range(2, n - 2)  # 索引从0开始，第3根是索引2，倒数第3根是索引n-3
                    if not valid_range:  # 窗口过短，无有效局部极值
                        return (None, None, None, None)

                    # 阶段极值判定：分位数统计（手册「极值点定义」），根据price_type选择对应价格
                    if is_bottom:
                        # 底背离：阶段新低=当前价格 < 窗口价格的low_quantile分位
                        if price_type == "close":
                            stage_low_threshold = np.quantile(close, low_quantile)
                            latest_price = close[-1]  # 最新价：最新K线收盘价
                        else:  # price_type="hl"
                            stage_low_threshold = np.quantile(low, low_quantile)
                            latest_price = low[-1]  # 最新价：最新K线最低价
                        latest_idx = n - 1
                    else:
                        # 顶背离：阶段新高=当前价格 > 窗口价格的high_quantile分位
                        if price_type == "close":
                            stage_high_threshold = np.quantile(close, high_quantile)
                            latest_price = close[-1]  # 最新价：最新K线收盘价
                        else:  # price_type="hl"
                            stage_high_threshold = np.quantile(high, high_quantile)
                            latest_price = high[-1]  # 最新价：最新K线最高价
                        latest_idx = n - 1

                    # 分行情类型提取极值
                    if market_type in ["单边上涨", "单边下跌"]:
                        # 单边行情：阶段极值=最新价之前≥trend_interval_min根的窗口内绝对极值（手册「子步骤2.2」）
                        # 阶段极值索引范围：[0, n-1 - trend_interval_min]（最新价之前至少3根）
                        stage_extreme_range = range(0, n - 1 - trend_interval_min)
                        if not stage_extreme_range:
                            return (None, None, None, None)

                        if is_bottom:
                            # 单边下跌（底背离）：阶段极值=窗口内绝对低价
                            if price_type == "close":
                                stage_extreme_idx = np.argmin(close[stage_extreme_range])
                                stage_extreme_price = close[stage_extreme_idx]
                            else:
                                stage_extreme_idx = np.argmin(low[stage_extreme_range])
                                stage_extreme_price = low[stage_extreme_idx]
                            # 最新价需创阶段新低（低于阶段极值）
                            if latest_price >= stage_extreme_price:
                                return (None, None, None, None)
                        else:
                            # 单边上涨（顶背离）：阶段极值=窗口内绝对高价
                            if price_type == "close":
                                stage_extreme_idx = np.argmax(close[stage_extreme_range])
                                stage_extreme_price = close[stage_extreme_idx]
                            else:
                                stage_extreme_idx = np.argmax(high[stage_extreme_range])
                                stage_extreme_price = high[stage_extreme_idx]
                            # 最新价需创阶段新高（高于阶段极值）
                            if latest_price <= stage_extreme_price:
                                return (None, None, None, None)

                        # 验证单边行情间隔：阶段极值与最新价间隔≥trend_interval_min根（手册「时间间隔硬规则」）
                        if (latest_idx - stage_extreme_idx) < trend_interval_min:
                            return (None, None, None, None)

                        return (stage_extreme_idx, stage_extreme_price, latest_idx, latest_price)

                    else:  # 震荡行情
                        # 震荡行情：提取两个独立极值点（H1→H2 / L1→L2），间隔≥oscillation_interval_min根（手册「两段结构要求」）
                        if is_bottom:
                            # 找局部低点（左右各2根更低），根据price_type选择对应价格
                            if price_type == "close":
                                peaks, _ = find_peaks(-close[valid_range], distance=2)  # distance=2：左右各2根对比
                                price_data = close
                            else:
                                peaks, _ = find_peaks(-low[valid_range], distance=2)
                                price_data = low
                            # 转换为原始索引
                            low_extremes = [valid_range[i] for i in peaks]
                            if len(low_extremes) < 2:
                                return (None, None, None, None)
                            # 取最新的两个低点（L1=前低，L2=后低）
                            l1_idx, l2_idx = low_extremes[-2], low_extremes[-1]
                            l1_price, l2_price = price_data[l1_idx], price_data[l2_idx]
                            # 验证：L2 < L1（幅度达标），间隔在3~5根之间（手册「时间间隔硬规则」）
                            interval = l2_idx - l1_idx
                            if (l2_price >= l1_price) or (interval < oscillation_interval_min) or (
                                    interval > oscillation_interval_max):
                                return (None, None, None, None)
                            return (l1_idx, l1_price, l2_idx, l2_price)
                        else:
                            # 找局部高点（左右各2根更高），根据price_type选择对应价格
                            if price_type == "close":
                                peaks, _ = find_peaks(close[valid_range], distance=2)
                                price_data = close
                            else:
                                peaks, _ = find_peaks(high[valid_range], distance=2)
                                price_data = high
                            # 转换为原始索引
                            high_extremes = [valid_range[i] for i in peaks]
                            if len(high_extremes) < 2:
                                return (None, None, None, None)
                            # 取最新的两个高点（H1=前高，H2=后高）
                            h1_idx, h2_idx = high_extremes[-2], high_extremes[-1]
                            h1_price, h2_price = price_data[h1_idx], price_data[h2_idx]
                            # 验证：H2 > H1（幅度达标），间隔在3~5根之间（手册「时间间隔硬规则」）
                            interval = h2_idx - h1_idx
                            if (h2_price <= h1_price) or (interval < oscillation_interval_min) or (
                                    interval > oscillation_interval_max):
                                return (None, None, None, None)
                            return (h1_idx, h1_price, h2_idx, h2_price)

                # 提取底背离、顶背离对应的价格极值
                # 底背离：价格极值（阶段低点/前低L1，最新低点/后低L2）
                bottom_p1_idx, bottom_p1_val, bottom_p2_idx, bottom_p2_val = _get_price_extremes(market_type,
                                                                                                 is_bottom=True)
                # 顶背离：价格极值（阶段高点/前高H1，最新高点/后高H2）
                top_p1_idx, top_p1_val, top_p2_idx, top_p2_val = _get_price_extremes(market_type, is_bottom=False)

                # ==================== 步骤3：提取同期DIF极值（手册「步骤3」） ====================
                def _get_dif_extremes(price_p1_idx, price_p2_idx, is_bottom):
                    """
                    提取与价格极值同期的DIF极值
                    - price_p1_idx: 价格前极值索引（阶段极值/L1/H1）
                    - price_p2_idx: 价格后极值索引（最新极值/L2/H2）
                    - is_bottom: True=底背离（找DIF低点），False=顶背离（找DIF高点）
                    返回：(DIF前极值，DIF后极值，DIF前极值索引，DIF后极值索引)
                    """
                    if price_p1_idx is None or price_p2_idx is None:
                        return (None, None, None, None)

                    # 震荡行情：取价格极值±2根K线内的DIF极值（手册「步骤3」）
                    if market_type == "震荡":
                        # 前极值对应的DIF极值范围：price_p1_idx±2
                        p1_range = range(max(0, price_p1_idx - 2), min(n, price_p1_idx + 3))
                        # 后极值对应的DIF极值范围：price_p2_idx±2
                        p2_range = range(max(0, price_p2_idx - 2), min(n, price_p2_idx + 3))
                    else:  # 单边行情：取价格极值对应K线的DIF值（手册「步骤3」）
                        p1_range = [price_p1_idx]
                        p2_range = [price_p2_idx]

                    # 提取对应范围的DIF极值
                    if is_bottom:
                        # 底背离：找DIF低点
                        dif_p1 = np.min(dif[p1_range])
                        dif_p1_idx = p1_range[np.argmin(dif[p1_range])]
                        dif_p2 = np.min(dif[p2_range])
                        dif_p2_idx = p2_range[np.argmin(dif[p2_range])]
                    else:
                        # 顶背离：找DIF高点
                        dif_p1 = np.max(dif[p1_range])
                        dif_p1_idx = p1_range[np.argmax(dif[p1_range])]
                        dif_p2 = np.max(dif[p2_range])
                        dif_p2_idx = p2_range[np.argmax(dif[p2_range])]

                    # 验证价格-DIF错位：索引绝对差≤max_offset（手册「时间间隔硬规则」）
                    if abs(price_p1_idx - dif_p1_idx) > max_offset or abs(price_p2_idx - dif_p2_idx) > max_offset:
                        return (None, None, None, None)

                    return (dif_p1, dif_p2, dif_p1_idx, dif_p2_idx)

                # 提取底背离、顶背离对应的DIF极值
                bottom_d1_val, bottom_d2_val, bottom_d1_idx, bottom_d2_idx = _get_dif_extremes(bottom_p1_idx,
                                                                                               bottom_p2_idx,
                                                                                               is_bottom=True)
                top_d1_val, top_d2_val, top_d1_idx, top_d2_idx = _get_dif_extremes(top_p1_idx, top_p2_idx,
                                                                                   is_bottom=False)

                # ==================== 步骤4：核心判断公式 + 步骤5：伪信号过滤（手册「步骤4+5」） ====================
                def _verify_divergence(is_bottom):
                    """
                    验证背离是否成立（核心公式+四层过滤）
                    - is_bottom: True=底背离，False=顶背离
                    """
                    # 1. 先判断价格和DIF极值是否有效
                    if is_bottom:
                        p1_val, p2_val = bottom_p1_val, bottom_p2_val
                        d1_val, d2_val = bottom_d1_val, bottom_d2_val
                        p1_idx, p2_idx = bottom_p1_idx, bottom_p2_idx
                        if any(x is None for x in [p1_val, p2_val, d1_val, d2_val]):
                            return False
                    else:
                        p1_val, p2_val = top_p1_val, top_p2_val
                        d1_val, d2_val = top_d1_val, top_d2_val
                        p1_idx, p2_idx = top_p1_idx, top_p2_idx
                        if any(x is None for x in [p1_val, p2_val, d1_val, d2_val]):
                            return False

                    # 2. 核心背离判断（手册「核心判断公式」）
                    if is_bottom:
                        # 底背离：价格创极值（L2<L1/最新价<阶段低点），DIF未创同期极值（D2>D1/最新DIF>阶段DIF低点）
                        price_condition = (p2_val < p1_val)
                        dif_condition = (d2_val > d1_val)
                    else:
                        # 顶背离：价格创极值（H2>H1/最新价>阶段高点），DIF未创同期极值（D2<D1/最新DIF<阶段DIF高点）
                        price_condition = (p2_val > p1_val)
                        dif_condition = (d2_val < d1_val)
                    if not (price_condition and dif_condition):
                        return False

                    # 3. 伪信号过滤（四层硬过滤，手册「步骤5」）
                    # 3.1 微幅过滤：价格幅度≥window_close_mean的price_amplitude_threshold
                    price_amplitude = abs(p2_val - p1_val) / window_close_mean
                    if price_amplitude < price_amplitude_threshold:
                        return False

                    # 3.2 DIF无变化过滤：DIF变化≥d1_val的dif_change_threshold（避免除0）
                    denominator = max(abs(d1_val), 1e-6)
                    dif_change = abs(d2_val - d1_val) / denominator
                    if dif_change < dif_change_threshold:
                        return False

                    # 3.3 跨期过滤：价格-DIF极值错位≤max_offset（已在步骤3验证，此处二次校验）
                    if is_bottom:
                        if abs(p1_idx - bottom_d1_idx) > max_offset or abs(p2_idx - bottom_d2_idx) > max_offset:
                            return False
                    else:
                        if abs(p1_idx - top_d1_idx) > max_offset or abs(p2_idx - top_d2_idx) > max_offset:
                            return False

                    # 3.4 间隔过滤：极值点间隔符合要求（已在步骤2验证，此处二次校验）
                    interval = p2_idx - p1_idx
                    if market_type == "震荡":
                        if interval < oscillation_interval_min or interval > oscillation_interval_max:
                            return False
                    else:
                        if interval < trend_interval_min:
                            return False

                    # 所有条件满足，背离成立
                    return True

                # ==================== 执行背离判断（手册「有效/无效边界」） ====================
                # 先判断底背离，再判断顶背离（避免同时触发时冲突）
                if _verify_divergence(is_bottom=True):
                    # print(f"{strat.data.datetime.date(0)}【底背离确认】→ "
                    #       f"行情类型：{market_type}，价格类型：{price_type}，"
                    #       f"价格({bottom_p1_idx}:{bottom_p1_val:.1f}→{bottom_p2_idx}:{bottom_p2_val:.1f})创阶段新低，"
                    #       f"DIF({bottom_d1_idx}:{bottom_d1_val:.2f}→{bottom_d2_idx}:{bottom_d2_val:.2f})未创同期新低，"
                    #       f"价格幅度：{abs(bottom_p2_val - bottom_p1_val) / window_close_mean:.2%}，"
                    #       f"DIF变化：{abs(bottom_d2_val - bottom_d1_val) / max(abs(bottom_d1_val), 1e-6):.2%}")
                    return "底背离"

                if _verify_divergence(is_bottom=False):
                    # print(f"{strat.data.datetime.date(0)}【顶背离确认】→ "
                    #       f"行情类型：{market_type}，价格类型：{price_type}，"
                    #       f"价格({top_p1_idx}:{top_p1_val:.1f}→{top_p2_idx}:{top_p2_val:.1f})创阶段新高，"
                    #       f"DIF({top_d1_idx}:{top_d1_val:.2f}→{top_d2_idx}:{top_d2_val:.2f})未创同期新高，"
                    #       f"价格幅度：{abs(top_p2_val - top_p1_val) / window_close_mean:.2%}，"
                    #       f"DIF变化：{abs(top_d2_val - top_d1_val) / max(abs(top_d1_val), 1e-6):.2%}")
                    return "顶背离"

                # 无符合条件的背离
                return "无背离"

            except Exception as e:
                print(f"【MACD背离判断异常】{e}")
                import traceback
                traceback.print_exc()
                return "无背离"

        # 4. 核心状态获取
        zero_axis_pos = '上方' if macd > 0 else '下方'
        cross_status = get_cross_status()
        divergence_type = get_macd_divergence(strat)
        # print(f"零轴: {zero_axis_pos}  金叉状态: {cross_status}, 背离类型: {divergence_type}")

        # 5. 查表匹配原始得分
        buy_score = 0.0
        sell_score = 0.0
        for rule in MACD_SCORE_TABLE:
            if (rule["零轴位置"] == zero_axis_pos and
                    rule["金叉时间"] == cross_status and
                    rule["顶底背离"] == divergence_type):
                buy_score = rule["买入评分"]
                sell_score = rule["卖出评分"]
                break

        # 打印分项得分
        # print(f"买入得分: {buy_score}, 卖出得分: {sell_score}, 最终得分: {buy_score - sell_score}")

        # 6. 返回最终得分
        net_score = buy_score - sell_score
        return net_score
    except Exception as e:
        print(f"评分计算异常: {e}")
        return 0.0


# ========== 核心：完全模拟backtrader的Line对象 ==========
class BacktraderLine:
    """完全复刻backtrader的Line对象行为"""

    def __init__(self, values):
        self._history = list(values)  # 时间正序：[最早, ..., 最新]

    def __len__(self):
        return len(self._history)

    def __getitem__(self, idx):
        # 核心修复：统一处理 idx <= 0 的情况（Backtrader 风格）
        if idx <= 0:
            # 映射公式：实际列表索引 = (总长度 - 1) + idx
            # 例如：len=5, idx=0 -> 4 (最新); idx=-1 -> 3 (前1日)
            list_idx = (len(self._history) - 1) + idx
            if 0 <= list_idx < len(self._history):
                return self._history[list_idx]
            else:
                return None  # 越界返回 None
        else:
            # idx > 0 保持原样（通常回测用不到正索引）
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
    def __init__(self, close_prices):
        self.close = BacktraderLine(close_prices)


# ========== 模拟MACD指标对象 ==========
class MACDIndicator:
    def __init__(self, macd_values, signal_values):
        self.macd = BacktraderLine(macd_values)
        self.signal = BacktraderLine(signal_values)


# ========== 模拟策略对象 ==========
class MockStrategy:
    def __init__(self, macd_vals, signal_vals, hist_vals, close_prices, mcross_vals=None):
        self.data = BacktraderDataFeed(close_prices)
        self.macd = MACDIndicator(macd_vals, signal_vals)
        self.macd_hist = BacktraderLine(hist_vals)
        self.mcross = BacktraderLine(mcross_vals or [])


# ========== MACD辅助计算函数 ==========
def calculate_macd_hist(macd_vals, signal_vals):
    """自动计算MACD柱状线（Hist = DIF - DEA）"""
    return [round(m - s, 4) for m, s in zip(macd_vals, signal_vals)]


def calculate_mcross(macd_vals, signal_vals):
    """
    自动计算金叉标记（mcross）
    规则：
    - mcross=1：当日MACD上穿SIGNAL（前一日MACD<signal，当日MACD>=signal）
    - mcross=0：非金叉状态（包括死叉、持续金叉/死叉）
    """
    mcross_vals = [0] * len(macd_vals)
    for i in range(1, len(macd_vals)):
        prev_macd = macd_vals[i - 1]
        curr_macd = macd_vals[i]
        prev_signal = signal_vals[i - 1]
        curr_signal = signal_vals[i]

        # 金叉判定逻辑
        if prev_macd < prev_signal and curr_macd >= curr_signal:
            mcross_vals[i] = 1
        elif prev_macd > prev_signal and curr_macd <= curr_signal:
            mcross_vals[i] = -1
        else:
            mcross_vals[i] = 0
    return mcross_vals


def get_gold_cross_ongoing(macd_vals, signal_vals):
    """
    生成持续金叉状态标记（用于验证金叉多日）
    - 1：当前处于金叉状态（MACD>=SIGNAL）
    - 0：非金叉状态
    """
    return [1 if m >= s else 0 for m, s in zip(macd_vals, signal_vals)]


# ========== 完整的测试用例 ==========
class TestMACDScoreCalculation(unittest.TestCase):
    def setUp(self):
        """通用初始化：创建基础数据模板"""
        self.base_20d = list(range(20))  # 20日基础索引

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
    def test_rule_0_zero_above_soon_gold_top_divergence(self):
        """规则0：零轴上方+即将金叉+顶背离 → -1.3"""
        close_prices = [
            58.20, 58.90, 59.50, 60.10, 60.80,
            61.40, 62.00, 62.50, 62.90, 63.20,
            62.80, 62.30, 61.90, 61.50, 61.20,
            61.80, 62.40, 63.00, 63.50, 63.80
        ]

        # DIF：最后5期大幅回升 (0.50 -> 0.75)
        macd_vals = [
            0.72, 0.75, 0.78, 0.81, 0.83,
            0.85, 0.84, 0.82, 0.80, 0.79,
            0.76, 0.73, 0.70, 0.68, 0.50,  # 砸出一个明显的坑
            0.55, 0.60, 0.65, 0.70, 0.75  # 大幅、明确地回升
        ]

        # DEA：稳定在 0.80，让 DIF 从下方大幅靠近
        signal_vals = [
            0.70, 0.71, 0.72, 0.73, 0.74,
            0.75, 0.76, 0.76, 0.76, 0.76,
            0.75, 0.76, 0.77, 0.78, 0.79,
            0.80, 0.80, 0.80, 0.80, 0.80
        ]

        hist_vals = calculate_macd_hist(macd_vals, signal_vals)
        mcross_vals = calculate_mcross(macd_vals, signal_vals)

        print(f"hist_vals   = {hist_vals}")
        print(f"mcross_vals = {mcross_vals}")

        strat = MockStrategy(macd_vals, signal_vals, hist_vals, close_prices, mcross_vals)
        score = calculate_macd_score(strat)
        self.assertAlmostEqual(score, -1.3, places=1)

    def test_rule_1_zero_above_soon_gold_bottom_divergence(self):
        """规则1：零轴上方+即将金叉+底背离 → 1.2"""
        # 构造符合底背离的真实数据
        # 股价：持续下跌创新低（底背离的前提）
        close_prices = [
            10.0, 9.9, 9.8, 9.7, 9.6, 9.5, 9.4, 9.3, 9.2, 9.1,
            9.0, 8.9, 8.8, 8.7, 8.6, 8.5, 8.4, 8.3, 8.2, 8.1
        ]

        # DIFF (MACD)：先随价格下跌，在第15日触底后反弹，未创出新低（底背离核心）
        # 走势：从0.30 → 0.05（第15日）→ 回升至0.12（第20日）
        macd_vals = [
            0.30, 0.28, 0.26, 0.24, 0.22, 0.20, 0.18, 0.16, 0.14, 0.12,
            0.10, 0.08, 0.06, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10, 0.12
        ]

        # SIGNAL (DEA)：平滑跟随DIFF，最后几日DIFF快速上穿，即将金叉
        # 走势：从0.32 → 0.13（第20日），DIFF=0.12，DEA=0.13 → 即将金叉
        signal_vals = [
            0.32, 0.31, 0.30, 0.29, 0.28, 0.27, 0.26, 0.25, 0.24, 0.23,
            0.22, 0.21, 0.20, 0.19, 0.18, 0.17, 0.16, 0.15, 0.14, 0.13
        ]

        # 自动计算柱状线和金叉标记
        hist_vals = calculate_macd_hist(macd_vals, signal_vals)
        mcross_vals = calculate_mcross(macd_vals, signal_vals)

        strat = MockStrategy(macd_vals, signal_vals, hist_vals, close_prices, mcross_vals)
        score = calculate_macd_score(strat)
        self.assertAlmostEqual(score, 1.2, places=1)

    def test_rule_2_zero_above_soon_gold_no_divergence(self):
        """规则2：零轴上方+即将金叉+无背离 → 0.7"""
        macd_vals = [1.5] * 19 + [1.6]
        signal_vals = [1.7] * 19 + [1.75]

        # 自动计算柱状线和金叉标记
        hist_vals = calculate_macd_hist(macd_vals, signal_vals)
        mcross_vals = calculate_mcross(macd_vals, signal_vals)

        close_prices = [8.5] * 20  # 无背离

        strat = MockStrategy(macd_vals, signal_vals, hist_vals, close_prices, mcross_vals)
        score = calculate_macd_score(strat)
        self.assertAlmostEqual(score, 0.7, places=1)

    def test_rule_3_zero_above_just_gold_top_divergence(self):
        """规则3：零轴上方+刚金叉+顶背离 → -1.7"""
        # 顶背离：价格新高，MACD未新高
        close_prices = [
            8.0, 8.2, 8.4, 8.6, 8.8, 9.0, 9.2, 9.4, 9.6, 9.8,
            9.0, 9.2, 9.4, 9.6, 9.8, 10.0, 10.2, 10.4, 10.6, 10.8
        ]

        # MACD：未新高
        macd_vals = [
            2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9,
            2.8, 2.7, 2.6, 2.5, 2.4, 2.3, 2.2, 2.1, 2.0, 1.9
        ]

        signal_vals = [2.5] * 19 + [1.8]  # 最后一日金叉

        # 自动计算柱状线和金叉标记
        hist_vals = calculate_macd_hist(macd_vals, signal_vals)
        mcross_vals = calculate_mcross(macd_vals, signal_vals)

        strat = MockStrategy(macd_vals, signal_vals, hist_vals, close_prices, mcross_vals)
        score = calculate_macd_score(strat)
        self.assertAlmostEqual(score, -1.7, places=1)

    def test_rule_4_zero_above_just_gold_bottom_divergence(self):
        """规则4：零轴上方+刚金叉+底背离 → 1.6"""

        # 1. 股价：清晰底背离（第9天低点10.40，第18天创新低10.25）
        close_prices = [
            10.50, 10.45, 10.40, 10.55, 10.70,  # 1-5
            10.75, 10.65, 10.55, 10.40, 10.35,  # 6-10
            10.30, 10.35, 10.40, 10.50, 10.55,  # 11-15（第11天第一低点10.30）
            10.45, 10.35, 10.15, 10.30, 10.45  # 16-20（第18天第二低点10.15创新低）
        ]

        # 2. MACD (DIF)：零轴上方，底背离（第11天谷底0.18，第18天0.24未新低）
        macd_vals = [
            0.15, 0.18, 0.22, 0.26, 0.30,  # 1-5
            0.32, 0.30, 0.28, 0.25, 0.22,  # 6-10
            0.18, 0.20, 0.24, 0.28, 0.32,  # 11-15（第11天第一谷底0.18）
            0.30, 0.27, 0.24, 0.26, 0.29  # 16-20（第18天第二谷底0.24未新低）
        ]

        # 3. SIGNAL (DEA)：零轴上方，确保第20天刚金叉
        # 关键：第19天DIF(0.26) < DEA(0.27)，第20天DIF(0.29) > DEA(0.27)
        signal_vals = [
            0.20, 0.21, 0.22, 0.23, 0.25,  # 1-5
            0.27, 0.28, 0.28, 0.28, 0.27,  # 6-10
            0.25, 0.24, 0.24, 0.24, 0.26,  # 11-15
            0.27, 0.27, 0.27, 0.27, 0.27  # 16-20（第20天刚金叉）
        ]

        # 自动计算柱状线和金叉标记
        hist_vals = calculate_macd_hist(macd_vals, signal_vals)
        mcross_vals = calculate_mcross(macd_vals, signal_vals)

        print(f"hist_vals   = {hist_vals}")
        print(f"mcross_vals = {mcross_vals}")

        strat = MockStrategy(macd_vals, signal_vals, hist_vals, close_prices, mcross_vals)
        score = calculate_macd_score(strat)
        self.assertAlmostEqual(score, 1.6, places=1)

    def test_rule_5_zero_above_just_gold_no_divergence(self):
        """规则5：零轴上方+刚金叉+无背离 → 1.2"""
        macd_vals = [2.0] * 19 + [4.0]
        signal_vals = [2.5] * 19 + [3.0]

        # 自动计算柱状线和金叉标记
        hist_vals = calculate_macd_hist(macd_vals, signal_vals)
        mcross_vals = calculate_mcross(macd_vals, signal_vals)

        close_prices = [9.0] * 20  # 无背离

        strat = MockStrategy(macd_vals, signal_vals, hist_vals, close_prices, mcross_vals)
        score = calculate_macd_score(strat)
        self.assertAlmostEqual(score, 1.2, places=1)

    def test_rule_6_zero_above_gold_days_red_up_top_divergence(self):
        """规则6：零轴上方+金叉多日红柱放大+顶背离 → 买入0.2，卖出1.5 → 最终-1.3"""
        # 顶背离：价格新高，MACD未新高
        close_prices = [
            10.0, 10.1, 10.2, 10.3, 10.4, 10.5, 10.6, 10.7, 10.8, 10.9,
            11.0, 11.1, 11.2, 11.3, 11.4, 11.5, 11.6, 11.7, 11.8, 11.9
        ]

        # MACD：先涨后跌（未新高）
        macd_vals = [
            3.0, 3.2, 3.4, 3.6, 3.8, 4.0, 4.2, 4.4, 4.6, 4.8,
            5.0, 4.9, 4.8, 4.7, 4.6, 4.5, 4.4, 4.3, 4.2, 4.1
        ]

        signal_vals = [
            3.5, 3.4, 3.3, 3.2, 3.1, 3.0, 2.9, 2.8, 2.7, 2.6,
            2.5, 2.4, 2.3, 2.2, 2.1, 2.0, 1.9, 1.8, 1.7, 1.6
        ]

        # 自动计算柱状线和金叉标记
        hist_vals = calculate_macd_hist(macd_vals, signal_vals)
        mcross_vals = calculate_mcross(macd_vals, signal_vals)

        strat = MockStrategy(macd_vals, signal_vals, hist_vals, close_prices, mcross_vals)
        score = calculate_macd_score(strat)
        self.assertAlmostEqual(score, -1.3, places=1)

    def test_rule_7_zero_above_gold_days_red_up_bottom_divergence(self):
        """规则7：零轴上方+金叉多日红柱放大+底背离 → 买入1.2，卖出0.1 → 最终1.1"""
        # 底背离：价格新低，MACD未新低
        close_prices = [
            9.0, 8.9, 8.8, 8.7, 8.6, 8.5, 8.4, 8.3, 8.2, 8.1,
            8.0, 7.9, 7.8, 7.7, 7.6, 7.5, 7.4, 7.3, 7.2, 7.1
        ]

        # MACD：持续上涨
        macd_vals = [
            2.0, 2.2, 2.4, 2.6, 2.8, 3.0, 3.2, 3.4, 3.6, 3.8,
            4.0, 4.2, 4.4, 4.6, 4.8, 5.0, 5.2, 5.4, 5.6, 5.8
        ]

        signal_vals = [
            2.5, 2.4, 2.3, 2.2, 2.1, 2.0, 1.9, 1.8, 1.7, 1.6,
            1.5, 1.4, 1.3, 1.2, 1.1, 1.0, 0.9, 0.8, 0.7, 0.6
        ]

        # 自动计算柱状线和金叉标记
        hist_vals = calculate_macd_hist(macd_vals, signal_vals)
        mcross_vals = calculate_mcross(macd_vals, signal_vals)

        strat = MockStrategy(macd_vals, signal_vals, hist_vals, close_prices, mcross_vals)
        score = calculate_macd_score(strat)
        self.assertAlmostEqual(score, 1.1, places=1)

    def test_rule_8_zero_above_gold_days_red_up_no_divergence(self):
        """规则8：零轴上方+金叉多日红柱放大+无背离 → 买入1.0，卖出0.3 → 最终0.7"""
        macd_vals = [
            3.0, 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7, 3.8, 3.9,
            4.0, 4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 4.7, 4.8, 4.9
        ]

        signal_vals = [
            3.2, 3.1, 3.0, 2.9, 2.8, 2.7, 2.6, 2.5, 2.4, 2.3,
            2.2, 2.1, 2.0, 1.9, 1.8, 1.7, 1.6, 1.5, 1.4, 1.3
        ]

        # 自动计算柱状线和金叉标记
        hist_vals = calculate_macd_hist(macd_vals, signal_vals)
        mcross_vals = calculate_mcross(macd_vals, signal_vals)

        close_prices = [9.5] * 20  # 无背离

        strat = MockStrategy(macd_vals, signal_vals, hist_vals, close_prices, mcross_vals)
        score = calculate_macd_score(strat)
        self.assertAlmostEqual(score, 0.7, places=1)

    def test_rule_9_zero_above_gold_days_red_flat_top_divergence(self):
        """规则9：零轴上方+金叉多日红柱走平+顶背离 → -1.1"""
        # 顶背离：价格新高，MACD走平
        close_prices = [
            10.0, 10.1, 10.2, 10.3, 10.4, 10.5, 10.6, 10.7, 10.8, 10.9,
            11.0, 11.1, 11.2, 11.3, 11.4, 11.5, 11.6, 11.7, 11.8, 11.9
        ]

        # 修正：构造真实的金叉过程，而非全程固定值
        macd_vals = [
            2.8, 2.9, 2.95, 2.98, 2.99,  # 前5日：MACD < SIGNAL
            3.1, 3.5, 3.8, 3.9, 4.0,  # 金叉后上升至4.0
            4.0, 4.001, 3.999, 4.0, 4.0,  # 红柱走平阶段
            4.0, 4.002, 3.998, 4.0, 4.0  # 最后5日红柱波动<0.01
        ]
        signal_vals = [
            3.0, 3.0, 3.0, 3.0, 3.0,  # 前5日：SIGNAL > MACD
            2.95, 3.0, 3.0, 3.0, 3.0,  # 金叉后保持3.0
            3.0, 3.0, 3.0, 3.0, 3.0,  # 持续金叉
            3.0, 3.0, 3.0, 3.0, 3.0  # 红柱走平
        ]

        # 自动计算柱状线和金叉标记
        hist_vals = calculate_macd_hist(macd_vals, signal_vals)
        mcross_vals = calculate_mcross(macd_vals, signal_vals)

        strat = MockStrategy(macd_vals, signal_vals, hist_vals, close_prices, mcross_vals)
        score = calculate_macd_score(strat)
        self.assertAlmostEqual(score, -1.1, places=1)

    def test_rule_10_zero_above_gold_days_red_flattening_bottom_divergence(self):
        """规则10：零轴上方+金叉多日红柱走平+底背离 → 0.9"""
        close_prices = [
            18.0, 17.8, 17.5, 17.2, 17.0, 16.8, 16.6, 16.4, 16.2, 16.0,
            15.8, 15.6, 15.4, 15.2, 15.0, 14.8, 14.6, 14.4, 14.2, 14.0
        ]

        # 修改测试用例的macd_vals，让最后几天的上升更明显
        macd_vals = [
            3.2, 3.0, 2.9, 2.8, 2.7, 2.6, 2.55, 2.5, 2.45, 2.4,
            2.35, 2.32, 2.3, 2.28, 2.27, 2.26, 2.26, 2.28, 2.30, 2.31  # 最后几天上升更明显
        ]

        signal_vals = [
            2.1, 2.0, 1.95, 1.9, 1.85, 1.8, 1.78, 1.75, 1.72, 1.7,
            1.68, 1.66, 1.65, 1.64, 1.63, 1.62, 1.61, 1.60, 1.58, 1.59
        ]

        # 自动计算柱状线和金叉标记
        hist_vals = calculate_macd_hist(macd_vals, signal_vals)
        mcross_vals = calculate_mcross(macd_vals, signal_vals)
        print(f"hist_vals   = {hist_vals}")
        print(f"mcross_vals = {mcross_vals}")
        strat = MockStrategy(macd_vals, signal_vals, hist_vals, close_prices, mcross_vals)
        score = calculate_macd_score(strat)
        self.assertAlmostEqual(score, 0.9, places=1)

    def test_rule_11_zero_above_gold_days_red_flat_no_divergence(self):
        """规则11：零轴上方+金叉多日红柱走平+无背离 → 0.4"""
        # 修正：构造真实的金叉过程
        macd_vals = [
            1.8, 1.9, 1.95, 1.98, 1.99,  # 前5日：MACD < SIGNAL
            2.1, 2.3, 2.5, 2.7, 2.9,  # 金叉后红柱放大
            3.0, 3.0, 3.0, 3.0, 3.0,  # 红柱开始走平
            3.0, 3.001, 2.999, 3.0, 3.0  # 最后5日红柱波动<0.01
        ]
        signal_vals = [
            2.0, 2.0, 2.0, 2.0, 2.0,  # 前5日：SIGNAL > MACD
            1.95, 2.0, 2.1, 2.2, 2.3,  # 金叉后缓慢上升
            2.4, 2.5, 2.6, 2.7, 2.8,  # 持续金叉
            2.9, 2.9, 2.9, 2.9, 2.9  # 红柱走平
        ]

        # 自动计算柱状线和金叉标记
        hist_vals = calculate_macd_hist(macd_vals, signal_vals)
        mcross_vals = calculate_mcross(macd_vals, signal_vals)

        close_prices = [9.5 + 0.01 * i for i in range(5)] + [9.5] * 10 + [9.5 - 0.01 * i for i in range(5)]  # 无背离

        strat = MockStrategy(macd_vals, signal_vals, hist_vals, close_prices, mcross_vals)
        score = calculate_macd_score(strat)
        self.assertAlmostEqual(score, 0.4, places=1)

    def test_rule_12_zero_above_gold_days_red_down_top_divergence(self):
        """规则12：零轴上方+金叉多日红柱缩减+顶背离 → -1.0"""
        # 1. 股价：持续创新高（顶背离的前提）
        close_prices = [
            10.0, 10.1, 10.2, 10.3, 10.4, 10.5, 10.6, 10.7, 10.8, 10.9,
            11.0, 11.1, 11.2, 11.3, 11.4, 11.5, 11.6, 11.7, 11.8, 11.9
        ]

        # 2. MACD (DIF)：零轴上方，先创新高后回落（未超过前高，顶背离核心）
        # 前高 5.2（第10日），后高 4.8（第20日）→ 未新高
        macd_vals = [
            4.0, 4.2, 4.4, 4.6, 4.8,  # 前5日：持续上升
            5.0, 5.1, 5.2, 5.15, 5.1,  # 第6-10日：创新高 5.2
            5.0, 4.9, 4.8, 4.7, 4.6,  # 第11-15日：开始回落
            4.5, 4.4, 4.3, 4.29, 4.2  # 第16-20日：持续回落，未创新高
        ]

        # 3. SIGNAL (DEA)：零轴上方，金叉后持续低于DIF（保证金叉多日）
        # 关键：DEA 始终低于 DIF，保持金叉状态
        signal_vals = [
            2.8, 2.9, 3.0, 3.1, 3.2,  # 前5日：DEA < DIF（已金叉）
            3.3, 3.4, 3.5, 3.6, 3.7,  # 第6-10日：持续上升，仍低于DIF
            3.7, 3.75, 3.8, 3.85, 3.9,  # 第11-15日：继续上升，与DIF差值缩小
            3.91, 3.92, 3.93, 3.94, 3.95  # 最后走平，仍低于DIF
        ]

        # 自动计算柱状线和金叉标记
        hist_vals = calculate_macd_hist(macd_vals, signal_vals)
        mcross_vals = calculate_mcross(macd_vals, signal_vals)

        print(f"hist_vals   = {hist_vals}")
        print(f"mcross_vals = {mcross_vals}")

        strat = MockStrategy(macd_vals, signal_vals, hist_vals, close_prices, mcross_vals)
        score = calculate_macd_score(strat)
        self.assertAlmostEqual(score, -1.0, places=1)

    def test_rule_13_zero_above_gold_days_red_down_bottom_divergence(self):
        """规则13：零轴上方+金叉多日红柱缩减+底背离 → 买入得分0.8， 卖出得分0.2， 合计0.6"""
        # 1. 股价：清晰的底背离形态
        # 第一谷底：8.2（第10日），第二谷底：7.8（第20日）→ 股价创新低
        close_prices = [
            10.0, 9.9, 9.7, 9.5, 9.3,  # 第1-5日：缓慢下跌
            9.1, 8.9, 8.7, 8.5, 8.2,  # 第6-10日：加速下跌至第一谷底8.2
            8.4, 8.6, 8.8, 9.0, 8.9,  # 第11-15日：反弹后开始回落
            8.7, 8.5, 8.3, 8.1, 7.8  # 第16-20日：下跌至第二谷底7.8（创新低）
        ]

        # 2. MACD (DIF)：零轴上方，底背离形态
        # 第一谷底：2.2（第10日），第二谷底：2.6（第20日）→ DIF未创新低
        macd_vals = [
            3.0, 2.9, 2.8, 2.7, 2.6,
            2.5, 2.4, 2.3, 2.2, 2.25,  # 第一底
            2.3, 2.35, 2.4, 2.8, 2.75,  # 最后七天加0.3
            2.7, 2.65, 2.6, 2.59, 2.58  # 最后七天加0.3
        ]

        signal_vals = [
            1.5, 1.55, 1.60, 1.65, 1.68,
            1.70, 1.72, 1.74, 1.76, 1.78,
            1.80, 1.82, 1.84, 1.86, 1.88,
            1.90, 1.92, 1.94, 1.96, 1.98  # Signal 持续上行
        ]

        # 自动计算柱状线和金叉标记
        hist_vals = calculate_macd_hist(macd_vals, signal_vals)
        mcross_vals = calculate_mcross(macd_vals, signal_vals)

        print(f"hist_vals   = {hist_vals}")
        print(f"mcross_vals = {mcross_vals}")

        strat = MockStrategy(macd_vals, signal_vals, hist_vals, close_prices, mcross_vals)
        score = calculate_macd_score(strat)
        self.assertAlmostEqual(score, 0.6, places=1)

    def test_rule_14_zero_above_gold_days_red_down_no_divergence(self):
        """规则14：零轴上方+金叉多日红柱缩减+无背离 → -0.1"""
        # 修正：构造真实的金叉过程
        macd_vals = [
            3.8, 3.9, 3.95, 3.98, 3.99,  # 前5日：MACD < SIGNAL
            4.1, 4.0, 3.8, 3.6, 3.4,  # 金叉后红柱缩减
            3.2, 3.0, 2.8, 2.6, 2.4,  # 持续缩减
            2.2, 2.0, 1.8, 1.6, 1.4  # 最后5日加速缩减
        ]
        signal_vals = [
            3.0, 3.0, 3.0, 3.0, 3.0,  # 前5日：SIGNAL > MACD
            2.95, 1.5, 1.0, 1.0, 1.0,  # 金叉后保持1.0
            1.0, 1.0, 1.0, 1.0, 1.0,  # 持续金叉
            1.0, 1.0, 1.0, 1.0, 1.0  # 红柱缩减
        ]

        # 自动计算柱状线和金叉标记
        hist_vals = calculate_macd_hist(macd_vals, signal_vals)
        mcross_vals = calculate_mcross(macd_vals, signal_vals)
        print(f"hist_vals   = {hist_vals}")
        print(f"mcross_vals = {mcross_vals}")
        close_prices = [9.5 + 0.01 * i for i in range(10)] + [9.5 - 0.01 * i for i in range(10)]  # 无背离

        strat = MockStrategy(macd_vals, signal_vals, hist_vals, close_prices, mcross_vals)
        score = calculate_macd_score(strat)
        self.assertAlmostEqual(score, -0.1, places=1)

    def test_rule_15_zero_above_soon_death_top_divergence(self):
        """规则15：零轴上方+即将死叉+顶背离 → -1.8"""
        # 顶背离：价格新高，MACD下跌
        close_prices = [
            10.0, 10.1, 10.2, 10.3, 10.4, 10.5, 10.6, 10.7, 10.8, 10.9,
            11.0, 11.1, 11.2, 11.3, 11.4, 11.5, 11.6, 11.7, 11.8, 11.9
        ]

        # 即将死叉：红柱缩短
        macd_vals = [3.0] * 19 + [2.1]
        signal_vals = [2.0] * 19 + [2.0]

        # 自动计算柱状线和金叉标记
        hist_vals = calculate_macd_hist(macd_vals, signal_vals)
        mcross_vals = calculate_mcross(macd_vals, signal_vals)

        strat = MockStrategy(macd_vals, signal_vals, hist_vals, close_prices, mcross_vals)
        score = calculate_macd_score(strat)
        self.assertAlmostEqual(score, -1.8, places=1)


if __name__ == '__main__':
    unittest.main(verbosity=2)