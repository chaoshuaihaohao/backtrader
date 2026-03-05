#!/usr/bin/env python
# -*- coding: utf-8; py-indent-offset:4 -*-
###############################################################################
# 修正核心：1. Line对象添加line_前缀，避免与方法重名 2. 支持外部line_is_xxx[0]/[-1]调用
#          3. 所有形态判断逻辑完整保留，兼容原有策略
###############################################################################
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

try:
    from backtrader import Indicator, MovAv
except ImportError:
    from . import Indicator, MovAv


class CDLPatterns(Indicator):
    '''
    单K线语义解析器（无命名冲突版本，支持外部line_is_xxx[0]/[-1]历史回溯调用）
    '''
    lines = (
        # 基础形态ID Line
        'pattern_id',
        # 核心：所有形态判断Line对象添加line_前缀，避免与is_xxx方法重名
        'line_is_dayangxian', 'line_is_dayinxian', 'line_is_yizixian',
        'line_is_guangtouyang', 'line_is_guangtouyin', 'line_is_guangjiaoyang', 'line_is_guangjiaoyin',
        'line_is_yangchuizi', 'line_is_yinchuizi', 'line_is_yangdaochui', 'line_is_yindaochui',
        'line_is_mubeishizi', 'line_is_diaorenxian', 'line_is_qingtingshizi', 'line_is_shizixing',
        'line_is_xingxian', 'line_is_xiaok', 'line_is_fangchuan',
        # 趋势分类判断Line（同样添加line_前缀）
        'line_is_jiduan', 'line_is_qiangli', 'line_is_fanzhuan', 'line_is_yujing', 'line_is_shuaijie', 'line_is_laju',
        'line_is_qiangli_bullish', 'line_is_qiangli_bearish',
        'line_is_fanzhuan_bullish', 'line_is_fanzhuan_bearish',
        'line_is_laju_sideways', 'line_is_shuaijie_bullish', 'line_is_shuaijie_bearish', 'line_is_shuaijie_sideways',
        'line_is_jiduan_bullish', 'line_is_jiduan_bearish',
        'line_is_bullish', 'line_is_bearish',
        # 原有其他趋势线（保持不变）
        'line_jiduan', 'line_qiangli', 'line_fanzhuan',
        'line_laju', 'line_shuaijie', 'line_yujing',
        'line_qiangli_bull', 'line_qiangli_bear',
        'line_fanzhuan_bull', 'line_fanzhuan_bear',
        'line_laju_side', 'line_shuaijie_side',
    )

    # 形态分类常量
    CAT_JIDUAN = "极端"
    CAT_QIANGLI = "强力"
    CAT_FANZHUAN = "反转"
    CAT_LAJU = "拉锯"
    CAT_SHUAIJIE = "衰竭"
    CAT_YUJING = "即将大跌"

    DIR_BULLISH = "看涨"
    DIR_BEARISH = "看跌"
    DIR_SIDEWAYS = "震荡"

    # 形态 ID 常量映射
    P_NONE = 0;
    P_DAYANG = 1;
    P_DAYIN = 2;
    P_YIZI = 3
    P_GT_YANG = 4;
    P_GT_YIN = 5;
    P_GJ_YANG = 6;
    P_GJ_YIN = 7
    P_CHUIZI_YANG = 8;
    P_CHUIZI_YIN = 9;
    P_DAOCHUI_YANG = 10;
    P_DAOCHUI_YIN = 11
    P_MUBEI = 12;
    P_DIAOREN = 13;
    P_QINGTING = 14;
    P_SHIZI_PU = 15;
    P_SHIZI_CHANGJIAO = 16
    P_XING_YANG = 17;
    P_XING_YIN = 18;
    P_XIAOK_YANG = 19;
    P_XIAOK_YIN = 20
    P_FANGCHUAN_YANG = 21;
    P_FANGCHUAN_YIN = 22

    # 形态编号→字符串 映射表
    pattern_id_to_name = {
        0: '无', 1: '大阳线', 2: '大阴线', 3: '一字线',
        4: '光头阳线', 5: '光头阴线', 6: '光脚阳线', 7: '光脚阴线',
        8: '阳锤子线', 9: '阴锤子线', 10: '阳倒锤子', 11: '阴倒锤子',
        12: '墓碑十字', 13: '吊人线', 14: '蜻蜓十字', 15: '普通十字星', 16: '长脚十字星',
        17: '阳星线', 18: '阴星线', 19: '小阳线', 20: '小阴线',
        21: '阳纺锤线', 22: '阴纺锤线'
    }

    # 形态字符串→编号 反向映射字典
    pattern_name_to_id = {v: k for k, v in pattern_id_to_name.items()}

    params = (
        ('body_ratio', 0.7),
        ('wick_ratio', 3.0),
        ('doji_ratio', 0.1),
    )

    def __init__(self):
        self._p_id = self.P_NONE
        self._direction = 0
        self.pattern_name = ""  # 保留当前形态名称（可选）

    # 编号转形态字符串
    def id_to_pattern_name(self, pattern_id):
        return self.pattern_id_to_name.get(pattern_id, "未知形态")

    # 字符串转形态编号
    def name_to_pattern_id(self, pattern_name):
        return self.pattern_name_to_id.get(pattern_name, self.P_NONE)

    # --- 单K线形态判断方法（无修改，保持原有逻辑）---
    def is_dayangxian(self):
        return self._p_id == self.P_DAYANG

    def is_dayinxian(self):
        return self._p_id == self.P_DAYIN

    def is_yizixian(self):
        return self._p_id == self.P_YIZI

    def is_guangtouyang(self):
        return self._p_id == self.P_GT_YANG

    def is_guangtouyin(self):
        return self._p_id == self.P_GT_YIN

    def is_guangjiaoyang(self):
        return self._p_id == self.P_GJ_YANG

    def is_guangjiaoyin(self):
        return self._p_id == self.P_GJ_YIN

    def is_yangchuizi(self):
        return self._p_id == self.P_CHUIZI_YANG

    def is_yinchuizi(self):
        return self._p_id == self.P_CHUIZI_YIN

    def is_yangdaochui(self):
        return self._p_id == self.P_DAOCHUI_YANG

    def is_yindaochui(self):
        return self._p_id == self.P_DAOCHUI_YIN

    def is_mubeishizi(self):
        return self._p_id == self.P_MUBEI

    def is_diaorenxian(self):
        return self._p_id == self.P_DIAOREN

    def is_qingtingshizi(self):
        return self._p_id == self.P_QINGTING

    def is_shizixing(self):
        return self._p_id in [self.P_SHIZI_PU, self.P_SHIZI_CHANGJIAO]

    def is_xingxian(self):
        return self._p_id in [self.P_XING_YANG, self.P_XING_YIN]

    def is_xiaok(self):
        return self._p_id in [self.P_XIAOK_YANG, self.P_XIAOK_YIN]

    def is_fangchuan(self):
        return self._p_id in [self.P_FANGCHUAN_YANG, self.P_FANGCHUAN_YIN]

    # --- 趋势分类判断方法（无修改，保持原有逻辑）---
    def is_jiduan(self):
        return self.is_yizixian() or self.is_dayangxian() or self.is_dayinxian()

    def is_qiangli(self):
        return (self.is_dayangxian() or self.is_dayinxian() or
                self.is_guangtouyang() or self.is_guangtouyin() or
                self.is_guangjiaoyang() or self.is_guangjiaoyin())

    def is_fanzhuan(self):
        return (self.is_yangchuizi() or self.is_yinchuizi() or
                self.is_yangdaochui() or self.is_yindaochui() or self.is_qingtingshizi())

    def is_yujing(self):
        return self.is_mubeishizi() or self.is_diaorenxian()

    def is_shuaijie(self):
        return self.is_xingxian() or self.is_xiaok()

    def is_laju(self):
        return self.is_shizixing() or self.is_fangchuan()

    def is_qiangli_bullish(self):
        return self.is_dayangxian() or self.is_guangtouyang() or self.is_guangjiaoyang()

    def is_qiangli_bearish(self):
        return self.is_dayinxian() or self.is_guangtouyin() or self.is_guangjiaoyin()

    def is_fanzhuan_bullish(self):
        return self.is_yangchuizi() or self.is_yangdaochui() or self.is_qingtingshizi()

    def is_fanzhuan_bearish(self):
        return self.is_yinchuizi() or self.is_yindaochui()

    def is_laju_sideways(self):
        return self.is_laju()

    def is_shuaijie_bullish(self):
        return self._p_id in [self.P_XIAOK_YANG, self.P_XING_YANG]

    def is_shuaijie_bearish(self):
        return self._p_id in [self.P_XIAOK_YIN, self.P_XING_YIN]

    def is_shuaijie_sideways(self):
        return self.is_shuaijie() and not (self.is_shuaijie_bullish() or self.is_shuaijie_bearish())

    def is_jiduan_bullish(self):
        return self.is_dayangxian() or (self.is_yizixian() and self._direction == 1)

    def is_jiduan_bearish(self):
        return self.is_dayinxian() or (self.is_yizixian() and self._direction == -1)

    def is_bullish(self):
        return self.is_jiduan_bullish() or self.is_qiangli_bullish() or self.is_shuaijie_bullish()

    def is_bearish(self):
        return self.is_jiduan_bearish() or self.is_qiangli_bearish() or self.is_shuaijie_bearish()

    # 内部方法：形态名称转编号
    def _get_pattern_id_by_name(self, pattern_name):
        return self.pattern_name_to_id.get(pattern_name, self.P_NONE)

    # --- 获取形态名称（无修改）---
    def get_pattern_name(self, o, h, l, c, p_c):
        body = c - o
        abs_body = abs(body)
        EPS = 1e-8
        full_range = max(h - l, EPS)
        hi_oc = max(o, c)
        lo_oc = min(o, c)
        u_sh = h - hi_oc
        d_sh = lo_oc - l

        body_pct = abs_body / full_range
        u_sh_pct = u_sh / full_range
        d_sh_pct = d_sh / full_range
        rel_volatility = full_range / max(p_c, EPS)
        safe_abs_body = max(abs_body, EPS)
        wick_body_ratio_down = d_sh / safe_abs_body
        wick_body_ratio_up = u_sh / safe_abs_body

        # 优先级1：一字线
        doji_straight_condition = (
                full_range < EPS
                and abs(o - h) < EPS
                and abs(o - l) < EPS
                and abs(o - c) < EPS
        )
        if doji_straight_condition:
            return '一字线'

        # 优先级2：十字星系列
        cross_star_common = (
                body_pct < 0.10
                and rel_volatility >= 0.005
        )
        if cross_star_common:
            if u_sh_pct > 0.4 and d_sh_pct > 0.4:
                return '长脚十字星'
            elif u_sh_pct > 0.6 and d_sh_pct < 0.2:
                return '墓碑十字'
            elif u_sh_pct < 0.2 and d_sh_pct > 0.6:
                return '蜻蜓十字'
            else:
                return '普通十字星'

        # 优先级3：大K线系列
        extreme_big_common = (
                body_pct > 0.85
                and rel_volatility >= 0.005
        )
        if extreme_big_common:
            if u_sh_pct < 0.05 and d_sh_pct < 0.05:
                return '大阳线' if body > 0 else '大阴线'
            elif u_sh_pct < 0.05:
                return '光头阳线' if body > 0 else '光头阴线'
            elif d_sh_pct < 0.05:
                return '光脚阳线' if body > 0 else '光脚阴线'
            else:
                return '大阳线' if body > 0 else '大阴线'

        # 优先级4：小阳线/小阴线
        small_fluct_common = (
                rel_volatility < 0.005
                and full_range > EPS
                and body_pct >= 0.10
                and body_pct <= 0.85
        )
        if small_fluct_common:
            return '小阳线' if body > 0 else '小阴线'

        # 优先级5：星线
        star_common = (
                0.10 <= body_pct < 0.30
                and rel_volatility >= 0.005
                and body_pct < 0.85
                and body_pct >= 0.10
        )
        if star_common:
            return '阳星线' if body > 0 else '阴星线'

        # 优先级6：锤子线
        hammer_common = (
                0.30 <= body_pct < 0.40
                and rel_volatility >= 0.005
                and u_sh_pct < 0.15
                and d_sh_pct > 0.50
                and wick_body_ratio_down > 3.0
        )
        if hammer_common:
            return '阳锤子线' if body > 0 else '阴锤子线'

        # 优先级7：吊人线
        hanging_man_condition = (
                0.30 <= body_pct < 0.40
                and rel_volatility >= 0.005
                and u_sh_pct > 0.15
                and body < 0
                and wick_body_ratio_down > 3.0
        )
        if hanging_man_condition:
            return '吊人线'

        # 优先级8：倒锤子线
        inverted_hammer_common = (
                0.30 <= body_pct < 0.40
                and rel_volatility >= 0.005
                and u_sh_pct > 0.50
                and d_sh_pct < 0.15
                and wick_body_ratio_up > 3.0
        )
        if inverted_hammer_common:
            return '阳倒锤子' if body > 0 else '阴倒锤子'

        # 优先级9：中等实体趋势类
        mid_body_common = (
                0.40 <= body_pct <= 0.85
                and rel_volatility >= 0.005
        )
        if mid_body_common:
            if u_sh_pct < 0.10:
                return '光头阳线' if body > 0 else '光头阴线'
            elif d_sh_pct < 0.10:
                return '光脚阳线' if body > 0 else '光脚阴线'
            else:
                return '大阳线' if body > 0 else '大阴线'

        # 优先级10：纺锤线
        spindle_common = (
                rel_volatility >= 0.005
                and wick_body_ratio_down <= 3.0
                and wick_body_ratio_up <= 3.0
        )
        if spindle_common:
            return '阳纺锤线' if body > 0 else '阴纺锤线'

        # 终极兜底
        return '阳纺锤线' if body > 0 else '阴纺锤线'

    # --- next方法（给带line_前缀的Line对象赋值，无命名冲突）---
    def next(self):
        o, h, l, c = self.data.open[0], self.data.high[0], self.data.low[0], self.data.close[0]
        p_c = self.data.close[-1] if len(self.data) > 1 else c

        # 1. 更新内部形态变量
        pattern_name_str = self.get_pattern_name(o, h, l, c, p_c)
        self._p_id = self._get_pattern_id_by_name(pattern_name_str)
        self.pattern_name = pattern_name_str
        self._direction = 1 if c >= o else -1

        # 2. 给基础形态ID Line赋值
        self.lines.pattern_id[0] = self._p_id

        # 3. 给单K线形态判断Line对象赋值（1=True，0=False）
        self.lines.line_is_dayangxian[0] = 1 if self.is_dayangxian() else 0
        self.lines.line_is_dayinxian[0] = 1 if self.is_dayinxian() else 0
        self.lines.line_is_yizixian[0] = 1 if self.is_yizixian() else 0
        self.lines.line_is_guangtouyang[0] = 1 if self.is_guangtouyang() else 0
        self.lines.line_is_guangtouyin[0] = 1 if self.is_guangtouyin() else 0
        self.lines.line_is_guangjiaoyang[0] = 1 if self.is_guangjiaoyang() else 0
        self.lines.line_is_guangjiaoyin[0] = 1 if self.is_guangjiaoyin() else 0
        self.lines.line_is_yangchuizi[0] = 1 if self.is_yangchuizi() else 0
        self.lines.line_is_yinchuizi[0] = 1 if self.is_yinchuizi() else 0
        self.lines.line_is_yangdaochui[0] = 1 if self.is_yangdaochui() else 0
        self.lines.line_is_yindaochui[0] = 1 if self.is_yindaochui() else 0
        self.lines.line_is_mubeishizi[0] = 1 if self.is_mubeishizi() else 0
        self.lines.line_is_diaorenxian[0] = 1 if self.is_diaorenxian() else 0
        self.lines.line_is_qingtingshizi[0] = 1 if self.is_qingtingshizi() else 0
        self.lines.line_is_shizixing[0] = 1 if self.is_shizixing() else 0
        self.lines.line_is_xingxian[0] = 1 if self.is_xingxian() else 0
        self.lines.line_is_xiaok[0] = 1 if self.is_xiaok() else 0
        self.lines.line_is_fangchuan[0] = 1 if self.is_fangchuan() else 0

        # 4. 给趋势分类判断Line对象赋值（1=True，0=False）
        self.lines.line_is_jiduan[0] = 1 if self.is_jiduan() else 0
        self.lines.line_is_qiangli[0] = 1 if self.is_qiangli() else 0
        self.lines.line_is_fanzhuan[0] = 1 if self.is_fanzhuan() else 0
        self.lines.line_is_yujing[0] = 1 if self.is_yujing() else 0
        self.lines.line_is_shuaijie[0] = 1 if self.is_shuaijie() else 0
        self.lines.line_is_laju[0] = 1 if self.is_laju() else 0
        self.lines.line_is_qiangli_bullish[0] = 1 if self.is_qiangli_bullish() else 0
        self.lines.line_is_qiangli_bearish[0] = 1 if self.is_qiangli_bearish() else 0
        self.lines.line_is_fanzhuan_bullish[0] = 1 if self.is_fanzhuan_bullish() else 0
        self.lines.line_is_fanzhuan_bearish[0] = 1 if self.is_fanzhuan_bearish() else 0
        self.lines.line_is_laju_sideways[0] = 1 if self.is_laju_sideways() else 0
        self.lines.line_is_shuaijie_bullish[0] = 1 if self.is_shuaijie_bullish() else 0
        self.lines.line_is_shuaijie_bearish[0] = 1 if self.is_shuaijie_bearish() else 0
        self.lines.line_is_shuaijie_sideways[0] = 1 if self.is_shuaijie_sideways() else 0
        self.lines.line_is_jiduan_bullish[0] = 1 if self.is_jiduan_bullish() else 0
        self.lines.line_is_jiduan_bearish[0] = 1 if self.is_jiduan_bearish() else 0
        self.lines.line_is_bullish[0] = 1 if self.is_bullish() else 0
        self.lines.line_is_bearish[0] = 1 if self.is_bearish() else 0

        # 5. 打印形态信息（调试用）
        print(f"p_id {self._p_id} p_name {self.pattern_name} open {o:.2f} close {c:.2f} high {h:.2f} low {l:.2f}\n")

        # 6. 给原有趋势线批量清零（跳过已赋值的Line对象）
        skip_lines = [
            self.lines.pattern_id,
            self.lines.line_is_dayangxian, self.lines.line_is_dayinxian, self.lines.line_is_yizixian,
            self.lines.line_is_guangtouyang, self.lines.line_is_guangtouyin, self.lines.line_is_guangjiaoyang,
            self.lines.line_is_guangjiaoyin,
            self.lines.line_is_yangchuizi, self.lines.line_is_yinchuizi, self.lines.line_is_yangdaochui,
            self.lines.line_is_yindaochui,
            self.lines.line_is_mubeishizi, self.lines.line_is_diaorenxian, self.lines.line_is_qingtingshizi,
            self.lines.line_is_shizixing,
            self.lines.line_is_xingxian, self.lines.line_is_xiaok, self.lines.line_is_fangchuan,
            self.lines.line_is_jiduan, self.lines.line_is_qiangli, self.lines.line_is_fanzhuan,
            self.lines.line_is_yujing, self.lines.line_is_shuaijie, self.lines.line_is_laju,
            self.lines.line_is_qiangli_bullish, self.lines.line_is_qiangli_bearish,
            self.lines.line_is_fanzhuan_bullish, self.lines.line_is_fanzhuan_bearish,
            self.lines.line_is_laju_sideways, self.lines.line_is_shuaijie_bullish, self.lines.line_is_shuaijie_bearish,
            self.lines.line_is_shuaijie_sideways,
            self.lines.line_is_jiduan_bullish, self.lines.line_is_jiduan_bearish,
            self.lines.line_is_bullish, self.lines.line_is_bearish
        ]
        for line in self.lines:
            if line not in skip_lines:
                line[0] = 0

        # 7. 原有趋势线赋值逻辑（保持不变）
        if self.is_jiduan():
            self.lines.line_jiduan[0] = self._direction
        if self.is_qiangli():
            self.lines.line_qiangli[0] = self._direction
        if self.is_fanzhuan():
            self.lines.line_fanzhuan[0] = self._direction
        if self.is_laju():
            self.lines.line_laju[0] = 2
        if self.is_shuaijie():
            self.lines.line_shuaijie[0] = self._direction
        if self.is_yujing():
            self.lines.line_yujing[0] = -1

        if self.is_qiangli_bullish():
            self.lines.line_qiangli_bull[0] = 1
        if self.is_qiangli_bearish():
            self.lines.line_qiangli_bear[0] = -1

        if self.is_fanzhuan_bullish():
            self.lines.line_fanzhuan_bull[0] = 1
        if self.is_fanzhuan_bearish():
            self.lines.line_fanzhuan_bear[0] = -1

        if self.is_laju_sideways():
            self.lines.line_laju_side[0] = 2

        if self.is_shuaijie_bullish():
            self.lines.line_shuaijie_side[0] = 1
        elif self.is_shuaijie_bearish():
            self.lines.line_shuaijie_side[0] = -1
        else:
            self.lines.line_shuaijie_side[0] = 2