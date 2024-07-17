#!/usr/bin/env python
# coding=utf-8
#######################################################################
#    > File Name: main.py
#    > Author: XXX
#    > Mail: XXX@qq.com
#    > Created Time: 2024年4月26日
#    > description:
#######################################################################
from dingtalkchatbot.chatbot import DingtalkChatbot
import datetime
import pandas as pd
import akshare as ak
import numpy as np
import re
from scipy.signal import find_peaks
import os

token = os.environ["TOKEN"]
secret = os.environ["SECRET"]
send_set = set()
transfer_date_dic = {'D': 'daily', 'W': 'weekly', 'M': 'monthly'}
def RD(N,D=3):   return np.round(N,D)

def MA(S,N):
    return pd.Series(S).rolling(N).mean().values

def HHV(S,N):
    return pd.Series(S).rolling(N).max().values

def LLV(S,N):
    return pd.Series(S).rolling(N).min().values

def EMA(S,N):
    return pd.Series(S).ewm(span=N, adjust=False).mean().values

def SLOPE_PRE(S, N):
    return pd.Series(S).rolling(N).apply(lambda x: np.polyfit(range(N), np.append(x, x[-1])[1:], deg=1)[0],
                                         raw=True).values

def KDJ(CLOSE,HIGH,LOW, N=9,M1=3,M2=3):
    RSV = (CLOSE - LLV(LOW, N)) / (HHV(HIGH, N) - LLV(LOW, N)) * 100
    K = EMA(RSV, (M1*2-1));    D = EMA(K,(M2*2-1));        J=K*3-D*2
    return K, D, J

def MACD(CLOSE,SHORT=12,LONG=26,M=9):
    DIF = EMA(CLOSE,SHORT)-EMA(CLOSE,LONG);
    DEA = EMA(DIF,M);      MACD=(DIF-DEA)*2
    return RD(DIF),RD(DEA),RD(MACD)

def BIAS(CLOSE,L1=6, L2=12, L3=24):
    BIAS1 = (CLOSE - MA(CLOSE, L1)) / MA(CLOSE, L1) * 100
    BIAS2 = (CLOSE - MA(CLOSE, L2)) / MA(CLOSE, L2) * 100
    BIAS3 = (CLOSE - MA(CLOSE, L3)) / MA(CLOSE, L3) * 100
    return RD(BIAS1), RD(BIAS2), RD(BIAS3)

def code2symbol(code, kind="shcode"):
    '''根据code代码开头数字转为为标准的symbol'''
    xcode = ''.join(c for c in code if c.isdigit())
    kind_list = ['shcode','sh.code','code.sh','codesh']
    # kind转为小写，在以上列表内，则自动按相应格式进行转换；否则只反馈6位数字代码。
    if kind.lower() in kind_list:
        prefix = ''.join(c for c in kind.replace('code','') if c.isalpha())
        char   = '.' if '.' in kind else ''
        if   kind.lower() == "shcode" or kind.lower() == "sh.code": symbol = f'sh{char}{xcode}' if (xcode[0] == "6" or xcode[:3] == "900")  else f'sz{char}{xcode}' if (xcode[0] == "0" or xcode[0] == "3" or xcode[0] == "2")  else f'bj{char}{xcode}' if (xcode[0] == "4" or xcode[0] == "8" or xcode[:3] == "920") else code
        elif kind.lower() == "codesh" or kind.lower() == "code.sh": symbol = f'{xcode}{char}sh' if (xcode[0] == "6" or xcode[:3] == "900")  else f'{xcode}{char}sz' if (xcode[0] == "0" or xcode[0] == "3" or xcode[0] == "2")  else f'{xcode}{char}bj' if (xcode[0] == "4" or xcode[0] == "8" or xcode[:3] == "920") else code
        return symbol if prefix[0].islower() else symbol.upper()
    else:return xcode

def find_max_min_point(df, k_name='k10'):
    '''获取数据的局部最大值和最小值的索引'''
    mindis = int((''.join(re.findall('\d+', k_name))))
    series = np.array(df[k_name])
    peaks, _ = find_peaks(series, distance=mindis)
    mins, _ = find_peaks(series*-1, distance=mindis)

    dt = {}
    condition_buy = (df[k_name].index.isin(mins.tolist())) & (df[k_name] < 0) & (df['MACD'] < 0)
    condition_sell = (df[k_name].index.isin(peaks.tolist())) & (df[k_name] > 0) & (df['MACD'] > 0)

    dt['BUY'], dt['SELL'] = condition_buy, condition_sell
    ret = pd.DataFrame(dt)
    return ret

def get_data(code, start_date, end_date, freq):
    '''获取综合数据'''
    s_date = (pd.to_datetime(start_date) + datetime.timedelta(days=-60)).strftime('%Y%m%d')
    end_date = (pd.to_datetime(end_date) + datetime.timedelta(days=+1)).strftime('%Y%m%d')
    if freq == 'D' or freq == 'W' or freq == 'M':
        df = ak.stock_zh_a_hist(symbol=code, period=transfer_date_dic[freq], start_date=s_date, end_date=end_date, adjust="qfq").iloc[:, :6]
        df.columns = ['date', 'open', 'close', 'high', 'low', 'volume', ]
        df["date"] = pd.to_datetime(df["date"])
    else:
        period = freq[:-1]
        df = ak.stock_zh_a_minute(symbol=code2symbol(code), period=period, adjust="qfq")
        df.columns = ['date', 'open', 'high', 'low', 'close', 'volume']
        df["date"] = pd.to_datetime(df["date"])
        df[df.columns.tolist()[1:]] = pd.DataFrame(df[df.columns.tolist()[1:]], dtype=float)

    df['volume'] = round(df['volume'].astype('float') / 10000, 2)

    # 计算主要指标
    for i in [10, 20, 60]:
        df['kp{}'.format(i)] = SLOPE_PRE(df['close'], i)
    df['K'], df['D'], df['J'] = KDJ(df['close'], df['high'], df['low'])
    df['DIF'], df['DEA'], df['MACD'] = MACD(df['close'])
    df['bias10'], df['bias20'], df['bias60'] = BIAS(df['close'], 10, 20, 60)
    # 获取最高最低点
    df = pd.concat([df, find_max_min_point(df, 'kp10')], axis=1)
    # 过滤日期
    df = df.loc[(df['date'] >= start_date) & (df['date'] <= end_date)]
    # 把date作为日期索引
    df.set_index(['date'], inplace=True)
    df.index.name=''
    return df

def get_individual_fund_flow_rank(code, indicator="今日", is_display=True):
    '''获取个股资金流动排名情况'''
    df = ak.stock_individual_fund_flow_rank(indicator=indicator)
    df = df[df['代码'] == code]
    for col in ['今日主力净流入-净额', '今日超大单净流入-净额', '今日大单净流入-净额', '今日中单净流入-净额', '今日小单净流入-净额']:
        df[col] = df[col].astype('float64')
    for col in df.columns.tolist():
        if str(df[col].dtype) == 'float64':
            df[col] = df[col].apply(lambda x: num2str(x))
    return df

def num2str(num):
    '''实现数值转换为万，亿单位，保留2位小数'''
    if num > 0:
        flag = 1
    else:
        flag = -1
    num = abs(num)
    level = 0
    while num > 10000:
        if level >= 2:
            break
        num /= 10000
        level += 1
    units = ['', '万', '亿']

    return '{}{}'.format(round(flag * num, 3), units[level])

def send_markdown(code, trade_type):
    webhook = 'https://oapi.dingtalk.com/robot/send?access_token={}'.format(token)
    now = datetime.datetime.now()
    end_date = now.strftime('%Y%m%d')
    # df = get_stock(code, start_date='20240101', end_date=end_date, freq='1d', count=600)
    df = get_data(code=code, start_date='20240701', end_date=end_date, freq='D')
    xcode = '买入' if trade_type == 'BUY' else '卖出'
    current_date = df[df[trade_type] == True].tail(1)
    if len(current_date) == 1:
        current_time = current_date.index.strftime('%Y-%m-%d')[0]

        # 创建机器器
        chart_bot = DingtalkChatbot(webhook, secret)
        # 发送markdown
        sub_key = "key_" + str(code) + str(current_time) + str(trade_type)
        yesterday = (now - datetime.timedelta(days=1)).strftime('%Y-%m-%d')
        if current_date == yesterday and sub_key not in send_set:
            # 加入set
            send_set.add(sub_key)
            # 计算数据
            open_price = current_date['open'].values[0]
            close_price = current_date['close'].values[0]
            high_price = current_date['high'].values[0]
            low_price = current_date['low'].values[0]
            kp10 = round(current_date['kp10'].values[0],3)
            bias10 = round(current_date['bias10'].values[0],3)
            fund = get_individual_fund_flow_rank(code).iloc[:, 1:]
            updown = fund.iloc[0][3]
            zhuli = fund.iloc[0][4]
            xiaodan = fund.iloc[0][-2]
            name = fund.iloc[0][1]
            code = fund.iloc[0][0]
            red_msg = f'<font color="#dd0000">级别: {xcode}</font>'
            orange_msg = f'<font color="#FFA500">时间: {current_time}</font>'

            chart_bot.send_markdown(
                title=f'每日复盘',
                text=f'### **{name}({code})**\n'
                     f'**{red_msg}**\n\n'
                     f'**{orange_msg}**\n\n'
                     f'open: {open_price} 元,\tclose: {close_price} 元\n\n'
                     f'high: {high_price} 元,\tlow: {low_price} 元\n\n'
                     f'今日涨跌幅: {updown} \n\n'
                     f'斜率: {kp10} \n\n'
                     f'乖离率: {bias10} \n\n'
                     f'今日主力净流入: {zhuli} \n\n'
                     f'今日小单净流入: {xiaodan} \n\n',
                is_at_all=False)


def func(code):
    send_markdown(code=code, trade_type='BUY')
    send_markdown(code=code, trade_type='SELL')


code_list = ['000977', '000612', '601877']
for code in code_list:
    func(code)
