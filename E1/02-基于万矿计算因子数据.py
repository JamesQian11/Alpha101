import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt

## 获取股票池
s_date = '2015-12-01'
e_date = '2016-01-01'
date = w.tdays(s_date, e_date, "preiod = D").Data[0]  # 日期函数
stocks = w.wset("sectorconstituent", "date=" + e_date + ";windcode=000906.SH").Data[1]  # 中证800股票池

## 获取日频量价原始数据
close = w.wsd(stocks, 'close', s_date, e_date, usedf=True)[1].reset_index().drop(columns=['index'])
# 改日期索引为数字，考虑后文rolling函数应用的便捷性
returns = w.wsd(stocks, 'pct_chg', s_date, e_date, usedf=True)[1].reset_index().drop(columns=['index'])
Open = w.wsd(stocks, 'open', s_date, e_date, usedf=True)[1].reset_index().drop(columns=['index'])
low = w.wsd(stocks, 'low', s_date, e_date, usedf=True)[1].reset_index().drop(columns=['index'])
vwap = w.wsd(stocks, 'vwap', s_date, e_date, usedf=True)[1].reset_index().drop(columns=['index'])
high = w.wsd(stocks, 'high', s_date, e_date, usedf=True)[1].reset_index().drop(columns=['index'])
cap = w.wsd(stocks, 'mkt_cap_ashare', s_date, e_date, usedf=True)[1].reset_index().drop(columns=['index'])
volume = w.wsd(stocks, 'volume', s_date, e_date, usedf=True)[1].reset_index().drop(columns=['index'])
ind = w.wss(stocks, "industry_citic", "unit=1;tradeDate=" + date[0].strftime("%Y%m%d") + ";industryType=1")
print(close)
## 计算因子值
start_1 = datetime.now()  # 记录计算用时

alpha_1 = alpha1(close, returns)
alpha_2 = alpha2(Open, close, volume)
alpha_3 = alpha3(Open, volume)
alpha_4 = alpha4(low)
alpha_5 = alpha5(Open, vwap, close)
alpha_6 = alpha6(Open, volume)
alpha_7 = alpha7(volume, close)
alpha_8 = alpha8(Open, returns)
alpha_9 = alpha9(close)
alpha_10 = alpha10(close)
alpha_11 = alpha11(vwap, close, volume)
alpha_12 = alpha12(volume, close)
alpha_13 = alpha13(volume, close)
alpha_14 = alpha14(Open, volume, returns)
alpha_15 = alpha15(high, volume)
alpha_16 = alpha16(high, volume)
alpha_17 = alpha17(volume, close)
alpha_18 = alpha18(close, Open)
alpha_19 = alpha19(close, returns)
alpha_20 = alpha20(Open, high, close, low)
alpha_21 = alpha21(volume, close)
alpha_22 = alpha22(high, volume, close)
alpha_23 = alpha23(high, close)
alpha_24 = alpha24(close)
alpha_25 = alpha25(volume, returns, vwap, high, close)
alpha_26 = alpha26(volume, high)
alpha_27 = alpha27(volume, vwap)
alpha_28 = alpha28(volume, high, low, close)
alpha_29 = alpha29(close, returns)
alpha_30 = alpha30(close, volume)
alpha_31 = alpha31(close, low, volume)
alpha_32 = alpha32(close, vwap)
alpha_33 = alpha33(Open, close)
alpha_34 = alpha34(close, returns)
alpha_35 = alpha35(volume, close, high, low, returns)
alpha_36 = alpha36(Open, close, volume, returns, vwap)
alpha_37 = alpha37(Open, close)
alpha_38 = alpha38(close, Open)
alpha_39 = alpha39(volume, close, returns)
alpha_40 = alpha40(high, volume)
alpha_41 = alpha41(high, low, vwap)
alpha_42 = alpha42(vwap, close)
alpha_43 = alpha43(volume, close)
alpha_44 = alpha44(high, volume)
alpha_45 = alpha45(close, volume)
alpha_46 = alpha46(close)
alpha_47 = alpha47(volume, close, high, vwap)
alpha_49 = alpha49(close)
alpha_50 = alpha50(volume, vwap)
alpha_51 = alpha51(close)
alpha_52 = alpha52(returns, volume, low)
alpha_53 = alpha53(close, high, low)
alpha_54 = alpha54(Open, close, high, low)
alpha_55 = alpha55(high, low, close, volume)
alpha_56 = alpha56(returns, cap)
alpha_57 = alpha57(close, vwap)
alpha_60 = alpha60(close, high, low, volume)
alpha_61 = alpha61(volume, vwap)
alpha_62 = alpha62(volume, high, low, Open, vwap)
alpha_64 = alpha64(high, low, Open, volume, vwap)
alpha_65 = alpha65(volume, vwap, Open)
alpha_66 = alpha66(vwap, low, Open, high)
alpha_68 = alpha41(high, low, vwap)
alpha_71 = alpha71(volume, close, low, Open, vwap)
alpha_72 = alpha72(volume, high, low, vwap)
alpha_73 = alpha73(vwap, Open, low)
alpha_74 = alpha74(volume, close, high, vwap)
alpha_75 = alpha75(volume, vwap, low)
alpha_77 = alpha77(volume, high, low, vwap)
alpha_78 = alpha78(volume, low, vwap)
alpha_81 = alpha81(volume, vwap)
alpha_83 = alpha83(high, low, close, volume)
alpha_84 = alpha84(vwap, close)
alpha_85 = alpha85(volume, high, close, low)
alpha_86 = alpha41(high, low, vwap)
alpha_88 = alpha88(volume, Open, low, high, close)
alpha_92 = alpha92(volume, high, low, close, Open)
alpha_94 = alpha94(volume, vwap)
alpha_95 = alpha95(volume, high, low, Open)
alpha_96 = alpha96(volume, vwap, close)
alpha_98 = alpha98(volume, Open, vwap)
alpha_99 = alpha99(volume, high, low)
alpha_100 = alpha41(high, low, vwap)
alpha_101 = alpha101(close, Open, high, low)

alpha_48 = alpha48(close, ind)  # 这些因子涉及行业中性化，计算时间较长
alpha_58 = alpha58(vwap, volume, ind)
alpha_59 = alpha59(vwap, volume, ind)
alpha_63 = alpha63(volume, close, vwap, Open, ind)
alpha_67 = alpha67(volume, vwap, high, ind)
alpha_69 = alpha69(volume, vwap, ind, close)
alpha_70 = alpha70(close, ind, vwap)
alpha_76 = alpha76(volume, vwap, low, ind)
alpha_79 = alpha79(volume, close, Open, ind, vwap)
alpha_80 = alpha80(Open, high, ind)
alpha_82 = alpha82(Open, volume, ind)
alpha_87 = alpha87(volume, close, vwap)
alpha_89 = alpha89(low, vwap, ind)
alpha_90 = alpha90(volume, close, ind, low)
alpha_91 = alpha91(close, ind, volume, vwap)
alpha_93 = alpha93(vwap, ind, volume, close)
alpha_97 = alpha97(volume, low, vwap, ind)
alpha_100 = alpha100(volume, close, low, high, ind)

# 因子放入列表
data_alpha = [alpha_1, alpha_2, alpha_3, alpha_4, alpha_5, alpha_6, alpha_7, alpha_8, alpha_9, alpha_10,
              alpha_11, alpha_12, alpha_13, alpha_14, alpha_15, alpha_16, alpha_17, alpha_18, alpha_19, alpha_20,
              alpha_21, alpha_22, alpha_23, alpha_24, alpha_25, alpha_26, alpha_27, alpha_28, alpha_29, alpha_30,
              alpha_31, alpha_32, alpha_33, alpha_34, alpha_35, alpha_36, alpha_37, alpha_38, alpha_39, alpha_40,
              alpha_41, alpha_42, alpha_43, alpha_44, alpha_45, alpha_46, alpha_47, alpha_48, alpha_49, alpha_50,
              alpha_51, alpha_52, alpha_53, alpha_54, alpha_55, alpha_56, alpha_57, alpha_58, alpha_59, alpha_60,
              alpha_61, alpha_62, alpha_63, alpha_64, alpha_65, alpha_66, alpha_67, alpha_68, alpha_69, alpha_70,
              alpha_71, alpha_72, alpha_73, alpha_74, alpha_75, alpha_76, alpha_77, alpha_78, alpha_79, alpha_80,
              alpha_81, alpha_82, alpha_83, alpha_84, alpha_85, alpha_86, alpha_87, alpha_88, alpha_89, alpha_90,
              alpha_91, alpha_92, alpha_93, alpha_94, alpha_95, alpha_96, alpha_97, alpha_98, alpha_99, alpha_100,
              alpha_101]

# 由于一些因子应用前几个月的数据进行计算，因此初始时期因子为错误值，所以取两年保留一年
for i in range(len(data_alpha)):
    data_alpha[i] = data_alpha[i].iloc[245:]  # 243,245,244,244,244/这些数字为某年交易日数量
df_101 = pd.concat(data_alpha)  # 合并各因子的数据
df_101.to_csv('data/alpha.csv')  # 储存数据至csv文件


## 提取数据并转换数据结构
# 因子数据转化为时间+股票的双重索引格式
def zhuanhuan(alpha_1, date, stocks, columns=['alpha']):
    alpha_year = alpha_1.fillna(value=0)
    alpha_year.set_index(date, inplace=True)
    index = pd.MultiIndex.from_product([date, stocks], names=['date', 'codes'])
    df = pd.DataFrame(alpha_year.stack(), columns=columns)
    alpha = pd.DataFrame(df.values, columns=columns, index=index)
    return alpha


date_list = ['2014-01-01', '2015-01-01', '2016-01-01', '2017-01-01', '2018-01-01', '2019-05-30']  # 测试区间，每年更换一次股票池
data_alpha = []
div = 101
# 从csv文件取出因子数据存进列表
for i in range(5):
    data_ = pd.read_csv('data/data_alpha_re' + str(i + 1) + '.csv')
    data_.drop(data_.columns[0], axis=1, inplace=True)
    data_alpha.append(data_)
# 将之前合并的因子进行分割，并转换数据结构为双重索引，存进二维列表
for i in range(5):
    data = []
    df = data_alpha[i]
    n = int(len(df) / div)
    date = w.wsd('000001.SZ', 'close', date_list[i], date_list[i + 1], period='D', usedf=True)[1].index
    stocks = w.wset("sectorconstituent", "date=" + date_list[i + 1] + ";windcode=000906.SH").Data[1]
    for j in range(div):
        d = zhuanhuan(df.iloc[n * j:n * (j + 1)], date, stocks, columns=['alpha' + str(j + 1)])
        data.append(d)
    data_alpha[i] = data
# 将各因子不同时期的数据进行合并，得到一维列表data存放各因子的双重索引可用数据
data = []
for j in range(div):
    d = pd.concat([data_alpha[0][j], data_alpha[1][j], data_alpha[2][j], data_alpha[3][j], data_alpha[4][j]])
    data.append(d)
