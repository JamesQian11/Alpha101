from scipy.stats import rankdata
from dateutil import parser
import numpy as np
import numpy.linalg as la
import pandas as pd
from datetime import datetime
import scipy.stats as stats
import matplotlib.pyplot as plt
from single_factor import *
import glob
import os


def file_preprocess(filename):
    input_csv = pd.read_csv(filename)
    input_csv.rename(columns={'Unnamed: 0': 'date'}, inplace=True)
    input_csv.index = input_csv['date']
    input_csv = input_csv.drop(columns='date')
    return input_csv.replace(0, np.nan)


# 中位数去极值
def extreme_process_MAD(sample):  # 输入的sample为时间截面的股票因子df数据
    factor_name = list(sample.columns)
    for name in factor_name:
        x = sample[name]
        median = x.median()
        MAD = abs(x - median).median()
        x[x > (median + 3 * 1.4826 * MAD)] = median + 3 * 1.4826 * MAD
        x[x < (median - 3 * 1.4826 * MAD)] = median - 3 * 1.4826 * MAD
        sample[name] = x
    return sample


# 中性化
def data_scale_neutral(sample, date):
    stocks = list(sample.index)
    ind = w.wss(stocks, "industry_citic", "unit=1;tradeDate=" + date + ";industryType=1", usedf=True)[1]  # 行业
    data_med = pd.get_dummies(ind, columns=['INDUSTRY_CITIC'])  # 生成0-1变量矩阵
    Incap = w.wss(stocks, "val_lnmv", "unit=1;tradeDate=" + date, usedf=True)[1]  # 市值
    others = w.wss(stocks, "tech_turnoverrate20,tech_revs20,risk_variance20", "unit=1;tradeDate=" + date, usedf=True)[
        1]  # 换手率、动量、波动率
    x = pd.concat([data_med], axis=1).fillna(value=0)  # ,others,Incap 这里选择中性化回归变量
    X = np.array(x)
    sample = sample.loc[list(x.index)]
    factor_name = list(sample.columns)
    for name in factor_name:
        y = np.array(sample[name])
        beta_ols = la.inv(X.T.dot(X)).dot(X.T).dot(y)  # 最小二乘法计算拟合值
        residual = y - X.dot(beta_ols)  # 取残差为中性化后的因子值
        sample[name] = residual
    return sample


# 标准化
def standardize(sample):
    factor_name = list(sample.columns)
    for name in factor_name:
        x = sample[name]
        sample[name] = (x - np.mean(x)) / (np.std(x))
    return sample


def in_data(close, volume, amount):
    close_data = file_preprocess(close)
    volume_data = file_preprocess(volume)
    amount_data = file_preprocess(amount)
    returns_data = close_data.pct_change(1)
    vwap_data = amount_data / volume_data

    return close_data, returns_data, volume_data, vwap_data


def cal_alpha(close, returns, volume, vwap):
    alpha001 = alpha1(close, returns)
    alpha007 = alpha7(volume, close)
    alpha011 = alpha11(vwap, close, volume)
    alpha021 = alpha21(volume, close)
    return alpha001, alpha007, alpha011, alpha021


def cal_IC(alpha, returns):
    alpha_MAD = extreme_process_MAD(alpha)
    alpha_MAD_SD = standardize(alpha_MAD)
    d = list(alpha_MAD_SD.index)  # 获取双索引中的日期索引
    print(len(d), d)
    ic_s = []
    for i in range(len(d)):
        alpha_MAD_SD_1 = alpha_MAD_SD.loc[d[i]]  # 所有的股票取截面的数据
        next_returns = returns.loc[d[i]]
        try:
            ic_s.append(next_returns.corr(alpha_MAD_SD_1))  # 计算因子值与下期收益的秩相关系数，并存进ic值列表
        except:
            ic_s.append(0)
        ic_df = pd.Series(ic_s, index=d)
        print(ic_df)
    return ic_df


def rank_csv(df):
    return df.rank(axis=1)


def cal_RankIC(alpha, returns):
    rank_returns = rank_csv(returns)
    rank_alpha = rank_csv(alpha)
    cal_IC(rank_alpha, rank_returns)


if __name__ == '__main__':
    print('--------------------Run-----------------')
    filename_close = '/Users/Vision/Desktop/03_Financial/00_GAM/AlphaNetV3/Data_XXX/S_DQ_ADJCLOSE.csv'
    filename_volume = '/Users/Vision/Desktop/03_Financial/00_GAM/AlphaNetV3/Data_XXX/S_DQ_VOLUME.csv'
    filename_amount = '/Users/Vision/Desktop/03_Financial/00_GAM/AlphaNetV3/Data_XXX/S_DQ_AMOUNT.csv'
    _close, _returns, _volume, _vwap = in_data(filename_close, filename_volume, filename_amount)
    # alpha001 = alpha1(_close, _returns)
    # _alpha001, _alpha007, _alpha011, _alpha021 = cal_alpha(_close, _returns, _volume, _vwap)
    # _alpha001.to_csv('alpha001.csv', header=True)
    # _alpha007.to_csv('alpha007.csv', header=True)
    # _alpha011.to_csv('alpha011.csv', header=True)
    # _alpha021.to_csv('alpha021.csv', header=True)
    # print('returns', _returns)
    # print('_alpha001', _alpha001)
    returns = _returns
    returns_rank = returns.rank(axis=1, pct=True)
    alpha = pd.read_csv('alpha001.csv', index_col=None)
    alpha.index = alpha['date']
    alpha = alpha.drop(columns='date')
    alpha_rank = alpha.rank(axis=1, pct=True)

    alpha_rank_MAD = extreme_process_MAD(alpha_rank)
    alpha_MAD_SD = standardize(alpha_rank_MAD)
    d = list(alpha_MAD_SD.index)  # 获取双索引中的日期索引
    print(len(d), d)
    ic_s = []
    for i in range(len(d)):
        if i < 1441:
            print(i)
            alpha_MAD_SD_1 = alpha_MAD_SD.loc[d[i]]  # 所有的股票取截面的数据
            next_returns = returns_rank.loc[d[i + 1]]
            try:
                ic_s.append(next_returns.corr(alpha_MAD_SD_1))  # 计算因子值与下期收益的秩相关系数，并存进ic值列表
            except:
                ic_s.append(0)
    ic_s = pd.Series(ic_s)  # 列表转化为Series，从而能够计算各统计量
    rate = len(ic_s[ic_s >= 0]) / len(ic_s)  # ic值大于0的比例
    IC_mean = ic_s.mean()  # ic均值
    IC_std = ic_s.std()  # ic标准差
    IC_IR = ic_s.mean() / ic_s.std()  # ic_IR用来衡量因子有效性
    stats = [IC_mean, IC_std, IC_IR, rate]

    ic_sum = ic_s.sum()

    #
    # alpha001_IC = cal_IC(_alpha001, _returns)
    # alpha001_IC.to_csv('alpha001_IC.csv', header=True)

    # print('_alpha007', _alpha001)
    # alpha007_IC = cal_IC(_alpha007, _returns)
    # alpha007_IC.to_csv('alpha007_IC.csv', header=True)
    #
    # print('_alpha011', _alpha011)
    # alpha011_IC = cal_IC(_alpha011, _returns)
    # alpha011_IC.to_csv('alpha011_IC.csv', header=True)
    #
    # print('_alpha021', _alpha021)
    # alpha021_IC = cal_IC(_alpha021, _returns)
    # alpha021_IC.to_csv('alpha021_IC.csv', header=True)
    print('--------------------finish-------------')

    # close_data = file_preprocess(filename_close)
    # def stddev(df, window):
    #     return df.rolling(window).std()
    # std = stddev(close_data, 10)
