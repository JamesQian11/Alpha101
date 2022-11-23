## 不同模块的代码应按顺序运行
from scipy.stats import rankdata
from dateutil import parser
import numpy as np
import numpy.linalg as la
import pandas as pd
from datetime import datetime
import scipy.stats as stats
import matplotlib.pyplot as plt
from single_factor import *
from data_prepare import *


def factor_save():
    file = '/Volumes/James/01_MyProjectTrees/00_GAM/dataset/分钟数据_20211114'
    amount_path = file + '/amount'
    close_path = file + '/close'
    high_path = file + '/high'
    low_path = file + '/low'
    open_path = file + '/open'
    returns_path = file + '/returns'
    vol_path = file + '/vol'

    for d in get_file_names_list(close_path):
        print(d)
        amount_name = 'amount_' + d.split('_')[-1]
        close_name = 'close_' + d.split('_')[-1]
        high_name = 'high_' + d.split('_')[-1]
        low_name = 'low_' + d.split('_')[-1]
        open_name = 'open_' + d.split('_')[-1]
        returns_name = 'returns_' + d.split('_')[-1]
        vol_name = 'vol_' + d.split('_')[-1]

        amount_names = f'{amount_path}/{amount_name}.csv'
        close_names = f'{close_path}/{close_name}.csv'
        high_names = f'{high_path}/{high_name}.csv'
        low_names = f'{low_path}/{low_name}.csv'
        open_names = f'{open_path}/{open_name}.csv'
        returns_names = f'{returns_path}/{returns_name}.csv'
        vol_names = f'{vol_path}/{vol_name}.csv'

        print(amount_names, close_names, high_names, low_names, open_names, returns_names, vol_names)
        amount_names_data = file_preprocess(amount_names)
        close_files_data = file_preprocess(close_names)
        high_names_data = file_preprocess(high_names)
        low_names_data = file_preprocess(low_names)
        open_names_data = file_preprocess(open_names)
        returns_names_data = file_preprocess(returns_names)
        vol_names_data = file_preprocess(vol_names)

        # alpha_1 = alpha1(close_files_data, returns_names_data)
        # alpha_2 = alpha2(open_names_data, close_files_data, vol_names_data)
        # alpha_3 = alpha3(open_names_data, vol_names_data)
        # alpha_4 = alpha4(low_names_data)
        # alpha_5 = alpha5(open_names_data, vwap, close)
        alpha_6 = alpha6(open_names_data, vol_names_data)
        alpha_7 = alpha7(vol_names_data, close_files_data)
        alpha_8 = alpha8(open_names_data, returns_names_data)
        alpha_9 = alpha9(close_files_data)
        alpha_10 = alpha10(close_files_data)
        alpha_33 = alpha33(open_names_data, close_files_data)

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

        # alpha001_name = vol_path + '/alpha001_' + vol_names.split('/')[-1].split('.')[0].split('_')[-1] + '.csv'
        # alpha002_name = vol_path + '/alpha002_' + vol_names.split('/')[-1].split('.')[0].split('_')[-1] + '.csv'
        # alpha003_name = vol_path + '/alpha003_' + vol_names.split('/')[-1].split('.')[0].split('_')[-1] + '.csv'
        # alpha004_name = vol_path + '/alpha004_' + vol_names.split('/')[-1].split('.')[0].split('_')[-1] + '.csv'
        # alpha005_name = vol_path + '/alpha005_' + vol_names.split('/')[-1].split('.')[0].split('_')[-1] + '.csv'
        alpha006_name = vol_path + '/alpha006_' + vol_names.split('/')[-1].split('.')[0].split('_')[-1] + '.csv'
        alpha007_name = vol_path + '/alpha007_' + vol_names.split('/')[-1].split('.')[0].split('_')[-1] + '.csv'
        alpha008_name = vol_path + '/alpha008_' + vol_names.split('/')[-1].split('.')[0].split('_')[-1] + '.csv'
        alpha009_name = vol_path + '/alpha009_' + vol_names.split('/')[-1].split('.')[0].split('_')[-1] + '.csv'
        alpha010_name = vol_path + '/alpha010_' + vol_names.split('/')[-1].split('.')[0].split('_')[-1] + '.csv'
        alpha033_name = vol_path + '/alpha033_' + vol_names.split('/')[-1].split('.')[0].split('_')[-1] + '.csv'

        # alpha_1.to_csv(alpha001_name, header=True)
        # alpha_2.to_csv(alpha002_name, header=True)
        # alpha_3.to_csv(alpha003_name, header=True)
        # alpha_4.to_csv(alpha004_name, header=True)
        # alpha_5.to_csv(alpha005_name, header=True)
        alpha_6.to_csv(alpha006_name, header=True)
        alpha_7.to_csv(alpha007_name, header=True)
        alpha_8.to_csv(alpha008_name, header=True)
        alpha_9.to_csv(alpha009_name, header=True)
        alpha_10.to_csv(alpha010_name, header=True)
        alpha_33.to_csv(alpha033_name, header=True)


## 数据预处理
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


# 数据预处理函数汇总
def data_process(sample, date):
    sample = extreme_process_MAD(sample)
    sample = data_scale_neutral(sample, date)
    sample = standardize(sample)
    return sample


def ic_fenxi(df, next_ret):
    d = list(next_ret.index)  # 获取双索引中的日期索引
    ic_s = []
    df = df.fillna(value=0)  # 异常值填充为0
    for i in range(len(d)):
        stock_v = stock_valid_df.loc[d[i]].dropna().values  # 获取当期有效股票池
        stock_v = list(set(stock_v) & set(df.loc[d[i]].index))
        dff = df.loc[d[i]].loc[stock_v]  # 提取当期可用因子值和下期收益数据
        r1 = dff['NEXT_RET']
        # 因子中性化
        dff[alpha] = data_scale_neutral(dff[alpha], d[i].strftime('%Y%m%d'))
        try:
            ic_s.append(r1.corr(dff[alpha], method='spearman'))  # 计算因子值与下期收益的秩相关系数，并存进ic值列表
        except:
            ic_s.append(0)
    ic_s = pd.Series(ic_s)  # 列表转化为Series，从而能够计算各统计量
    rate = len(ic_s[ic_s >= 0]) / len(ic_s)  # ic值大于0的比例
    IC_mean = ic_s.mean()  # ic均值
    IC_std = ic_s.std()  # ic标准差
    IC_IR = ic_s.mean() / ic_s.std()  # ic_IR用来衡量因子有效性
    stats = [IC_mean, IC_std, IC_IR, rate]
    return stats, ic_s


def cal_IC_IR(fac, retu):
    row = list(retu.index)
    ic_s = []
    for i in range(len(row)):
        if i <= 208:
            alpha = fac.loc[row[i]]
            ret = retu.loc[row[i + 31]]
            print(row[i], row[i + 31])
            print("IC", ret.corr(alpha))
            ic_s.append(ret.corr(alpha))
    ic_s = pd.Series(ic_s)
    ic_s.index = row[:209]
    rate = len(ic_s[ic_s >= 0]) / len(ic_s)  # ic值大于0的比例
    IC_mean = ic_s.mean()  # ic均值
    IC_std = ic_s.std()  # ic标准差
    IC_IR = ic_s.mean() / ic_s.std()  # ic_IR用来衡量因子有效性
    stats = [IC_mean, IC_std, IC_IR, rate]
    return ic_s, stats


if __name__ == '__main__':
    print("*********************************************************************************")
    alpha006_path = '/Volumes/James/01_MyProjectTrees/00_GAM/dataset/分钟数据_20211114/alpha006/alph006_20210401.csv'
    alpha006 = file_preprocess(alpha006_path)
    alpha006 = extreme_process_MAD(alpha006)
    alpha006 = standardize(alpha006)

    returns = '/Volumes/James/01_MyProjectTrees/00_GAM/dataset/分钟数据_20211114/returns/returns_20210401.csv'
    returns = file_preprocess(returns)

    d = list(returns.index)  # 获取双索引中的日期索引
    ic_s = []
    for i in range(len(d)):
        if i < 210:
            alpha = alpha006.loc[d[i]]  # 所有的股票取截面的数据
            next_returns = returns.loc[d[i + 30]]
            try:
                ic_s.append(next_returns.corr(alpha, method='spearman'))  # 计算因子值与下期收益的秩相关系数，并存进ic值列表
            except:
                ic_s.append(0)
    ic_s = pd.Series(ic_s)  # 列表转化为Series，从而能够计算各统计量
    IC_mean = ic_s.mean()  # ic均值
    IC_std = ic_s.std()  # ic标准差
    IC_IR = ic_s.mean() / ic_s.std()  # ic_IR用来衡量因子有效性
    stats = [IC_mean, IC_std, IC_IR, rate]
    IC_sum = ic_s.sum()

    # ic_cul0 = []  # 累计IC值储存列表
    # for i in range(len(data)):
    #     d = list(next_ret.index.levels[0])
    #     lc = data[i].loc[d]  # 根据调仓频率筛选数据
    #     lc['NEXT_RET'] = next_ret.NEXT_RET
    #     if i < len(data) - 4:
    #         list_ic, ic_s = ic_fenxi(lc, next_ret, alpha=['alpha' + str(i + 1)])  # 计算IC统计量
    #     elif i > len(data) - 5:
    #         list_ic, ic_s = ic_fenxi(lc, next_ret, alpha=['fengge' + str(i + 1 - 30)])  # 计算风格因子的IC统计量
    #     print(list_ic)
    #     ic_cul0.append(ic_s)
    #
    # # 计算累计IC
    # ic_cul = ic_cul0.copy()
    # for i in range(len(ic_cul)):
    #     x = ic_cul[i]
    #     for j in range(1, len(x)):
    #         x[j] = x[j] + x[j - 1]
    #     ic_cul[i] = x



    """
    202101-202106  IC IR Returns
    ic_cul0 = []  # 累计IC值储存列表
    for i in range(len(data)):
        d = list(next_ret.index.levels[0])
        lc = data[i].loc[d]  # 根据调仓频率筛选数据
        lc['NEXT_RET'] = next_ret.NEXT_RET
        if i < len(data) - 4:
            list_ic, ic_s = ic_fenxi(lc, next_ret, alpha=['alpha' + str(i + 1)])  # 计算IC统计量
        elif i > len(data) - 5:
            list_ic, ic_s = ic_fenxi(lc, next_ret, alpha=['fengge' + str(i + 1 - 30)])  # 计算风格因子的IC统计量
        print(list_ic)
        ic_cul0.append(ic_s)

        # 计算累计IC
    ic_cul = ic_cul0.copy()
    for i in range(len(ic_cul)):
        x = ic_cul[i]
        for j in range(1, len(x)):
            x[j] = x[j] + x[j - 1]
        ic_cul[i] = x

    
    # for index, row in alpha006.iterrows():
    #     print(index)
    #     sample = extreme_process_MAD(row)

    
    ## 提取数据并转换数据结构
    # 因子数据转化为时间+股票的双重索引格式
    def zhuanhuan(alpha_1,date,stocks,columns = ['alpha']):
        alpha_year = alpha_1.fillna(value = 0)
        alpha_year.set_index(date,inplace = True)
        index = pd.MultiIndex.from_product([date,stocks],names = ['date', 'codes'])
        df = pd.DataFrame(alpha_year.stack(),columns = columns)
        alpha = pd.DataFrame(df.values,columns = columns,index = index)
        return alpha
    date_list = ['2017-01-01','2020-01-01'] # 测试区间，每年更换一次股票池
    div = 10
    df_101 = pd.read_csv('data/alpha.csv') # 从csv文件取出因子数据
    df_101.drop(df_101.columns[0], axis=1, inplace=True)
    
    # 将之前合并的因子进行分割，并转换数据结构为双重索引三维数据结构，存进二维列表
    data = []
    data_alpha = [] # pandas结构用于相关性分析
    n = int(len(df_101)/div)
    date = w.wsd('000001.SZ','close',date_list[0],date_list[1],period = 'D',usedf = True)[1].index
    stocks = w.wset("sectorconstituent", "date="+date_list[1]+";windcode=000906.SH").Data[1]
    for j in range(div):
        alpha_j = df_101.iloc[n*j:n*(j+1)]
        data_alpha.append(alpha_j)
        d = zhuanhuan(alpha_j,date,stocks,columns = ['alpha'+str(j+1)]) 
        data.append(d)    
        
    # 与传统量价因子相关性分析
    corr_list = []
    for k in range(div):
        for i in range(22,len(data_fengge[0])):
            corr_1,corr_2,corr_3,corr_4 = [],[],[],[]
            corr_1.append(data_fengge[0].iloc[i].corr(data_alpha[k].iloc[i-22]))
            corr_2.append(data_fengge[1].iloc[i].corr(data_alpha[k].iloc[i-22]))
            corr_3.append(data_fengge[2].iloc[i].corr(data_alpha[k].iloc[i-22]))
            corr_4.append(data_fengge[3].iloc[i].corr(data_alpha[k].iloc[i-22]))
        corr_list.append([np.array(corr_1).mean(),np.array(corr_2).mean(),np.array(corr_3).mean(),np.array(corr_4).mean()])
    for corr_ in corr_list:
        print(corr_)  
        
    # 加入传统量价因子对照
    for i in range(4):        
        d = zhuanhuan(data_fengge[i].iloc[22:],date,stocks,columns = ['fengge'+str(i+1)]) 
        data.append(d)
     
     
    ## 股票池筛选，可交易，非新股，非PT，ST，涨跌停的股票
    s_date = '2017-01-01'  
    e_date = '2020-01-01'
    Period = 'M'
    def get_stocks(trDate):
        trDate = trDate.strftime('%Y-%m-%d')
        stocks_800 = w.wset("sectorconstituent", "date="+trDate+";windcode=000906.SH").Data[1]
        status = w.wss(stocks_800, "trade_status,maxupordown,riskwarning,ipo_date", tradeDate=trDate, usedf=True)[1]
        date_least=w.tdaysoffset(-6,trDate,'Period=M').Data[0][0]   
        trade_codes=list(status[(status['TRADE_STATUS']=='交易')&(status['IPO_DATE']<=date_least)&(status['MAXUPORDOWN']==0)&(status['RISKWARNING']=='否')].index)    
        return trade_codes
    trade_d = w.tdays(s_date, e_date, Period=Period,usedf=True).Data[0]
    stock_valid = []
    for i in range(len(trade_d)):
        stock_valid.append(get_stocks(trade_d[i]))
    stock_valid_df = pd.DataFrame(stock_valid,index = trade_d)
    
    
    ## 获取下期收益数据，用于因子测试
    date_list = [s_date,e_date]
    date = w.wsd('000001.SZ','close',date_list[0],date_list[1],period = Period,usedf = True)[1].index
    stocks = w.wset("sectorconstituent", "date="+date_list[1]+";windcode=000906.SH").Data[1]
    d1 = w.tdaysoffset(1, date[0], Period = Period,usedf=True).Data[0][0].strftime('%Y-%m-%d')  # 时间往后推一个周期，并且格式转化为字符串
    d2 = w.tdaysoffset(1, date[-1],Period = Period, usedf=True).Data[0][0].strftime('%Y-%m-%d')
    next_ret_ = w.wsd(stocks, "pct_chg", d1, d2, usedf=True, Period = Period)[1].fillna(value = 0)
    f = lambda x: x/100  # 万矿收益率数据单位为100%，这里换算成小数
    next_ret_ = next_ret_.applymap(f)
    next_ret = zhuanhuan(next_ret_,date,stocks,columns = ['NEXT_RET'])
    
    
     
     
     ## IC测试
    def ic_fenxi(df,next_ret,alpha = ['alpha']):
        d = list(next_ret.index.levels[0]) # 获取双索引中的日期索引
        ic_s = []
        df = df.fillna(value = 0) # 异常值填充为0
        for i in range(len(d)):
            stock_v = stock_valid_df.loc[d[i]].dropna().values # 获取当期有效股票池
            stock_v = list(set(stock_v) & set(df.loc[d[i]].index))
            dff = df.loc[d[i]].loc[stock_v] # 提取当期可用因子值和下期收益数据
            r1 = dff['NEXT_RET']
            # 因子中性化
            dff[alpha] = data_scale_neutral(dff[alpha],d[i].strftime('%Y%m%d'))
            try:
                ic_s.append(r1.corr(dff[alpha],method='spearman')) # 计算因子值与下期收益的秩相关系数，并存进ic值列表
            except:
                ic_s.append(0)
        ic_s = pd.Series(ic_s) # 列表转化为Series，从而能够计算各统计量
        rate = len(ic_s[ic_s>=0])/len(ic_s)  # ic值大于0的比例
        IC_mean = ic_s.mean()  # ic均值
        IC_std = ic_s.std()  # ic标准差
        IC_IR = ic_s.mean()/ic_s.std()  # ic_IR用来衡量因子有效性
        stats = [IC_mean,IC_std,IC_IR,rate]
        return stats,ic_s
        
    ic_cul0 = []  # 累计IC值储存列表
    for i in range(len(data)):
        d = list(next_ret.index.levels[0])
        lc = data[i].loc[d] # 根据调仓频率筛选数据
        lc['NEXT_RET'] = next_ret.NEXT_RET
        if i < len(data)-4:
            list_ic,ic_s = ic_fenxi(lc,next_ret,alpha = ['alpha'+str(i+1)]) # 计算IC统计量
        elif i > len(data)-5:
            list_ic,ic_s = ic_fenxi(lc,next_ret,alpha = ['fengge'+str(i+1-30)]) # 计算风格因子的IC统计量
        print(list_ic)
        ic_cul0.append(ic_s) 
        
    # 计算累计IC
    ic_cul = ic_cul0.copy()
    for i in range(len(ic_cul)):
        x = ic_cul[i]
        for j in range(1,len(x)):
            x[j] = x[j]+x[j-1]
        ic_cul[i] = x
        
    # 各组累计IC时序图
    y1,y2,y3,y4,y5,y6,y7,y8,y9 = ic_cul[0],ic_cul[1],ic_cul[2],ic_cul[3],ic_cul[4],ic_cul[-4],ic_cul[-3],ic_cul[-2],ic_cul[-1]
    d = list(next_ret.index.levels[0]) # 获取时间范围作为横轴
    plt.subplots(figsize=(15,5))  # 图的长宽设置
    plt.rcParams['font.sans-serif'] = ['SimHei'] # 图中显示中文
    plt.rcParams['axes.unicode_minus'] = False   # 正常显示负号
    plt.plot(d,y1,label='alpha1');plt.plot(d,y2,label='alpha2');plt.plot(d,y3,label='alpha3');plt.plot(d,y4,label='alpha4')
    plt.plot(d,y5,label='alpha5');plt.plot(d,y6,label='对数总市值');plt.plot(d,y7,label='20日收益率')
    plt.plot(d,y8,label='20日换手率');plt.plot(d,y9,label='20日波动率')
    plt.legend()
    plt.title('Rank IC值累积图（因子不中性化，T=20）')
    plt.xlabel('回测区间')
    plt.ylabel("IC值")
    
    
    ## 分层测试
    def return_fenxi(df,d,num,alpha = 'alpha'):
        df.fillna(value = 0,inplace = True)
        return_s = []
        for i in range(len(d)):
            stock_v = stock_valid_df.loc[d[i]].dropna().values
            stock_v = list(set(stock_v) & set(df.loc[d[i]].index))
            dff = df.loc[d[i]].loc[stock_v]
            x = dff[alpha]
            x = data_scale_neutral(x,d[i].strftime('%Y%m%d'))
            if x.sum() != 0:  # 筛选掉因子值异常期
                df_i = dff.sort_values(alpha)
                return_list = []
                for j in range(num):
                    n1 = round(len(df_i)*j/num)
                    n2 = round(len(df_i)*(j+1)/num)
                    df_j = df_i.iloc[n1:n2]
                    return_j = df_j['NEXT_RET'].mean()+1
                    return_list.append(return_j)
                return_s.append(return_list)
        x = np.array(return_s).T  # 二维列表转化为二维数组转置
        return_s = [list(i) for i in x]
        # 增加多空组
        top_bottom = []
        for i in range(len(return_s[0])):
            top_bottom.append(return_s[-1][i]-return_s[0][i]+1)
        return_s.append(top_bottom)
        # 计算累计收益
        for i in range(len(return_s)):
            x = return_s[i]
            for j in range(1,len(x)):
                x[j] = x[j]*x[j-1]
            return_s[i] = x
        culmu = [re[-1] for re in return_s]
        # 累计收益与分组次序的两种相关系数
        r1=pd.Series(culmu[:10])
        r2 = pd.Series([i for i in range(10)])
        R = [abs(r1.corr(r2,method='spearman')),abs(r1.corr(r2))]
        return culmu,return_s,R
        
    def fencengceshi(data,next_ret,num):
        data_re,data_re1,data_re2 = [],[],[]
        for i in range(len(data)):
            lc = data[i]
            #lc.fillna(value = 0,inplace = True)
            d = list(next_ret.index.levels[0])
            lc = data[i].loc[d]
            lc['NEXT_RET'] = next_ret.NEXT_RET
            list_re1,list_re,list_re2 = return_fenxi(lc,d,num,alpha = data[i].columns[0])
            data_re.append(list_re)
            data_re1.append(list_re1)
            data_re2.append(list_re2)
            print(data_re1[i],',',data_re2[i]) # 各因子最终各组累计收益
        return data_re
      
    df = fencengceshi(data,next_ret,num=10) # df为各因子分组累计收益序列，分层收益在函数运行时输出
    dff = fencengceshi(data,next_ret,num=1) # 计算等权基准线
    
    # 分层各组收益时序图，探究单个因子
    n = 0
    y1,y2,y3,y4,y5,y6,y7,y8,y9 = df[n][1],df[n][2],df[n][3],df[n][4],df[n][5],df[n][6],df[n][7],df[n][8],df[n][9]
    y10,y11 = df[n][10],df[n][-1]
    d = list(next_ret.index.levels[0])
    plt.subplots(figsize=(15,5))  # 图的长宽设置
    plt.rcParams['font.sans-serif'] = ['SimHei'] # 图中显示中文
    plt.rcParams['axes.unicode_minus'] = False   # 正常显示负号
    plt.plot(d,y1,label='group1');plt.plot(d,y2,label='group2');plt.plot(d,y3,label='group3');plt.plot(d,y4,label='group4')
    plt.plot(d,y5,label='group5');plt.plot(d,y6,label='group6');plt.plot(d,y7,label='group7');plt.plot(d,y8,label='group8')
    plt.plot(d,y9,label='group9');plt.plot(d,y10,label='group10');plt.plot(d,y11,label='top-bottom')
    plt.legend()
    plt.title('alpha1 各组累计净值变化曲线（因子不中性化，T=20）')
    plt.xlabel('回测区间')
    plt.ylabel("净值")
    
    # 多空组收益净值变化，探究多个因子
    y1,y2,y3,y4,y5,y6,y7,y8,y9 = df[0][-1],df[1][-1],df[2][-1],df[3][-1],df[4][-1],df[-4][-1],df[-3][-1],df[-2][-1],df[-1][-1]
    bench = dff[0][0] # 基准线
    d = list(next_ret.index.levels[0])
    plt.subplots(figsize=(15,5))  # 图的长宽设置
    plt.rcParams['font.sans-serif'] = ['SimHei'] # 图中显示中文
    plt.rcParams['axes.unicode_minus'] = False   # 正常显示负号
    plt.plot(d,y1,label='alpha1');plt.plot(d,y2,label='alpha2');plt.plot(d,y3,label='alpha3');plt.plot(d,y4,label='alpha4')
    plt.plot(d,y5,label='alpha5');plt.plot(d,y6,label='对数总市值');plt.plot(d,y7,label='20日收益率')
    plt.plot(d,y8,label='20日换手率');plt.plot(d,y9,label='20日波动率');plt.plot(d,bench,label='中证800')
    plt.legend()
    plt.title('多空组累计净值变化曲线（因子不中性化，T=20）')
    plt.xlabel('回测区间')
    plt.ylabel("净值")
    """
