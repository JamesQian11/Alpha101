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


def cal_IC_IR():
    alpha_path = '/Users/Vision/MyTrees/00_GAM/dataset/test_IC/alpha009'
    returns_path = '/Users/Vision/MyTrees/00_GAM/dataset/test_IC/returns'
    start_time = 20210401
    end_time = 20210430
    day_list = get_time(start_time, end_time)
    IC = pd.DataFrame()
    for day_time in day_list:
        returns_name = 'returns_' + day_time
        alpha_name = 'alph009_' + day_time
        alpha_path_name = f'{alpha_path}/{alpha_name}.csv'
        returns_files_name = f'{returns_path}/{returns_name}.csv'
        print(alpha_path_name)

        single_factor = file_preprocess(alpha_path_name)
        single_factor = extreme_process_MAD(single_factor)
        single_factor = standardize(single_factor)
        returns = file_preprocess(returns_files_name)

        d = list(single_factor.index)  # 获取双索引中的日期索引
        ic_s = []
        for i in range(len(d)):
            if i < 210:
                alpha = single_factor.loc[d[i]]  # 所有的股票取截面的数据
                next_returns = returns.loc[d[i + 30]]
                try:
                    ic_s.append(next_returns.corr(alpha, method='spearman'))  # 计算因子值与下期收益的秩相关系数，并存进ic值列表
                except:
                    ic_s.append(0)
        day_time = alpha_path_name.split('/')[-1].split('.')[0].split('_')[-1]
        IC[day_time] = pd.Series(ic_s, index=single_factor.index[30:])  # 列表转化为Series，从而能够计算各统计量
    IC_names = alpha_path.split('/')[-1] + '_IC.csv'
    IC.to_csv(IC_names, header=True)
    # IC_IR = ic_s.mean() / ic_s.std()

    print(IC)


if __name__ == '__main__':
    print("*********************************************************************************")
    IC_path = '/Users/Vision/MyTrees/00_GAM/alpha101/alpha101-new-master/alpha101_IC'
    IC_list = []
    for d in get_file_names_list(IC_path):
        IC_name = f'{IC_path}/{d}.csv'
        IC = pd.read_csv(IC_name)
        IC.index = IC["time"]
        IC = IC.drop(columns=["time"])
        ic_cul = IC.copy().replace(np.nan, 0.)
        ic_cul = ic_cul.to_numpy().flatten(order='F')
        print(d)
        print(ic_cul.mean())
        for j in range(1, len(ic_cul)):
            ic_cul[j] = ic_cul[j] + ic_cul[j - 1]
        # print('ic_cul', ic_cul[len(ic_cul)-1])
        IC_list.append(ic_cul.tolist())
    values = IC_list[:][-1]
    # 各组累计IC时序图
    y1, y2, y3, y4, y5, y6, y7, y8, y9, y10, y11, y12, y13, y14, y15, y16, y17, y18, y19, y20, y21, y22, y23, y24, y25, y26, y27, y28, y29, y30, y31, y32, y33, y34, y35, y36, y37, y38, y39, y40, y41, y42, y43, y44, y45, y46, y47 = \
        IC_list[0], IC_list[1], IC_list[2], IC_list[3], IC_list[4], IC_list[5], IC_list[6], IC_list[7], IC_list[8], \
        IC_list[
            9], IC_list[10], IC_list[11], IC_list[12], IC_list[13], IC_list[14], IC_list[15], IC_list[16], IC_list[17], \
        IC_list[18], IC_list[19], IC_list[20], IC_list[21], IC_list[22], IC_list[23], IC_list[24], IC_list[25], IC_list[
            26], \
        IC_list[27], IC_list[28], IC_list[29], IC_list[30], IC_list[31], IC_list[32], IC_list[33], IC_list[34], IC_list[
            35], \
        IC_list[36], IC_list[37], IC_list[38], IC_list[39], IC_list[40], IC_list[41], IC_list[42], IC_list[43], IC_list[
            44], \
        IC_list[45], IC_list[46]

    plt.subplots(figsize=(35, 15))  # 图的长宽设置
    plt.plot(y1, label='alpha1')
    plt.plot(y2, label='alpha2')
    plt.plot(y3, label='alpha3')
    plt.plot(y4, label='alpha4')
    plt.plot(y5, label='alpha5')
    plt.plot(y6, label='alpha6')
    plt.plot(y7, label='alpha7')
    plt.plot(y8, label='alpha8')
    plt.plot(y9, label='alpha9')
    plt.plot(y10, label='alpha10')
    plt.plot(y11, label='alpha11')
    plt.plot(y12, label='alpha12')
    plt.plot(y13, label='alpha13')
    plt.plot(y14, label='alpha14')
    plt.plot(y15, label='alpha15')
    plt.plot(y16, label='alpha16')
    plt.plot(y17, label='alpha17')
    plt.plot(y18, label='alpha18')
    plt.plot(y19, label='alpha19')
    plt.plot(y20, label='alpha20')
    plt.plot(y21, label='alpha21')
    plt.plot(y22, label='alpha22')
    plt.plot(y23, label='alpha23')
    plt.plot(y24, label='alpha24')
    plt.plot(y25, label='alpha25')
    plt.plot(y26, label='alpha26')
    plt.plot(y27, label='alpha27')
    plt.plot(y28, label='alpha28')
    plt.plot(y29, label='alpha29')
    plt.plot(y30, label='alpha30')
    plt.plot(y31, label='alpha31')
    plt.plot(y32, label='alpha32')
    plt.plot(y33, label='alpha33')
    plt.plot(y34, label='alpha34')
    plt.plot(y35, label='alpha35')
    plt.plot(y36, label='alpha36')
    plt.plot(y37, label='alpha37')
    plt.plot(y38, label='alpha38')
    plt.plot(y39, label='alpha39')
    plt.plot(y40, label='alpha40')
    plt.plot(y41, label='alpha41')
    plt.plot(y42, label='alpha42')
    plt.plot(y43, label='alpha43')
    plt.plot(y44, label='alpha44')
    plt.plot(y45, label='alpha45')
    plt.plot(y46, label='alpha46')
    plt.plot(y47, label='alpha47')
    # plt.plot(y8, label='alpha28')
    # plt.plot(y9, label='alpha29')
    # plt.plot(y10, label='alpha30')
    # plt.plot(y10, label='alpha20')
    # plt.plot(y1, label='alpha21')
    # plt.plot(y2, label='alpha22')
    # plt.plot(y3, label='alpha23')
    # plt.plot(y4, label='alpha24')
    # plt.plot(y5, label='alpha25')
    # plt.plot(y6, label='alpha26')
    # plt.plot(y7, label='alpha27')
    # plt.plot(y8, label='alpha28')
    # plt.plot(y9, label='alpha29')
    # plt.plot(y10, label='alpha30')
    plt.legend()
    plt.title('Cumulative plot of IC values')
    plt.xlabel('time')
    plt.ylabel("IC")
    plt.show()

    # y1 = ic_cul
    # # d = list(next_ret.index.levels[0])  # 获取时间范围作为横轴
    # plt.subplots(figsize=(15, 5))  # 图的长宽设置
    # plt.plot(y1, label='alpha009')
    #
    # plt.legend()
    # plt.title('Rank IC accumulation')
    # plt.xlabel('time')
    # plt.ylabel("value")
    # plt.show()
    # print('finish')

    # a = ic_cul.shape[1]
    # for i in range(ic_cul.shape[1]):
    #     for j in range(ic_cul.shape[0]):
    #         print(j, i)
    #         ic_cul[j, i] = ic_cul[j, i] + ic_cul[j - 1, i]

    # ic_cul = ic_cul0.copy()
    # for i in range(len(ic_cul)):
    #     x = ic_cul[i]
    #     for j in range(1, len(x)):
    #         x[j] = x[j] + x[j - 1]
    #     ic_cul[i] = x

    # df["cumsum"] = ic_cul.cumsum(axis=0)

    # b = columns[i-1]
    # columns[i] = float(columns[i]) + float(columns[i - 1])

    # for i in range(len(ic_columns)):
    #     x = ic_cul[i]
    # for j in range(1, len(x)):
    #     x[j] = x[j] + x[j - 1]
    # ic_cul[i] = x  # 计算每个i的 factor的累积IC

    # ic_cul0 = []  # 累计IC值储存列表 把所有的IC放在一个list中
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
    #     ic_cul[i] = x # 计算每个i的 factor的累积IC

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

alpha001_IC
ic_cul -931.6247433958802
alpha002_IC
ic_cul 785.9667757452729
alpha003_IC
ic_cul 1488.4236319463546
alpha004_IC
ic_cul 2322.5834621408476
alpha005_IC
ic_cul 3045.5565772261302
alpha006_IC
ic_cul 1413.664231372915
alpha007_IC
ic_cul 972.5594584882492
alpha008_IC
ic_cul 544.1211195766044
alpha009_IC
ic_cul 1528.760519163585
alpha010_IC
ic_cul 1275.3254071184365
alpha011_IC
ic_cul 2007.9637453258235
alpha012_IC
ic_cul 631.7418334118887
alpha013_IC
ic_cul 213.45800520009445
alpha014_IC
ic_cul 1222.4066923453188
alpha015_IC
ic_cul 883.7173743745097
alpha016_IC
ic_cul 1250.239821112333
alpha017_IC
ic_cul 678.8818880409005
alpha018_IC
ic_cul 2546.4684180051786
alpha019_IC
ic_cul 0.0
alpha020_IC
ic_cul 274.3269478486829
alpha021_IC
ic_cul -8.458845744607736
alpha022_IC
ic_cul 104.75269859915629
alpha023_IC
ic_cul 815.1471519166341
alpha024_IC
ic_cul 1894.1159127707824
alpha025_IC
ic_cul 2848.0261463493503
alpha026_IC
ic_cul 1461.2291847304964
alpha027_IC
ic_cul 0.0
alpha028_IC
ic_cul 2525.964839217969
alpha029_IC
ic_cul 380.32970277348204
alpha030_IC
ic_cul 949.5131357721305
alpha031_IC
ic_cul 142.81102052358892
alpha032_IC
ic_cul 1365.7889640874498
alpha033_IC
ic_cul 3036.2142794115357
alpha034_IC
ic_cul 1079.8946859660919
alpha035_IC
ic_cul -826.2822269927502
alpha036_IC
ic_cul 213.0994813241426
alpha037_IC
ic_cul 335.15290044502785
alpha038_IC
ic_cul 3349.1737210438027
alpha039_IC
ic_cul 0.0
alpha040_IC
ic_cul 2469.8722542464598
alpha041_IC
ic_cul 213.18718509203737
        alpha042_IC
        ic_cul 3424.0393458924527
alpha043_IC
ic_cul 593.7946783882719
alpha044_IC
ic_cul 1449.2226197228463
alpha045_IC
ic_cul 410.17919386550545
alpha046_IC
ic_cul -112.8089586492029
alpha047_IC
ic_cul 2461.757573390222
    
    """