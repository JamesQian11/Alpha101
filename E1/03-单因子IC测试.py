## 股票池筛选，可交易，非新股，非PT，ST，涨跌停的股票
def get_stocks(trDate):
    trDate = trDate.strftime('%Y-%m-%d')
    stocks_800 = w.wset("sectorconstituent", "date=" + trDate + ";windcode=000906.SH").Data[1]
    status = w.wss(stocks_800, "trade_status,maxupordown,riskwarning,ipo_date", tradeDate=trDate, usedf=True)[1]
    date_least = w.tdaysoffset(-6, trDate, 'Period=M').Data[0][0]
    trade_codes = list(status[(status['TRADE_STATUS'] == '交易') & (status['IPO_DATE'] <= date_least) & (
                status['MAXUPORDOWN'] == 0) & (status['RISKWARNING'] == '否')].index)
    return trade_codes


trade_d = w.tdays("2014-01-02", "2019-05-30", Period='M', usedf=True).Data[0]
stock_valid = []
for i in range(len(trade_d)):
    stock_valid.append(get_stocks(trade_d[i]))
stock_valid_df = pd.DataFrame(stock_valid, index=trade_d)

## 获取下期收益数据
Period = 'M'
# 时间列表设置避免取到重复数据
date_list = ['2014-01-02', '2015-01-09', '2016-01-08', '2017-01-06', '2018-01-05', '2019-05-30']
date_listd = ['2015-01-08', '2016-01-07', '2017-01-05', '2018-01-04', '2019-04-30']
date_listw = ['2015-01-02', '2016-01-01', '2016-12-30', '2017-12-29', '2019-05-30']
date_listm = ['2015-01-01', '2016-01-01', '2017-01-01', '2018-01-01', '2019-04-30']
next_ret = []
bench_date = []  # 之后分层测试可视化需要用到
for i in range(5):
    date = w.wsd('000001.SZ', 'close', date_list[i], date_listm[i], period=Period, usedf=True)[1].index
    stocks = w.wset("sectorconstituent", "date=" + date_listm[i] + ";windcode=000906.SH").Data[1]
    d1 = w.tdaysoffset(1, date[0], Period=Period, usedf=True).Data[0][0].strftime('%Y-%m-%d')  # 时间往后推一个周期，并且格式转化为字符串
    d2 = w.tdaysoffset(1, date[-1], Period=Period, usedf=True).Data[0][0].strftime('%Y-%m-%d')
    bench_date.append(d1)
    bench_date.append(d2)
    next_ret_ = w.wsd(stocks, "pct_chg", d1, d2, usedf=True, Period=Period)[1].fillna(value=0)
    f = lambda x: x / 100  # 万矿收益率数据单位为100%，这里换算成小数
    next_ret_ = next_ret_.applymap(f)
    next_ret.append(zhuanhuan(next_ret_, date, stocks, columns=['NEXT_RET']))  # 转化下期收益数据为双重索引
next_ret = pd.concat(next_ret)  # 合并5年的下期收益数据


## IC测试
def ic_fenxi(df, next_ret, alpha=['alpha']):
    d = list(next_ret.index.levels[0])  # 获取双索引中的日期索引
    ic_s = []
    df = df.fillna(value=0)  # 异常值填充为0
    for i in range(len(date_)):
        stock_v = stock_valid_df.loc[d[i]].dropna().values  # 获取当期有效股票池
        stock_v = list(set(stock_v) & set(df.loc[d[i]].index))
        dff = df.loc[d[i]].loc[stock_v]  # 提取当期可用因子值和下期收益数据
        # dff = dff.sort_values(alpha,ascending=False).iloc[:round(len(dff)/10)] # 提取top组的数据分析
        ic_s.append(dff['NEXT_RET'].corr(dff[alpha], method='spearman'))  # 计算因子值与下期收益的秩相关系数，并存进ic值列表
    ic_s = pd.Series(ic_s)  # 列表转化为Series，从而能够计算各统计量
    rate = len(ic_s[ic_s >= 0]) / len(ic_s)  # ic值大于0的比例
    IC_mean = ic_s.mean()  # ic均值
    IC_std = ic_s.std()  # ic标准差
    IC_IR = ic_s.mean() / ic_s.std()  # ic_IR用来衡量因子有效性
    stats = [IC_mean, IC_std, IC_IR, rate]
    return stats


for i in range(len(data)):
    d = list(next_ret.index.levels[0])
    lc = data[i].loc[d]  # 根据调仓频率筛选数据
    lc['NEXT_RET'] = next_ret.NEXT_RET
    list_ic = ic_fenxi(lc, next_ret, alpha=['alpha' + str(i + 1)])  # 计算IC统计量
    print(list_ic)
