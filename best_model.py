# -*- coding: utf-8 -*-
# @Time : ${DATE} ${TIME}
# @Author : 马卓安 ma_zhuoan@163.com


import datetime
import bisect
import joblib
import warnings

import pandas as pd
import numpy as np
from datetime import datetime as dt

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, mean_absolute_error, mean_squared_error, r2_score
from sklearn.svm import NuSVR

warnings.filterwarnings("ignore")


def read_data(data_file_name1, data_file_name2):
    df1 = pd.read_excel(data_file_name1, sheet_name='Sheet1')
    df2 = pd.read_excel(data_file_name2, sheet_name='Sheet1')
    return df1, df2

def clean_data(data1, data2, read_clean_data=False):
    if read_clean_data: return pd.read_csv('all_data.csv')
    # 处理定增因子.xlsx
    data1.replace('-', np.nan, inplace=True)
    data1.dropna(inplace=True)
    data1.drop_duplicates(inplace=True)
    holding_days = pd.to_datetime(data1['限售股份解禁日']) - pd.to_datetime(data1['发行日期'])
    holding_days = holding_days.apply(lambda x: x.days)
    data1['解禁日年化收益'] = data1['解禁日绝对收益'] / holding_days * 365
    data1['盈利'] = data1['解禁日年化收益'].apply(lambda x: 1 if x > 0 else 0)
    data1['发行月份'] = data1['发行日期'].apply(lambda x:str(x)[:7])
    data1['月份'] = data1['发行日期'].apply(lambda x:str(x)[5:7])
    data1.drop(columns=['限售股份解禁日','证监会通过公告日', '最新报告期', '调整报告期', '前次报告期', '基金变化', '年度报告期1',
                        '上市日期', '解禁日收盘价（定点复权）', '沪深300涨跌幅', '解禁日相对沪深300', '中证500涨跌幅',
                        '解禁日相对中证500', '行业涨跌幅', '解禁日相对行业'], inplace=True)
    # 处理商业模式涉及财务指标20221230数据版.xlsx
    data2.drop(columns=['最新报告期', '年度报告期1', '年度报告期2', '年度报告期3', '成立日期','上市日期'], inplace=True)
    data2.replace('-', np.nan, inplace=True)
    data2['存货周转率'].replace(0, np.nan, inplace=True)
    data2.dropna(inplace=True)
    data2.drop_duplicates(inplace=True)
    data1.set_index('代码')
    data2.set_index('代码')
    # 合并数据
    all_data = pd.concat([data1, data2], axis=1, join='inner')
    all_data.to_csv('all_data.csv', index=False)
    return all_data

def rbf_svm_reg(all_data, num_factor=25):
    if num_factor > 0:
        factor = FACTOR[:num_factor]
    else:
        factor = FACTOR
    # 对all_data.csv 2022年1月13日版，0-1077 2020前数据， 1078-1161 2021数据 1161- 2022数据
    nontest = all_data.iloc[:1078].copy()
    test = all_data.iloc[1078:].copy()
    # 归一化
    mu = nontest[factor].mean(axis=0)
    sigma = nontest[factor].std(axis=0)
    nontest[factor] = (nontest[factor] - mu) / sigma
    test[factor] = (test[factor] - mu) / sigma
    # 剔除异常值
    cond = nontest[factor] < -3
    nontest[factor][cond] = -3
    cond = nontest[factor] > 3
    nontest[factor][cond] = 3
    cond = test[factor] < -3
    test[factor][cond] = -3
    cond = test[factor] > 3
    test[factor][cond] = 3
    # rbf核回归
    target='解禁日年化收益'
    X_nontest = nontest[factor]
    y_nontest = nontest[target]
    X_test = test[factor]
    y_test = test[target]
    X_train, X_valid, y_train, y_valid = train_test_split(X_nontest, y_nontest,test_size=0.1)
    # 目前高斯核SVM回归表现最佳
    svmreg = NuSVR(kernel='rbf')
    svmreg.fit(X_train, y_train)
    joblib.dump(svmreg, 'rbfsvmreg.pkl')
    y_test_pred = svmreg.predict(X_test)
    l2 = mean_squared_error(y_test, y_test_pred)
    l1 = mean_absolute_error(y_test, y_test_pred)
    r2 = r2_score(y_test, y_test_pred)
    print('l2:{}, l1:{}, r2:{}'.format(l2, l1, r2))
    test['预测解禁日年化收益'] = pd.Series(y_test_pred, index=X_test.index)
    return test
    
    


if __name__ == '__main__':
    FACTOR = ['股价', '前次占比', '总市值','发行价格', '增发占增发后自由流通市值', 'PS百分位', '基金占比',
              '净利FY0', '总资产增速3','净利润增速', '净利FY2', '总资产增速1', '净资产增速', 'PE百分位',
              '发行前20日股价涨跌幅', '净利FY1','预计募资', '行业20日涨跌幅', 'ROE', 'ROE1', '投入资本回报率1',
              'ROE波动率', '研发营收比','发行前20日相对行业涨跌幅', '总资产增速2', '货币资金/有息负债3', '上市年数',
              '毛利率3', '存货周转率','成立年数', '经营性现金流净额/净利润', '营业收入/长期资本', '投入资本回报率3',
              '经营性现金流增速', 'ROE3','现金收入比', '财务杠杆系数', '经营活动产生的现金流净额/营业收入',
              '货币资金/有息负债1', '期间费用率','未来1年净利润增速', '机构持股比例', '投入资本回报率2', '经营资产占比',
              '毛利率1', '未来2年净利润增速','ROE2', '毛利率波动性', '资产专用度', '经营杠杆系数', '经营层持股比例',
              '毛利率变异系数', '货币资金/有息负债2', '固定资产周转率', '闪发', '毛利率2', '换手率']
    discount = True # 是否使用折扣率作为因子
    num_factor = 25 # 使用的因子数量
    read_clean_data = True # 读取干净数据文件
    if discount:
        FACTOR = ['折扣率'] + FACTOR
    DATA_FILE_NAME1 = '/home/sam/data/定增因子.xlsx'
    DATA_FILE_NAME2 = '/home/sam/data/商业模式涉及财务指标 20221230数据版.xlsx'
    data1, data2 = read_data(DATA_FILE_NAME1, DATA_FILE_NAME2)
    all_data = clean_data(data1, data2, read_clean_data)
    test = rbf_svm_reg(all_data, num_factor)
    bought = test[test['预测解禁日年化收益'] > 0]
    model_perform = bought.groupby(['发行月份'])['解禁日年化收益'].mean()
    model_perform.to_csv('model_perform.csv')


