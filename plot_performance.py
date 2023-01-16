# -*- coding: utf-8 -*-
# @Time : 2023/1/16 上午10:11
# @Author : 马卓安 ma_zhuoan@163.com

import pandas as pd
import matplotlib.pyplot as plt

def plot_monthly_return(score_perform, model_perform):
    plt.figure(figsize=(15, 5))
    score_perform = score_perform[-21:] # 选取22年8月后数据
    score_perform.set_index('发行月份', inplace=True)
    plt.plot(score_perform['解禁日年化收益'])
    plt.plot(model_perform['解禁日年化收益'])
    plt.legend(labels=['打分表', '模型'])
    plt.ylabel('解禁日年化收益率')
    plt.xlabel('发行月份')
    plt.xticks(rotation=30)
    plt.show()

if __name__ == '__main__':
    model_perform = pd.read_csv('model_perform.csv')
    score_perform = pd.read_csv('score_perform.csv')
    plot_monthly_return(score_perform, model_perform)