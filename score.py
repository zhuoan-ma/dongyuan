import pandas as pd
import numpy as np
import warnings

from sklearn.metrics import accuracy_score, recall_score, f1_score

warnings.filterwarnings ("ignore")


def read_data(data_file_name):
    raw_data = pd.read_excel(data_file_name, sheet_name='Sheet1')
    return raw_data

def clean_data(raw_data, factor=None):
    if not factor:
        data = raw_data.copy()
    else:
        data = raw_data[factor+['代码', '发行日期']].copy()
    data.replace('-', np.nan, inplace=True)
    data.dropna(inplace=True)
    data.drop_duplicates(inplace=True)
    holding_days = pd.to_datetime(raw_data['限售股份解禁日']) - pd.to_datetime(raw_data['发行日期'])
    holding_days = holding_days.apply(lambda x: x.days)
    data['解禁日年化收益'] = raw_data['解禁日绝对收益'] / holding_days * 365
    data['盈利'] = data['解禁日年化收益'].apply(lambda x: 1 if x > 0 else 0)
    return data

def score_model(factor):
    if factor == '预计募资':
        bins = [0, 5, 9, 18, 1000]
        labels = [100, 75, 50, 25]
    elif factor == '总市值':
        bins = [0, 45, 77, 169, 1000]
        labels = [100, 75, 50, 25]
    elif factor == '增发占增发后自由流通市值':
        bins = [0, 0.1, 0.2, 10]
        labels = [100, 66, 33]
    elif factor == '基金占比':
        bins = [-1, 0, 0.02, 0.09, 10]
        labels = [100, 75, 50, 25]
    elif factor == '发行前20日相对行业涨跌幅':
        bins = [-10, -0.03, 0.02, 0.09, 10]
        labels = [25, 50, 75, 100]
    elif factor == '换手率':
        bins = [0, 0.01, 0.02, 0.04, 10]
        labels = [25, 50, 75, 100]
    elif factor == '股价':
        bins = [0, 6, 11, 18, 10000]
        labels = [100, 75, 50, 25]
    return bins, labels

def purpose_score(purpose):
    if purpose == '配套融资':
        return 100
    elif purpose == '项目融资':
        return 66
    else:
        return 33

def score(data, factor):
    for f in factor:
        if f == '增发目的':
            data['增发目的'] = data['增发目的'].apply(lambda x:purpose_score(x))
        else:
            bins, labels = score_model(f)
            data[f] = pd.cut(data[f], bins=bins, labels=labels)
    return data

def final_score(data, factor, weight):
    df = data[factor].values
    w = np.array(weight)
    data['总分'] = pd.Series(np.sum(df*w, axis=1), index=data.index)
    data['购买'] = pd.Series(np.zeros((data.shape[0])))
    bought = data.query('总分 > 60').index
    data['购买'].loc[bought] = 1
    return data

def evaluate(data):
    y_true = data['盈利']
    y_pred = data['购买']
    accuracy = accuracy_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 =f1_score(y_true, y_pred)
    return accuracy, recall, f1


if __name__ == '__main__':
    FACTOR = ['预计募资','总市值','增发占增发后自由流通市值','增发目的',
              '基金占比','发行前20日相对行业涨跌幅','换手率','股价'] # 当前打分表可实现的因子
    DATA_FILE_NAME = '/home/sam/data/FactorData.xlsx'
    WEIGHT = [0.08, 0.08, 0.08, 0.08,
              0.33, 0.13, 0.07, 0.13] # 对应因子权重
    raw_data = read_data(DATA_FILE_NAME)
    data = clean_data(raw_data, FACTOR)
    score_data = score(data, FACTOR)
    final_score_data = final_score(score_data, FACTOR, WEIGHT)
    final_score_data.to_csv('score.csv', index=False)
    accuracy, recall, f1 = evaluate(final_score_data)
    print('Accuracy:{}, Recall:{}, F1:{}'.format(accuracy, recall, f1))
    final_score_data['发行月份'] = final_score_data['发行日期'].apply(lambda x: str(x)[:7])
    bought = final_score_data[final_score_data['购买'] == 1]
    score_perform = bought.groupby(['发行月份'])['解禁日年化收益'].mean()
    score_perform.to_csv('score_perform.csv')
