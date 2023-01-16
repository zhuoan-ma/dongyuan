# -*- coding: utf-8 -*-
# @Time : 2023/1/16 上午11:06
# @Author : 马卓安 ma_zhuoan@163.com

import pickle
import lightgbm as lgb
import pydotplus
import matplotlib.pyplot as plt

if __name__ == '__main__':
     with open('../lgbclf.pkl', 'rb') as lgbclf_file:
         lgbclf = pickle.load(lgbclf_file)
     tree_plot = lgb.create_tree_digraph(lgbclf,tree_index=0)
     tree_plot.view() # 生成Digraph.gv和Digraph.gv.pdf文件
