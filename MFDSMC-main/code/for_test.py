
import pandas as pd
import numpy as np
import joblib,json
from utils import rename_with_feat40, metricsScores
import warnings
warnings.filterwarnings("ignore")

#  提取最优特征子集名称
feat146_cols = pd.read_csv(r'.\results\AAAA_featureSelect\feat_importance_sorted_rename.csv')
feat73_cols = feat146_cols['feat_name'][:73]


# 读取特征名称
with open('./data/feat146.json', 'r') as f:     
    feat146 = json.load(f)

# 加载最优模型
model = joblib.load(r'.\model\xgb_clf_mod_feat73.model')
imputer = joblib.load('.\model\imputer.pkl')
minmax = joblib.load('.\model\minmax.pkl')


# 读取测试集数据
metrics = 'Sen, Spe, Pre, F1, MCC, ACC, AUC, AUPR, tn, fp, fn, tp, thres'.split(', ')
# 测试集1
X_test1 = pd.read_csv(r'data\feature-encoded\for-train-test\test1-closeby6-encoded-feat146.csv')
X_test1, renamed_columns = rename_with_feat40(r'data\feature-encoded\for-train-test\test1-closeby6-encoded-feat146.csv') 

X_test1[feat146].replace('na', np.nan,inplace=True)
X_test11 = imputer.transform(X_test1[feat146]) 
X_test11 = minmax.transform(X_test11)

X_test11 = pd.DataFrame(X_test11,columns=feat146)
pred1 = model.predict_proba(X_test11.loc[:,feat73_cols].values)[:,1]

result_df1 = pd.concat([pd.Series(metrics),pd.Series(metricsScores(X_test1['#class'].values,pred1))],axis=1).T
result_df1.columns = result_df1.iloc[0]  # 将第一行设置为列标题
result_df1 = result_df1[1:]  # 去掉原来的第一行
print('The prediction result:\n', result_df1.iloc[:,:-5])


#  测试集2
# X_test2 = pd.read_csv(r'data\feature-encoded\for-train-test\test2-closeby6-encoded-feat146.csv')

# X_test2, renamed_columns = rename_with_feat40(r'data\feature-encoded\for-train-test\test2-closeby6-encoded-feat146.csv')
# print(X_test2.columns.tolist())

# X_test2[feat146].replace('na', np.nan,inplace=True)
# X_test22 = imputer.transform(X_test2[feat146])
# X_test22 = minmax.transform(X_test22)
# X_test22 = pd.DataFrame(X_test22,columns=feat146)
# pred2 = model.predict_proba(X_test22.loc[:,feat73_cols].values)[:,1]
# result_df2 = pd.concat([pd.Series(metrics),pd.Series(metricsScores(X_test2['#class'].values,pred2))],axis=1).T
# result_df2.columns = result_df2.iloc[0]  # 将第一行设置为列标题
# result_df2 = result_df2[1:]  # 去掉原来的第一行

# print(result_df2)
