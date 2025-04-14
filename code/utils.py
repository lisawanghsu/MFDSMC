import json, os
import pandas as pd
import numpy as np
from sklearn import preprocessing
from xgboost import XGBClassifier
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier, VotingClassifier, StackingClassifier
from sklearn.metrics import confusion_matrix, recall_score, precision_score, f1_score, matthews_corrcoef, \
    accuracy_score, roc_auc_score, precision_recall_curve,auc
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold, cross_val_score, GridSearchCV

from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
import joblib

from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
import sklearn.model_selection as sk_model_selection
import matplotlib.pyplot as plt
from pathlib import Path

def metricsScores(y_true, pred_proba, thres=0.50):
    '''
    函数功能：评估指标计算
    :param y_true: 真实标签
    :param pred_proba: 预测值
    :param thres: 分类阈值,默认为0.5
    :return: 计算的指标列表，依次为[Sen, Spe, Pre, F1, MCC, ACC, AUC, PRC, tn, fp, fn, tp, thres]
    '''

    y_pred = [(0. if item < thres else 1.) for item in pred_proba]  # pred_proba > thres
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()  # 混淆矩阵tn, fp, fn, tp
    Sen = recall_score(y_true, y_pred)  # 召回率、灵敏度
    Spe = tn / (tn + fp)  # 特异性
    Pre = precision_score(y_true, y_pred)  # 精确度
    F1 = f1_score(y_true, y_pred)  # F1-scores
    MCC = matthews_corrcoef(y_true, y_pred)  # 马修斯相关系数
    ACC = accuracy_score(y_true, y_pred)  # 准确度
    AUC = roc_auc_score(y_true, pred_proba)  # ROC曲线下围面积
    precision_prc, recall_prc, _ = precision_recall_curve(y_true, pred_proba)  # P-R曲线中x,y轴的数组
    PRC = auc(recall_prc, precision_prc)  # P-R曲线下围面积
    metrics_list = [Sen, Spe, Pre, F1, MCC, ACC, AUC, PRC, tn, fp, fn, tp, thres]
    return metrics_list


def fillMeanMaxMin(preXtrain,testData=None,fillMethod="mean"):
    '''
    函数功能：对训练和测试数据（默认没有）以均值填充缺失值并进行maxmin归一化
    :param preXtrain: 训练数据
    :param testData: 测试数据
    :param fillMethod: 缺失值填充方法，默认为均值。
                    可选方法有allowed_fillMethod = ["mean", "median", "most_frequent", "constant"]
    :return: 返回填充和归一化的训练和测试数据（如果有）
    '''
    
    from sklearn import preprocessing
    from sklearn.impute import SimpleImputer
    # 训练集数据均值填充，使用训练集数据填充验证集
    imp_mean = SimpleImputer(strategy=fillMethod)
    Xtrain = imp_mean.fit_transform(preXtrain)
    # Xtest = imp_mean.transform(testData)
    # 归一化处理
    maxmin = preprocessing.MinMaxScaler()
    Xtrain = maxmin.fit_transform(Xtrain)
    # Xtest = maxmin.transform(Xtest)
    return Xtrain
   

def fillMeanMaxMin1(preXtrain,testData=None,fillMethod="mean"):
    '''
    函数功能：对训练和测试数据（默认没有）以均值填充缺失值并进行maxmin归一化
    :param preXtrain: 训练数据
    :param testData: 测试数据
    :param fillMethod: 缺失值填充方法，默认为均值。
                    可选方法有allowed_fillMethod = ["mean", "median", "most_frequent", "constant"]
    :return: 返回填充和归一化的训练和测试数据（如果有）
    '''
    
    from sklearn import preprocessing
    from sklearn.impute import SimpleImputer
    # 训练集数据均值填充，使用训练集数据填充验证集
    imp_mean = SimpleImputer(strategy=fillMethod)
    Xtrain = imp_mean.fit_transform(preXtrain)
   
    # 归一化处理
    maxmin = preprocessing.MinMaxScaler()
    Xtrain = maxmin.fit_transform(Xtrain)
    # temp = []
    # if testData:
    #     for i in range(len(testData)):
    #         temp.append(imp_mean.transform(testData[i].values))
    Xtest1 = imp_mean.transform(testData[0].values)
    Xtest2 = imp_mean.transform(testData[1].values)
    Xtest1 = maxmin.transform(Xtest1)
    Xtest2 = maxmin.transform(Xtest2)
    return Xtrain,Xtest1,Xtest2
    
def fillMeanMaxMin2(preXtrain,testData=None,fillMethod="mean"):
    '''
    函数功能：对训练和测试数据（默认没有）以均值填充缺失值并进行maxmin归一化
    :param preXtrain: 训练数据
    :param testData: 测试数据
    :param fillMethod: 缺失值填充方法，默认为均值。
                    可选方法有allowed_fillMethod = ["mean", "median", "most_frequent", "constant"]
    :return: 返回填充和归一化的训练和测试数据（如果有）
    '''
    
    from sklearn import preprocessing
    from sklearn.impute import SimpleImputer
    # 训练集数据均值填充，使用训练集数据填充验证集
    imp_mean = SimpleImputer(strategy=fillMethod)
    Xtrain = imp_mean.fit_transform(preXtrain)
   
    # 归一化处理
    maxmin = preprocessing.MinMaxScaler()
    Xtrain = maxmin.fit_transform(Xtrain)
    # temp = []
    # if testData:
    #     for i in range(len(testData)):
    #         temp.append(imp_mean.transform(testData[i].values))
    Xtest = imp_mean.transform(testData.values)
    
    Xtest = maxmin.transform(Xtest)
   
    return Xtrain,Xtest,imp_mean,maxmin


# def trainFillMean(preXtrain,fillMethod="mean"):
#     '''
#     allowed_fillMethod = ["mean", "median", "most_frequent", "constant"]
#     '''
#     #填充
#     pipe = Pipeline(steps=[('missing_values', SimpleImputer(missing_values=np.nan, strategy=fillMethod, )),
#                            ('minmax_scaler', MinMaxScaler())])
#     Xtrain = pipe.fit_transform(preXtrain)
#     return Xtrain

def readData(test=True, without_feats=[5,6,7,8,9,10,11,12,13,14,15,16,17,18,19],save=False):
    '''
    函数功能：读取训练集和测试集数据,并进行数据预处理，包括特征列删除（缺失值超过70%）和特征缺失值填充。
    :param test: 是否读取测试集数据
    :param without_feats: 去除的特征列
    :param save: 是否保存数据    
    :return: 返回训练集和测试集数据
    '''

    #训练集数据读取
    train_data = pd.read_csv(r'.\data\feature-encoded\train_pos_neg_closeby6_20250118-manual-del.csv')
    print('由synMall注释并去除有大量缺失值特征列后的训练数据：', train_data.shape)
    featnames = train_data.columns.values

    # PredDSMC采用的全部33维特征(除了7个保守性特征)
    merge_pos_neg_feat33 = pd.concat([pd.read_csv(r'E:\WLH\bioinformatics\孙建辉\代码和数据\train_pos_closeby6.vcf',sep='\t',header=None),
                                    pd.read_csv(r'E:\WLH\bioinformatics\孙建辉\代码和数据\train_neg_closeby6.vcf',sep='\t',header=None)], axis=0).drop(without_feats,axis=1)
    train_data_X = pd.merge(train_data.iloc[:,:-1], merge_pos_neg_feat33, left_on='#CHROM	POS	ID	REF	ALT'.split(),right_on=[0,1,2,3,4], how='left')

    X_train = train_data_X.drop([0,1,2,3,4],axis=1).iloc[:,5:]
    Y_train = train_data.iloc[:,-1]
    print('X_train:', X_train.columns, X_train.shape)
    
    def readTestData():
        #------------------------------------------------------------
        #测试集2数据读取
        test2_data = pd.read_csv(r'.\data\feature-encoded\test2_pos_neg_closeby6_20250119.csv')
        print('由synMall注释并去除有大量缺失值特征列后的test2_data:', test2_data.shape)
        test2_data = test2_data[featnames]

        # PredDSMC采用的全部33维特征
        test2_merge_pos_neg_feat33 = pd.concat([pd.read_csv(r'E:\WLH\bioinformatics\孙建辉\代码和数据\test2_pos_closeby6_process_cs_quchong_closeby_match_features.txt',sep='\t',header=None),
                                        pd.read_csv(r'E:\WLH\bioinformatics\孙建辉\代码和数据\test2_neg_closeby6_process_cs_quchong_closeby_match_features.txt',sep='\t',header=None)], axis=0).drop(without_feats,axis=1)
        test2_data_X = pd.merge(test2_data.iloc[:,:-1], test2_merge_pos_neg_feat33, left_on='#CHROM	POS	ID	REF	ALT'.split(),right_on=[0,1,2,3,4], how='left')

        X_test2 = test2_data_X.drop([0,1,2,3,4],axis=1).iloc[:,5:]
        Y_test2 = test2_data.iloc[:,-1]
        print('test2:', X_test2.columns, X_test2.shape)
        
        #------------------------------------------------------------

        #测试集1数据读取
        test1_data = pd.read_csv(r'.\data\feature-encoded\test1_pos_neg_closeby6_20250119.csv')
        print('由synMall注释并去除有大量缺失值特征列后的test1_data:', test1_data.shape)
        test1_data = test1_data[featnames]

        # PredDSMC采用的全部33维特征
        test1_merge_pos_neg_feat33 = pd.concat([pd.read_csv(r'data\feature-encoded\test1_pos_closeby6_process_cs_quchong_closeby_match_features.txt',sep='\t',header=None,na_values=['nan', 'na', 'Na', '']),
                                        pd.read_csv(r'data\feature-encoded\test1_neg_closeby6_process_cs_quchong_closeby_match_features.txt',sep='\t',header=None,na_values=['nan', 'na', 'Na', ''])], axis=0).drop([5,6,7,8,9,10,11,12,13,14,15,16,17,18,19],axis=1)
        test1_data_X = pd.merge(test1_data.iloc[:,:-1], test1_merge_pos_neg_feat33, left_on='#CHROM	POS	ID	REF	ALT'.split(),right_on=[0,1,2,3,4], how='left')

        X_test1 = test1_data_X.drop([0,1,2,3,4],axis=1).iloc[:,5:].astype(float)
        Y_test1 = test1_data.iloc[:,-1]
        print('test1:', X_test1.columns, X_test1.shape)    

        assert X_test1.columns.all() == X_test2.columns.all()
        return X_test1, Y_test1, X_test2, Y_test2
    
    Xtrain_after_del,columns_remain = delete70NaN(X_train)
    print('特征列超过70%缺失值的列是:',set(X_train.columns.values)-set(X_train.columns.values[columns_remain]))

    feature_remain = X_train.columns.values[columns_remain]
    pd.DataFrame({'feat_index':columns_remain, 'feat_name':feature_remain}).to_csv(f'data/feature-encoded/for-train-test/feat_remain{Xtrain_after_del.shape[1]}.csv',index=False)


    if test: 
        X_test1, Y_test1, X_test2, Y_test2 = readTestData()   
        X_train, X_test1,X_test2 = fillMeanMaxMin1(Xtrain_after_del,[X_test1.iloc[:,columns_remain],X_test2.iloc[:,columns_remain]])
    else:
        X_train = fillMeanMaxMin(Xtrain_after_del)
    if test and save:
        columns=feature_remain.tolist()+['#class']
        print(columns)
        pd.concat([pd.DataFrame(X_train),Y_train], axis=1).to_csv(f'data/feature-encoded/for-train-test/train-closeby6-encoded-feat{X_train.shape[1]}.csv',index=False,header=columns)
        pd.concat([pd.DataFrame(X_test2),Y_test2], axis=1).to_csv(f'data/feature-encoded/for-train-test/test2-closeby6-encoded-feat{X_train.shape[1]}.csv',index=False,header=columns)
        pd.concat([pd.DataFrame(X_test1),Y_test1], axis=1).to_csv(f'data/feature-encoded/for-train-test/test1-closeby6-encoded-feat{X_train.shape[1]}.csv',index=False,header=columns)

def delete70NaN(Xtrain,threshold = 0.7):
    '''
    函数功能：删除缺失值多于70%的特征列
    :param Xtrain: 输入数据
    :param threshold: 删除阈值，默认为0.7
    :return: 返回去除超过70%缺失值列后的训练数据，以及去除70%缺失值列后的剩余列
    '''

    columnNumber = Xtrain.shape[1]  #原始列数
    columns_remain = []
    Xtrain = pd.DataFrame(Xtrain)
    Xtrain.columns =[i for i in range(0,columnNumber)]
    for i in range(0, columnNumber):
        num = Xtrain[i].isna().sum()
        proportion = num / len(Xtrain[i])
        # print(i, ":", proportion)
        if proportion < threshold:
            columns_remain.append(i)
    afterTrain = Xtrain.loc[:, columns_remain]
    return afterTrain,columns_remain

def delete70NaN1(Xtrain,threshold = 0.7):
    '''
    函数功能：删除缺失值多于70%的特征列
    :param Xtrain: 输入数据
    :param threshold: 删除阈值，默认为0.7
    :return: 返回去除超过70%缺失值列后的训练数据，以及去除70%缺失值列后的剩余列
    '''
    columnNumber = Xtrain.shape[1]  #原始列数
    columns_remain = []
    if isinstance(Xtrain, pd.DataFrame):        
        for i in range(0, columnNumber):
            num = Xtrain.iloc[:,i].isna().sum()
            proportion = num / len(Xtrain.iloc[:,i])
            # print(i, ":", proportion)
            if proportion < threshold:
                columns_remain.append(Xtrain.columns.values[i])
    else:        
        Xtrain = pd.DataFrame(Xtrain)
        Xtrain.columns =[i for i in range(0,columnNumber)]
        for i in range(0, columnNumber):
            num = Xtrain[i].isna().sum()
            proportion = num / len(Xtrain[i])
            # print(i, ":", proportion)
            if proportion < threshold:
                columns_remain.append(i)
    afterTrain = Xtrain.loc[:, columns_remain]
    return afterTrain,columns_remain

def model_eval(Xtrain, Ytrain, Xtest, Ytest, model, savepred = None):
    '''
    在全部训练集上训练模型，然后在测试集上得到预测值，并计算测试集上的评价指标
    :param Xtrain: 训练数据特征
    :param Ytrain:  训练数据标签
    :param Xtest: 测试数据特征
    :param Ytest: 测试数据标签
    :param model: 评估采用的分类器
    :param savepred: 是否存储预测结果，默认否，若存储，则此处输入待保存文件名
    :return:  返回测试集上各评价指标
    '''

    model.fit(Xtrain,Ytrain)
    ypred = model.predict_proba(Xtest)[:,1]
    print(ypred)
    if savepred != None:
        temp_ypred = [round(item, 4) for item in ypred]
        temp_ypred = pd.DataFrame(temp_ypred)
        ypred_ytrue = pd.concat([temp_ypred,Ytest],ignore_index=True,axis=1)
        ypred_ytrue.to_csv(savepred,index = None,header=None)
    score = metricsScores(Ytest,ypred)
    return score

def fold(model,X_train, Y_train, cv = 10 ):
    '''
    功能：在训练集上进行k折交叉验证，返回各项指标在k折上的平均结果
    :param model:  分类器
    :param X_train:  训练数据的特征
    :param Y_train:  训练数据的标签
    :param cv:  k折交叉验证，默认为十折
    :return: k折各项指标结果的最后一列，即返回各项指标在k折上的平均结果
    '''

    metrics_list = []
    kf = StratifiedKFold(n_splits=cv)
    for train_index, test_index in kf.split(X_train, Y_train):
        Xtrain_cross, Xtest_cross = X_train[train_index], X_train[test_index]
        Ytrain_cross, Ytest_cross = Y_train[train_index], Y_train[test_index]
        model.fit(Xtrain_cross, Ytrain_cross)
        y_pred = model.predict_proba(Xtest_cross)[:, 1]
        metrics_list.append(metricsScores(Ytest_cross, y_pred))
    metrics_list.append(np.mean(metrics_list, axis=0))
    print(model.__class__.__name__, metrics_list[-1])
    metrics_list = pd.DataFrame(metrics_list)

    return metrics_list.iloc[-1]

def params_search(X, Y, method):
    '''
    函数功能：对模型进行参数搜索，并保存（保存目录为model)各种模型下的使用最优参数训练得到的模型。
    :param X: 训练数据特征
    :param Y: 训练数据标签
    :param method: 模型名称，可选值为['svm', 'rfc', 'adaboost','xgboost','gbdt','vc']
    :return: 无
    '''
    os.makedirs("model", exist_ok=True)
    def train_svm_classifier(X, Y):
        """
        :param kernel: select from linear, poly, rbf, sigmoid
        :param C: penalty parameter C of the error term
        """
        print ("Training svm classifier")
        # 定义支持向量机超参数的搜索范围
        parameters = {
            'C': [0.1, 1, 10, 100],  # 惩罚参数，值越大对误分类惩罚越大
            'kernel': ['linear', 'rbf', 'poly'],  # 核函数类型
            'degree': [2, 3, 4],  # 多项式核函数的次数
            'gamma': ['scale', 'auto']  # rbf、poly等核函数的系数
        }
        clf = GridSearchCV(SVC(probability=True), param_grid=parameters, scoring="roc_auc", cv=10, verbose=1, n_jobs=-1)
        # Y = Y.astype(np.float64)
        # # print(X.dtype, Y.dtype)
        clf.fit(X, Y)
        print ("Training finished")
        print ("The best parameters: ", clf.best_params_)
        # y_result = clf.predict(X)
        # y_prob = clf.predict_proba(X)
        # print(y_result,y_prob)        
        joblib.dump(clf.best_estimator_, f"model/svm_clf_mod_feat{X.shape[1]}.model")

    def train_rfc_classifier(X, Y):
        """
        :param n_estimators: the number of estimators for the forest
        """
        print("Training random forest classifier")
        parameters = {
            "n_estimators": [20, 50, 100, 200, 500, 1000, 2000, 3000, 4000, 5000],
        }
        clf = GridSearchCV(RandomForestClassifier(), param_grid=parameters, scoring="roc_auc", cv=10, verbose=1, n_jobs=-1)
        clf.fit(X, Y)
        print("Training finished")
        print("The best parameters:", clf.best_params_)
        y_result = clf.predict(X)
        y_prob = clf.predict_proba(X)
        joblib.dump(clf.best_estimator_, f"model/rfc_clf_mod_feat{X.shape[1]}.model")


    def train_adaboost_classifier(X, Y):
        """
        :param n_estimators: number of estimators
        :param learning_rate: learning rate
        :param algorithm: SAMME or SAMME.R, but SAMME.R typically better
        """
        print("Training Adabooost classifier")
        parameters = {
            "n_estimators": [20, 50, 100, 200, 500, 1000, 2000, 3000, 5000],
            "learning_rate": [0.01, 0.1, 1, 10],
            # "algorithm": ["SAMME.R"],
        }
        clf = GridSearchCV(AdaBoostClassifier(), param_grid=parameters, scoring="roc_auc", cv=10, verbose=1, n_jobs=-1)
        clf.fit(X, Y)
        print("Training finished")
        print("The best parameters:", clf.best_params_)
        y_result = clf.predict(X)
        y_prob = clf.predict_proba(X)
        joblib.dump(clf.best_estimator_, f"model/adaboost_clf_mod_feat{X.shape[1]}.model")


    def train_gbdt_classifier(X, Y):
        """
        :param n_estimators: number of estimators
        :param learning_rate: learning rate
        :param criterion: function of measuring the quality of a split:friedman_mse (generally better), mse, mae
        """
        print("Training Gradient Boosting Decision Tree")
        parameters = {
            "n_estimators": [100, 500, 1000, 2000],
            "learning_rate": [0.01, 0.1, 1, 10],
            "criterion": ["mse", "friedman_mse", "mae"]
        }
        clf = GridSearchCV(GradientBoostingClassifier(), param_grid=parameters, scoring="roc_auc", cv=10, verbose=1, n_jobs=-1)
        clf.fit(X, Y)
        print("Training finished")
        print("The best parameters:", clf.best_params_)
        y_result = clf.predict(X)
        y_prob = clf.predict_proba(X)
        joblib.dump(clf.best_estimator_, f"model/gbdt_clf_mod_feat{X.shape[1]}.model")


    def train_vc_classifier(X, Y):
        """
        :param estimators: list of tuples, containing the name and classifier type
        :param voting: different prediction criterions: hard, soft
        """
        print("Training voting classifier")
        parameters = {
            "voting": ["soft"],
        }
        clf = GridSearchCV(VotingClassifier([("RF", RandomForestClassifier(n_estimators=1000, random_state=1)),
                            ("GBDT", GradientBoostingClassifier(n_estimators=1000))]),
                            param_grid=parameters, scoring="roc_auc", cv=10, verbose=1, n_jobs=-1)
        clf.fit(X, Y)
        print("Training finished")
        print("The best parameters:", clf.best_params_)
        y_result = clf.predict(X)
        joblib.dump(clf.best_estimator_, f"model/vc_clf_mod_feat{X.shape[1]}.model")


    def train_mlp_classifier(X, Y):
        """
        No explicit parameters, if you like to change the network, then change it here
        """
        print("Training MLP classifier")
        # 定义多层感知机超参数的搜索范围
        parameters = {
            'hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 50)],  # 隐藏层神经元数量
            'activation': ['relu', 'tanh','logistic'],  # 激活函数
            'learning_rate': ['constant', 'adaptive'],  # 学习率策略
            'alpha': [0.0001, 0.001, 0.01]  # L2正则化系数
        }

        # 创建多层感知机分类器对象
        mlp = MLPClassifier(max_iter=500, random_state=42)
        # clf = MLPClassifier(hidden_layer_sizes=(64, 32 ), activation="relu", solver="adam",
        #                         alpha=1e-4, batch_size=64, learning_rate='constant', learning_rate_init=1e-3,
        #                         power_t=0.5, max_iter=500, shuffle=True, random_state=None, tol=1e-4,
        #                         verbose=False, warm_start=False, momentum=0.9, nesterovs_momentum=True,
        #                         early_stopping=False, validation_fraction=0.1, beta_1=0.9, beta_2=0.999,
        #                         epsilon=1e-8)
        clf = GridSearchCV(mlp, param_grid=parameters, scoring="roc_auc", cv=10, verbose=1, n_jobs=-1)
        clf.fit(X, Y)
        print("Training finished")
        y_result = clf.predict(X)
        y_prob = clf.predict_proba(X)
        joblib.dump(clf, f"model/mlp_clf_mod_feat{X.shape[1]}.model")


    def train_xgb_classifier(X, Y):
        """
        Including multiple parameters to be evaluated
        """
        print("Training XGBoost classifier")
        parameters = {
            "booster": ["gbtree"],
            "n_estimators": [100, 500, 1000, 2000],
            "learning_rate": [0.01, 0.05, 0.1],
            #"gamma": np.arange(0, 10, 0.2),
            "max_depth": [5, 6, 7, 8, 9, 10 ],
    #         "reg_alpha": [1, 0.1, 0.01, 0.001, 0],
    #         "reg_lambda": [1, 0.1, 0.01, 0.001, 0],
            #"scale_pos_weight": np.arange(1, 11, 1),
            #"subsample": [0.8, 0.2, 0.4, 0.6, 1.0],
            #"colsample_bylevel": np.arange(0, 1.1, 0.1),
            #"colsample_bytree": np.arange(0, 1.2, 0.2),
            #"min_child_weight": np.arange(0, 10, 1),
        }
        clf = GridSearchCV(XGBClassifier(), param_grid=parameters, scoring="roc_auc", cv=10, verbose=1, n_jobs=-1)
        clf.fit(X, Y)
        print("Training finished")
        print("The best parameters:", clf.best_params_)
        joblib.dump(clf.best_estimator_, f"model/xgb_clf_mod_feat{X.shape[1]}.model")

    def train_dt_classifier(X, Y):
        """
        Including multiple parameters to be evaluated
        """
        print("Training decision tree classifier")
        # 定义要调参的超参数范围
        parameters = {
            'max_depth': [3, 5, 7, 10],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['sqrt', 'log2', None]
        }
        clf = GridSearchCV(DecisionTreeClassifier(), param_grid=parameters, scoring="roc_auc", cv=10, verbose=1, n_jobs=-1)
        clf.fit(X, Y)
        print("Training finished")
        print("The best parameters:", clf.best_params_)
        joblib.dump(clf.best_estimator_, f"model/dt_clf_mod_feat{X.shape[1]}.model")

    def train_lr_classifier(X, Y):
        """
        Including multiple parameters to be evaluated
        """
        print("Training logistic regression classifier")
        # 定义要调参的超参数范围
        # 定义逻辑回归超参数的搜索范围
        parameters = {
            'C': [0.001, 0.01, 0.1, 1, 10, 100],  # 正则化强度的倒数，值越小正则化越强
            'penalty': ['l1', 'l2'],  # 正则化类型，l1为Lasso，l2为Ridge
            'solver': ['liblinear', 'lbfgs', 'saga']  # 优化算法
        }
        clf = GridSearchCV(LogisticRegression(), param_grid=parameters, scoring="roc_auc", cv=10, verbose=1, n_jobs=-1)
        clf.fit(X, Y)
        print("Training finished")
        print("The best parameters:", clf.best_params_)
        joblib.dump(clf.best_estimator_, f"model/lr_clf_mod_feat{X.shape[1]}.model")        
    # %%
    train_x = X
    train_y = Y
    if method == "svm":
        train_svm_classifier(train_x, train_y)
    elif method == "rfc":
        train_rfc_classifier(train_x, train_y)
    elif method == "adaboost":
        train_adaboost_classifier(train_x, train_y)
    elif method == "gbdt":
        train_gbdt_classifier(train_x, train_y)
    elif method == "vc":
        train_vc_classifier(train_x, train_y)
    elif method == "mlp":
        train_mlp_classifier(train_x, train_y)
    elif method == "xgboost":
        train_xgb_classifier(train_x, train_y)
    elif method == "dt":
        train_dt_classifier(train_x, train_y)
    elif method == "lr":
        train_lr_classifier(train_x, train_y)
    elif method == 'all':
        train_svm_classifier(train_x, train_y)
        train_rfc_classifier(train_x, train_y)
        train_adaboost_classifier(train_x, train_y)
        train_gbdt_classifier(train_x, train_y)
        train_vc_classifier(train_x, train_y)
        train_mlp_classifier(train_x, train_y)
        train_xgb_classifier(train_x, train_y)
        
    else:
        print("Parameters wrongly specified")

def params_default(X, Y):
    '''
    函数功能：对模型使用默认参数训练
    :param X: 训练数据特征
    :param Y: 训练数据标签
    返回值：
        所有模型名称
        各模型在训练集上的十折交叉验证结果'''
    models = [RandomForestClassifier( random_state=1),
              GradientBoostingClassifier(random_state=1),
              DecisionTreeClassifier(random_state=1),
              LogisticRegression(random_state=1),
              XGBClassifier(random_state=1),
              MLPClassifier(hidden_layer_sizes=(64, 32), activation="relu", solver="adam"),
              SVC(gamma="scale", probability=True,random_state=1),
              AdaBoostClassifier(random_state=1),
            #   VotingClassifier([("RF", RandomForestClassifier( random_state=1)),
            #                     ("GBDT", GradientBoostingClassifier(random_state=1)),
            #                     ("XGB", XGBClassifier(random_state=1)),
            #                     ('MLP', MLPClassifier(hidden_layer_sizes=(64, 32), activation="relu", solver="adam")),
            #                     ('SVM', SVC(gamma="scale", probability=True,random_state=1)),                                
            #                     ('DT', DecisionTreeClassifier(random_state=1)),
            #                     ('LR', LogisticRegression())],voting="soft"),
            #   StackingClassifier([("RF", RandomForestClassifier( random_state=1)),
            #                     ("GBDT", GradientBoostingClassifier(random_state=1)),
            #                     ("XGB", XGBClassifier(random_state=1)),
            #                     ('MLP', MLPClassifier(hidden_layer_sizes=(64, 32), activation="relu", solver="adam")),
            #                     ('SVM', SVC(gamma="scale", probability=True,random_state=1)),                               
            #                     ('DT', DecisionTreeClassifier(random_state=1)),                              
            #                     ],final_estimator=LogisticRegression(random_state=1) )
            ]
    modelnames = []
    modelresults = []
    for model in models:
    
        print(model)
        # 十折交叉验证
        metrics_cross = cross_val_score(model,X,Y,cv=10,scoring='roc_auc')
           
        
        print(model.__class__.__name__)
        modelnames.append(model.__class__.__name__)
        print('cv10:',metrics_cross.mean())
        modelresults.append(metrics_cross.mean())

    return modelnames,modelresults


def selectFeature(train,label,model):
    '''

    :param train:
    :param label:
    :param model:
    :return:
    '''
    # 特征重要性排序, 可更改其他方式
    clf = ExtraTreesClassifier(random_state=1)
    clf = clf.fit(train, label)
    feature_order = clf.feature_importances_
    feature_order = pd.DataFrame({'feat_name':train.columns,'feat_importance':feature_order}) 
    print(feature_order) # 0  0.116739  0.051316  0.395378  0.436567
    feature_order = feature_order.sort_values(by='feat_importance',ascending=False)
    os.makedirs('results/AAAA_featureSelect',exist_ok=True)
    
    pd.DataFrame(feature_order).to_csv('results/AAAA_featureSelect/feat_importance_sorted.csv',index=False)

    #贪心算法确定特征子集
    auc = 0
    importFeature = []
    results = []
    os.makedirs('results/AAAA_featureSelect_auc',exist_ok=True)
    
    for i in feature_order['feat_name']:
        importFeature.append(i)
        train = pd.DataFrame(train)
        featureSet = train.loc[:,importFeature]
        CV = StratifiedKFold(n_splits=10)#8389058127314367
        # lastAuc = auc
        # print('lastAuc',lastAuc)
        auc = sk_model_selection.cross_val_score(model, featureSet, label,scoring='roc_auc', cv=CV, n_jobs=-1).mean()
        auc = round(auc,5)
        # if lastAuc-0.01 > auc:
        #     importFeature.remove(i)
        #     break
        results.append([len(importFeature),auc])
        print(f'with {len(importFeature)} features of auc:',auc)         
    results = pd.DataFrame(results,columns=['feat_num','auc'])
    results.to_csv(f'results/AAAA_featureSelect_auc/xgb_feat_num_auc.csv',index=False)
    plot_metric(results['auc'],ylabel='AUC',x=results['feat_num'].values.tolist(),xlabel='feature number',saved=True)    
        
        
        
    return results["auc"].max()

def plot_metric( yvalues,  ylabel, x= None, rotation=45, xlabel=None, saved=False):
    '''
    绘制某项指标的变化趋势图，并存储绘制结果
    :param x: 横向坐标的数值
    :param yvalues: 纵向指标的数值
    :param ylabel: 指标名称,为字符串类型
    :param plot_title: 图的标题名称，默认为空
    :return: 无返回值
      '''

    plt.figure(figsize=(16, 9))
    if x == None:
        x = range(1, len(yvalues) + 1)
    plt.plot(x, yvalues, 'ro-', color='#4169E1', alpha=1)
    plt.xticks(np.arange(min(x), max(x) + 1, 3.0),rotation=rotation)
    plt.grid(ls='--')
    plt.legend()
    # if plot_title:
    #     plt.title(plot_title, fontsize=18)
    plt.tick_params(axis='both', which='major', labelsize=12)
    plt.ylabel(ylabel, fontsize=14)
    if xlabel:
        plt.xlabel(xlabel,fontsize=14)

    # 保存文件
    Path("results/Plot").mkdir(exist_ok=True)
    if saved:    
        plt.savefig(f"results/Plot/{xlabel}_{ylabel}.png", dpi=300, format="png")
    plt.show()

def plot_metric1(yvalues, ylabel, x=None, rotation=45, xlabel=None, saved=False):
    '''
    绘制某项指标的变化趋势图，并存储绘制结果
    :param x: 横向坐标的数值
    :param yvalues: 纵向指标的数值
    :param ylabel: 指标名称,为字符串类型
    :param plot_title: 图的标题名称，默认为空
    :return: 无返回值
    '''
    plt.figure(figsize=(16, 9))
    if x is None:
        x = range(1, len(yvalues) + 1)
    
    # 绘制主曲线
    plt.plot(x, yvalues, 'ro-', color='#4169E1', alpha=1)
    
    # 找到y值最大的点
    max_y = max(yvalues)
    max_x = x[yvalues.index(max_y)] if isinstance(yvalues, list) else x[np.argmax(yvalues)]
    
    # 绘制最大值垂直线
    plt.axvline(x=max_x, color='r', linestyle='--', alpha=0.5)
    
    # 标注最大值点
    plt.text(max_x, max_y, f'({max_x}, {max_y:.3f})', 
             fontsize=12, color='red',
             bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))
    
    # 设置x轴刻度
    plt.xticks(np.arange(min(x), max(x) + 1, 3.0), rotation=rotation)
    plt.grid(ls='--')
    plt.legend()
    
    # 设置标签
    plt.tick_params(axis='both', which='major', labelsize=12)
    plt.ylabel(ylabel, fontsize=14)
    if xlabel:
        plt.xlabel(xlabel, fontsize=14)

    # 保存文件
    Path("results/Plot").mkdir(exist_ok=True)
    if saved:    
        plt.savefig(f"results/Plot/{xlabel}_{ylabel}.png", dpi=300, format="png")
    plt.show()

def plot_metric2(yvalues, ylabel, x=None, rotation=45, xlabel=None, saved=False, vline_x=None):
    '''
    绘制某项指标的变化趋势图，并存储绘制结果
    :param x: 横向坐标的数值
    :param yvalues: 纵向指标的数值
    :param ylabel: 指标名称,为字符串类型
    :param plot_title: 图的标题名称，默认为空
    :param vline_x: 指定绘制垂直线的x值
    :return: 无返回值
    '''
    plt.figure(figsize=(16, 9))
    if x is None:
        x = range(1, len(yvalues) + 1)
    
    # 绘制主曲线
    plt.plot(x, yvalues, 'ro-', color='#4169E1', alpha=1)
    
    # 如果指定了vline_x，绘制垂直线并标注y值
    if vline_x is not None:
        # 找到对应的y值
        idx = x.index(vline_x) if isinstance(x, list) else np.where(x == vline_x)[0][0]
        vline_y = yvalues[idx]
        
        # 绘制垂直线
        plt.axvline(x=vline_x, color='r', linestyle='--', alpha=0.5)
        
        # 标注y值
        plt.text(vline_x, vline_y, f'({vline_x}, {vline_y:.3f})', 
                 fontsize=12, color='red',
                 bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))
    
    # ... 其余代码保持不变 ...
    plt.xticks(np.arange(min(x), max(x) + 1, 3.0), rotation=rotation)
    plt.grid(ls='--')
    plt.legend()
    plt.tick_params(axis='both', which='major', labelsize=12)
    plt.ylabel(ylabel, fontsize=14)
    if xlabel:
        plt.xlabel(xlabel, fontsize=14)

    # 保存文件
    Path("results/Plot").mkdir(exist_ok=True)
    if saved:    
        plt.savefig(f"results/Plot/{xlabel}_{ylabel}.png", dpi=300, format="png")
    plt.show()


def rename_with_feat40(inputfile, saved=True):
    '''
    函数功能：将当前输入文件中的部分特征名称修改为feat_40中的特征名称
    inputfile: str, 待更新特征名称的输入文件路径
    saved: bool, 是否保存修改后的文件，默认为True
    return: 
        feat_146, 修改后的特征数据
        feat_40_dict, feat_40的特征名称字典
    '''

    feat_40 = pd.read_csv(r'E:\WLH\bioinformatics\my_papers_codes\pred-dsmcplus\data\8tool_conser_TFBs_TE_dPSIZ_DSP_SilVA_header.txt', sep='\t', header=None)
    feat_40_dict = {}
    for item in range(feat_40.shape[1]):   
        feat_40_dict[str(item)] = feat_40.loc[0,item]
    # temp = pd.DataFrame(feat_40_dict, index=[0]).transpose()  #temp shape:(45,1)
    # temp.to_csv(r'results\feat_40.csv')  #save feat_40_dict
    feat_146 = pd.read_csv(inputfile)
    print(f'Origial feature names:\n {feat_146.columns}')
    feat_146.rename(columns=feat_40_dict, inplace=True)
    if saved:
        filename =os.path.splitext((inputfile))[0]
        feat_146.to_csv(f'{filename}_rename.csv', index=False)
    return feat_146, feat_40_dict


if __name__ == '__main__':
    feat_146, feat_40 = rename_with_feat40(r'E:\WLH\bioinformatics\my_papers_codes\pred-dsmcplus\data\feature-encoded\for-train-test\train-closeby6-encoded-feat146.csv')#, saved=False)
    print(f'Rename feature 146 :\n{feat_146["feat_name"]}')
    print(feat_40)