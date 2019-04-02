import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os, sys
import time
import seaborn as sb
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn import svm, tree, linear_model, neighbors, naive_bayes, ensemble, discriminant_analysis, gaussian_process
from xgboost import XGBClassifier
from sklearn import ensemble
from sklearn import linear_model
from sklearn import naive_bayes
from sklearn import neighbors
from sklearn import svm
from sklearn import tree

from sklearn import model_selection
from sklearn.metrics import precision_score, recall_score,f1_score, accuracy_score, make_scorer, classification_report
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, cross_validate, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.externals import joblib
import shap

CURR_PATH = os.path.dirname(os.path.realpath(__file__))
DATA_RAW = os.path.join(CURR_PATH, os.pardir, 'data', 'raw')
TEST = os.path.join(CURR_PATH, os.pardir, 'data', 'TEST')
#(os.sep).join(os.getcwd().split(os.sep)[:-1])+os.sep + 'data'+ os.sep + 'raw'
PROCESSED = os.path.join(CURR_PATH, os.pardir, 'data', 'processed')
STATS = os.path.join(PROCESSED, 'stats')

## preprocess and EDA ###

def preprocess(transf = False):
    df = pd.read_csv(PROCESSED + os.sep  +'stats_summary.csv')
    # # log transform
    if transf==True:
        for col in ['pratio', 'pratio_submin','kurtosis']:
            df[col] = np.log(df[col])
        # # square transform
        for col in ['inter_q', 'std']:
            df[col] = (df[col])**2
        df = df.rename(columns = {'pratio':'log(pratio)',
                                  'pratio_submin': 'log(pratio_sub)',
                                  'kurtosis': 'log(kurt)',
                                  'inter_q':'inter_q^2',
                                  'std': 'std^2'})

    df = df.dropna(axis = 0)
    df.head()
    data = df[df.columns[~df.columns.isin(['id', 'gender'])]]
    df['gender'] = np.where(df['gender']=='F', 1,0)
    target = df['gender']
    return df, data, target





## machine learning related ###


def get_pipe_pca(scaler):
    return [('scaler', scaler), ('pca', PCA())]


def get_pipe_nopca(scaler):
    return [('scaler', scaler)]


def get_class_weight(dict_params):
    if 'class_weight' in dict_params.keys():
        return dict_params['class_weight']
    elif 'scale_pos_weight' in dict_params.keys():
        return dict_params['scale_pos_weight']
    else:
        return 'None'


def train_cls(cls, pipelinesteps, X_train, y_train, kfolds, scoring):
    idx = 0
    df_cls = pd.DataFrame(
        columns=['Model Name', 'Parameters', 'class_weight', 'Pipeline', 'Time', 'Train Accuracy Mean', 'Train F1 Mean',
                 'CV F1 Mean',
                 'CV F1 Std', 'CV Precision Mean', 'CV Recall Mean', 'CV Accuracy Mean'])
    for cl in cls:
        steps = pipelinesteps.copy()
        steps.append(cl)
        ml_pipe = Pipeline(steps)
        cv_results = cross_validate(ml_pipe, X_train, y_train, cv=kfolds, scoring=scoring, return_train_score=True)
        clname = cl[1].__class__.__name__
        df_cls.loc[idx, 'Model Name'] = clname
        df_cls.loc[idx, 'Parameters'] = str(cl[1].get_params())
        df_cls.loc[idx, 'class_weight'] = get_class_weight(cl[1].get_params())
        df_cls.loc[idx, 'Pipeline'] = pipelinesteps
        df_cls.loc[idx, 'Time'] = cv_results['fit_time'].mean()
        df_cls.loc[idx, 'Train Accuracy Mean'] = cv_results['train_Accuracy'].mean()
        df_cls.loc[idx, 'Train F1 Mean'] = cv_results['train_F1_score'].mean()
        df_cls.loc[idx, 'CV F1 Mean'] = cv_results['test_F1_score'].mean()
        df_cls.loc[idx, 'CV F1 Std'] = cv_results['test_F1_score'].std()
        df_cls.loc[idx, 'CV Precision Mean'] = cv_results['test_Precision'].mean()
        df_cls.loc[idx, 'CV Recall Mean'] = cv_results['test_Recall'].mean()
        df_cls.loc[idx, 'CV Accuracy Mean'] = cv_results['test_Accuracy'].mean()
        df_cls.loc[idx, 'CV Accuracy Mean'] = cv_results['test_Accuracy'].mean()
        idx += 1
    df_cls.to_csv(PROCESSED + os.sep + 'cl_comparison_.csv', index=False, sep=';')
    return df_cls


def trial_dataparams(data, target):
    random_state= 42

    X, y = data, target

    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.1, random_state=random_state, stratify=y)

    kfolds = StratifiedKFold(n_splits=5, shuffle =True, random_state=random_state)
    scoring = {'Precision': make_scorer(precision_score), 'Recall': make_scorer(recall_score),  'F1_score': make_scorer(f1_score), 'Accuracy': make_scorer(accuracy_score)}


    cls_balanced = [
        ('dtc', tree.DecisionTreeClassifier(class_weight = 'balanced', random_state=random_state)),
        ('rfc', ensemble.RandomForestClassifier(n_estimators = 100,class_weight = 'balanced', random_state=random_state)),
        ('lr', linear_model.LogisticRegressionCV(class_weight = 'balanced', random_state=random_state)),
        ('svc', svm.SVC(probability=True,class_weight = 'balanced', random_state=random_state)),
       ('xgb', XGBClassifier(n_estimators = 100, objective= 'binary:logistic',scale_pos_weight =13 , random_state=random_state))
    ]

    cls = [
        ('dtc', tree.DecisionTreeClassifier(random_state=random_state)),
        ('bc', ensemble.BaggingClassifier(n_estimators = 100, random_state=random_state)),
        ('gbc', ensemble.GradientBoostingClassifier(n_estimators = 100, random_state=random_state)),
        ('rfc', ensemble.RandomForestClassifier(n_estimators = 100, random_state=random_state)),
        ('lr', linear_model.LogisticRegressionCV(random_state=random_state)),
        ('knn', neighbors.KNeighborsClassifier()),
        ('svc', svm.SVC(probability=True, random_state=random_state)),
       ('xgb', XGBClassifier(n_estimators = 100, objective= 'binary:logistic', random_state=random_state))
    ]


    dfout = pd.DataFrame()
    for scaler in [StandardScaler(), RobustScaler(quantile_range=(2.5, 97.5))]:
        for pipelinesteps in [get_pipe_pca(scaler), get_pipe_nopca(scaler)]:
            for cls_train in [cls, cls_balanced]:
                dfout_i = train_cls(cls_train, pipelinesteps, X_train = X_train, y_train= y_train, kfolds = kfolds, scoring = scoring)
                dfout = dfout.append(dfout_i, ignore_index=True)
    dfout = dfout.sort_values(by = 'CV F1 Mean', ascending = False)
    dfout.to_csv(PROCESSED + os.sep + 'trial_dataparams_.csv', index = False, sep = ';')


def randomforest_grids(data, target):
    random_state = 42
    X, y = data, target

    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.1, random_state=random_state, stratify=y)

    kfolds = StratifiedKFold(n_splits=5, shuffle =True, random_state=random_state)

    scoring = {'Precision': make_scorer(precision_score), 'Recall': make_scorer(recall_score),  'F1_score': make_scorer(f1_score), 'Accuracy': make_scorer(accuracy_score)}

    steps = [('scaler', StandardScaler()),
             # ('pca', PCA()),
            ('rfc', ensemble.RandomForestClassifier())]

    ml_pipe = Pipeline(steps)
    param_grid={'rfc__class_weight': [{0: 1, 1: 2}], #[{0: 1, 1: 3}, {0: 1, 1: 5}, {0: 1, 1: 7},{0: 1, 1: 9}, {0: 1, 1: 11}, {0: 1, 1: 13}, {0: 1, 1: 15}],
            'rfc__criterion': ['gini'],
            'rfc__max_depth': list(np.arange(10,100,10)) + [None],
            'rfc__max_features': list(range(1,X.shape[1])),
            'rfc__min_samples_leaf': np.arange(10,100, 10),
            #'rfc__min_samples_split': np.linspace(0.01, 1.0, 100, endpoint=True),
            'rfc__n_estimators': [100],


                'rfc__random_state': [42],
            'rfc__verbose': [0],
            'rfc__warm_start': [False]
           }

    grid_clf = GridSearchCV(
        param_grid = param_grid,
        estimator = ml_pipe,
        scoring = scoring,
        refit='F1_score',
        cv = kfolds.split(X_train, y_train),
        n_jobs=-1,
        verbose = 1
        )
    grid_clf.fit(X_train, y_train)
    df_grid =  pd.DataFrame(grid_clf.cv_results_)
    clf_best = grid_clf.best_estimator_
    df_grid.to_csv( PROCESSED+os.sep +'rfc_grid.csv', sep = ';', index = False)
    joblib.dump(clf_best,PROCESSED+os.sep +'rfc_grid_bestclf.pkl', compress = 1)


    y_pred = clf_best.predict(X_test)
    print(classification_report(y_test, y_pred))
    df_clrepo= pd.DataFrame(classification_report(y_test, y_pred,output_dict=True))
    df_clrepo.to_csv(PROCESSED + os.sep + 'classification_report.csv', sep = ';')

def print_best_model():
    clf_best = joblib.load(PROCESSED+os.sep +'rfc_grid_bestclf.pkl')
    best_model = clf_best.steps[1][1]
    print(best_model)

def plot_bestRF_importance(data):
    clf_best = joblib.load(PROCESSED+os.sep +'rfc_grid_bestclf.pkl')
    best_model = clf_best.steps[1][1]
    df_importance = pd.DataFrame({'Feature':data.columns, 'Importance':best_model.feature_importances_}).sort_values(by = 'Importance', ascending = False)

    f, ax = plt.subplots(1,1, figsize=(10, 8))
    _ = plt.rcParams.fontsize = 30
    _ = sb.barplot(y  = df_importance['Feature'],x = df_importance['Importance'], ax = ax)
    _ = plt.title('RF Feature Importance', fontsize  =20)
    _ = plt.xticks(fontsize=20);
    _ = plt.yticks(fontsize=20);
    _ = ax.set(xlabel = '', ylabel = '')
    fname = PROCESSED + os.sep + 'plots' + os.sep + 'RF_feature_importance.png'
    plt.tight_layout()
    plt.savefig(fname, dpi = 100)

def plot_shap_importance(data, target):
    clf_best = joblib.load(PROCESSED+os.sep +'rfc_grid_bestclf.pkl')
    best_model = clf_best.steps[1][1]
    random_state = 42
    X, y = data, target
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.1, random_state=random_state, stratify=y)
    explainer = shap.TreeExplainer(best_model)
    shap_values = (explainer.shap_values(X_train))
    sb.set()
    shap.summary_plot(shap_values, X.columns,show=False)
    plt.tight_layout()
    plt.savefig(PROCESSED + os.sep + 'SHAP_importance.png', dpi = 100)
    plt.title('SHAP Feature Importance')
# visualize the first prediction's explanation (use matplotlib=True to avoid Javascript)

def plot_pca_importance(data, target):
    X, y = data, target
    scaler = StandardScaler()
    scaler.fit(X)
    X=scaler.transform(X)
    pca_n = PCA()
    X_pca_n = pca_n.fit_transform(X)
    scaler.fit(X)
    pca_n.explained_variance_ratio_.sum()
    f, ax = plt.subplots(1,1, figsize=(10, 8))
    sb.barplot(y=data.columns, x = pca_n.explained_variance_ratio_, ax = ax)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.title('PCA Feature Importance',
              fontsize=20)