# data analysis and wrangling
import pandas as pd
import numpy as np
import random as rnd
import math
import statsmodels.api as sm
from statsmodels.genmod.generalized_linear_model import GLM
from statsmodels.genmod import families
from statsmodels.stats.outliers_influence import variance_inflation_factor
from scipy import stats

# visualization
import seaborn as sns
import matplotlib.pyplot as plt

# machine learning
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.metrics import roc_curve, roc_auc_score

# formatting outputs
pd.set_option('display.max_columns', 500)
sns.set(rc={'figure.figsize':(11.7,8.27)})
pd.set_option('display.float_format', lambda x: f'{x:.3f}')
#%%
# load Data into Dataframe
df = pd.read_csv(r'/Users/mark/Documents/Jupyter Files/data1.csv')
print(df.head())
#%%
print(df.info())
print(df.describe())
print('-'*48)
print(df.describe(include=['O']))
#%%
# Rename column, replace missing values with mean based on location and one hot encode the location
df = df.rename(columns={"Sensor5.1":"Sensor6"})
df = df.drop(['OPXVolume'], axis = 1)
df['Sensor3'] = df.Sensor3.replace({"-":np.nan}).astype(float)
location_encoded = pd.get_dummies(df['Location'])
df = pd.concat([location_encoded,df], axis = 1)
df = df.fillna(df.groupby('Location').transform('mean'))

df = df.drop(['Location', 'B'], axis=1) # don't need both A and B
del location_encoded
print(df.head())
#%%
# function to automatically split independent and dependent variables
def variable_split(df):
    y = df.Target
    X = df.drop(['Target'], axis=1)
    return X, y
#%%
# log transforming to determine whether the first assumption is satisfied
def logit_results_lt(df):
    X, target = variable_split(df)
    cont_var = X.drop(['A'],axis=1).columns.tolist() # continuous variables only
    
    for var in cont_var:
        X[f'{var}:Log_{var}'] = X[var].apply(lambda x: x*np.log(x+1))
    
    cols = cont_var + X.columns.tolist()[-len(cont_var):]
    
    lt = X[cols]
    lt_constant = sm.add_constant(lt, prepend = False)
    
    logit_results = GLM(target, lt_constant, family=families.Binomial()).fit()
    
    print(logit_results.summary())
    del logit_results, target, cont_var, cols, lt, lt_constant, X
    return
#%%
# a visual function to see if the variables are linear to the log odds
def visual_logit(df):
    X, y = variable_split(df)
    X_cols = X.columns.to_list()
    X_constant = sm.add_constant(X, prepend=False)
    logit_results = GLM(y, X_constant, family=families.Binomial()).fit()
    predicted = logit_results.predict(X_constant)
    log_odds = np.log(predicted / (1 - predicted))
    def multi_scatter(df, var, n_rows, n_cols):
        fig=plt.figure()
        for i, var_name in enumerate(var):
            ax=fig.add_subplot(n_rows,n_cols,i+1)
            plt.scatter(x=df[var_name].values, y=log_odds)
            ax.set_title(var_name+" Scatter")
        fig.tight_layout()  # Improves appearance a bit.
        plt.show()
    multi_scatter(df, X, 3, 4)
    del logit_results, X, y, X_cols, X_constant, predicted
    return 
#%%
# function to calculate the VIF
def calc_vif(df):
    vif = pd.DataFrame()
    vif["variables"] = df.columns
    vif["VIF"] = [variance_inflation_factor(df.values, i) for i in range(df.shape[1])]
    return(vif)
#%%
print(logit_results_lt(df))
visual_logit(df)
#%%
# some variables are not linear to log odds, one solution is to square the variable to see if the square 
# can satisfy the assumption
df_2 = np.power(df,2)
print(logit_results_lt(df_2))
visual_logit(df_2)
#%%
X, y = variable_split(df)
X_cols = X.columns.to_list()
X_constant = sm.add_constant(X, prepend=False)

logit_results = GLM(y, X_constant, family=families.Binomial()).fit()
print(logit_results.summary())
#%%
# this is to determine the cooks distance and std residual for influence and outliers
inf = logit_results.get_influence()
summ_df = inf.summary_frame()
diag_df = summ_df.loc[:,['cooks_d']]
diag_df['std_resid'] = stats.zscore(logit_results.resid_pearson)
diag_df['std_resid'] = diag_df.loc[:,'std_resid'].apply(lambda x: np.abs(x))
print(diag_df)
#%%
# standard threshold is 4 divided by the length of the independent variables
cook_threshold = 4/len(X)
print(f'Threshold for Cooks Distance = {cook_threshold}')
fig = inf.plot_index(y_var = "cooks", threshold = cook_threshold)
plt.axhline(y=cook_threshold, ls = "--", color = 'red')
fig.tight_layout(pad=2)
outliers = diag_df[diag_df['cooks_d'] > cook_threshold]
prop_outliers = round(100*(len(outliers) / len(X)),1)
print(f'Proportion of data points that are highly influential = {prop_outliers}%')
#%%
print(outliers.sort_values("cooks_d", ascending=False).head(15))
# looking at the biggest outlier
print(X.iloc[2708])
print('-'*20)
print(y.iloc[2708])
#%%
out_ind = list(outliers.index.values) 
df_clean = df.drop(index = out_ind)
df_clean.head(30)
#%%
X, y = variable_split(df_clean)
X_constant = sm.add_constant(X, prepend=False)
print(calc_vif(X_constant))
corr_matrix = df_clean.corr()
sns.heatmap(corr_matrix, annot=True)
plt.show()
#%%
X_constant_drop = X_constant.drop(['A', 'Sensor2'], axis=1)
print(calc_vif(X_constant_drop))
corr_matrix = df_clean.drop(['A', 'Sensor2'], axis=1).corr()
sns.heatmap(corr_matrix, annot=True)
plt.show()
#%%
logit_results = GLM(y, X_constant, family=families.Binomial()).fit()
fig = plt.figure(figsize=(8,5))
ax = fig.add_subplot(111, title="Residual Series Plot",
                    xlabel="Index Number", ylabel="Deviance Residuals")

ax.plot(X.index.tolist(), stats.zscore(logit_results.resid_deviance))
plt.axhline(y=0, ls="--", color='red');
#%%
data = df_clean[['MonthlyRunTime','FlowRate','Sensor3','Sensor6', 'Target']].reset_index(drop = True)
print(data.head())
print(data.describe())
# VIF check
X, y = variable_split(data)
X_constant_data = sm.add_constant(X, prepend=False)
print(calc_vif(X_constant_data))
# correlation matrix check
corr_matrix = data.corr()
sns.heatmap(corr_matrix, annot=True)
plt.show()
# significance check
logit_model = GLM(y, X_constant_data, family=families.Binomial())
logit_results = logit_model.fit()
print(logit_results.summary())
#%%
# test train split as 80:20 ratio
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=10)
lr = LogisticRegression()
lr.fit(X_train,y_train)
pred = lr.predict(X_test)
score = lr.score(X_test,y_test)
print(f'Logistic Regression Score = {score}')
#%%
cm = metrics.confusion_matrix(y_test, pred)
print(cm)
# confusion matrix to see false negatives and positives
plt.figure(figsize=(9,9))
sns.heatmap(cm, annot=True, fmt=".3f", linewidths=.5, square = True, cmap = 'Blues_r')
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
all_sample_title = 'Accuracy Score: {0}'.format(score)
plt.title(all_sample_title, size = 15)
#%%
pred_prob1 = lr.predict_proba(X_test)
fpr1, tpr1, thresh1 = roc_curve(y_test, pred_prob1[:,1], pos_label=1)
random_probs = [0 for i in range(len(y_test))]
p_fpr, p_tpr, _ = roc_curve(y_test, random_probs, pos_label=1)
auc_score1 = roc_auc_score(y_test, pred_prob1[:,1])
print(f'AUC Score = {auc_score1}')
plt.style.use('seaborn')
# plot roc curves
plt.plot(fpr1, tpr1, linestyle='--',color='orange', label='Logistic Regression')
plt.plot(p_fpr, p_tpr, linestyle='--', color='blue')
# title
plt.title('ROC curve')
# x label
plt.xlabel('False Positive Rate')
# y label
plt.ylabel('True Positive rate')

plt.legend(loc='best')
plt.savefig('ROC',dpi=300)
plt.show()
#%%
print('intercept ', lr.intercept_[0])
print('classes', lr.classes_)
pd.DataFrame({'coeff': lr.coef_[0]}, index=X.columns)
# odds based on coefficients
coef = pd.DataFrame({'coeff': lr.coef_[0]}, index=X.columns)
coef['odds'] = np.exp(coef['coeff'])
print(coef)












