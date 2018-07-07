from pandas import read_csv
import pandas as pd
train = pd.read_csv("//Users/chitrasekar2k5/Desktop/Machine Learning/working_MLA/local copy_kaggle/Competition2_house prices dataset/train.csv")
test = pd.read_csv("//Users/chitrasekar2k5/Desktop/Machine Learning/working_MLA/local copy_kaggle/Competition2_house prices dataset/test.csv")
""" STEP 1: As per documentation, we are removing outliers first"""
"""checking outliers using scatter plot for sale price and gr liv area before removing"""
import matplotlib.pyplot as plt
#plt.scatter(train.GrLivArea,train.SalePrice)
#plt.show()
#print(train.GrLivArea.count())
#rmvset = train[train.GrLivArea > 4000]
#print(rmvset.Id)
train = train[train.GrLivArea <= 4000]
#plt.scatter(train.GrLivArea,train.SalePrice)
#print(train.GrLivArea.count())
#print(test.GrLivArea.count())
#print(train.shape,test.shape)
"""STEP 2: Save Id column for future reference and drop them from train and test set"""
train_id = train['Id']
test_id = test['Id']
train = train.drop(['Id'], axis = 1)
test = test.drop(['Id'], axis = 1)
#print(train.shape,test.shape)
"""STEP 3: Eliminate skewness in the target variable using log transformation"""
import numpy as np
train.SalePrice = np.log1p(train.SalePrice)
#print(train.SalePrice.head(2))
"""STEP 4: To perform feature engineering we combine train and test set for consistency while remembering index so we can split later."""
y = train.SalePrice.reset_index(drop =True)
train_set = train.drop(['SalePrice'], axis = 1)
test_set = test
alldata_set = pd.concat([train_set,test_set]).reset_index(drop = True)
"""STEP 5: Handling missing values - NAs"""
"""check total missing columns count
nulls = np.sum(alldata_set.isnull())
nullcols = nulls.loc[(nulls != 0)]
dtypes = alldata_set.dtypes
dtypes2 = dtypes.loc[(nulls != 0)]
info = pd.concat([nullcols,dtypes2], axis = 1).sort_values(by=0, ascending = False)"""
"""change NA's with none"""
alldata_set["PoolQC"] = alldata_set["PoolQC"].fillna("None")
alldata_set["MiscFeature"] = alldata_set["MiscFeature"].fillna("None")
alldata_set["Alley"] = alldata_set["Alley"].fillna("None")
alldata_set["Fence"] = alldata_set["Fence"].fillna("None")
alldata_set["FireplaceQu"] = alldata_set["FireplaceQu"].fillna("None")
alldata_set["LotFrontage"] = alldata_set.groupby("Neighborhood")["LotFrontage"].transform(lambda x:x.fillna(x.median()))
for col in ('GarageType','GarageFinish','GarageQual', 'GarageCond'):
    alldata_set[col] = alldata_set[col].fillna('None')
for col in ('GarageArea', 'GarageCars'):
    alldata_set[col] = alldata_set[col].fillna(0)
for col in ('BsmtFinSF1','BsmtFinSF2','BsmtUnfSF','TotalBsmtSF','BsmtFullBath','BsmtHalfBath'):
    alldata_set[col] = alldata_set[col].fillna(0)
for col in ('BsmtQual','BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2'):
    alldata_set[col] = alldata_set[col].fillna('None')
alldata_set["MasVnrType"] = alldata_set["MasVnrType"].fillna("None")
alldata_set["MasVnrArea"] = alldata_set["MasVnrArea"].fillna(0)
alldata_set['MSZoning'] = alldata_set['MSZoning'].fillna(alldata_set["MSZoning"].mode()[0])
alldata_set = alldata_set.drop(["Utilities"], axis =1)
alldata_set["Functional"] = alldata_set["Functional"].fillna("Typ")
alldata_set["GarageYrBlt"] = alldata_set["GarageYrBlt"].fillna(alldata_set["GarageYrBlt"].mode()[0])
alldata_set["Electrical"] = alldata_set["Electrical"].fillna(alldata_set["Electrical"].mode()[0])
alldata_set["KitchenQual"] = alldata_set["KitchenQual"].fillna(alldata_set["KitchenQual"].mode()[0])
alldata_set["Exterior1st"] = alldata_set["Exterior1st"].fillna(alldata_set["Exterior1st"].mode()[0])
alldata_set["Exterior2nd"] = alldata_set["Exterior2nd"].fillna(alldata_set["Exterior2nd"].mode()[0])
alldata_set["SaleType"] = alldata_set["SaleType"].fillna(alldata_set["SaleType"].mode()[0])
"""again check total missing columns count
nulls = np.sum(alldata_set.isnull())
nullcols = nulls.loc[(nulls != 0)]
dtypes = alldata_set.dtypes
dtypes2 = dtypes.loc[(nulls != 0)]
info = pd.concat([nullcols,dtypes2], axis = 1).sort_values(by=0, ascending = False)
print(info)
print(alldata_set.describe())"""
"""STEP 6: Transforming some numerical variables which are really categorical by nature"""
#print(alldata_set.dtypes)
alldata_set["MSSubClass"] = alldata_set["MSSubClass"].apply(str)
alldata_set["OverallCond"] = alldata_set["OverallCond"].apply(str)
alldata_set["YrSold"] = alldata_set["YrSold"].apply(str)
alldata_set["MoSold"] = alldata_set["MoSold"].apply(str)
"""STEP 7: Label encoding categorical variables that contains info in ordering set"""
from sklearn.preprocessing import LabelEncoder
cols = ('FireplaceQu', 'BsmtQual', 'BsmtCond', 'GarageQual', 'GarageCond',
        'ExterQual', 'ExterCond','HeatingQC', 'PoolQC', 'KitchenQual', 'BsmtFinType1',
        'BsmtFinType2', 'Functional', 'Fence', 'BsmtExposure', 'GarageFinish', 'LandSlope',
        'LotShape', 'PavedDrive', 'Street', 'Alley', 'CentralAir', 'MSSubClass', 'OverallCond',
        'YrSold', 'MoSold')
# process columns, apply LabelEncoder to categorical features
for c in cols:
    lbl = LabelEncoder()
    lbl.fit(list(alldata_set[c].values))
    alldata_set[c] = lbl.transform(list(alldata_set[c].values))
# shape
"""print('Shape alldata_set: '.format(alldata_set.shape))
print(alldata_set['FireplaceQu'])"""
"""STEP 7: Skew transformation features"""
from scipy.stats import skew
numeric_dtypes = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
numerics2 = []
for i in alldata_set.columns:
    if alldata_set[i].dtype in numeric_dtypes:
        numerics2.append(i)
skew_features = alldata_set[numerics2].apply(lambda x: skew(x)).sort_values(ascending=False)
skews = pd.DataFrame({'skew':skew_features})
#print(skews)
from scipy.special import boxcox1p
from scipy.stats import boxcox_normmax
high_skew = skew_features[skew_features > 0.5]
high_skew = high_skew
skew_index = high_skew.index
for i in skew_index:
    alldata_set[i]= boxcox1p(alldata_set[i], boxcox_normmax(alldata_set[i]+1))
skew_features2 = alldata_set[numerics2].apply(lambda x: skew(x)).sort_values(ascending=False)
skews2 = pd.DataFrame({'skew':skew_features2})
#print(skews2)
"""STEP 8: creating new features"""
alldata_set['Total_sqr_footage'] = (alldata_set['BsmtFinSF1'] + alldata_set['BsmtFinSF2'] +
                                 alldata_set['1stFlrSF'] + alldata_set['2ndFlrSF'])

alldata_set['Total_Bathrooms'] = (alldata_set['FullBath'] + (0.5*alldata_set['HalfBath']) +
                               alldata_set['BsmtFullBath'] + (0.5*alldata_set['BsmtHalfBath']))

alldata_set['Total_porch_sf'] = (alldata_set['OpenPorchSF'] + alldata_set['3SsnPorch'] +
                              alldata_set['EnclosedPorch'] + alldata_set['ScreenPorch'] +
                             alldata_set['WoodDeckSF'])


#simplified features
alldata_set['haspool'] = alldata_set['PoolArea'].apply(lambda x: 1 if x > 0 else 0)
alldata_set['has2ndfloor'] = alldata_set['2ndFlrSF'].apply(lambda x: 1 if x > 0 else 0)
alldata_set['hasgarage'] = alldata_set['GarageArea'].apply(lambda x: 1 if x > 0 else 0)
alldata_set['hasbsmt'] = alldata_set['TotalBsmtSF'].apply(lambda x: 1 if x > 0 else 0)
alldata_set['hasfireplace'] = alldata_set['Fireplaces'].apply(lambda x: 1 if x > 0 else 0)
#print(alldata_set.shape)
#print(alldata_set.dtypes)
"""STEP 9: Since sklearn im.fit() does not accept strings we have to convert our objects to dummy variables"""
finaldata_set = pd.get_dummies(alldata_set).reset_index(drop=True)
#print(finaldata_set.dtypes)
"""STEP 10: Re-split the model into train and test"""
#print(y.shape)
X = finaldata_set.iloc[:len(y),:]
testing_set = finaldata_set.iloc[len(X):,:]
#print(X.shape)
#print(testing_set.shape)
"""STEP 11: Baseline model"""
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import make_pipeline

#Build our model method
lm = LinearRegression()

#Build our cross validation method
kfolds = KFold(n_splits=10, shuffle=True, random_state=23)

#build our model scoring function
def cv_rmse(model):
    rmse = np.sqrt(-cross_val_score(model, X, y,
                                   scoring="neg_mean_squared_error",
                                   cv = kfolds))
    return(rmse)


#second scoring metric
def cv_rmsle(model):
    rmsle = np.sqrt(np.log(-cross_val_score(model, X, y,
                                           scoring = 'neg_mean_squared_error',
                                           cv=kfolds)))
    return(rmsle)
benchmark_model = make_pipeline(RobustScaler(),lm).fit(X=X, y=y)
(cv_rmse(benchmark_model).mean())

"""STEP 12: my mean rsme is so high which will be reduced in feature selection process"""
coeffs = pd.DataFrame(list(zip(X.columns, benchmark_model.steps[1][1].coef_)), columns=['Predictors', 'Coefficients'])
(coeffs.sort_values(by='Coefficients', ascending=False))
"""STEP 13: Embedded methods"""
#RIDGE REGRESSION
from sklearn.linear_model import RidgeCV
def ridge_selector(k):
    ridge_model = make_pipeline(RobustScaler(),
                                RidgeCV(alphas = [k],
                                        cv=kfolds)).fit(X, y)

    ridge_rmse = cv_rmse(ridge_model).mean()
    return(ridge_rmse)
r_alphas = [.0001, .0003, .0005, .0007, .0009,
          .01, 0.05, 0.1, 0.3, 1, 3, 5, 10, 15, 20, 30, 50, 60, 70, 80]

ridge_scores = []
for alpha in r_alphas:
    score = ridge_selector(alpha)
    ridge_scores.append(score)
plt.plot(r_alphas, ridge_scores, label='Ridge')
plt.legend('center')
plt.xlabel('alpha')
plt.ylabel('score')

ridge_score_table = pd.DataFrame(ridge_scores, r_alphas, columns=['RMSE'])
(ridge_score_table)
alphas_alt = [14.5, 14.6, 14.7, 14.8, 14.9, 15, 15.1, 15.2, 15.3, 15.4, 15.5]

ridge_model2 = make_pipeline(RobustScaler(),
                            RidgeCV(alphas = alphas_alt,
                                    cv=kfolds)).fit(X, y)

(cv_rmse(ridge_model2).mean())
ridge_model2.steps[1][1].alpha_
#LASSO REGRESSION
from sklearn.linear_model import LassoCV


alphas = [0.00005, 0.0001, 0.0003, 0.0005, 0.0007,
          0.0009, 0.01]
alphas2 = [0.00005, 0.0001, 0.0002, 0.0003, 0.0004, 0.0005,
           0.0006, 0.0007, 0.0008]


lasso_model2 = make_pipeline(RobustScaler(),
                             LassoCV(max_iter=1e7,
                                    alphas = alphas2,
                                    random_state = 42)).fit(X, y)
scores = lasso_model2.steps[1][1].mse_path_

plt.plot(alphas2, scores, label='Lasso')
plt.legend(loc='center')
plt.xlabel('alpha')
plt.ylabel('RMSE')
plt.tight_layout()
#plt.show()
lasso_model2.steps[1][1].alpha_
(cv_rmse(lasso_model2).mean())
coeffs = pd.DataFrame(list(zip(X.columns, lasso_model2.steps[1][1].coef_)), columns=['Predictors', 'Coefficients'])
used_coeffs = coeffs[coeffs['Coefficients'] != 0].sort_values(by='Coefficients', ascending=False)
#print(used_coeffs.shape)
#print(used_coeffs)
used_coeffs_values = X[used_coeffs['Predictors']]
(used_coeffs_values.shape)
overfit_test2 = []
for i in used_coeffs_values.columns:
    counts2 = used_coeffs_values[i].value_counts()
    zeros2 = counts2.iloc[0]
    if zeros2 / len(used_coeffs_values) * 100 > 99.5:
        overfit_test2.append(i)

(overfit_test2)



#Elastic Net (L1 and L2 penalty)
from sklearn.linear_model import ElasticNetCV

e_alphas = [0.0001, 0.0002, 0.0003, 0.0004, 0.0005, 0.0006, 0.0007]
e_l1ratio = [0.8, 0.85, 0.9, 0.95, 0.99, 1]

elastic_cv = make_pipeline(RobustScaler(),
                           ElasticNetCV(max_iter=1e7, alphas=e_alphas,
                                        cv=kfolds, l1_ratio=e_l1ratio))

elastic_model3 = elastic_cv.fit(X, y)
(cv_rmse(elastic_model3).mean())
(elastic_model3.steps[1][1].l1_ratio_)
(elastic_model3.steps[1][1].alpha_)
from sklearn.pipeline import make_pipeline

#setup models
ridge = make_pipeline(RobustScaler(),
                      RidgeCV(alphas = alphas_alt, cv=kfolds))

lasso = make_pipeline(RobustScaler(),
                      LassoCV(max_iter=1e7, alphas = alphas2,
                              random_state = 42, cv=kfolds))

elasticnet = make_pipeline(RobustScaler(),
                           ElasticNetCV(max_iter=1e7, alphas=e_alphas,
                                        cv=kfolds, l1_ratio=e_l1ratio))
#prepare dataframes
stackX = np.array(X)
stacky = np.array(y)

#scoring

print("cross validated scores")

for model, label in zip([ridge, lasso, elasticnet],
                     ['RidgeCV', 'LassoCV', 'ElasticNetCV']):

    SG_scores = cross_val_score(model, stackX, stacky, cv=kfolds,
                               scoring='neg_mean_squared_error')
    print("RMSE", np.sqrt(-SG_scores.mean()), "SD", scores.std(), label)
#stack_gen_model = stack_gen.fit(stackX, stacky)
em_preds = elastic_model3.predict(testing_set)
lasso_preds = lasso_model2.predict(testing_set)
ridge_preds = ridge_model2.predict(testing_set)
stack_preds = ((0.3*em_preds) + (0.4*lasso_preds) + (0.3*ridge_preds))
submission = pd.read_csv("/Users/chitrasekar2k5/Desktop/Machine Learning/working_MLA/local copy_kaggle/Competition2_house prices dataset/sample_submission.csv")
submission.iloc[:,1] = np.expm1(stack_preds)
submission.to_csv("/Users/chitrasekar2k5/Desktop/Machine Learning/working_MLA/local copy_kaggle/Competition2_house prices dataset/final_submission.csv", index=False)
