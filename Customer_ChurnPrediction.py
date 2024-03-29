import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import sweetviz as sv
import pickle
import joblib
import statsmodels.formula.api as smf
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline 
from sklearn.pipeline import make_pipeline
from feature_engine.outliers import Winsorizer
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sqlalchemy import create_engine


df = pd.read_csv(r"C:/Users/ashut/OneDrive/Documents/Mentorness/Customer Churn Prediction/Customer_Churn.csv")
engine = create_engine("mysql+pymysql://{user}:{pw}@localhost/{db}"
                       .format(user="root",
                               pw="2607", 
                               db="mentorness")) 

df.to_sql('customer',con = engine, if_exists = 'replace', index = False)
sql = 'select * from customer;'
customer_full = pd.read_sql_query(sql, engine)

customer_full.head()

customer_full.describe()

customer_full.tenure.value_counts()

customer_full.info()


#EDA
df.head()
df.shape
df.isnull().sum()
df.head(10)
df.reset_index(inplace = True, drop = True)
df.info()

X = pd.DataFrame(df['tenure'])
Y = pd.DataFrame(df['TotalCharges'])

#FeatureEngineering
numeric_features = ['tenure']
df.plot(kind = 'box', subplots = True, sharey = False, figsize = (15, 8))
plt.subplots_adjust(wspace = 0.75)
plt.show()

winsor = Winsorizer(capping_method = 'iqr', 
                          tail = 'both', 
                          fold = 1.5,
                          variables = numeric_features)

winsor
#DataPreprocessing
num_pipeline = Pipeline(steps = [('impute', SimpleImputer(strategy = 'mean'))])
outlier_pipeline = Pipeline(steps = [('winsor', winsor)])

num_pipeline

outlier_pipeline

preprocessor = ColumnTransformer(transformers = [('num', num_pipeline, numeric_features)])
print(preprocessor)

preprocessor1 = ColumnTransformer(transformers = [('wins', outlier_pipeline, numeric_features)])
print(preprocessor1)

impute_data = preprocessor.fit(X)
df['tenure'] = pd.DataFrame(impute_data.transform(X))

X2 = pd.DataFrame(df['tenure'])
winz_data = preprocessor1.fit(X2)

df['tenure'] = pd.DataFrame(winz_data.transform(X))


df.head(10)

joblib.dump(impute_data, 'meanimpute')

joblib.dump(winz_data, 'winzor')


df.plot(kind = 'box', subplots = True, sharey = False, figsize = (15, 8))

plt.subplots_adjust(wspace = 0.75) 
plt.show()

plt.bar(height = df.TotalCharges, x = np.arange(1, 7044, 1))

plt.hist(df.TotalCharges) 

plt.bar(height = df.tenure, x = np.arange(1, 7044, 1))

plt.hist(df.tenure)

report = sv.analyze(df)
report.show_html('EDAreport.html')

plt.scatter(x = df['tenure'], y = df['TotalCharges']) 

## Measure the strength of the relationship between two variables using Correlation coefficient.

np.corrcoef(df.tenure, df.TotalCharges)

# Covariance
cov_output = np.cov(df.tenure, df.TotalCharges)[0, 1]
cov_output

# wcat.cov()

dataplot = sns.heatmap(df.corr(), annot = True, cmap = "YlGnBu")


# # Linear Regression using statsmodels package
# Simple Linear Regression
model = smf.ols('TotalCharges ~ tenure', data = df).fit()

model.summary()

pred1 = model.predict(pd.DataFrame(df['tenure']))

pred1


# Regression Line
plt.scatter(df.tenure, df.TotalCharges)
plt.plot(df.tenure, pred1, "r")
plt.legend(['Observed data', 'Predicted line'])
plt.show()


# Error calculation (error = AV - PV)
res1 = df.TotalCharges - pred1

print(np.mean(res1))

res_sqr1 = res1 * res1
mse1 = np.mean(res_sqr1)
rmse1 = np.sqrt(mse1)
rmse1




plt.scatter(x = np.log(df['tenure']), y = df['TotalCharges'], color = 'brown')
np.corrcoef(np.log(df.tenure), df.TotalCharges) #correlation

model2 = smf.ols('TotalCharges ~ np.log(tenure)', data = df).fit()
model2.summary()


pred2 = model2.predict(pd.DataFrame(df['tenure']))

# Regression Line
plt.scatter(np.log(df.tenure), df.TotalCharges)
plt.plot(np.log(df.tenure), pred2, "r")
plt.legend(['Observed data', 'Predicted line'])
plt.show()

# Error calculation
res2 = df.TotalCharges - pred2
res_sqr2 = res2 * res2
mse2 = np.mean(res_sqr2)
rmse2 = np.sqrt(mse2)
rmse2


# ## Exponential transformation


plt.scatter(x = df['tenure'], y = np.log(df['TotalCharges']), color = 'orange')
np.corrcoef(df.tenure, np.log(df.TotalCharges)) #correlation

model3 = smf.ols('np.log(TotalCharges) ~ tenure', data = df).fit()
model3.summary()

pred3 = model3.predict(pd.DataFrame(df['tenure']))

# Regression Line
plt.scatter(df.tenure, np.log(df.TotalCharges))
plt.plot(df.tenure, pred3, "r")
plt.legend(['Observed data', 'Predicted line'])
plt.show()


pred3_at = np.exp(pred3)
print(pred3_at)

res3 = df.TotalCharges - pred3_at
res_sqr3 = res3 * res3
mse3 = np.mean(res_sqr3)
rmse3 = np.sqrt(mse3)
rmse3


# ## Polynomial transformation 


X = pd.DataFrame(df['tenure'])


Y = pd.DataFrame(df['TotalCharge'])


model4 = smf.ols('np.log(TotalCharges) ~ tenure + I(tenure*tenure)', data = df).fit()
model4.summary()

pred4 = model4.predict(pd.DataFrame(df))
print(pred4)


plt.scatter(X['tenure'], np.log(Y['TotalCharges']))
plt.plot(X['tenure'], pred4, color = 'red')
plt.plot(X['tenure'], pred3, color = 'green', label = 'linear')
plt.legend(['Transformed Data', 'Polynomial Regression Line', 'Linear Regression Line'])
plt.show()

pred4_at = np.exp(pred4)
pred4_at

# Error calculation
res4 = df.TotalCharges - pred4_at
res_sqr4 = res4 * res4
mse4 = np.mean(res_sqr4)
rmse4 = np.sqrt(mse4)
rmse4


# ### Choose the best model using RMSE
data = {"MODEL":pd.Series(["SLR", "Log model", "Exp model", "Poly model"]), "RMSE":pd.Series([rmse1, rmse2, rmse3, rmse4])}
table_rmse = pd.DataFrame(data)

table_rmse

# # Evaluate the best model
# Data Split
train, test = train_test_split(df, test_size = 0.2, random_state = 0)

plt.scatter(train.tenure, np.log(train.TotalCharges))

plt.figure(2)
plt.scatter(test.tenure, np.log(test.TotalCharges))

# Fit the best model on train data
finalmodel = smf.ols('np.log(TotalCharges) ~ tenure + I(tenure*tenure)', data = train).fit()


# Predict on test data
test_pred = finalmodel.predict(test)
pred_test_AT = np.exp(test_pred)

# Model Evaluation on Test data
test_res = test.AT - pred_test_AT
test_sqrs = test_res * test_res
test_mse = np.mean(test_sqrs)
test_rmse = np.sqrt(test_mse)

test_rmse

# Prediction on train data
train_pred = finalmodel.predict(pd.DataFrame(train))
pred_train_AT = np.exp(train_pred)
pred_train_AT

# Model Evaluation on train data
train_res = train.AT - pred_train_AT
train_sqrs = train_res * train_res
train_mse = np.mean(train_sqrs)
train_rmse = np.sqrt(train_mse)

train_rmse





poly_model = make_pipeline(PolynomialFeatures(degree = 2), LinearRegression())
poly_model.fit(df[['tenure']], df[['TotalCharges']])

pickle.dump(poly_model, open('poly_model.pkl', 'wb'))


### testing on new data
# Load the saved pipelines

impute = joblib.load('meanimpute')
winsor = joblib.load('winzor')
poly_model = pickle.load(open('poly_model.pkl', 'rb'))


wcat_test = pd.read_csv(r"C:/Users/ashut/Downloads/Simple Linear Regression/Simple Linear Regression/datasets/wc-at_test.csv")

clean1 = pd.DataFrame(impute.transform(wcat_test), columns = wcat_test.select_dtypes(exclude = ['object']).columns)

clean2 = pd.DataFrame(winsor.transform(clean1), columns = clean1.columns)

prediction = pd.DataFrame(poly_model.predict(clean2), columns = ['Pred_AT'])

final = pd.concat([prediction, wcat_test], axis = 1)

final