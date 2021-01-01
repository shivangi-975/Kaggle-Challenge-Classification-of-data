#!/usr/bin/env python
# coding: utf-8

# In[101]:


import numpy as np
import pandas as pd
import sklearn
import seaborn as sns
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import warnings;
warnings.filterwarnings('ignore');
from imblearn.over_sampling import RandomOverSampler, BorderlineSMOTE 


# In[102]:


df=pd.read_csv('train_input.csv')


# In[103]:


df_copy = df.copy()


# In[104]:


lastcolumn_np = df.iloc[:,-1:].to_numpy()
lastcolumn_op=np.ravel(lastcolumn_np)


# In[105]:


df_copy.drop(['Target Variable (Discrete)','Feature 16','Feature 17'], axis=1, inplace=True)


# In[106]:


imp_mean = IterativeImputer(random_state=0)
imp_mean.fit(df_copy)
IterativeImputer(random_state=0)
data = imp_mean.transform(df_copy)


# In[108]:


corr_features = set()
correlation_matrix = pd.DataFrame(data).corr()
for i in range(len(correlation_matrix.columns)):
    for j in range(i):
        if abs(correlation_matrix.iloc[i, j]) > 0.8:
            colname = correlation_matrix.columns[i]
            corr_features.add(colname)
to_drop=np.array(list(corr_features))


# In[109]:


np_drop = np.delete(data,to_drop,1)


# In[110]:


Dict = {17:10, 12:10, 11:10, 16:10, 10:10, 9:10}


# In[111]:


over = RandomOverSampler(random_state=0, sampling_strategy=Dict)


# In[112]:


X_resampled, y_resampled = over.fit_resample(np_drop, lastcolumn_op)


# In[113]:


sm = BorderlineSMOTE(random_state=0, k_neighbors=2, kind='borderline-2')


# In[13]:


X_resampled, y_resampled = sm.fit_resample(X_resampled, y_resampled)


# In[17]:


n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
max_features = ['auto', 'sqrt']
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
max_depth.append(None)
min_samples_split = [2, 5, 10]
min_samples_leaf = [1, 2, 4]
bootstrap = [True, False]
params = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}
cv_result = RandomizedSearchCV(RandomForestClassifier(),params,cv=3,scoring='accuracy',random_state = 5)
cv_result.fit(X_resampled, y_resampled)
cv_result.best_params_


# In[114]:


classifier_op = RandomForestClassifier(n_estimators = 2000, min_samples_split=2,min_samples_leaf=1,max_features='auto',
                               max_depth=70, random_state = 42,bootstrap=False)
classifier_op.fit(X_resampled, y_resampled)


# In[115]:


df_2=pd.read_csv('iith_foml_2020_test.csv')


# In[122]:


df_2_copy = df_2.copy()
df_2_copy.drop(['Feature 16','Feature 17'], axis=1, inplace=True)


# In[123]:


data_2 = imp_mean.transform(df_2_copy)


# In[124]:


np_drop_2 = np.delete(data_2, to_drop,1)


# In[125]:


data_pred = classifier_op.predict(np_drop_2)


# In[126]:


col_id = np.linspace(1,426,426,dtype=int)


# In[127]:


Final_df = pd.DataFrame(columns=['Id','Category'])


# In[128]:


Final_df['Id'] = col_id


# In[129]:


Final_df['Category'] = data_pred


# In[130]:


Final_df.to_csv('test_ouput.csv', index=False)


# In[ ]:





# In[ ]:




