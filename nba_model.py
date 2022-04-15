# -*- coding: utf-8 -*-
"""
Created on Wed Mar  2 08:58:47 2022

@author: User
"""

import pandas as pd
import os

import re
import requests 
from bs4 import BeautifulSoup as bs
import pandas as pd
from datetime import datetime
from datetime import timedelta
import joblib

from datapreprocessing import *
from scrapy import *
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.ensemble import StackingClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB

import joblib
from sklearn.preprocessing import StandardScaler

import sklearn
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
import xgboost
from  pymongo import MongoClient
from sklearn.preprocessing import StandardScaler


scaler = StandardScaler()
pd.options.mode.chained_assignment = None

elo_ =  True


path = './model'
scaler = joblib.load(path + '/scalert_all.bin')

regressor =  not True
if regressor:
  xgboost_model = joblib.load(path +  '/xgboost_all_regressor.model')
  svc = joblib.load(path + '/svr_all_regressor.model')
  stacking = joblib.load(path + '/stacking_all__regressor.model')
  voting = joblib.load(path + '/voting_all_regressor.model')
  RF = joblib.load(path + '/RF_all_regressor.model')
else:
  xgboost_model = joblib.load(path + '/xgboost_all.model')
  svc = joblib.load(path + '/svc_all.model')
  stacking = joblib.load(path + '/stacking_all.model')
  voting = joblib.load(path + '/voting_all.model')
  lr = LogisticRegression()
yr = 2022


rf = RandomForestClassifier(random_state = 1)
gnb = GaussianNB()
svm_model = SVC(gamma = 'auto')


estimators = [
    ('rf', rf), ('svm', svm_model)
]
clf = StackingClassifier(
    estimators = estimators, final_estimator = gnb

)



s_date = '20220210'
now = datetime.now() -  timedelta(hours= 8) 
year, month, day = str(now.year), str(now.month), str(now.day)
month = '0'+ month if len(month) != 2  else month
day = '0'+ day if len(day) != 2  else day
date = year  + month  +day
# col = ['visitor_pts', 'ts_pct_away_advanced', 'trb_pct_away_advanced',
#        'ast_away_opp', 'blk_away_opp', 'pts_away_opp',
#        'matchup_away_win_percent', 'matchup_away_lose_percent', 'ast_home_opp',
#        'pts_home_opp', 'def_rtg_home_opp_advanced', 'pts_home',
#        'current_home_win', 'current_home_lose']

process = not True
if   process:
    import os 
    path = r'./nba_current'
    data = pd.DataFrame()
    for i in os.listdir(path):
        
        if '_20' in i and 'orgin' not in i :
            yr = i.split('.csv')[0][-4:]
            if int(yr) > 2008:
                temp = pd.read_csv(path + '/' + i)
                temp['year'] = yr
                
                temp = preprocessing(temp, not True,int(yr) ,10, prev_game = False) 
                data = pd.concat([data,temp])
    
    data['date'] = data['id'].apply(lambda x: x[:-4])               
    
    data = data.loc[:, ~data.columns.str.contains('Unnamed')]     
    
    data.to_csv('./nba_preprocessing/nba_preprocessed.csv', index = not True)



# yr = 2022
# t_ = pd.read_csv('./nba_current/nba_{}_{}.csv'.format(yr-1, yr))
data = pd.read_csv('./nba_preprocessing/nba_preprocessed.csv').sort_values('id')
# data.iloc[:, :-2].to_csv('./nba_preprocessing/nba_preprocessed.csv', index = not True)
data['date'] = data['id'].apply(lambda x: x[:-4])


# data = pd.concat([data_current,t_processed])
# data = data.loc[:, ~data.columns.str.contains('Unnamed')]
# data_current = data[data['year'] == yr]
# data[data['year'] != yr]
# t_processed = preprocessing(t_, not True, 2022,10, prev_game = False) 
# t__ = data_current.sort_values('id')['visitor_pts'].reset_index(drop = True) - t_processed.sort_values('id')['visitor_pts'].reset_index(drop = True)



def team_edit(x):
    team_dict = {'Charlotte Bobcats':'Charlotte Hornets','New Jersey Nets':'Brooklyn Nets','New Orleans Hornets':'New Orleans Pelicans',
    'Seattle SuperSonics':'Oklahoma City Thunder'
    }    
    if x in team_dict.keys():
        return team_dict[x]
    else:
        return x


data['home_team_name'] = data['home_team_name'].apply(lambda x: team_edit(x))
data['visitor_team_name'] = data['visitor_team_name'].apply(lambda x: team_edit(x))


data['date'] = data['date'].astype(str)
data = data.loc[:, ~data.columns.str.contains('Unnamed')]
data = data.loc[:, ~data.columns.str.contains('plus')]
data = data.loc[:, ~data.columns.str.contains('bpm')]
data = data.loc[:, ~data.columns.str.contains('overtimes')]
data = data.loc[:, ~data.columns.str.contains('remarks')]
data = data.loc[:, ~data.columns.str.contains('result')]

train_odds = pd.read_csv(r'./nba_odd_preprocessing - nba_odd_preprocessing.csv')[['date','home_Team','away_Team','away_ML', 'home_ML']]
test_odds = pd.read_csv(r'./nba odds 2021-22_processed.csv').rename(columns = {'away_Date':'date'})


import pymongo

client = pymongo.MongoClient('mongodb://localhost:27017/')
db = client['nba']
collection = db['odds']

date_ = datetime.strptime(s_date, '%Y%m%d') + timedelta(days = 1)
date_ = date_.strftime('%Y-%m-%d')

cursor = collection.find({'date' : {'$gte':date_}})
db_odds = pd.DataFrame()
for x in cursor:
    db_odds = pd.concat([db_odds, pd.DataFrame(x,index =[0])])



db_odds = db_odds[['home','away','home_prob','away_prob','date']]
db_odds['true_away_probability'] = db_odds['away_prob']/(db_odds['away_prob'] + db_odds['home_prob'])
db_odds['true_home_probability'] = db_odds['home_prob']/(db_odds['away_prob'] + db_odds['home_prob'])
db_odds['date'] = db_odds['date'].apply(lambda x : datetime.strftime(datetime.strptime(x,'%Y-%m-%d')-timedelta(days = 1), '%Y%m%d'))



odds = pd.concat([train_odds, test_odds])
odds['date'] = odds.astype(str)
# odds = odds[(odds['date'] != '20080330'&odds['home_Team'] != 'Boston')]
for col in odds.columns:
    try :
        if col != 'date':
            odds[col] = odds[col].apply(lambda x: 1 if x == 'NL' else x)
            odds[col] = odds[col].astype(float)
            odds[col] = odds[col].apply(lambda x : x/(x+100)if x>0 else (-x)/(-x + 100))
            
    except Exception as e :
        print(e)
        pass
    
odds['true_away_probability'] = odds['away_ML'] /(odds['away_ML'] +odds['home_ML'])
odds['true_home_probability'] = odds['home_ML'] /(odds['away_ML'] +odds['home_ML'])




dict_ ={'Brooklyn': 'Brooklyn Nets',
 'GoldenState': 'Golden State Warriors',
 'Indiana': 'Indiana Pacers',
 'Chicago': 'Chicago Bulls',
 'Washington': 'Washington Wizards',
 'Boston': 'Boston Celtics',
 'Cleveland': 'Cleveland Cavaliers',
 'Philadelphia': 'Philadelphia 76ers',
 'Houston': 'Houston Rockets',
 'Orlando': 'Orlando Magic',
 'OklahomaCity': 'Oklahoma City Thunder',
 'Sacramento': 'Sacramento Kings',
 'Denver': 'Denver Nuggets',
 'Dallas': 'Dallas Mavericks',
 'Milwaukee': 'Milwaukee Bucks',
 'LAClippers': 'Los Angeles Clippers',
 'NewYork': 'New York Knicks',
 'Charlotte': 'Charlotte Hornets',
 'Toronto': 'Toronto Raptors',
 'NewOrleans': 'New Orleans Pelicans',
 'SanAntonio': 'Sacramento Kings',
 'Phoenix': 'Phoenix Suns',
 'Utah': 'Utah Jazz',
 'Atlanta': 'Atlanta Hawks',
 'Miami': 'Miami Heat',
 'Detroit': 'Detroit Pistons',
 'Memphis': 'Memphis Grizzlies',
 'Portland': 'Portland Trail Blazers',
 'LALakers': 'Los Angeles Lakers',
 'Minnesota': 'Minnesota Timberwolves',
 'Oklahoma City': 'Oklahoma City Thunder',
 'LA Clippers': 'Los Angeles Clippers',
 'NewJersey': 'Brooklyn Nets',
 'Seattle': 'Seattle'}

odds['away_Team'] = odds['away_Team'].apply(lambda x : dict_[x] if x in dict_ else x)
odds['home_Team'] = odds['home_Team'].apply(lambda x : dict_[x] if x in dict_ else x)

db_odds['away'] = db_odds['away'].apply(lambda x : dict_[x] if x in dict_ else x)
db_odds['home'] = db_odds['home'].apply(lambda x : dict_[x] if x in dict_ else x)
db_odds = db_odds.rename(columns = {'away':'away_Team','home':'home_Team'})

odds_new = pd.concat([odds, db_odds])[['date','home_Team','away_Team','true_home_probability','true_away_probability']]


data = data.merge(odds_new, left_on = ['date', 'home_team_name'], right_on = ['date', 'home_Team'])
    

train_date = '20220210'
train_data = data[data['date'].astype(str) < train_date]
# test_data = data[(data['date']>=train_date) & (data['date']<s_date)]

test_data = data[(data['date'].astype(str) >= train_date)&(data['date'].astype(str)< '20221112')]
# y_test = test_data['id']
if elo_:
    col = [
      'current_home_win',
      'current_home_lose',
      'def_rtg_home_opp_advanced',
      'def_rtg_home_advanced',
      'pts_home_opp',
      'drb_away',
      'pts_away',
      'pts_home',
      'pts_away_opp',
        'efg_pct_away_advanced',
        'ts_pct_home_advanced',
        'ts_pct_home_opp_advanced',
        'ts_pct_away_advanced',
      'fg_home_opp',
      'blk_away_opp',
      'efg_pct_home_advanced',
      'efg_pct_home_opp_advanced',
      'ast_away_opp',
      'trb_home_opp',
        'orb_pct_away_opp_advanced',
      'trb_away_opp', 'true_home_probability','true_away_probability','home_elo','away_elo']
else:
    col = [
    'true_away_probability',
    # 'true_home_probability',
    
    ]
    
    
    # col = [
    #   'current_home_win',
    #   'current_home_lose',
    #   'def_rtg_home_opp_advanced',
    #   'def_rtg_home_advanced',
    #   'pts_home_opp',
    #   'drb_away',
    #   'pts_away',
    #   'pts_home',
    #   'pts_away_opp',
    #     'efg_pct_away_advanced',
    #     'ts_pct_home_advanced',
    #     'ts_pct_home_opp_advanced',
    #     'ts_pct_away_advanced',
    #   'fg_home_opp',
    #   'blk_away_opp',
    #   'efg_pct_home_advanced',
    #   'efg_pct_home_opp_advanced',
    #   'ast_away_opp',
    #   'trb_home_opp',
    #     'orb_pct_away_opp_advanced',
    #   'trb_away_opp', 'true_home_probability','true_away_probability', 'id']
    
    
    # col = [
    #   'current_home_win',
    #   'current_home_lose',
    #   'def_rtg_home_opp_advanced',
    #   'def_rtg_home_advanced',
    #   'pts_home_opp',
    #   'drb_away',
    #   'pts_away',
    #   'pts_home',
    #   'pts_away_opp',
    #     'efg_pct_away_advanced',
    #     'ts_pct_home_advanced',
    #     'ts_pct_home_opp_advanced',
    #     'ts_pct_away_advanced',
    #   'fg_home_opp',
    #   'blk_away_opp',
    #   'efg_pct_home_advanced',
    #   'efg_pct_home_opp_advanced',
    #   'ast_away_opp',
    #   'trb_home_opp',
    #     'orb_pct_away_opp_advanced',
    #   'trb_away_opp', 'true_home_probability','true_away_probability' , 'id']
    
    # col = [
    #  'true_home_probability','true_away_probability','id']


# col = list(train_data.loc[:, train_data.columns.str.contains('advanced')].columns)
# remove_list = ['box_score_text_y','box_score_text_x','attendance_y','attendance_x','away_Team','home_Team']

# [col.remove(x) for x in train_data.loc[:, train_data.columns.str.contains('game_start_time')].columns]
# [col.remove(x) for x in remove_list]
# col += list(train_data.loc[:, train_data.columns.str.contains('true')].columns)
# col.append('id')

if elo_:
    from datetime import datetime, timedelta
    elo = pd.read_csv('./elo_rating.csv')
    elo['team'] = elo['team'].astype('str')
    elo['d'] = elo['d'].astype('str')
    elo['d'] = elo['d'].apply(lambda x : (datetime.strptime(x, '%Y%m%d') + timedelta(days = 1)).strftime('%Y%m%d'))

    # elo.index = elo['d'] + elo['team']
    elo = elo[['d','team','y']]
    def elo_rating(x, col):
        date = x['date']
        team_elo = elo[(elo['d'] < date)&(elo['team'] == x[col])].iloc[-1]['y']
        return team_elo
    
        
    train_data['date'] = train_data['date'].astype(str)
    test_data['date'] = test_data['date'].astype(str)
    train_data['home_elo'] = train_data.apply(lambda x: elo_rating(x, 'home_team_name'), axis = 1)
    train_data['away_elo'] = train_data.apply(lambda x: elo_rating(x, 'visitor_team_name'), axis = 1)
    
    test_data['home_elo'] = test_data.apply(lambda x: elo_rating(x, 'home_team_name'), axis = 1)
    test_data['away_elo'] = test_data.apply(lambda x: elo_rating(x, 'visitor_team_name'), axis = 1)
    # train_data = elo.merge(train_data, left_on = ['d','team'], right_on = ['date','visitor_team_name'])
    # train_data = elo.merge(train_data, left_on = ['d','team'], right_on = ['date','home_team_name'])
    
    # test_data = elo.merge(test_data,left_on = ['d','team'], right_on = ['date','visitor_team_name'])
    # test_data = elo.merge(test_data,left_on = ['d','team'], right_on = ['date','home_team_name'])
# col.remove(list(train_data.loc[:, train_data.columns.str.contains('game_start_time')].columns))

    # train_data['prob_y_x'] =1/(1  + 10**((train_data['y_y']- train_data['y_x'])/400))
    # train_data['prob_y_y'] = 1/(1  + 10**((train_data['y_x']- train_data['y_y'])/400))
    
    # test_data['prob_y_x'] =1/(1  + 10**((test_data['y_y']- test_data['y_x'])/400))
    # test_data['prob_y_y'] = 1/(1  + 10**((test_data['y_x']- test_data['y_y'])/400))

train_data.index = train_data['id']
# train = train_data[col].dropna()
test_data.index = test_data['id']
test_data = test_data.loc[~test_data['current_home_win'].duplicated()]
# test = test_data[col].dropna()
train = train_data
test = test_data
# 


train_id = train['id'].to_list()
test_id = test['id'].to_list()
train = train.select_dtypes(exclude=['object','datetime64'])
test = test.select_dtypes(exclude=['object','datetime64'])


train = train.dropna()
test = test.dropna()



y_train =train_data[train_data.index.isin(train.index)][['true_home_pts','true_visitor_pts']]
y_test = test_data[['true_home_pts','true_visitor_pts']]

# train['id'] = train_id

# test['id'] = test_id
train = train.loc[:,~train.columns.str.contains('year')]
train = train.loc[:,~train.columns.str.contains('date')]

test = test.loc[:,~test.columns.str.contains('year')]
test = test.loc[:,~test.columns.str.contains('date')]


# y_train = y_train[y_train['id'].isin(train['id'])][['true_home_pts', 'true_visitor_pts']]
# y_test = y_test[y_test['id'].isin(test['id'])][['true_home_pts', 'true_visitor_pts']]

train = train.loc[:,~train.columns.str.contains('true_home_pts')]
test = test.loc[:,~test.columns.str.contains('true_home_pts')]

train = train.loc[:,~train.columns.str.contains('true_visitor_pts')]
test = test.loc[:,~test.columns.str.contains('true_visitor_pts')]

y_train['result'] = np.where(y_train['true_home_pts'] - y_train['true_visitor_pts'] >0 ,1, -1)

y_test['result'] = np.where(y_test['true_home_pts'] - y_test['true_visitor_pts'] >0 ,1, -1)

# train['odds'] = np.where(train['true_home_probability'] - train['true_away_probability'] >0 , 1, -1)
# test['odds'] = np.where(test['true_home_probability'] - test['true_away_probability'] >0 , 1, -1)


# train = train.sort_values('id').drop(columns = ['id'])
# test = test.sort_values('id').drop(columns = ['id'])

# col = train.columns
test = test.sort_values('id')
y_test = y_test.sort_values('id')

train = train[col]
test = test[col]

# y_train = y_train.loc[train.index]
# y_test = y_test.loc[test.index]
train = train.loc[:,~train.columns.duplicated()]
test = test.loc[:,~test.columns.duplicated()]
col = train.columns

train_index = train.index
test_index = test.index
train = pd.DataFrame(scaler.fit_transform(train))
test = pd.DataFrame(scaler.fit_transform(test))
train.index = train_index
test.index = test_index

train.columns = col
test.columns = col

# param = {
#     'n_estimators': [300,400,500,600,700],
#     'max_depth': [10,11,12,13],
#     'learning_rate': [0.01,0.001],
# }
# param = {'learning_rate': 0.01, 'max_depth': 11, 'n_estimaors': 700}
# param = {'learning_rate': 0.01, 'max_depth': 10, 'n_estimators': 300}

# param = {'learning_rate': 0.01, 'max_depth': 12, 'n_estimators': 400}
param = {'learning_rate': 0.01, 'max_depth': 10, 'n_estimators': 400}

from sklearn.experimental import  enable_halving_search_cv
from sklearn.feature_selection import RFECV,RFE
xgboost_model = xgboost.XGBClassifier(**param)


# selector =  RFECV(lr, step = 1, cv= 5)

# selector.fit(train, y_train['result'])
# selector = sklearn.model_selection.HalvingGridSearchCV(estimator = xgboost_model, param_grid = param)
# selector.fit(train, y_train['result'])


xgboost_model.fit(train, y_train['result'] )
lr.fit(train, y_train['result'] ) 
clf.fit(train,y_train['result'])


# joblib.dump(xgboost_model, './xgboost_model_new_elo.model')
# joblib.dump(lr, './lr_new_elo.model')
# joblib.dump(lr, './stack_new_elo.model')
y_pre = xgboost_model.predict(test)
y_lr = lr.predict(test)
y_test['pred'] = y_pre
# y_test['date'] = y_test['id'].apply(lambda x: x[:-4])
y_test['lr'] = y_lr
y_test['clf'] = clf.predict(test)
print(sklearn.metrics.accuracy_score(y_test['pred'], y_test['result'] ))
print(sklearn.metrics.accuracy_score(y_test['lr'], y_test['result'] ))
print(sklearn.metrics.accuracy_score(y_test['clf'], y_test['result'] ))
# print(sklearn.metrics.accuracy_score(y_test[y_test['date']>= '20220210']['pred'],y_test[ y_test['date']>= '20220210']['result'] ))

    

# joblib.dump(xgboost_model, './model/xgboost_model_fix.model')
# import matplotlib.pyplot as plt
# (train.columns,xgboost_model.feature_importances_)
# df = pd.DataFrame(train.columns)
# df['feature_importances'] = xgboost_model.feature_importances_
# df.nlargest(20, 'feature_importances')[0].to_list()

# np.unique(y_test, return_counts=True)



