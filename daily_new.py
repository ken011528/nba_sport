# -*- coding: utf-8 -*-
"""
Created on Wed Mar  2 16:58:06 2022

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
elo = pd.read_csv('./elo_rating.csv')

path = './model'
# xgboost_model = joblib.load(path + '/xgboost_model_new.model')


regressor =  not True
if regressor:
  xgboost_model = joblib.load(path +  '/xgboost_all_regressor.model')

else:

  xgboost_model = joblib.load('./xgboost_model_new.model')

yr = 2022

s_date = '20220413'


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
  'trb_away_opp', 'true_home_prob','true_away_prob']



current_season_data = pd.read_csv('./nba_current/nba_{}_{}_orgin.csv'.format(yr-1, yr))
current_season_data['date'] = current_season_data['id'].apply(lambda x: x[:-4])
current_season_data['year'] = yr
scrapy =     True

now = datetime.now() -  timedelta(hours= 8) 
year, month, day = str(now.year), str(now.month), str(now.day)
month = '0'+ month if len(month) != 2  else month
day = '0'+ day if len(day) != 2  else day
date = year  + month  +day

if scrapy:
  
  new_data = get_data()
  new_data['date'] = new_data['id'].apply(lambda x: x[:-4]).astype(str)
  new_data['result'] = np.where(new_data['home_pts'] - new_data['visitor_pts'] >0 , 1, -1)
  new_data.to_csv('./nba_current/nba_{}_{}.csv'.format(yr-1, yr), index = not True)

  # current_season_data_ = current_season_data[(current_season_data['date'] ==date)]
  current_season_data_ = current_season_data[current_season_data['date'] >= date]
  current_season_data_ = pd.concat([new_data, current_season_data_])
  current_season_data_.to_csv('./nba_current/nba_{}_{}_orgin.csv'.format(yr-1, yr), index = not True)
  scrapy = not   True
  current_season_data_ = current_season_data[current_season_data['date']<=date]   
  new_data = current_season_data_
else:
  current_season_data_ = current_season_data[current_season_data['date']<=date]    
  new_data = current_season_data_



test = preprocessing(new_data, not True, 2022,10, prev_game = False) 
test['date'] = test['id'].apply(lambda x: x[:-4]).astype(str)

import pymongo

client = pymongo.MongoClient('mongodb://localhost:27017/')
db = client['nba']
collection = db['odds']

date_ = datetime.strptime(s_date, '%Y%m%d') + timedelta(days = 1)
date_ = date_.strftime('%Y-%m-%d')

cursor = collection.find({'date' : {'$gte':date_}})
odds = pd.DataFrame()
for x in cursor:
    odds = pd.concat([odds, pd.DataFrame(x,index =[0])])
    
odds['true_away_prob'] = odds['away_prob']/(odds['away_prob'] + odds['home_prob'])
odds['true_home_prob'] = odds['home_prob']/(odds['away_prob'] + odds['home_prob'])

odds ['date'] = odds['date'].apply(lambda x: (datetime.strptime(x, '%Y-%m-%d') - timedelta(days = 1)).strftime('%Y%m%d'))
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
odds['away'] = odds['away'].apply(lambda x: dict_[x] if x in dict_ else x)
odds['home'] = odds['home'].apply(lambda x: dict_[x] if x in dict_ else x)

current_season_data_process = test[(test['date'] >= s_date)]
# current_season_data_process = test[(test['date'] >= s_date)&(test['date']< '20220312')]



current_season_data_process = current_season_data_process[~pd.isnull(current_season_data_process['visitor_pts'])]

current_season_data_process = current_season_data_process.merge(odds, left_on = ['date', 'visitor_team_name'], right_on = ['date', 'away'])
odds[odds['date'] == s_date]['away']


y_test = current_season_data_process[['true_home_pts', 'true_visitor_pts','home_team_name','visitor_team_name' ,'id','date']]
y_test['result'] = pd.DataFrame(np.where(current_season_data_process['true_home_pts'] - current_season_data_process['true_visitor_pts']>0, 1, -1))


current_season_data_process = current_season_data_process.sort_values('id')
y_test = y_test.sort_values('id')



current_season_data_process_ = current_season_data_process[col]



current_season_data_process_col = current_season_data_process_[col]



current_season_data_process_scaler = current_season_data_process_col

list_ = []
for i in col:
  if 'year' not in i.lower() and 'true' not in i:
    list_.append(i)
    

list_.append('true_home_prob')
list_.append('true_away_prob')
list_ = col


current_season_data_process_scaler = current_season_data_process_scaler.loc[:,~current_season_data_process_scaler.columns.str.contains('true_home_pts')]
current_season_data_process_scaler = current_season_data_process_scaler.loc[:,~current_season_data_process_scaler.columns.str.contains('true_visitor_pts')]

current_season_data_process_scaler = current_season_data_process_scaler.fillna(current_season_data_process_scaler.mean())
current_season_data_process_scaler_index = current_season_data_process_scaler.index


current_season_data_process_scaler = pd.DataFrame(scaler.fit_transform(current_season_data_process_scaler))
current_season_data_process_scaler.index = current_season_data_process_scaler_index
current_season_data_process_scaler.columns = list_

current_season_data_process_scaler = current_season_data_process_scaler.rename(columns = {'true_home_prob':'true_home_probability', 'true_away_prob':'true_away_probability'})

def bet(x):
    bet_profit = 2

    if x['away_pred_pro']> x['true_away_prob']:
        fraction = (x['away_pred_pro']*float(x['away_odds'])-x['away_pred_pro'])/float(x['away_odds'])
        return -bet_profit*fraction/6
    else:
        fraction = (x['home_pred_pro']*float(x['home_odds'])-x['home_pred_pro'])/float(x['home_odds'])
        return bet_profit*fraction/6


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
  'trb_away_opp', 'true_home_probability','true_away_probability']


result = pd.DataFrame()

current_season_data_process_scaler = current_season_data_process_scaler[col]
# current_season_data_process_scaler = current_season_data_process_scaler.loc[:, ~current_season_data_process_scaler.columns.str.contains('true')]

result = pd.DataFrame(xgboost_model.predict_proba(current_season_data_process_scaler))
result.index = current_season_data_process_scaler.index

result = result.rename(columns = {1:'home_pred_pro',0:'away_pred_pro'})
# result['awway_probabitily'] = y_xgboost.iloc[:,-1]
result['home_team_name'] = current_season_data_process['home_team_name']
result['visitor_team_name'] = current_season_data_process['visitor_team_name']
result['result_pred'] = np.where(result['home_pred_pro'] > result['away_pred_pro'] , 1, -1)
result['id'] = current_season_data_process['id']
result['date'] = current_season_data_process.sort_values('id')['date']

t = result.merge(odds, left_on = ['date','visitor_team_name', 'home_team_name'], right_on = ['date','away','home'])
t['bet'] = t.apply(lambda x: bet(x),axis = 1)
t = t.merge(y_test, on = ['id','home_team_name','visitor_team_name'])
t = t[['id','home_team_name', 'visitor_team_name', 'true_home_prob', 'true_away_prob', 'home_pred_pro', 'away_pred_pro','bet','home_odds','away_odds','result','result_pred']]
t['odd_win'] = np.where(t['true_home_prob']> t['true_away_prob'] , 1, -1)
print(sklearn.metrics.accuracy_score(result['result_pred'], y_test['result']))

print(sklearn.metrics.accuracy_score(result['result_pred_lr'], y_test['result']))
print(sklearn.metrics.accuracy_score(t['odd_win'], y_test['result']))
print(result)

def net(pred, home_odds, away_odds, true_result):
    if pred ==true_result:
        if true_result== 1:
            return float(home_odds)*1000 - 1000
        else:
            return float(away_odds)*1000 - 1000
    else:
        return -1000
    
t['net'] = t.apply(lambda x: net(x['result_pred'], x['home_odds'], x['away_odds'], x['result']), axis = 1)
t['lr_net'] = t.apply(lambda x: net(x['result_pred_lr'], x['home_odds'], x['away_odds'], x['result']), axis = 1)
t['date'] = t['id'].apply(lambda x: x[:-4])

for idx, group in     t.groupby('date'):
    print('---'*10 + 'xgboost_model' + '---'*10)
    print(idx, group['net'].sum())
    print('---'*10 + 'lr' + '---'*10)
    print(idx, group['lr_net'].sum())
print('---'*10 + 'xgboost_model' + '---'*10)
print(t['net'].sum())
print('---'*10 + 'lr' + '---'*10)
print(t['lr_net'].sum())

t['pred_odds'] = np.where(t['home_pred_pro'] - t['true_home_prob']>0, 1, -1)
# t__ = t[t['result_pred'] == t['result_pred_lr']]
t__ = t[(t['result_pred_lr'] == t['result_pred_stack'])]
def bet_or_not(x):
    if (x['result_pred'] == 1 and x['home_pred_pro'] > x['true_home_prob']) or (x['result_pred'] == -1 and x['away_pred_pro'] > x['true_away_prob']):
        return 1
    else:
        return -1
t__['bet_'] = t__.apply(lambda x:   bet_or_not(x), axis = 1 )
t_bet = t__[t__['bet_'] == -1]
t_bet['net'].sum()
print(sklearn.metrics.accuracy_score(t_bet['result_pred_lr'], t_bet['result']))

