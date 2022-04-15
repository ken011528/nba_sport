import pandas as pd
import os
import re
import numpy as np
import requests 
from bs4 import BeautifulSoup as bs
import pandas as pd
from sklearn.preprocessing import StandardScaler
pd.options.mode.chained_assignment = None


# pd.read_csv('/content/drive/MyDrive/Colab Notebooks/nba/nba_preprocessing/nba_preprocessing_2009_2021_full')



def preprocessing(df,train,yr,rolling_window, prev_game = True, spread = True):    
    prev_season = pd.read_csv('./nba_current/all.csv').sort_values('id')
    playoff = pd.read_csv('./nba_current/playoff.csv')
    prev_season = prev_season[prev_season['year'] <= yr-1]
    prev_season['date'] = prev_season['id'].apply(lambda x:x[:-4])
    playoff['year'] = playoff['date'].apply(lambda x: str(x)[:4])

    regula = pd.DataFrame()
    for idx ,group in prev_season.groupby('year'):
      date = playoff[playoff['year'] == str(idx)]['date'].values[0]
      
      regula = pd.concat([regula, group[group['date']< str(date)]])

    all_season = pd.concat([regula, df])
    

    # all_season = pd.concat([prev_season, df])


    df['date'] = df.apply(lambda x: x['id'][:-4] , axis = 1)
    team_dict = {'Charlotte Bobcats':'Charlotte Hornets','New Jersey Nets':'Brooklyn Nets','New Orleans Hornets':'New Orleans Pelicans',
    'Seattle SuperSonics':'Oklahoma City Thunder'
    }
    df['visitor_team_name'] = df['visitor_team_name'].apply(lambda x: team_dict[x] if x in team_dict else x)
    df['home_team_name'] = df['home_team_name'].apply(lambda x: team_dict[x] if x in team_dict else x)
    #   df['year'] = df['year'].apply(lambda x : str(x))
    # result['year'] = pd.to_datetime(result['year'])
    df['date_game'] = pd.to_datetime(df['date'].apply(lambda x: x[:4] + '-' + x[4:6] + '-' + x[6:8]))

    df['result'] = np.where(df['home_pts']- df['visitor_pts'] >0 ,1, -1)
    
    all_season['date'] = all_season.apply(lambda x: x['id'][:-4] , axis = 1)
    
    all_season['visitor_team_name'] = all_season['visitor_team_name'].apply(lambda x: team_dict[x] if x in team_dict else x)
    all_season['home_team_name'] = all_season['home_team_name'].apply(lambda x: team_dict[x] if x in team_dict else x)
#     all_season['year'] = all_season['year'].apply(lambda x : str(x))
    # result['year'] = pd.to_datetime(result['year'])
    all_season['date_game'] = pd.to_datetime(all_season['date'].apply(lambda x: x[:4] + '-' + x[4:6] + '-' + x[6:8]))
    
    all_season['result'] = np.where(all_season['home_pts']- all_season['visitor_pts'] >0 ,1, -1)
    
    
    
    y = df[['id','visitor_pts','home_pts']]
    y = y.rename(columns = {'visitor_pts':'true_visitor_pts','home_pts':'true_home_pts'})
    
    home = pd.DataFrame()
    away = pd.DataFrame()


    result = all_season

    if spread:
      for idx_, (idx_tuple, group) in enumerate(df.sort_values('date').groupby(['year','home_team_name'])):
        yr = idx_tuple[0]
        team  = idx_tuple[-1]
        away_group = df[(df['visitor_team_name'] == team)&(df['year']== yr)]
        
        
        for col in group.columns:
          if group[col].dtype != 'O' and group[col].dtype != '<M8[ns]' and col != 'result':
            if prev_game:
              group[col] = group[col].shift(1).fillna(0)
              group[col] = group[col].rolling(rolling_window).sum()/(rolling_window-1)

              away_group[col] = away_group[col].shift(1).fillna(0)
              away_group[col] = away_group[col].rolling(rolling_window).sum()/(rolling_window-1)

            else:
              group[col] = group[col].shift(1)
              visitor_pts = group[col].values[1:]
              for idx, i in enumerate(group.index[1:]):
                if idx != 0:
                  # group.at[group.index[idx],col +'_roll'] = visitor_pts[:idx].mean()
                  group.at[group.index[idx],col +'_roll'] = visitor_pts[:idx].mean()
              group.at[group.index[-1], col +'_roll'] = visitor_pts.mean()
              group[col] = group[col+'_roll']
              group = group.drop(columns = col+'_roll')

              away_group[col] = away_group[col].shift(1)
              visitor_pts = away_group[col].values[1:]
              for idx, i in enumerate(away_group.index[1:]):
                if idx != 0:
                  away_group.at[away_group.index[idx],col +'_roll'] = visitor_pts[:idx].mean()
              away_group.at[away_group.index[-1], col +'_roll'] = visitor_pts.mean()
              away_group[col] = away_group[col+'_roll']
              away_group = away_group.drop(columns = col+'_roll')

            # away_group[col] = away_group[col].rolling(rolling_window).mean()
            if train:
              playoff_ = playoff[(playoff['year']== str(yr))]
              date = str(playoff_['date'].values[0])
              group  = group[group['date']<date]
              away_group = away_group[away_group['date'] < date]
 
        
        group['away_possessions'] = 0.5*(group['fga_away'] + 0.4*group['fta_away'] - 1.07*(group['orb_away']/(group['orb_away']+group['drb_home'])*(group['fga_away']-group['fg_away'])) + group['tov_away']+ group['fga_home'] + 0.4*group['fta_home'] - 1.07*(group['orb_home']/(group['drb_away']+group['orb_home'])*(group['fga_home']-group['fg_home'])) + group['tov_home'])
        group['home_possessions'] = 0.5*(group['fga_home'] + 0.4*group['fta_home'] - 1.07*(group['orb_home']/(group['orb_home']+group['drb_away'])*(group['fga_home']-group['fg_home'])) + group['tov_home']+ group['fga_away'] + 0.4*group['fta_away'] - 1.07*(group['orb_away']/(group['drb_home']+group['orb_away'])*(group['fga_away']-group['fg_away'])) + group['tov_away'])
        
        away_group['away_possessions'] = 0.5*(away_group['fga_away'] + 0.4*away_group['fta_away'] - 1.07*(away_group['orb_away']/(away_group['orb_away']+away_group['drb_home'])*(away_group['fga_away']-away_group['fg_away'])) + away_group['tov_away']+ away_group['fga_home'] + 0.4*away_group['fta_home'] - 1.07*(away_group['orb_home']/(away_group['drb_away']+away_group['orb_home'])*(away_group['fga_home']-away_group['fg_home'])) + away_group['tov_home'])
        away_group['home_possessions'] = 0.5*(away_group['fga_home'] + 0.4*away_group['fta_home'] - 1.07*(away_group['orb_home']/(away_group['orb_home']+away_group['drb_away'])*(away_group['fga_home']-away_group['fg_home'])) + away_group['tov_home']+ away_group['fga_away'] + 0.4*away_group['fta_away'] - 1.07*(away_group['orb_away']/(away_group['drb_home']+away_group['orb_away'])*(away_group['fga_away']-away_group['fg_away'])) + away_group['tov_away'])
        # for col in group.columns:
        #   if group[col].dtype != 'O' and group[col].dtype != '<M8[ns]' and col != 'result' and 'possessions' not in col:
        #       group[col] = group[col]*100/group['away_possessions']
              

        
        
        for idx, r in group.iterrows():
            date = r['date']
            home_team, away_team = r['home_team_name'], r['visitor_team_name']
            temp = result[(result['date'] < date)&(result['visitor_team_name'] == away_team)&(result['home_team_name']==home_team)]
            # away_possessions = 0.5*(r['fga_away'] + 0.4*r['fta_away'] - 1.07*(r['orb_away']/(r['orb_away']+r['drb_away_opp'])*(r['fga_away']-r['fg_away'])) + r['tov_away']+ r['fga_away_opp'] + 0.4*r['fta_away_opp'] - 1.07*(r['orb_away_opp']/(r['drb_away']+r['orb_away_opp'])*(r['fga_away_opp']-r['fg_away_opp'])) + r['tov_away_opp'])
            # home_posessions = 0.5*(r['fga_home'] + 0.4*r['fta_home'] - 1.07*(r['orb_home']/(r['orb_home']+r['drb_home_opp'])*(r['fga_home']-r['fg_home'])) + r['tov_home']+ r['fga_home_opp'] + 0.4*r['fta_home_opp'] - 1.07*(r['orb_home_opp']/(r['drb_home']+r['orb_home_opp'])*(r['fga_home_opp']-r['fg_home_opp'])) + r['tov_home_opp'])
            matchup_home_win_percent = len(temp[temp['result'] == 1])/len(temp)
            matchup_home_lose_percent = len(temp[temp['result'] == -1])/len(temp)
            w = temp[temp['result'] == 1]
    #             group.at[idx,'matchup_percent_home'] = len(w)/len(temp)
            current_match = group[(group['date'] < date)&(group['home_team_name']==home_team)].iloc[-rolling_window:]
            cuurent_win = current_match[current_match['result'] == 1]
            current_lose = current_match[current_match['result'] == -1]
            group.at[idx,'matchup_home_win_percent'] = matchup_home_win_percent
            group.at[idx,'matchup_home_lose_percent'] = matchup_home_lose_percent
            group.at[idx,'current_home_win'] = len(cuurent_win)/rolling_window
            group.at[idx,'current_home_lose'] = len(current_lose)/rolling_window
            # group.at[idx,'away_possessions'] = away_possessions
            # group.at[idx,'home_possessions'] = home_possessions


        group['home_rest_day'] = group.date_game.diff()
        group['home_rest_day'] = group['home_rest_day'].dt.days

        for idx, r in away_group.iterrows():
            date = r['date']
            home_team, away_team = r['home_team_name'], r['visitor_team_name']
            temp = result[(result['date'] < date)&(result['visitor_team_name'] == away_team)&(result['home_team_name']==home_team)]
            matchup_away_win_percent = len(temp[temp['result'] == 1])/len(temp)
            matchup_away_lose_percent = len(temp[temp['result'] == -1])/len(temp)
            w = temp[temp['result'] == 1]
            # group.at[idx,'matchup_percent_home'] = len(w)/len(temp)
            current_match = group[(group['date'] < date)&(group['visitor_team_name']==away_team)].iloc[-rolling_window:]
            cuurent_win = current_match[current_match['result'] == 1]
            current_lose = current_match[current_match['result'] == -1]
            away_group.at[idx,'matchup_away_win_percent'] = matchup_away_win_percent
            away_group.at[idx,'matchup_away_lose_percent'] = matchup_away_lose_percent
            away_group.at[idx,'current_away_win'] = len(cuurent_win)/rolling_window
            away_group.at[idx,'current_away_lose'] = len(current_lose)/rolling_window

        away_group['away_rest_day'] = away_group.date_game.diff()
        away_group['away_rest_day'] = away_group['away_rest_day'].dt.days
        
        away = pd.concat([away, away_group])
        home = pd.concat([home, group])
    else:
      for idx_, (idx_tuple, group) in enumerate(df.sort_values('date').groupby(['year','home_team_name'])):
        yr = idx_tuple[0]
        team  = idx_tuple[-1]
        away_group = df[(df['visitor_team_name'] == team)&(df['year']== yr)]
        group = pd.concat([group , away_group])
        
        for col in group.columns:
          if group[col].dtype != 'O' and group[col].dtype != '<M8[ns]' and col != 'result':
            if prev_game:
              group[col] = group[col].shift(1).fillna(0)
              group[col] = group[col].rolling(rolling_window).sum()/(rolling_window-1)

            else:
              group[col] = group[col].shift(1)
              visitor_pts = group[col].values[1:]
              for idx, i in enumerate(group.index[1:]):
                if idx != 0:
                  # group.at[group.index[idx],col +'_roll'] = visitor_pts[:idx].mean()
                  group.at[group.index[idx],col +'_roll'] = visitor_pts[:idx].mean()
              group.at[group.index[-1], col +'_roll'] = visitor_pts.mean()
              group[col] = group[col+'_roll']
              group = group.drop(columns = col+'_roll')

            # away_group[col] = away_group[col].rolling(rolling_window).mean()
            if train:
              playoff_ = playoff[(playoff['year']== str(yr))]
              date = str(playoff_['date'].values[0])
              group  = group[group['date']<date]
 
        
        group['away_possessions'] = 0.5*(group['fga_away'] + 0.4*group['fta_away'] - 1.07*(group['orb_away']/(group['orb_away']+group['drb_home'])*(group['fga_away']-group['fg_away'])) + group['tov_away']+ group['fga_home'] + 0.4*group['fta_home'] - 1.07*(group['orb_home']/(group['drb_away']+group['orb_home'])*(group['fga_home']-group['fg_home'])) + group['tov_home'])
        group['home_possessions'] = 0.5*(group['fga_home'] + 0.4*group['fta_home'] - 1.07*(group['orb_home']/(group['orb_home']+group['drb_away'])*(group['fga_home']-group['fg_home'])) + group['tov_home']+ group['fga_away'] + 0.4*group['fta_away'] - 1.07*(group['orb_away']/(group['drb_home']+group['orb_away'])*(group['fga_away']-group['fg_away'])) + group['tov_away'])
        
        
        
        for idx, r in group.iterrows():
            date = r['date']
            home_team, away_team = r['home_team_name'], r['visitor_team_name']
            temp = result[(result['date'] < date)&(result['visitor_team_name'] == away_team)&(result['home_team_name']==home_team)]
            # away_possessions = 0.5*(r['fga_away'] + 0.4*r['fta_away'] - 1.07*(r['orb_away']/(r['orb_away']+r['drb_away_opp'])*(r['fga_away']-r['fg_away'])) + r['tov_away']+ r['fga_away_opp'] + 0.4*r['fta_away_opp'] - 1.07*(r['orb_away_opp']/(r['drb_away']+r['orb_away_opp'])*(r['fga_away_opp']-r['fg_away_opp'])) + r['tov_away_opp'])
            # home_posessions = 0.5*(r['fga_home'] + 0.4*r['fta_home'] - 1.07*(r['orb_home']/(r['orb_home']+r['drb_home_opp'])*(r['fga_home']-r['fg_home'])) + r['tov_home']+ r['fga_home_opp'] + 0.4*r['fta_home_opp'] - 1.07*(r['orb_home_opp']/(r['drb_home']+r['orb_home_opp'])*(r['fga_home_opp']-r['fg_home_opp'])) + r['tov_home_opp'])
            matchup_home_win_percent = len(temp[temp['result'] == 1])/len(temp)
            matchup_home_lose_percent = len(temp[temp['result'] == -1])/len(temp)
            w = temp[temp['result'] == 1]
    #             group.at[idx,'matchup_percent_home'] = len(w)/len(temp)
            current_match = group[(group['date'] < date)&(group['home_team_name']==home_team)].iloc[-rolling_window:]
            cuurent_win = current_match[current_match['result'] == 1]
            current_lose = current_match[current_match['result'] == -1]
            group.at[idx,'matchup_home_win_percent'] = matchup_home_win_percent
            group.at[idx,'matchup_home_lose_percent'] = matchup_home_lose_percent
            group.at[idx,'current_home_win'] = len(cuurent_win)/rolling_window
            group.at[idx,'current_home_lose'] = len(current_lose)/rolling_window



        group['home_rest_day'] = group.date_game.diff()
        group['home_rest_day'] = group['home_rest_day'].dt.days
        away = pd.concat([away, away_group])
        home = pd.concat([home, group])


    col_list = []
    for col in result.columns:
        if 'away' in col:
            if len(col.split('_away')[0]) >1:
                col = col.split('_away')[0] + '_home_opp' + col.split('_away')[-1]
            else:
                col = col.split('_away')[0] +'_home_opp'
        if col == 'visitor_pts':
              col = 'home_pts_opp'
        col_list.append(col)
    col_list += ['matchup_home_win_percent','matchup_home_lose_percent','current_home_win','current_home_lose','home_rest_day','home_poessions','away_poessions']

    away_col_list = []
    for col in result.columns:
        if col == 'home_pts':
              col = 'away_pts_opp'

        if 'home'in col and 'team_name' not in col:
            if len(col.split('_home')[0]) >1:
                col = col.split('_home')[0] + '_away_opp' + col.split('_home')[-1]
            else:
                col = col.split('_home')[0] +'_away_opp'


        away_col_list.append(col)
    away_col_list += ['matchup_away_win_percent','matchup_away_lose_percent','current_away_win','current_away_lose','away_rest_day','home_poessions','away_poessions']

    away = away.reset_index(drop = True)
    home = home.reset_index(drop = True)

    if spread:
      for col in home:
        if '_away' in col:
          if 'advanced' not in col:
            new_col = col.split('_away')[0]
            new_col = new_col +'_home_opp'
          else:
            new_col = col.split('_away')[0] + '_home_opp_advanced'  
          if home[col].dtype != 'O' and home[col].dtype != '<M8[ns]' and col != 'result':
            home[col] = home[col]*100/home['away_possessions']
          
          home = home.rename(columns = {col:new_col})
        elif '_home' in col:
          if home[col].dtype != 'O' and home[col].dtype != '<M8[ns]' and col != 'result':
            home[col] = home[col]*100/home['home_possessions']
      
    if spread:
      for col in away:
        if '_home' in col:
          if 'advanced' not in col:
            new_col = col.split('_home')[0]
            new_col = new_col +'_away_opp'
          else:
            new_col = col.split('_home')[0] + '_away_opp_advanced'
          if away[col].dtype != 'O' and away[col].dtype != '<M8[ns]' and col != 'result':

              away[col] = away[col]*100/away['home_possessions']
          
          away = away.rename(columns = {col:new_col})
        elif '_away' in col:
          if away[col].dtype != 'O' and away[col].dtype != '<M8[ns]' and col != 'result':
            
              away[col] = away[col]*100/away['away_possessions']
  

    away = away.rename(columns = {'home_possessions_x':'home_possessions','away_possessions_x':'home_possessions_opp','home_possessions_y':'away_possessions_opp','away_possessions_y':'away_possessions','home_pts':'away_opp_pts'})
    home = home.rename(columns = {'home_possessions_x':'home_possessions','away_possessions_x':'home_possessions_opp','home_possessions_y':'away_possessions_opp','away_possessions_y':'away_possessions','visitor_pts':'home_opp_pts'})

    # for col in away.columns:
    #   if away[col].dtype != 'O' and away[col].dtype != '<M8[ns]' and col != 'result':
    #     away[col] = result['']

    # result = pd.concat([away,home], axis = 1)
    if spread:
      result = away.merge(home, on = ['id','visitor_team_name','home_team_name'])
    else:
      result = home
    result = result.loc[:,~result.columns.duplicated()]
    # result = result[result['Unnamed: 0'] == 0]
    result = result.merge(y, on = 'id')
    result['year'] = yr
    # result['away_possessions'] = 0.5*(result['fga_away'] + 0.4*result['fta_away'] - 1.07*(result['orb_away']/(result['orb_away']+result['drb_away_opp'])*(result['fga_away']-result['fg_away'])) + result['tov_away']+ result['fga_away_opp'] + 0.4*result['fta_away_opp'] - 1.07*(result['orb_away_opp']/(result['drb_away']+result['orb_away_opp'])*(result['fga_away_opp']-result['fg_away_opp'])) + result['tov_away_opp'])
    # result['home_possessions'] = 0.5*(result['fga_home'] + 0.4*result['fta_home'] - 1.07*(result['orb_home']/(result['orb_home']+result['drb_home_opp'])*(result['fga_home']-result['fg_home'])) + result['tov_home']+ result['fga_home_opp'] + 0.4*result['fta_home_opp'] - 1.07*(result['orb_home_opp']/(result['drb_home']+result['orb_home_opp'])*(result['fga_home_opp']-result['fg_home_opp'])) + result['tov_home_opp'])


    # col = ['home_pts','home_pts_opp','visitor_pts','away_pts_opp','fg_pct_away', 'fg3_pct_away', 'ft_pct_away', 'orb_away', 'trb_away','trb_away',
    #    'ast_away', 'tov_away',  'away_pts_opp', 
    #    'fg_pct_home', 'fg3_pct_home', 'ft_pct_home','trb_home',
    #    'orb_home', 'trb_home', 'ast_home', 'tov_home', 'matchup_home_win_percent', 'matchup_home_lose_percent',
    #    'matchup_away_win_percent', 'matchup_away_lose_percent', 'away_rest_day','current_home_win','current_home_lose','current_away_win','current_away_lose',
    #    'home_rest_day']
    # col = ['visitor_pts', 'away_pts_opp', 'fga_away', 'tov_away',
    #    'trb_pct_away_advanced', 'def_rtg_away_advanced', 'tov_away_opp',
    #    'matchup_away_win_percent', 'matchup_away_lose_percent']
    
    # result = result[col]
    
    
#     thresold = 0.8
#     corr = result.corr()
#     upper = corr.where(np.triu(np.ones(corr.shape) ,k = 1).astype(bool))
#     collinear_col = [column for column in upper.columns if any(upper[column]>thresold)]
#     print(collinear_col)
    # collinear_col = ['away_pts_opp',
    #   'trb_away',
    #   'trb_away',
    #   'away_pts_opp',
    #   'trb_home',
    #   'trb_home']
    # result = result.drop(columns = collinear_col)
    
    # scaler = StandardScaler()
    # columns_name = result.columns
    # result = pd.DataFrame(scaler.fit_transform(result))
    
    # result.columns = columns_name
    
    return result

