import pandas as pd
import numpy as np
import json

with open('data_config.json', 'r') as fp:
    config = json.load(fp)

data_players = pd.read_csv(config['players_data'])
data_matches = pd.read_csv(config['matches_data'])
data_tournaments = pd.read_csv(config['tournaments_data'])
data_teams = pd.read_csv(config['teams_data'])
data_bbb = pd.read_csv(config['bbb_data'])
data_batting_scorecard = pd.read_csv(config['batting_scorecard_data'])
data_bowling_scorecard = pd.read_csv(config['bowling_scorecard_data'])

def get_batting_stats(player_pkey):
    df = data_batting_scorecard[(data_batting_scorecard['player_pkey'] == player_pkey) & (data_batting_scorecard['status'] == 'YES')]
    if len(df)==0:
        return [0, 0, 0, 0]
    else:
        runs = sum(df['runs'])
        matches = len(df['match_pkey'].unique())
        balls = sum(df['balls'])
        outs = len(df) - sum(df['is_dismissed'])
        if outs==0:
            outs=1
        if balls==0:
            balls=1

        return [runs/matches, matches, runs/balls * 100, runs/outs]

def get_bowling_stats(player_pkey):
    df = data_bowling_scorecard[data_bowling_scorecard['player_pkey'] == player_pkey]
    if len(df)==0:
        return [0, 0, 0, 0, 0]
    else:
        runs = sum(df['runs'])
        matches = len(df['match_pkey'].unique())
        balls = sum(df['balls'])
        wickets = sum(df['wickets'])

        if wickets==0:
            return [wickets/matches, matches, runs/balls * 6, 0, 0]
        else:
            return [wickets / matches, matches, runs / balls * 6, runs / wickets, balls / wickets]

'''
def get_mwc(player_pkey, role='batsman'):

    if role=='bowler':
        temp = data_bbb[data_bbb['bowler_pkey'] == player_pkey]
        if len(temp) == 0:
            return 1
        matches = list(temp['match_pkey'].unique())
        win = [0, 0, 0, 0]
        lose = [0, 0, 0, 0]
        for mat in matches:
            temp = data_bbb[data_bbb['match_pkey'] == mat]
            a = temp[temp['bowler_pkey']==player_pkey]
            inn1 = a['innings'][0]
            inn2 = 3-inn1
            for_runs = sum(temp[temp['innings']==inn2]['runs'])
            ag_runs = sum(temp[temp['innings']==inn1]['runs'])
            if for_runs > ag_runs:
                win[0] += sum(a['runs'])
            
'''
def get_CBR(player_pkey, phase='powerplay'):

    if phase == 'powerplay':
        df = data_bbb[(data_bbb['bowler_pkey'] == player_pkey) & (data_bbb['over_num'] <= 6)]
    elif phase == 'death':
        df = data_bbb[(data_bbb['bowler_pkey'] == player_pkey) & (data_bbb['over_num'] > 15)]

    runs = sum(df['runs'])
    balls = len(df)
    wickets = sum(df['is_dismissal'])

    if balls == 0:
        return 0
    else:
        cbr = 3*runs/(wickets+balls/6+wickets*runs/balls)
        return cbr

def get_BP(player_pkey, phase='powerplay'):
    if phase == 'powerplay':
        df = data_bbb[(data_bbb['batsman_pkey'] == player_pkey) & (data_bbb['over_num'] <= 6)]
    elif phase == 'death':
        df = data_bbb[(data_bbb['batsman_pkey'] == player_pkey) & (data_bbb['over_num'] > 15)]

    if len(df) == 0:
        return 0
    else:
        runs = sum(df['runs'])
        balls = len(df)
        outs = sum(df['is_dismissal'])
        if outs==0:
            outs=1
        fours = len(df[df['runs']==4])
        sixes = len(df[df['runs']==6])

        if phase == 'powerplay':
            return 0.458 * runs + 0.398 * runs/outs + 0.325 * runs/balls * 100 + 0.406 * fours + 0.417 * sixes
        elif phase == 'death':
            return 0.458 * runs + 0.325 * runs / balls * 100 + 0.406 * fours + 0.417 * sixes