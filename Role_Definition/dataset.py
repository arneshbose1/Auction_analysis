from re import I

import numpy
import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
import random
import math
import json


class PlayerDataset(Dataset):
    def __init__(self, config, role='batsman', split='train'):
        # config file
        self.config = config

        # phase dictionary
        self.phases_dict = {'first': 1, 'second': 5, 'third': 9, 'fourth': 13, 'fifth': 17}

        # loading the datasets
        self.data_players = pd.read_csv(config['players_data'])
        self.data_matches = pd.read_csv(config['matches_data'])
        self.data_tournaments = pd.read_csv(config['tournaments_data'])
        self.data_teams = pd.read_csv(config['teams_data'])
        self.data_bbb = pd.read_csv(config['bbb_data'])
        self.data_batting_scorecard = pd.read_csv(config['batting_scorecard_data'])
        self.data_bowling_scorecard = pd.read_csv(config['bowling_scorecard_data'])

        if role == 'batsman':
            print("Initializing Batsman dataset:")
            dftemp = self.data_batting_scorecard[self.data_batting_scorecard['status'] == 'YES']
            data = list(dftemp['player_pkey'].unique())

        else:
            print("Intialising Bowler dataset:")
            data = list(self.data_bbb['bowler_pkey'].unique())

        random.Random(4).shuffle(data)
        n = int(math.floor(0.8 * len(data)))

        if split == 'train':
            data = data[:n]
        elif split =='val':
            data = data[n:]
        else:
            data=data[:1]

        self.data_dict = []
        for i in range(len(data)):
            temp = {}
            pkey = data[i]
            temp['player_pkey'] = pkey
            temp['tendency'] = self.get_tendency(pkey, role)
            temp['career'] = self.get_career(pkey, role)
            temp['role'] = self.get_role(pkey, r=role)
            temp['tendency'] = temp['tendency'].astype(np.float32)
            temp['career'] = temp['career'].astype(np.float32)

            self.data_dict.append(temp)


    def get_tendency(self, player_pkey, role='batsman'):
        """
        Function to compute phase wise strike rates/ economy rates of a player for each phase

        params:
            player_pkey - Player Primary Key

        returns:
            array of phase wise strike rates for the given player
        """

        if role == 'batsman':
            strike_rates = []
            for phase in range(5):
                # Extract the ball by ball data of the player for current phase
                player_bbb = self.data_bbb[(self.data_bbb['batsman_pkey'] == player_pkey) & (self.data_bbb['over_num'] >= 1 + 4 * phase) &
                                           (self.data_bbb['over_num'] < 1 + 4 * (phase+1))]
                # Compute strike rate from total runs scored and balls faced
                if len(player_bbb) == 0:
                    sr = 0
                else:
                    sr = np.sum(player_bbb['runs_batsman']) / len(player_bbb) * 100 * len(player_bbb['match_pkey'].unique())
                strike_rates.append(round(sr, 2))

            return np.array(strike_rates)

        else:
            economy_rates = []
            for phase in range(5):
                player_bbb = self.data_bbb[(self.data_bbb['bowler_pkey'] == player_pkey) & (self.data_bbb['over_num'] >= 1 + 4 * phase) &
                                           (self.data_bbb['over_num'] < 1 + 4 * (phase + 1))]

                if len(player_bbb) == 0:
                    er = 0
                else:
                    er = np.sum(player_bbb['runs']) / len(player_bbb) * 6 * len(player_bbb['match_pkey'].unique())
                economy_rates.append(round(er, 2))

            return np.array(economy_rates)

    def get_career(self, player_pkey, role='batsman'):
        """
        Function to get career data historically for a player

        params:
            player_pkey - Player Primary Key

        returns:
            mean, std. dev, dismissal rate, no. of innings
        """

        if role == 'batsman':
            df = self.data_batting_scorecard[(self.data_batting_scorecard['player_pkey'] == player_pkey) & (self.data_batting_scorecard['status'] == 'YES')]
            balls = np.array(df['balls'])
            runs = np.array(df['runs'])
            not_dismissed = np.sum(df['is_dismissed'])

            if len(df) > 0:
                arr = [np.sum(runs), np.mean(balls), np.std(balls), 1 - (not_dismissed / len(balls)), len(balls)]
            else:
                arr = [0, 0, 0, 0, 0]

            return np.array(arr)

        else:
            df = self.data_bowling_scorecard[self.data_bowling_scorecard['player_pkey'] == player_pkey]
            balls = np.array(df['balls'])
            wickets = np.sum(np.array(df['wickets']))
            if wickets == 0:
                w=1
            else:
                w=wickets
            if len(df) > 0:
                arr = [wickets, np.mean(balls), np.std(balls), np.sum(balls)/w, len(df)]
            else:
                arr = [0, 0, 0, 0, 0]

            return np.array(arr)

    def get_role(self, player_pkey, r='batsman'):

        if r=='batsman':
            role = np.zeros(3)
            df = self.data_batting_scorecard[(self.data_batting_scorecard['player_pkey'] == player_pkey) & (
                            self.data_batting_scorecard['position'] <=4)]
            role[0] = len(df)

            df = self.data_batting_scorecard[(self.data_batting_scorecard['player_pkey'] == player_pkey) & (
                    self.data_batting_scorecard['position'] >= 5) & (self.data_batting_scorecard['position'] <=7)]
            role[1] = len(df)

            df = self.data_batting_scorecard[(self.data_batting_scorecard['player_pkey'] == player_pkey) & (
                    self.data_batting_scorecard['position'] >= 8)]
            role[2] = len(df)

            return np.array(np.argmax(role))

        else:
            role = np.zeros(3)

            df = self.data_bbb[(self.data_bbb['bowler_pkey'] == player_pkey) & (
                        self.data_bbb['over_num'] <=6)]
            role[0] = len(df)

            df = self.data_bbb[(self.data_bbb['bowler_pkey'] == player_pkey) & (
                    self.data_bbb['over_num'] > 6) & (self.data_bbb['over_num'] < 16)]
            role[1] = len(df)

            df = self.data_bbb[(self.data_bbb['bowler_pkey'] == player_pkey) & (
                    self.data_bbb['over_num'] >= 16)]
            role[2] = len(df)

            return np.array(np.argmax(role))


    def __len__(self):
        return len(self.data_dict)

    def __getitem__(self, idx):
        annot = self.data_dict[idx]
        temp = {}

        temp['player_pkey'] = annot['player_pkey']
        temp['tendency'] = torch.from_numpy(annot['tendency'])
        temp['career'] = torch.from_numpy(annot['career'])
        temp['role'] = torch.from_numpy(annot['role'])
        return temp


if __name__ == '__main__':
    with open('data_config.json') as fp:
        config = json.load(fp)
    dataset = PlayerDataset(config, 'batsman')

    print(dataset[0])
