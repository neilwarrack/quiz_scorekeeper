#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
import math
    
class Player:
    def __init__(self, name, number):
        self.name = name
        self.raw_scores = []
        self.number = number
    def add_score(self, score):
        self.raw_scores.append(score)
    def weight_scores(self, alphas):
        self.weighted_scores = [x*y for x,y in zip(alphas,self.raw_scores)]
    def acumulate_scores(self):
        raw_scores_np = np.array(self.raw_scores, np.float)
        raw_scores_np[np.isnan(raw_scores_np)] = 0
        cumulative_scores = []
        for i in range(len(self.raw_scores)):
            cumulative_scores.append(0)
            for j in range(i+1):
                cumulative_scores[i] += raw_scores_np[j]
        self.cumu_scores = cumulative_scores
    def acumulate_weighted_scores(self):
        weighted_scores_np = np.array(self.weighted_scores, np.float)
        weighted_scores_np[np.isnan(weighted_scores_np)] = 0
        cumu_weighted_scores = []
        for i in range(len(self.weighted_scores)):
            cumu_weighted_scores.append(0)
            for j in range(i+1):
                cumu_weighted_scores[i] += weighted_scores_np[j]
        self.cumu_w_scores = cumu_weighted_scores
    def add_colour(self, colour):
        self.colour = colour
    def filter_scores(self):
        raw_scores_np = np.array(self.raw_scores, np.float)
        self.scores = np.ma.masked_invalid(raw_scores_np).compressed()
    def filter_w_scores(self):
        weighted_scores_np = np.array(self.weighted_scores, np.float)
        self.wscores = np.ma.masked_invalid(weighted_scores_np).compressed()
    def filter_rounds(self):
        rounds = np.arange(len(self.raw_scores))
        rounds += 1
        idx = np.isfinite(np.array(self.raw_scores, np.float))
        self.played_rounds = np.ma.masked_equal(np.array(rounds*idx),0).compressed()
    def all_rounds(self):
        return np.arange(len(self.raw_scores))
        
def weight(players, rnd):
    num = 0
    denum = 0
    for key, pl in players.items():
        pl_av = 0
        rnd_plyd = 0
        for i in range(len(pl.raw_scores)):
            if not math.isnan(pl.raw_scores[i]):
                pl_av += pl.raw_scores[i]
                rnd_plyd += 1
        if not math.isnan(pl.raw_scores[rnd-1]):
            pl_rndScr = pl.raw_scores[rnd-1]
            pl_av    *= 1/rnd_plyd
            pl_num    = pl_av*pl_rndScr
            num   += pl_num
            denum += pl_rndScr*pl_rndScr
    alpha = num/denum
    return alpha

# old scores
X = float('nan')
quiz_1 = [[],[],[],[],[],[]]
#########    brns, bckt, wazz, arns, kirk, frew
#examp1
quiz_1[0] = (9612, X   , 5094, 5612, 5634, 5135)
quiz_1[1] = (X   , 5228, 5386, 5631, 5346, 5688)
quiz_1[2] = (9867, 5904, 5043, X   , 5847, 5620)
quiz_1[3] = (9207, 5775, 5812, 5630, 5266, X   )
quiz_1[4] = (9254, 5953, X   , 5769, 5919, 5597)
quiz_1[5] = (8776, 5751, 5361, 5850, X   , 5276)
#exampl2 
# quiz_1[0] = (9612, X   , 5094, 7012, 5634, 5135)
# quiz_1[1] = (X   , 1228, 5386, 7631, 5346, 9688)
# quiz_1[2] = (9867, 2904, 5043, X   , 5847, 8620)
# quiz_1[3] = (9207, 1775, 5812, 7630, 5266, X   )
# quiz_1[4] = (9254, 1953, X   , 7769, 5919, 2597)
# quiz_1[5] = (8776, 1751, 5361, 7850, X   , 1276)
#examp3
# quiz_1[0] = (9612, X   , 5094, 5612, 5634, 5135)
# quiz_1[1] = (X   , 1528, 1538, 1563, 1534, 1568)
# quiz_1[2] = (9867, 5904, 5043, X   , 5847, 5620)
# quiz_1[3] = (9207, 5775, 5812, 5630, 5266, X   )
# quiz_1[4] = (9254, 5953, X   , 5769, 5919, 5597)
# quiz_1[5] = (8776, 5751, 5361, 5850, X   , 5276)
#examp4
# quiz_1[0] = (5612, X   , 5094, 5612, 5634, 5135)
# quiz_1[1] = (X   , 1528, 2038, 1763, 2134, 1968)
# quiz_1[2] = (5867, 5904, 5043, X   , 5847, 5620)
# quiz_1[3] = (5207, 5775, 5812, 5630, 5266, X   )
# quiz_1[4] = (5254, 5953, X   , 5769, 5919, 5597)
# quiz_1[5] = (5776, 5751, 5361, 5850, X   , 5276)
#examp5
# quiz_1[0] = (5612, X   , 5094, 5612, 5634, 5135)
# quiz_1[1] = (X   , 1128, 1038, 1263, 1134, 1268)
# quiz_1[2] = (5867, 5904, 5043, X   , 5847, 5620)
# quiz_1[3] = (5207, 5775, 5812, 5630, 5266, X   )
# quiz_1[4] = (9254, 9953, X   , 9769, 8919, 9597)
# quiz_1[5] = (5776, 5751, 5361, 5850, X   , 5276);
#TRUE EXAMPLE
# quiz_1[0] = (9612, X   , 8094, 5612, 6634, 5135)
# quiz_1[1] = (X   , 5228, 6386, 3631, 3346, 1688)
# quiz_1[2] = (8867, 7904, 8043, X   , 5847, 6620)
# quiz_1[3] = (9207, 9775, 2812, 5630, 5266, X   )
# quiz_1[4] = (9254, 4953, X   , 5769, 5919, 5597)
# quiz_1[5] = (8776, 7751, 7361, 8850, X   , 7276)


players = {}
players_names = ["burns", "beckett", "waz", "airns", "kirk", "frew"]

for i in range(6):
    name = players_names[i]
    players[name] = Player(name, i)
    scores_old = [x[i] for x in quiz_1]

    for j in range(6):
        players[name].add_score(scores_old[j])

players["burns"].add_colour("blue")
players["beckett"].add_colour("red")
players["waz"].add_colour("indigo")
players["airns"].add_colour("gold")
players["kirk"].add_colour("tab:cyan")
players["frew"].add_colour("magenta")



for key, player in players.items():
    player.filter_rounds()
    player.filter_scores()
    player.acumulate_scores()



alphas = []
for r in range(6):
    a = weight(players, r+1)
    alphas.append(a)

for key, player in players.items():
    player.weight_scores(alphas)
    player.acumulate_weighted_scores()
    player.filter_w_scores()


plt.figure(figsize=(20,10))
for key, player in players.items():
    
    # plot true scores
    real = plt.subplot(321)
    real.set_title('TRUE')
    #real.set_xlabel('round')
    plt.plot(player.played_rounds, player.scores, label=key.format('o'), color=player.colour)
    plt.plot(player.played_rounds, player.scores, 'o', color=player.colour)

    # plot cumulative true scores
    plt.subplot(323)
    plt.plot(player.all_rounds(), player.cumu_scores, label=key.format('o'), color=player.colour)
    plt.plot(player.all_rounds(), player.cumu_scores, 'o', color=player.colour)
    plt.legend(loc=0)

    
    # plot weighted scores
    weighted = plt.subplot(322)
    weighted.set_title('ADJUSTED')
    plt.plot(player.played_rounds, player.wscores, label=key.format('o'), color=player.colour)
    plt.plot(player.played_rounds, player.wscores, 'o', color=player.colour)

    # plot cummulative weighted scores
    plt.subplot(324)
    plt.plot(player.all_rounds(), player.cumu_w_scores, label=key.format('o'), color=player.colour)
    plt.plot(player.all_rounds(), player.cumu_w_scores, 'o', color=player.colour)


    plt.subplot(325)
    f_rnd = len(player.cumu_scores) - 1
    
    plt.bar(player.name, player.cumu_scores, color=player.colour)
    plt.annotate(round(player.cumu_scores[f_rnd],0), (player.name, player.cumu_scores[f_rnd]))

    plt.subplot(326)
    f_rnd = len(player.cumu_w_scores) - 1
    plt.bar(player.name, player.cumu_w_scores, color=player.colour)
    plt.annotate(round(player.cumu_w_scores[f_rnd],0), (player.name, player.cumu_w_scores[f_rnd]))

plt.savefig('EXAMPL1.pdf')
