#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np

class Player:
    def __init__(self, name):
        self.name = name
        self.raw_scores = []
    def add_score(self, score):
        self.raw_scores.append(score)
    def add_colour(self, colour):
        self.colour = colour
    def filter_scores(self):
        raw_scores_np = np.array(self.raw_scores, np.float)
        self.scores = np.ma.masked_invalid(raw_scores_np).compressed()
    def filter_rounds(self):
        rounds = np.arange(len(self.raw_scores))
        rounds += 1
        idx = np.isfinite(np.array(self.raw_scores, np.float))
        self.played_rounds = np.ma.masked_equal(np.array(rounds*idx),0).compressed()
        


# old scores
X = float('nan')
quiz_1 = [[],[],[],[],[],[]]
#########    brns, bckt, wazz, arns, kirk, frew
quiz_1[0] = (9612, X   , 8094, 5612, 6634, 5135)
quiz_1[1] = (X   , 5228, 6386, 3631, 3346, 1688)
quiz_1[2] = (8867, 7904, 8043, X   , 5847, 6620)
quiz_1[3] = (9207, 9775, 2812, 5630, 5266, X   )
quiz_1[4] = (9254, 4953, X   , 5769, 5919, 5597)
quiz_1[5] = (8776, 7751, 7361, 8850, X   , 7276)

players = {}
players_names = ["burns", "beckett", "waz", "airns", "kirk", "frew"]
p1 = Player("Burns")
for i in range(6):
    name = players_names[i]
    players[name] = Player(name)
    scores_old = [x[i] for x in quiz_1]

    for j in range(6):
        players[name].add_score(scores_old[j])

players["burns"].add_colour("blue")
players["beckett"].add_colour("red")
players["waz"].add_colour("indigo")
players["airns"].add_colour("gold")
players["kirk"].add_colour("tab:cyan")
players["frew"].add_colour("magenta")

#scores_brns_mod = players["burns"].scores
#scores_bckt_mod = players["beckett"].scores


# linear fit
# for key, player in players.items():
#     player.filter_rounds()
#     player.filter_scores()
#     #print(player.name,":", player.scores, player.played_rounds)
    
#     plt.plot(player.played_rounds, player.scores, 'o', label=key.format('o'), color=player.colour)
#     function = np.poly1d(np.polyfit(player.played_rounds, player.scores, 1))
#     plt.plot((1, len(player.raw_scores)), (function(1), function(len(player.raw_scores))),color=player.colour)


for key, player in players.items():
    player.filter_rounds()
    player.filter_scores()
    print(player.name,":", player.scores, player.played_rounds)
    
    plt.plot(player.played_rounds, player.scores, label=key.format('o'), color=player.colour)
    plt.plot(player.played_rounds, player.scores, 'o', color=player.colour)

    
    
plt.legend(numpoints=1)

plt.show()
