#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
import math
    
class Player:
    def __init__(self, name):
        self.name = name
        self.raw_scores = []
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

SMALL_SIZE = 8
MEDIUM_SIZE = 10
BIGGER_SIZE = 12
plt.rc('font', size=BIGGER_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=BIGGER_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=BIGGER_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=BIGGER_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=MEDIUM_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
X = float('nan')

####################################################################
#players_names = ["burns", "beckett", "waz", "airns", "kirk", "frew", "ash"]
players_names = ["graeme", "chris", "waz", "calum", "bnb"]
n_players = 5
n_rounds = 6
quiz_1 = np.empty((n_rounds, 0)).tolist()

#########    brns, bckt, wazz, arns, kirk, frew, ash
filename="quizz.pdf"
#            greame, chrs, wax, cal, bnb
quiz_1[0] = (1,        2,   1,   1,   6)
quiz_1[1] = (2,        2,   1,   3,   3)
quiz_1[2] = (6,      4,     2,  7,   8)
quiz_1[3] = (5,       6,    6,   4,  5)
quiz_1[4] = (13,     7,   13,   14,  12)
quiz_1[5] = (7,      5,    5.5,  4,  6 )

# quiz_1[0] = (1,      2,   1,   1,   6)
#quiz_1[1] = (8853, X   , 4591, 5767, 5442, 1630, 8961)
#quiz_1[2] = (X   , 6018, 5689, 4480, 3141, 4627, 1830)
#quiz_1[3] = (8322, 8636, 4381, X   , 3813, 7478, 2733)
#quiz_1[4] = (13342,15797,X   , 4704,10716, 9273, 8533)
#quiz_1[5] = (7218, 4350, 5558, 2653, 6053, 4853, X   )
#quiz_1[6] = (9967, 11001,14731,9951, X   , 11631, 7547)

titles = {"graeme":"Big Graeme", "chris" : "Chris the Dish", "waz" : "Wazypoos",  "calum":"Calum and Joni", "bnb":"Bruce and Becca"}

# titles = {
#     "burns"   : "The Champ Champ",
#     "beckett" : "Quizzee Pascale",
#     "waz"     : "Neil The Deal",
#     "airns"   : "Marvellous Mattz",
#     "kirk"    : "Zoro",
#     "frew"    : "Cunning Chris Roberts",
#     "ash"     : "Quiz-BomB"
# }
#######################################################################
players = {}
for i, nm in enumerate(players_names):
    name = nm
    players[name] = Player(titles[name])
    scrs = [x[i] for x in quiz_1]

    for j in range(n_rounds):
        players[name].add_score(scrs[j])

# players["burns"].add_colour("blue")
# players["beckett"].add_colour("red")
# players["waz"].add_colour("green") # indigo
# players["airns"].add_colour("brown") # gold
# players["kirk"].add_colour("tab:cyan")
# players["frew"].add_colour("magenta")
# players["ash"].add_colour("indigo")

players["graeme"].add_colour("blue")
players["chris"].add_colour("red")
players["waz"].add_colour("green") # indigo
players["calum"].add_colour("brown") # gold
players["bnb"].add_colour("tab:cyan")
#players["frew"].add_colour("magenta")
#players["ash"].add_colour("indigo")



for key, player in players.items():
    player.filter_rounds()
    player.filter_scores()
    player.acumulate_scores()



alphas = []
for r in range(n_rounds):
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
    plt.plot(player.played_rounds, player.scores, label=player.name.format('o'), color=player.colour)
    plt.plot(player.played_rounds, player.scores, 'o', color=player.colour)
    plt.grid(True)
    
    # plot cumulative true scores
    plt.subplot(323)
    plt.plot(player.all_rounds(), player.cumu_scores, label=player.name.format('o'), color=player.colour)
    plt.plot(player.all_rounds(), player.cumu_scores, 'o', color=player.colour)
    #plt.legend(loc=0, bbox_to_anchor=[0.275, 1.1 ])
    #plt.legend(loc=0, bbox_to_anchor=[0.275, 0.0 ])
    plt.legend(loc=0)
    plt.grid(True)
    
    # plot weighted scores
    weighted = plt.subplot(322)
    weighted.set_title('ADJUSTED')
    plt.plot(player.played_rounds, player.wscores, label=player.name.format('o'), color=player.colour)
    plt.plot(player.played_rounds, player.wscores, 'o', color=player.colour)
    plt.grid(True)
    
    # plot cummulative weighted scores
    plt.subplot(324)
    plt.plot(player.all_rounds(), player.cumu_w_scores, label=player.name.format('o'), color=player.colour)
    plt.plot(player.all_rounds(), player.cumu_w_scores, 'o', color=player.colour)
    plt.grid(True)

    plt.subplot(325)
    f_rnd = len(player.cumu_scores) - 1
    plt.xticks(rotation=15)
    plt.grid(True)
    plt.bar(player.name, player.cumu_scores, color=player.colour)
    plt.annotate(round(player.cumu_scores[f_rnd],3), (player.name, player.cumu_scores[f_rnd]))

    plt.subplot(326)
    f_rnd = len(player.cumu_w_scores) - 1
    plt.xticks(rotation=15)
    plt.grid(True)
    plt.bar(player.name, player.cumu_w_scores, color=player.colour)
    plt.annotate(round(player.cumu_w_scores[f_rnd],3), (player.name, player.cumu_w_scores[f_rnd]))

plt.savefig(filename)
#plt.show()
