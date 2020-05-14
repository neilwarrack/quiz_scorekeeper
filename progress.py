#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np

def zero_to_none(values):
    """Replace every 0 with 'nan' and return a copy."""
    return [None if x==0 else x for x in values]

def polyfit(scores_mod, rounds):
    print("in poly")
    print(scores_mod)
    scores_mod_nan=np.array(scores_mod, np.float)
    #print(scores_mod)
    idx = np.isfinite(scores_mod_nan)
    rounds = np.array(rounds*idx)
    scores = np.array(scores_mod_nan*idx)
    print(idx)
    print(rounds)
    print(scores_mod)
    print(scores)
    rounds_plot = np.ma.masked_equal(rounds,0)
    print("rounds_plot:", rounds_plot)
    scores_plot = np.ma.masked_equal(scores,0)
    print("scores_plot:", scores_plot)
    print(rounds_plot.compressed())
    
    scores_mod = scores_mod

    print(scores_plot)
    print(scores_plot.compressed())

    ab = np.polyfit(rounds_plot.compressed(), scores_plot.compressed(), 1)
    return ab, rounds_plot




quiz_1 = [[],[],[],[],[],[]]
#########    brns, bckt, wazz, arns, kirk, frew
quiz_1[0] = (9612, 0   , 8094, 5612, 6634, 5135)
quiz_1[1] = (0   , 5228, 6386, 3631, 3346, 1688)
quiz_1[2] = (8867, 7904, 8043, 0   , 5847, 6620)
quiz_1[3] = (9207, 9775, 2812, 5630, 5266, 0   )
quiz_1[4] = (9254, 4953, 0   , 5769, 5919, 5597)
quiz_1[5] = (8776, 7751, 7361, 8850, 0   , 7276)

scores_brns = [x[0] for x in quiz_1]
scores_bckt = [x[1] for x in quiz_1]

#scores_brns_mod = zero_to_none(scores_brns)
#scores_bckt_mod = zero_to_none(scores_bckt)
scores_brns_mod = scores_brns
scores_bckt_mod = scores_bckt

rounds = [1,2,3,4,5,6]

plt.plot(rounds, scores_brns_mod, 'o',label="Burns".format('o'), color="blue")
plt.plot(rounds, scores_bckt_mod, 'o',label="Beckett".format('o'), color="red")

ab_brns, rounds_brns = polyfit(scores_brns_mod, rounds)

scores_brns_mod=np.array(scores_brns_mod, np.float)
idx = np.isfinite(scores_brns_mod)
rounds_brns=np.array(rounds*idx)
print("brns scores:", scores_brns_mod)
rounds_plot_brns = np.ma.masked_equal(rounds_brns,0)
scores_plot_brns = np.ma.masked_equal(scores_brns,0)
#print(rounds_plot_brns)
ab = np.polyfit(rounds_plot_brns.compressed(), scores_plot_brns.compressed(), 1)


#print(ab, ab_brns)
#print(rounds_plot_brns, rounds_brns)
#plt.plot(np.unique(rounds_brns), np.poly1d(ab_brns)(np.unique(rounds_brns)),color="blue")
plt.plot(np.unique(rounds_plot_brns), np.poly1d(ab)(np.unique(rounds_plot_brns)),color="blue")


plt.legend(numpoints=1)


plt.show()
