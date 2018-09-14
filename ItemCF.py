# -*- coding: utf-8 -*-
"""
use suprise.KNNBasic() algorithm to compute the similarity between Items
using users' rating
"""

from MovieLens import MovieLens
from surprise import KNNBasic
import heapq
from collections import defaultdict
from operator import itemgetter

# testSubject = '85' # Angels and Insects (1995)
testSubject = '412' # Age of Innocence
k = 10

ml = MovieLens()
data = ml.loadMovieLensLatestSmall()

trainSet = data.build_full_trainset()

sim_options = {'name': 'cosine',
               'user_based': False # item_item
               }

model = KNNBasic(sim_options=sim_options)
model.fit(trainSet)
simsMatrix = model.compute_similarities()

testUserInnerID = trainSet.to_inner_uid(testSubject)

testUserRatings = trainSet.ur[testUserInnerID]

# Get the top K items we rated
kNeighbors = heapq.nlargest(k, testUserRatings, key=lambda t: t[1])

# Music From Another Room (1998) 10.0
# My Life Without Me (2003) 10.0
# Simply Irresistible (1999) 9.96058526610992
# Set It Off (1996) 9.96
# April Fool's Day (1986) 9.928008768935916
# Heart and Souls (1993) 9.927705686154587
# Heartburn (1986) 9.924273370877941
# Grace of My Heart (1996) 9.914766859820652
# Boys Life (1995) 9.914209019256287
# Creepshow 2 (1987) 9.913648635607869
# Stepfather, The (1987) 9.907185199445436


# tuning by threshold
kNeighbors = []
for rating in testUserRatings:
    if rating[1] > 4.0:
        kNeighbors.append(rating)

# Simply Irresistible (1999) 20.727223504541914
# Cat's Eye (1985) 20.644828138914125
# Under the Tuscan Sun (2003) 20.63129972419603
# Rainmaker, The (1997) 20.622033152527152
# White Squall (1996) 20.61512459204586
# Slap Shot (1977) 20.581093873631396
# Holes (2003) 20.555678479297388
# Iron Man 3 (2013) 20.51070770110813
# Three Men and a Little Lady (1990) 20.506840058203093
# Empire Records (1995) 20.484993711920097
# Snow Falling on Cedars (1999) 20.45286651646317

# Get similar items to stuff we liked (weighted by rating)
candidates = defaultdict(float)
for itemID, rating in kNeighbors:
    similarityRow = simsMatrix[itemID]
    for innerID, score in enumerate(similarityRow):
        candidates[innerID] += score * (rating / 5.0)

# Build a dictionary of stuff the user has already seen
watched = {}
for itemID, rating in trainSet.ur[testUserInnerID]:
    watched[itemID] = 1

# Get top-rated items from similar users:
pos = 0
for itemID, ratingSum in sorted(candidates.items(), key=itemgetter(1), reverse=True):
    if not itemID in watched:
        movieID = trainSet.to_raw_iid(itemID)
        print(ml.getMovieName(int(movieID)), ratingSum)
        pos += 1
        if (pos > 10):
            break
