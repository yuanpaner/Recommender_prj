# -*- coding: utf-8 -*-
"""
use suprise.KNNBasic() algorithm to compute the similarity between Items
using users' rating

And recommend the most similar items(movies) to movie:412
"""

from MovieLens import MovieLens
from surprise import KNNBasic
import heapq
from collections import defaultdict
from operator import itemgetter
import MyDump


testSubject = '85' # user, not movie - Angels and Insects (1995)
# testSubject = '412' # Age of Innocence
k = 10

# ml = MovieLens()
# data = ml.loadMovieLensLatestSmall()

ml = None
data = None
ml, data, _ = MyDump.LoadMovieLensData(True)
if ml == None or data == None:
    ml = MovieLens()
    data = ml.loadMovieLensLatestSmall()



trainSet = data.build_full_trainset()

sim_options = {'name': 'cosine',
               'user_based': False # item_item
               }

print(f'testUser {testSubject}, the ratings are:')
for (movieID, rating) in sorted(ml.getUserRatings(int(testSubject)), key=lambda x: x[1], reverse = True):
    print(f'\t{ml.movieID_to_name[movieID]}\t:{rating}')



# model = KNNBasic(sim_options=sim_options)
# model.fit(trainSet)
# simsMatrix = model.compute_similarities()

simsMatrix = None
_,_,simsMatrix = MyDump.Load('item_similarity',1)
if simsMatrix is None:

    model = KNNBasic(sim_options=sim_options)
    model.fit(trainSet) # calculate the similarity
    simsMatrix = model.compute_similarities()

    MyDump.Save('item_similarity', data = simsMatrix, verbose = 1)




testUserInnerID = trainSet.to_inner_uid(testSubject)
testUserRatings = trainSet.ur[testUserInnerID]

# Get the top K items we rated
kNeighbors = heapq.nlargest(k, testUserRatings, key=lambda t: t[1])

# tuning by threshold, using movies higher than threshold to work as neighbors to get similar movies to recommended
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
