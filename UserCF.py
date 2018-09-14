# -*- coding: utf-8 -*-
"""
not predict the rating score, so hard to make heavy use of AlgoBase class
"""

from MovieLens import MovieLens
from surprise import KNNBasic
import heapq
from collections import defaultdict
from operator import itemgetter

testSubject = '85'
k = 10

# Load our data set and compute the user similarity matrix
ml = MovieLens()
data = ml.loadMovieLensLatestSmall()

# print(f'test user {testSubject}, the ratings are:')
# for (movieID, rating) in sorted(ml.getUserRatings(int(testSubject)), key=lambda x: x[1], reverse = True):
#     print(f'\t{ml.movieID_to_name[movieID]}\t:{rating}')



trainSet = data.build_full_trainset()

sim_options = {'name': 'cosine',
               'user_based': True
               }

model = KNNBasic(sim_options=sim_options)
model.fit(trainSet)
simsMatrix = model.compute_similarities()

# Get top N similar users to our test subject
# (Alternate approach would be to select users up to some similarity threshold - try it!)
testUserInnerID = trainSet.to_inner_uid(testSubject)
similarityRow = simsMatrix[testUserInnerID]

similarUsers = []
for innerID, score in enumerate(similarityRow):
    if (innerID != testUserInnerID):
        similarUsers.append( (innerID, score) )


# kNeighbors = heapq.nlargest(k, similarUsers, key=lambda t: t[1])
# Inception (2010) 3.3
# Star Wars: Episode V - The Empire Strikes Back (1980) 2.4
# Bourne Identity, The (1988) 2.0
# Crouching Tiger, Hidden Dragon (Wo hu cang long) (2000) 2.0
# Dark Knight, The (2008) 2.0
# Good, the Bad and the Ugly, The (Buono, il brutto, il cattivo, Il) (1966) 1.9
# Departed, The (2006) 1.9
# Dark Knight Rises, The (2012) 1.9
# Back to the Future (1985) 1.9
# Gravity (2013) 1.8
# Fight Club (1999) 1.8

# get similar users by threshold
kNeighbors = []
for rating in similarUsers:
    if rating[1] > 0.95:
        kNeighbors.append(rating)
# Star Wars: Episode IV - A New Hope (1977) 114.57068319140309
# Matrix, The (1999) 107.72095292088618
# Star Wars: Episode V - The Empire Strikes Back (1980) 88.09116645357186
# Fight Club (1999) 79.26558201621258
# Back to the Future (1985) 78.78807368067915
# Raiders of the Lost Ark (Indiana Jones and the Raiders of the Lost Ark) (1981) 78.77028125945898
# American Beauty (1999) 77.32300806156537
# Toy Story (1995) 76.37713266677879
# Godfather, The (1972) 76.21072562503657
# Star Wars: Episode VI - Return of the Jedi (1983) 74.71908773556109
# Lord of the Rings: The Fellowship of the Ring, The (2001) 74.37234120218191


# Get the stuff they rated, and add up ratings for each item, weighted by user similarity
candidates = defaultdict(float)
for similarUser in kNeighbors:
    innerID = similarUser[0]
    userSimilarityScore = similarUser[1]
    theirRatings = trainSet.ur[innerID]
    for rating in theirRatings:
        candidates[rating[0]] += (rating[1] / 5.0) * userSimilarityScore

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
