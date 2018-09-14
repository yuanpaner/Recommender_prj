# -*- coding: utf-8 -*-
""" test many algorithms using the same evaluation data
    by wrapping those algorithm with an EvaluatedAlgorithm instance
    compare algorithms

    steps:
    1. create an EvaluationData instance given a data set
    2. create an EvaluatedAlgorithm instance for the algorithm to test
    3. call Evaluate function in EvaluatedAlgorithm, passing in the same  EvaluationData to measure the algorithm's performnace

    AddAlgorithm(algorithm)
    Evaluate()
        dataset: EvaluatedDataSet
        algorithms: EvaluatedAlgorithm[]
"""
from EvaluationData import EvaluationData
from EvaluatedAlgorithm import EvaluatedAlgorithm
from ContentKNNAlgorithm import ContentKNNAlgorithm

from MovieLens import MovieLens
from surprise import SVD
from surprise import NormalPredictor
from surprise import KNNBasic

import random
import numpy as np
import time
import sys

import MyDump

loader = True

class Evaluator:

    algorithms = []

    def __init__(self, dataset, rankings, load = False):
        ed = None
        if load:
            _, _, ed = MyDump.Load('EvaluationData', 1)
        if ed == None :
            ed = EvaluationData(dataset, rankings)
            if load:
                MyDump.Save(file_name = 'EvaluationData', data = ed, verbose = 1)
        self.dataset = ed

    def AddAlgorithm(self, algorithm, name):
        alg = EvaluatedAlgorithm(algorithm, name)
        self.algorithms.append(alg)

    def LegendPrint():
        print("\nLegend:\n")
        print("RMSE:      Root Mean Squared Error. Lower values mean better accuracy.")
        print("MAE:       Mean Absolute Error. Lower values mean better accuracy.")
        if (doTopN):
            print("HR:        Hit Rate; how often we are able to recommend a left-out rating. Higher is better.")
            print("cHR:       Cumulative Hit Rate; hit rate, confined to ratings above a certain threshold. Higher is better.")
            print("ARHR:      Average Reciprocal Hit Rank - Hit rate that takes the ranking into account. Higher is better." )
            print("Coverage:  Ratio of users for whom recommendations above a certain threshold exist. Higher is better.")
            print("Diversity: 1-S, where S is the average similarity score between every possible pair of recommendations")
            print("           for a given user. Higher means more diverse.")
            print("Novelty:   Average popularity rank of recommended items. Higher means more novel.")

    def Evaluate(self, doTopN, load = False):
        """ use "RMSE", "MAE" to measure prediction accuracy by spliting the data into train:test = 75:25
            "HR", "cHR", "ARHR" to measure the topN recommendation
        """
        results = {}
        for algorithm in self.algorithms:
            print("\nEvaluating ", algorithm.GetName(), "...")
            results[algorithm.GetName()] = algorithm.Evaluate(self.dataset, doTopN, verbose = True, load = load)

        # Print results
        print("\n")

        if (doTopN):
            print("{:<10} {:<10} {:<10} {:<10} {:<10} {:<10} {:<10} {:<10} {:<10}".format(
                    "Algorithm", "RMSE", "MAE", "HR", "cHR", "ARHR", "Coverage", "Diversity", "Novelty"))
            for (name, metrics) in results.items():
                print("{:<10} {:<10.4f} {:<10.4f} {:<10.4f} {:<10.4f} {:<10.4f} {:<10.4f} {:<10.4f} {:<10.4f}".format(
                        name, metrics["RMSE"], metrics["MAE"], metrics["HR"], metrics["cHR"], metrics["ARHR"],
                                      metrics["Coverage"], metrics["Diversity"], metrics["Novelty"]))
        else:
            print("{:<10} {:<10} {:<10}".format("Algorithm", "RMSE", "MAE"))
            for (name, metrics) in results.items():
                print("{:<10} {:<10.4f} {:<10.4f}".format(name, metrics["RMSE"], metrics["MAE"]))

        # LegendPrint()

    def SampleTopNRecs(self, ml, load = False, testSubject=85, k=10):
        """ after predicting the GetAntiTestSetForUser data set,
            return the sorted top-k items
        """
        for algo in self.algorithms:
            print("\nUsing recommender ", algo.GetName())

            trainSet = self.dataset.GetFullTrainSet()
            testSet = self.dataset.GetAntiTestSetForUser(testSubject)
            if load:
                predictions, algo_sample, _ = MyDump.Load(algo.GetName() + '_sampleTopNRecs', 1)
                if algo_sample == None or predictions == None:
                    print("\nBuilding recommendation model...")
                    algo.GetAlgorithm().fit(trainSet)
                    print("Computing recommendations...")
                    predictions = algo.GetAlgorithm().test(testSet)
                    MyDump.Save(algo.GetName() + '_sampleTopNRecs', predictions, algo, None, 1)
            else :
                print("\nBuilding recommendation model...")
                # trainSet = self.dataset.GetFullTrainSet()
                algo.GetAlgorithm().fit(trainSet)

                print("Computing recommendations...")
                # testSet = self.dataset.GetAntiTestSetForUser(testSubject)

                predictions = algo.GetAlgorithm().test(testSet)


            recommendations = []

            print ("\nWe recommend:")
            for userID, movieID, actualRating, estimatedRating, _ in predictions:
                intMovieID = int(movieID)
                recommendations.append((intMovieID, estimatedRating))

            recommendations.sort(key=lambda x: x[1], reverse=True)

            for ratings in recommendations[:10]:
                print(ml.getMovieName(ratings[0]), ratings[1])


# def LoadMovieLensData():
#     ml = MovieLens()
#     print("\nLoading movie ratings...")
#     data = ml.loadMovieLensLatestSmall()
#     print("Computing movie popularity ranks so we can measure novelty later...")
#     rankings = ml.getPopularityRanks() # for novelty
#     return (ml, data, rankings)


def CompareSVDRandom():
    np.random.seed(0)
    random.seed(0)
    # loader = True
    start_t = time.time()

    # Load up common data set for the recommender algorithms
    # (evaluationData, rankings) = LoadMovieLensData()
    # MyDump.Save('ratingsDataset', data = evaluationData, verbose = 1)
    # MyDump.Save('rankings', data = rankings, verbose = 1)

    # _,_,evaluationData = MyDump.Load('ratingsDataset', 1)
    # _,_,rankings = MyDump.Load('rankings',1)
    # if evaluationData == None or rankings == None:
    #     (_, evaluationData, rankings) = MyDump.LoadMovieLensData() # meat
    #     MyDump.Save('ratingsDataset', data = evaluationData, verbose = 1)
    #     MyDump.Save('rankings', data = rankings, verbose = 1)
    _, evaluationData, rankings = MyDump.LoadMovieLensData(loader)
    print(f'------time consumption: {time.time() - start_t} for LoadMovieLensData()')
    start_t = time.time()


    # Construct an Evaluator to, you know, evaluate them
    evaluator = Evaluator(dataset = evaluationData, rankings = rankings, load = loader)

    print(f'------time consumption: {time.time() - start_t} for create an evaluator instance')
    start_t = time.time()

    # Throw in an SVD recommender
    SVDAlgorithm = SVD(random_state=10)
    evaluator.AddAlgorithm(SVDAlgorithm, "SVD")

    # Just make random recommendations
    Random = NormalPredictor()
    evaluator.AddAlgorithm(Random, "Random")


    start_t = time.time()
    evaluator.Evaluate(True, loader) # doTopN, loader
    print(f'------time consumption: {time.time() - start_t} for evaluator.Evaluate()')
    start_t = time.time()

def ContentRecs():
    """ for this content-based(item) recommender
        calculate items' cosine similarity matrix in alg.fit()

        I don't test `HitRate` for `topN` recommendation, because here it's impossible to do that. What I recommend by this algorithm are all the movies the user haven't rated.
    """
    np.random.seed(0)
    random.seed(0)
    # loader = True

    # Load up common data set for the recommender algorithms
    # print(f'call ContentRecs()\nloader = {loader}')
    (ml, evaluationData, rankings) = MyDump.LoadMovieLensData(loader)

    # Construct an Evaluator to, you know, evaluate them
    evaluator = Evaluator(evaluationData, rankings, load = True)

    contentKNN = ContentKNNAlgorithm()
    evaluator.AddAlgorithm(contentKNN, "ContentKNN")

    # Just make random recommendations
    Random = NormalPredictor()
    evaluator.AddAlgorithm(Random, "Random")

    evaluator.Evaluate(False, True) # not topN, able load

    # recommend 10(default) items
    evaluator.SampleTopNRecs(ml)

def BehaviorBasedCF():
    np.random.seed(0)
    random.seed(0)

    # Load up common data set for the recommender algorithms
    (ml, evaluationData, rankings) = MyDump.LoadMovieLensData(loader)

    # Construct an Evaluator to, you know, evaluate them
    evaluator = Evaluator(evaluationData, rankings, load = True)

    # User-based KNN
    UserKNN = KNNBasic(sim_options = {'name': 'cosine', 'user_based': True})
    evaluator.AddAlgorithm(UserKNN, "User KNN")

    # Item-based KNN
    ItemKNN = KNNBasic(sim_options = {'name': 'cosine', 'user_based': False})
    evaluator.AddAlgorithm(ItemKNN, "Item KNN")

    # Just make random recommendations
    Random = NormalPredictor()
    evaluator.AddAlgorithm(Random, "Random")

    evaluator.Evaluate(False)  # load is also False, cause simsMatrix needs to be loaded; I haven't figured it out.

    evaluator.SampleTopNRecs(ml)

def test():
    print('test the function dictionary')

functionDict = {
    "SvdRandom": CompareSVDRandom,
    "ContentRecs": ContentRecs,
    "BehaviorBasedCF": BehaviorBasedCF,
    "test":test
}

def main(functionName):
    if not functionName in functionDict.keys():
        raise ValueError('wrong function name')
    functionDict[functionName]()

if __name__ == '__main__':
    if len(sys.argv) > 1:
        funcName = sys.argv[1]
    else:
        # funcName = "test"
        funcName = input("Please input function name: ")

    main(funcName)
