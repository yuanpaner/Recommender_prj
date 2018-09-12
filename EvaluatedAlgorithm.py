# -*- coding: utf-8 -*-
""" build on top of surpriselib
    algorithm: AlgoBase
    Evaluate(EvaluationData)

    collaborate with:
        RecommenderMetrics
        EvaluationData
    used by:
        Evaluator
"""
from RecommenderMetrics import RecommenderMetrics
from EvaluationData import EvaluationData
import os
from surprise import dump

class EvaluatedAlgorithm:
    folder_path = './DumpFiles'
    def __init__(self, algorithm, name):
        self.algorithm = algorithm
        self.name = name

    def Evaluate(self, evaluationData, doTopN, n=10, verbose=True):
        metrics = {}
        # Compute accuracy
        if (verbose):
            print("\nEvaluating accuracy...")


        if not os.path.exists(self.__class__.folder_path):
            print("path doesn't exist. trying to make")
            os.makedirs(self.__class__.folder_path)

        file_name = self.__class__.folder_path+'/'+self.name+'_acc'
        if os.path.exists(file_name):
            _, self.algorithm = dump.load(file_name)
            print(f'{file_name} file exist, only need to load the trained algorithm')
        else:
            self.algorithm.fit(evaluationData.GetTrainSet()) # 75%
            dump.dump(file_name, algo=self.algorithm)
            print(f'{file_name} SAVE sucessfully')

        predictions = self.algorithm.test(evaluationData.GetTestSet())
        metrics["RMSE"] = RecommenderMetrics.RMSE(predictions)
        metrics["MAE"] = RecommenderMetrics.MAE(predictions)

        if (doTopN):
            # Evaluate top-10 with Leave One Out testing
            if (verbose):
                print("Evaluating top-N with leave-one-out...")
            self.algorithm.fit(evaluationData.GetLOOCVTrainSet())
            leftOutPredictions = self.algorithm.test(evaluationData.GetLOOCVTestSet())
            # Build predictions for all ratings not in the training set
            allPredictions = self.algorithm.test(evaluationData.GetLOOCVAntiTestSet())
            # Compute top 10 recs for each user
            topNPredicted = RecommenderMetrics.GetTopN(allPredictions, n)
            if (verbose):
                print("Computing hit-rate and rank metrics...")
            # See how often we recommended a movie the user actually rated
            metrics["HR"] = RecommenderMetrics.HitRate(topNPredicted, leftOutPredictions)
            # See how often we recommended a movie the user actually liked
            metrics["cHR"] = RecommenderMetrics.CumulativeHitRate(topNPredicted, leftOutPredictions)
            # Compute ARHR
            metrics["ARHR"] = RecommenderMetrics.AverageReciprocalHitRank(topNPredicted, leftOutPredictions)

            #Evaluate properties of recommendations on full training set
            if (verbose):
                print("Computing recommendations with full data set...")
            self.algorithm.fit(evaluationData.GetFullTrainSet())
            allPredictions = self.algorithm.test(evaluationData.GetFullAntiTestSet())
            topNPredicted = RecommenderMetrics.GetTopN(allPredictions, n)
            if (verbose):
                print("Analyzing coverage, diversity, and novelty...")
            # Print user coverage with a minimum predicted rating of 4.0:
            metrics["Coverage"] = RecommenderMetrics.UserCoverage(  topNPredicted,
                                                                   evaluationData.GetFullTrainSet().n_users,
                                                                   ratingThreshold=4.0)
            # Measure diversity of recommendations:
            metrics["Diversity"] = RecommenderMetrics.Diversity(topNPredicted, evaluationData.GetSimilarities())

            # Measure novelty (average popularity rank of recommendations):
            metrics["Novelty"] = RecommenderMetrics.Novelty(topNPredicted,
                                                            evaluationData.GetPopularityRankings())

        if (verbose):
            print("Analysis complete.")

        # SaveAlg()

        return metrics


    def GetName(self):
        return self.name

    def GetAlgorithm(self):
        return self.algorithm
