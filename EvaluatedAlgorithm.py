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
import MyDump

class EvaluatedAlgorithm:
    folder_path = './DumpFiles'
    def __init__(self, algorithm, name):
        self.algorithm = algorithm
        self.name = name


    def Evaluate(self, evaluationData, doTopN, n=10, verbose=True, load = False):
        metrics = {}
        # Compute accuracy
        if (verbose):
            print("Evaluating accuracy...")
        pr = None
        alg = None
        if load:
            file_name = self.name + '_acc'
            pr, alg, _ = MyDump.Load(file_name,1)
        if pr == None or alg == None:
            self.algorithm.fit(evaluationData.GetTrainSet()) # 75%
            predictions = self.algorithm.test(evaluationData.GetTestSet())
            if load:
                MyDump.Save(file_name, predictions, self.algorithm, None, 1)
        else:
            self.algorithm = alg
            predictions = pr
        metrics["RMSE"] = RecommenderMetrics.RMSE(predictions)
        metrics["MAE"] = RecommenderMetrics.MAE(predictions)
            # if load:
            #     MyDump.Save(file_name, predictions, self.algorithm, None, 1)


        if (doTopN):
        # if False:
            # Evaluate top-10 with Leave One Out testing
            if (verbose):
                print("Evaluating top-N with leave-one-out...")

            pr_top = None
            alg_top = None
            if load:
                file_name = self.name + '_top' + str(n)
                pr_top, alg_top, _ = MyDump.Load(file_name,1)
            if pr_top == None or alg_top == None:
                self.algorithm.fit(evaluationData.GetLOOCVTrainSet())
                leftOutPredictions = self.algorithm.test(evaluationData.GetLOOCVTestSet())
                # Build predictions for all ratings not in the training set
                allPredictions = self.algorithm.test(evaluationData.GetLOOCVAntiTestSet())
                # Compute top 10 recs for each user
                topNPredicted = RecommenderMetrics.GetTopN(allPredictions, n)

                pr_data = {'leftOutPredictions':leftOutPredictions, 'topNPredicted':topNPredicted}
                if load:
                    MyDump.Save(file_name, pr_data , self.algorithm, None, 1)
            else:
                self.algorithm = alg_top
                leftOutPredictions = pr_top['leftOutPredictions']
                topNPredicted = pr_top['topNPredicted']


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

            pr_top = None
            alg_top = None
            if load:
                file_name = self.name + '_fulldata'
                pr_top, alg_top, _ = MyDump.Load(file_name,1)
            if pr_top == None or alg_top == None:
                self.algorithm.fit(evaluationData.GetFullTrainSet())
                allPredictions = self.algorithm.test(evaluationData.GetFullAntiTestSet())
                topNPredicted = RecommenderMetrics.GetTopN(allPredictions, n)
                pr_data = {'allPredictions':allPredictions,
                            'topNPredicted':topNPredicted}
                if load:
                    MyDump.Save(file_name, pr_data , self.algorithm, None, 1)
            else:
                self.algorithm = alg_top
                allPredictions = pr_top['allPredictions']
                topNPredicted = pr_top['topNPredicted']


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

    def Evaluate_v1(self, evaluationData, doTopN, n=10, verbose=True):
        metrics = {}
        # Compute accuracy
        if (verbose):
            print("Evaluating accuracy...")

        self.algorithm.fit(evaluationData.GetTrainSet()) # 75%
        predictions = self.algorithm.test(evaluationData.GetTestSet())

        # self.algorithm.fit(evaluationData.GetTrainSet()) # 75%
        # predictions = self.algorithm.test(evaluationData.GetTestSet())

        # self.algorithm.fit(evaluationData.GetTrainSet()) # 75%
        # predictions = self.algorithm.test(evaluationData.GetTestSet())
        metrics["RMSE"] = RecommenderMetrics.RMSE(predictions)
        metrics["MAE"] = RecommenderMetrics.MAE(predictions)

        if (doTopN):
        # if False:
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
