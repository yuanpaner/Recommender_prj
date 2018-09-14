# driver file
from EvaluationData import EvaluationData
from EvaluatedAlgorithm import EvaluatedAlgorithm
from ContentKNNAlgorithm import ContentKNNAlgorithm

from MovieLens import MovieLens
from surprise import SVD, SVDpp, NormalPredictor, KNNBasic
from RBMAlgorithm import RBMAlgorithm
from AutoRecAlgorithm import AutoRecAlgorithm

import random
import numpy as np
import time
import sys

from Evaluator import Evaluator
import MyDump

loader = True

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

def MF():
    """ the idea behind is math, latent features
        implementation is simple out of library source
    """
    np.random.seed(0)
    random.seed(0)
    ml, evaluationData, rankings = MyDump.LoadMovieLensData(loader)
    evaluator = Evaluator(evaluationData, rankings, loader)

    mySVD = SVD(random_state=10)
    evaluator.AddAlgorithm(mySVD, "SVD") # the same with before
    mySVDpp = SVDpp(random_state=10)
    evaluator.AddAlgorithm(mySVDpp, "SVDpp")
    Random = NormalPredictor()
    evaluator.AddAlgorithm(Random, "Random")

    evaluator.Evaluate(doTopN = False, load = loader)
    evaluator.SampleTopNRecs(ml, loader)

def RBMtest():
    np.random.seed(0)
    random.seed(0)

    ml, evaluationData, rankings = MyDump.LoadMovieLensData(loader)

    # Construct an Evaluator to, you know, evaluate them
    evaluator = Evaluator(evaluationData, rankings, loader)

    # RBM
    # able to tune by trying more parameter combination
    myRBM = RBMAlgorithm(epochs=20)
    evaluator.AddAlgorithm(myRBM, "RBM")

    Random = NormalPredictor()
    evaluator.AddAlgorithm(Random, "Random")

    evaluator.Evaluate(doTopN = False, load = loader)

    evaluator.SampleTopNRecs(ml, loader)

def AutoRec():
    np.random.seed(0)
    random.seed(0)

    ml, evaluationData, rankings = MyDump.LoadMovieLensData(loader)
    evaluator = Evaluator(evaluationData, rankings, loader)

    myAutoRec= RBMAlgorithm()
    evaluator.AddAlgorithm(myAutoRec, "AutoRec")

    Random = NormalPredictor()
    evaluator.AddAlgorithm(Random, "Random")

    evaluator.Evaluate(doTopN = False, load = loader)

    evaluator.SampleTopNRecs(ml, loader)

def test():
    print('test the function dictionary')

functionDict = {
    "SvdRandom": CompareSVDRandom,
    "ContentRecs": ContentRecs,
    "BehaviorBasedCF": BehaviorBasedCF,
    "MF": MF,
    "RBM": RBMtest,
    "AutoRec": AutoRec,
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
