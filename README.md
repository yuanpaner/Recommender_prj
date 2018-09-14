# Recommender_prj
I'm working on organizing the project again.

# Run result  

* python3 Evaluator.py SvdRandom  
pick the algorithm, train, predict, and measure it.  
train and measure take about 522s; measure(calculate RMSE, HR, Coverage and etc.) takes about 118s.
```shell

Loading movie ratings...
Computing movie popularity ranks so we can measure novelty later...
------time consumption: 0.6481490135192871 for LoadMovieLensData()

Estimating biases using als...
Computing the cosine similarity matrix...
Done computing similarity matrix.
------time consumption: 50.82963490486145 for create an evaluator instance

Evaluating  SVD ...
Evaluating accuracy...
Evaluating top-N with leave-one-out...
Computing hit-rate and rank metrics...
Computing recommendations with full data set...
Analyzing coverage, diversity, and novelty...
Computing the cosine similarity matrix...
Done computing similarity matrix.
Analysis complete.

Evaluating  Random ...
Evaluating accuracy...
Evaluating top-N with leave-one-out...
Computing hit-rate and rank metrics...
Computing recommendations with full data set...
Analyzing coverage, diversity, and novelty...
Computing the cosine similarity matrix...
Done computing similarity matrix.
Analysis complete.


Algorithm  RMSE       MAE        HR         cHR        ARHR       Coverage   Diversity  Novelty   
SVD        0.9034     0.6978     0.0298     0.0298     0.0112     0.9553     0.0445     491.5768  
Random     1.4385     1.1478     0.0149     0.0149     0.0043     1.0000     0.0676     545.3663  
------time consumption: 522.2393069267273 for evaluator.Evaluate()
```
# Note
If you want to run the program on your own computer sucessfully, some folders with files, which can be downloaded from websites are needed:

```shell
--ml-latest-small  
    --ratings.csv // data files
--ml-20m
    --ratings.csv
--DumpFiles  // use to save calculated result which includes algorithm, prediction and dataset
    57K rankings
    2.9M  Random_top10
    3.1M Random_acc
    3.6M ratingsDataset
    10M SVD_acc
    10M SVD_top10
    290M Random_fulldata
    358M SVD_fulldata
    829M EvaluationData  // huge because multiple combinations of dataset and the similarty matrix
```
