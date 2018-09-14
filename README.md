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

* python3 Evaluator.py ContentRecs  
I don't test `HitRate` for `topN` recommendation, because here it's impossible to do that.   
What I recommend by this algorithm comes from all the movies the user haven't rated.  

```shell
# Evaluating  ContentKNN and Random
Algorithm  RMSE       MAE       
ContentKNN 0.9375     0.7263    
Random     1.4385     1.1478    

# Using recommender  ContentKNN
# for userId = 85 by default
We recommend(movie predicted_scores):
Presidio, The (1988) 3.841314676872932
Femme Nikita, La (Nikita) (1990) 3.839613347087336
Wyatt Earp (1994) 3.8125061475551796
Shooter, The (1997) 3.8125061475551796
Bad Girls (1994) 3.8125061475551796
The Hateful Eight (2015) 3.812506147555179
True Grit (2010) 3.812506147555179
Open Range (2003) 3.812506147555179
Big Easy, The (1987) 3.7835412549266985
Point Break (1991) 3.764158410102279

# Using recommender  Random
We recommend:
Dangerous Minds (1995) 5
Escape from New York (1981) 5
Cinema Paradiso (Nuovo cinema Paradiso) (1989) 5
Beavis and Butt-Head Do America (1996) 5
Antz (1998) 5
Mighty Aphrodite (1995) 5
First Knight (1995) 5
Circle of Friends (1995) 5
Wolf (1994) 5
Finding Nemo (2003) 5
```


* python3 Evaluator.py BehaviorBasedCF

```shell
Evaluating  User KNN ...
Evaluating accuracy...
Computing the cosine similarity matrix...
Done computing similarity matrix.
Analysis complete.

Evaluating  Item KNN ...
Evaluating accuracy...
Computing the cosine similarity matrix...
Done computing similarity matrix.
Analysis complete.

Evaluating  Random ...
Evaluating accuracy...
Analysis complete.


Algorithm  RMSE       MAE       
User KNN   0.9961     0.7711    
Item KNN   0.9995     0.7798    
Random     1.4385     1.1478    


Using recommender  User KNN

Building recommendation model...
Computing the cosine similarity matrix...
Done computing similarity matrix.
Computing recommendations...

We recommend:
One Magic Christmas (1985) 5
Step Into Liquid (2002) 5
Art of War, The (2000) 5
Taste of Cherry (Ta'm e guilass) (1997) 5
King Is Alive, The (2000) 5
Innocence (2000) 5
MaelstrÃ¶m (2000) 5
Faust (1926) 5
Seconds (1966) 5
Amazing Grace (2006) 5

Using recommender  Item KNN

Building recommendation model...
Computing the cosine similarity matrix...
Done computing similarity matrix.
Computing recommendations...

We recommend:
Life in a Day (2011) 5
Under Suspicion (2000) 5
Asterix and the Gauls (AstÃ©rix le Gaulois) (1967) 5
Find Me Guilty (2006) 5
Elementary Particles, The (Elementarteilchen) (2006) 5
Asterix and the Vikings (AstÃ©rix et les Vikings) (2006) 5
From the Sky Down (2011) 5
Vive L'Amour (Ai qing wan sui) (1994) 5
Vagabond (Sans toit ni loi) (1985) 5
Ariel (1988) 5

Using recommender  Random

Building recommendation model...
Computing recommendations...

We recommend:
Sleepers (1996) 5
Beavis and Butt-Head Do America (1996) 5
Fear and Loathing in Las Vegas (1998) 5
Happiness (1998) 5
Summer of Sam (1999) 5
Bowling for Columbine (2002) 5
Babe (1995) 5
Birdcage, The (1996) 5
Carlito's Way (1993) 5
Wizard of Oz, The (1939) 5
```

# Todo   

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

    ContentKNN_acc
    ml
    ml-latest-small-item-similarity

```
