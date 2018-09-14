# Recommender_prj  
I'm working on organizing the project again.  

# Framework  
`Run.py` file is used to run the chosen algorithm, by command line `python3 Run.py algorithm_name`.  
`EvaluationData` class is a wrapper of the data set, containing multiple ways to split the data.  
`EvaluatedAlgorithm` is a wrapper of the algorithm which inherits from `surprise.AlgoBase` class.  
`RecommenderMetrics` is a wrapper of the metrics used to measure the recommender, including `RMSE`, `HitRate` and etc.  

# Think about it...  

* Data to use  
The data is even more important than the algorithm to choose.  
Here I use movielen 100,000 rating data set, which is large enough to get a good performance for some methods to recommendation.  
Large data set introduces computation and sparsity issues.  
Also, it's better to evaluate the dataset per se, like finding the outliers.

* Hybrid  
Recommendation is based on what we have.  
Only one algorithm or method is far from good enough in the real world situation.  
To deal with cold-start problem, content-based one is a good choice; or content-based one considering the new user's geo-information.  

Recommendation is hard to evaluate; accuracy is not good enough but a proper one I use to test the recommender offline.  
To consider novelty, diversity, we might combine the random choice of high-score items from different categories with what we recommend based on the prediction.   

* Real-World case  
Recommendation metrics are related to the purpose, especially the business purpose.  
And some social ethical issues are also needed to consider to avoid some trouble.  

# Implementation  

## Behavior-Based, CF - Measuring Similarity and Sparsity  

CF has the problem of data sparsity. The data is even more important than the algorithm to choose.  
Here the 100,000 rating data aren't enough to generate good similarity data, but I assume that I have enough data with high quality(it only contains users who have rated at least 20 movies, in the real world it might not be like this).  
Sparsity also introduces some computational challenge; how to store the data efficiently. (sparse vector/matrix).  

* Pro:  
works well in practice  
* Con:  
limited scalability -- Apache Spark  
sparsity, sensitive to data noise  

https://surprise.readthedocs.io/en/stable/knn_inspired.html  
* (user-based)Cosine - similarity  
adjusted cosine  
Normalize the data to deal with people's different measure scale. But the normalization is only meaningful when the user has rated a lot of stuff.  

* (item-based)Pearson similarity  
the same thing as adjusted cosine basically  

- [ ] try to build a filter to get high quality neighbors to measure the similarity instead of using all the other users/items to save computation.  
- [ ] try to use `KNNWithMeans()` to do the user-based CF  

- [ ] try Apache Spark if I have chance  

**According to the output, the user-based CF is very different from the item-based one.**  

It's different when we are doing the recommendation from calculating the accuracy or other measurement.  



other method not used often  

* Spearman rank correlation  
Pearson similarity based on ranks, not ratings.  
* Mean Square difference (MSD)  
* Jaccard Similarity  


## Model-Based by Matrix Factorization, learn the features  
latent features  

* PCA  
* SVD  `R = M * Sigma * U_t`  
implemented by SGD(surprise library has done this).  

The performance is the best so far compared with content-based, behavior-based ones. Impressive.  

Using machine learning achieve the parameters, has good performance but hard to explain.  `TuningSVD.py`.  

- [ ] Try more parameter combinations, ex learning rate, epoch.  Surprise supports the tool to do this.  

## Deep Learning in Recommender by tensorflow  
Recommendation is a pattern recognition problem actually, which is looking for a very complex pattern based on behavior of other people. So MF can be modeled by NN.  
Here use Tensorflow Framework to implement NN.  
I don't deal with the missing data, so the DL does **NOT** perform good enough. But it doesn't mean it is bad, just I need to find the right tool to deal with it. That's what I want to try next.  
- [ ] deal with input sparsity.  

* RBM(restricted boltzmann machine)  

visible layer -> hidden layer -> output layer  
`visible layer` used for forward pass  
convert rating scores into categorical data; the input dimension is movie# * score_categorical_bit.  
`hidden layer` used for backward pass  

Problem is sparsity.  

* another way  

Basically it's not so much different from RBM. It normalize the score to 0 - 1, so the input layer contains `numItems` nodes.  
When predicting, denormalizing is needed.  

* CNN, session-based  
It deals with sequence, like clickstream; it's really complex.  

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

# Reference and Source  
Coursera relevant course.  
Udemy relevant course.  
http://technocalifornia.blogspot.com/2014/08/introduction-to-recommender-systems-4.html?m=1&from=singlemessage&isappinstalled=0  
https://www.youtube.com/watch?v=E8aMcwmqsTg&list=PLLssT5z_DsK9JDLcT8T62VtzwyW9LNepV&index=55  
https://stackoverflow.com/questions/5253362/how-to-build-a-movie-recommender-system  





# Run result  

* python3 Run.py SvdRandom  
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

* python3 Run.py ContentRecs  
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


* python3 Run.py BehaviorBasedCF  

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



* python3 UserCF.py  
```shell  
testUser 85, the ratings are:  
Jumanji (1995)	:5.0  
GoldenEye (1995)	:5.0  
Braveheart (1995)	:5.0  
Jerky Boys, The (1995)	:5.0  
LÃ©on: The Professional (a.k.a. The Professional) (LÃ©on) (1994)	:5.0  
Pulp Fiction (1994)	:5.0  
Stargate (1994)	:5.0  
Shawshank Redemption, The (1994)	:5.0  
Star Trek: Generations (1994)	:5.0  
Clear and Present Danger (1994)	:5.0  
Speed (1994)	:5.0  
True Lies (1994)	:5.0  
Fugitive, The (1993)	:5.0  
Jurassic Park (1993)	:5.0  
Terminator 2: Judgment Day (1991)	:5.0  
Mission: Impossible (1996)	:5.0  
Rock, The (1996)	:5.0  
...  


Star Wars: Episode IV - A New Hope (1977) 114.57068319140309  
Matrix, The (1999) 107.72095292088618  
Star Wars: Episode V - The Empire Strikes Back (1980) 88.09116645357186  
Fight Club (1999) 79.26558201621258  
Back to the Future (1985) 78.78807368067915  
Raiders of the Lost Ark (Indiana Jones and the Raiders of the Lost Ark) (1981) 78.77028125945898  
American Beauty (1999) 77.32300806156537  
Toy Story (1995) 76.37713266677879  
Godfather, The (1972) 76.21072562503657  
Star Wars: Episode VI - Return of the Jedi (1983) 74.71908773556109  
Lord of the Rings: The Fellowship of the Ring, The (2001) 74.37234120218191  
```  


* python3 ItemCF.py  
```shellKiss of Death (1995) 16.910437073265502  
Amos & Andrew (1993) 16.861270021975354  
Edge of Seventeen (1998) 16.853845983977223  
Get Real (1998) 16.840092759084882  
Grace of My Heart (1996) 16.83866418909583  
Relax... It's Just Sex (1998) 16.825893097731395  
My Crazy Life (Mi vida loca) (1993) 16.825163372963015  
Set It Off (1996) 16.820045947032426  
Bean (1997) 16.81043113102984  
Joe's Apartment (1996) 16.804698282071367  
Lost & Found (1999) 16.78956315445952  

```  


* python3 Run.py MF  
```shell  
...  
Algorithm  RMSE       MAE  
SVD        0.9039     0.6984  
SVD++      0.8943     0.6887  
Random     1.4359     1.1493  
...  
Using recommender  SVD  

Building recommendation model...  
Computing recommendations...  

We recommend:  
Gladiator (1992) 4.520884890007874  
Philadelphia Story, The (1940) 4.420701711947352  
Stand by Me (1986) 4.395958975217837  
Moon (2009) 4.372613693384055  
Happiness (1998) 4.369493252705134  
American Graffiti (1973) 4.353470600109924  
And Your Mother Too (Y tu mamÃ¡ tambiÃ©n) (2001) 4.349192492630499  
Wallace & Gromit: A Close Shave (1995) 4.3154412154304085  
Band of Brothers (2001) 4.315414828016616  
Seven Samurai (Shichinin no samurai) (1954) 4.311102920673881  

Using recommender  SVD++  

Building recommendation model...  
Computing recommendations...  

We recommend:  
Lock, Stock & Two Smoking Barrels (1998) 4.60422766627621  
The Imitation Game (2014) 4.457817607681913  
Amadeus (1984) 4.322529285260794  
Indiana Jones and the Last Crusade (1989) 4.289355864317423  
Happiness (1998) 4.260333724293291  
Harry Potter and the Sorcerer's Stone (a.k.a. Harry Potter and the Philosopher's Stone) (2001) 4.229511982111833  
Wings of Desire (Himmel Ã¼ber Berlin, Der) (1987) 4.2247114809512345  
Grand Day Out with Wallace and Gromit, A (1989) 4.223352198994352  
Charade (1963) 4.208953110263677  
Snatch (2000) 4.204477673848874  

Using recommender  Random  

Building recommendation model...  
Computing recommendations...  

We recommend:  
Usual Suspects, The (1995) 5  
Legends of the Fall (1994) 5  
Trainspotting (1996) 5  
Titanic (1997) 5  
Happiness (1998) 5  
Big Daddy (1999) 5  
Spider-Man 2 (2004) 5  
Blade Runner (1982) 5  
Stand by Me (1986) 5  
Indiana Jones and the Last Crusade (1989) 5  
```  
* python3 Run.py AutoRec  
```shell  
...  
Evaluating  RBM ...  
Evaluating accuracy...  

Trained epoch  0  
...  
Trained epoch  19  

Processing user  0  
...  
Processing user  650  

Analysis complete.  


Algorithm  RMSE       MAE  
RBM        1.1897     0.9935  
Random     1.4385     1.1478  

Using recommender  RBM  
Computing recommendations...  
We recommend:  
Howl's Moving Castle (Hauru no ugoku shiro) (2004) 3.3286826610565186  
Harry Potter and the Deathly Hallows: Part 2 (2011) 3.3260419368743896  
Hurt Locker, The (2008) 3.3108229637145996  
Downfall (Untergang, Der) (2004) 3.3095009326934814  
Harry Potter and the Half-Blood Prince (2009) 3.30783748626709  
Amores Perros (Love's a Bitch) (2000) 3.3013195991516113  
My Neighbor Totoro (Tonari no Totoro) (1988) 3.2996013164520264  
Moon (2009) 3.2982006072998047  
There Will Be Blood (2007) 3.2948057651519775  
Blood Diamond (2006) 3.2938945293426514  

Using recommender  Random  
...  
```  
