# Recommender_prj

sundog-education.com/RecSys  

anaconda  


## Run the project by Terminal  

```shell
python3 path_name/GettingStarted.py
```


Fun problem to solve. Try to find people's preference.


data-driven  

forms & purpose  
    things
    content ( New York Times, articles -- read instead of buy )
    music( content-based )
    people( on line dating )
    search results
YOU  
    explicit feedback: star review  
        ratings -- extra effort to users -- sparse -- low quality; if can persuasive users give  it to you  
        diff standard -- normalize -- cutural diff...
    implicit: purchase data, video viewing data, click data  
        not sparsity, but might not reliable, haha, pornography detection engines  
        fraud  
        purchases, resistant to fraud, amazon recsys is so good, cuz data is so great  
        consume, youtube, minutes to watch, consumption of your time

what source of data I have about my user's interest and start from here.  


MovieLens
No worry about data source

## Terminology  

top-N  
    candidate generation
    ranking
    filtering  


## Evaluation  
it depends: culture, business,  


accuracy is not the best but the best we can do without online test case.  

* accuracy
train/test
cross validation  
measuring
    accuracy(offline)
    - MAE mean absolute
    - RMSE root mean square  -- Netflix prize, not valuable in real problem
    BellKor -- though accuracy is not everything  

Top-N Hit Rate - Many Ways offline
    evaluating top-n recommenders
    - hit rate
    leave-one-out cross validation  
        hold back one rating per user
        test ability to recommend that left out movie in our top-N lists
    - average reciprocal hit rate(ARHR)
    more user focus metrics  
    - cHR
    - rHR

RMSE and hit rate aren't always related.  

* Coverage, Diversity and Novelty !!!  

Coverage  

Diversity  -- the same types   
    similarity between recommendation pairs
    1 - S
Novelty  -- popular or obscure  
    familiar popular items and what we call serendipitous discovery of new items.
    novelty and trust.


* Churn, Responsiveness and A/B Test  
online A/B test -- only way  -- matters more than anyting
perceived quality  


http://surpriselib.com  

## Content - Based
> Here I recommend movies just based on movies themselves, like genre, year to release(narrow).

Measure the similarity based on genres.   
Mathematical Ways:  
    cosine similarity
        0 : not similar
        1: the same  


Able to do more:
IMDB to get information, like, reviews, actors, directors, music and etc..


## Code Snippet
```python
class MyOwnAlgorithm(AlgoBase):
    """docstring for MyOwnAlgorithm."""
    def __init__(self):
        AlgoBase.__init__()
    def estimate(self, user, item):
        """ inner ID, not raw ID
            predict
        """
        return 3
```
