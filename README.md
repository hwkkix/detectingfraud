## Problem Statement 

Financial fraud is a problem that affects the entire world economy.  Despite the importance of fully understanding it, according the [doctoral thesis](http://bth.diva-portal.org/smash/record.jsf?pid=diva2%3A955852&dswid=4451) of Edgar Alonso Lopez-Rojas, there is a general lack of understanding of fraudulent behavior in financial domains today.

We lack understanding primarily due to our inability to research the subject and we cannot research the subject because of our lack of financial transaction data.  To help to address this situation two simulation models (Payment Simulator (PaySim) and Retail Store Simulator (RetSim)) were developed to generate fictional transactional data, of a fraudulent and non-fraudulent nature.  Both behaviors, fraudulent and non-fraudulent, wsa codified in the simulators using available data with the intent to apply machine learning methodologies in hopes to better understand and stop financial fraud.

We will focus on a modified version of the [PaySim dataset on Kaggle](https://www.kaggle.com/ntnu-testimon/paysim1) that Edgar was so kind to have shared with the rest of us.  It is through using this dataset, applying methods of scaling, featuring engineering, and machine learning models that we will begin to better understand the behaviors that serve as leading indicators that a transaction is fraudulent.  Only then will we be better positioned to protect the global financial interests of it's law abiding citizens.

![eye](https://cloud.githubusercontent.com/assets/22734960/26037132/1547440a-38bb-11e7-9616-59c5651b1d34.png)

## An Exploratory Data Analysis

A methodical EDA (exploratory data analysis) is comprised of four steps.

1. The first of these steps involves loading the data into memory and storing it in a dataframe. When a dataset does not fit into memory there are alternative methods of doing this - though I am unsure of what those are.

```markdown
df = pd.read_csv('train.csv')
# making dataframe df from 'train' dataset
# completes step 1 of a methodical EDA
```

2. In step two we will make the data set a "tidy data" set by following the guidelines established by [Hadley Wickham](http://vita.had.co.nz/papers/tidy-data.html)

```markdown
df.shape
# Curious as to the size of the data set

df.head()
# curious as to the content of the data set
```
# Why look at the head of the data set?

I chose to begin step two of my EDA, tidying up the data set, by looking at the head of the data frame. In doing this I can take a look at my column headers as well as the first 5 results where I am looking for the presence of the 3 indicators of a tidy data set:

1. Each variable forms a column
2. Each observation forms a row
3. Each type of observational unit forms a table

Though not all of these are going to be appparent with only 5 observations listed, we can capture any of the obvious ones and correct them right away.

According to Hadley, the 5 most frequent causes of messy data are as follows:

1. Column headers are values and not variable names
2. Mulitple variables are stored in one column
3. Variables are stored in both rows and colummns
4. Observational units types of varying kinds are stored in the same table
5. Multiple instances of a single observational type is found in numerous tables

At the on set alll of these stipulations seems to be in order. I do however take issue with the ordering of the columns. Fixed variables should come first (left-most) followed by the measured variables. Currently, all we have are fixed variables and the flow of the columns makes sense to me with the exception of the 'id' column. It is my belief that this is a unique identifier for each transaction and as such it makes more sense to me that this column be moved to the far left side of the data frame. I've also corrected the column header 'oldbalanceOrg' to 'oldbalanceOrig' thinking that the typo ought to be corrected.

As the analysis continued to deepen I will began to realize that the column header change was unneeded and the change was dropped.  The 'id' column was soon ignored too once my analysis highlighted that it was skewing the results.  Columns like this result in "overfitting" of a model to the training data set.  Oh well - lesson learned.  Code that was eventually abandoned is as follows:

```markdown
df.shape
# Curious as to the size of the data set

df = df[['id', 'step', 'type', 'amount', 'nameOrig', 'oldbalanceOrg',
         'newbalanceOrig', 'nameDest', 'oldbalanceDest', 'newbalanceDest',
         'isFraud']]
df.head()
# moving 'id' column to the left end of df

df = df.rename(columns={'oldbalanceOrg': 'oldbalanceOrig'})
# correcting 'oldbalanceOrig' column header

df.head()
# confirming my changes
# completes step 2 of a methodical EDA
```
3. Step three will use descriptive statistics and exploratory visualizations to understand the data at a macro level

### What can we learn from our tabular analysis?

By looking at crosstab tables (an example of one is below) we can learn from our data the following:

- There are 445,383 observations none of which have null values
- 576 transactions were fraudulant (276 cash_out & 300 transfer)
- There are 5 different transaction types for us to take into account.  The msot frequest of these transaction types is the Cash_Out type.
- Out of the 445,383 transactions we have 445,338 unique customers originating the transaction.  This means we have 45 repeat originators.  However, of these repeat originators none appear more than twice.
- The mean transaction amount was 178,918 with the largest transaction being 60,642,000.
- 50% of origin account new balances are 0, indicating they were drained, be it on purpose or by fraud
- 69% (174,616) of the origin accounts having new balances of 0 ened up that way through Cash_Out and Transfer transaction types
- 88% (150,954) of the destination accounts having new balances of 0 ended up that way through Payment transaction types
- These same accounts have old and new balances equal to each other indicating to me they may be acting like pass through accounts.
- The average transfer amount was 903,518.
- I prefer tables when analyzing data despite most peoples preference for graphs

```markdown
df.isnull().sum()
# no need for imputation
id                0
step              0
type              0
amount            0
nameOrig          0
oldbalanceOrig    0
newbalanceOrig    0
nameDest          0
oldbalanceDest    0
newbalanceDest    0
isFraud           0
dtype: int64

pd.crosstab(df.type, df.isFraud == 1)
# Looking at transaction types for fraudulent transactions
| isFraud  | False  | True |
| -------  | -----  | --- |
| type|
| CASH_IN  | 97801  | 0 |
| CASH_OUT | 156358 | 276 |
|  DEBIT   | 2933   | 0 |
| PAYMENT  | 150954 | 0 |
| TRANSFER | 36761  | 300 |

pd.crosstab(df.type, df.newbalanceOrig == 0.0)
# '''Looking at post transfer balances equal to $0
# of orgin accounts by transaction type'''

d.crosstab(df.type, df.newbalanceDest == 0.0)
# '''Looking at post transfer balances equal to $0
# of destination accounts by transaction type'''

pd.crosstab(df.type, df.oldbalanceDest == df.newbalanceDest)
# '''Looking for destination accounts having pre
# and post transaction balances equal to each other'''

pd.crosstab(df.type, df.amount == 0, values=df.amount, aggfunc=np.mean)
# Looking at average transaction amount by transaction type
```
### What can we learn from our analysis through visualization?

By looking at the below visualizations we can learn from our data the following:

- Due to the disparity in our data the ability to zoom offered through plotly is exeremely helpful here.  At the greatest zoom many of these charts are difficult to analyze.
- Graphics slowed my notebook and they need some further work
- Though large in transaction size the fraudulent transactions were not large in number

![chart](https://cloud.githubusercontent.com/assets/22734960/26038010/07674f94-38cd-11e7-8093-e46f0dd2d42f.jpg)

4. In step four we will aggregate the data and explore the group properties in a much more detailed manner

### What can we learn from our micro level analysis?

By looking at the below visualizations and tables we can learn from our data the following:

- There were 276 fraudulent transactions of the Cash_Out type with a mean amount of 1,587,794 in local currency.  Half of all fraudulent transactions of this type were in the amount of 407,495
- There were 300 fraudulent transactions of the Transfer type with a mean amount of 1,484,329 in local currency.  Half of all fraudulent transactions of this type were in the amount of 463,901
- Only 130 (47%) fruadulent transactions were below 100,000
- New balances for the destination accounts were 0 75% of the time for the fraudulent Transfer transaction types.  curious 

```markdown
df.groupby(['isFraud', 'type']).describe()
# looking at data by transaction type and fraudulent or not

df.groupby(['type', df.amount < 100000]).isFraud.value_counts().unstack()
# determining fraudulent transactions over $100,000

pd.crosstab(df.type, df.isFraud).plot.bar()
# completes step 4 of a methodical EDA
```
### What is my data coupled with my intuition telling me the story is here?

A typical fraudulent account is not frequent in nature occuring only about .1% of the all transactions.  When a fraudulent transaction ois successful however it is devistanting, draining it's victim's account balance to $0 in nearly every case.  The average fruadulent transaction amount being 1.5 million.  Of the 5 transaction types the two that are most susceptible are Cash Out and Transfer transactions.  The risk of a fraudulent Transfer transaction taking place actually increases to 1 percent.  Interestingly the ending balance on the destination accounts on Cash Out transactions seemingly represents the increase for the fraudulent funds deposited where in the case of the Transfer transactions the funds seem to disappear.  

![investigation](https://cloud.githubusercontent.com/assets/22734960/26038261/59b1b762-38d2-11e7-98ab-08c6b98431b8.png)

## Data Preprocessing

Preprocessing of the data to get it in a pipe-ready state will be taken care of by two scripy files - the first for training the data set and the second for the test data for whc=ich we want to make predictions upon.  The only difference between the two spripts will be an indexing of dataframes where the training script accounts for the 'isFraud' column which is the column upon which we train the model on.  Of course the test set will be lacking this column as the entire goal is to predict the value that goes in the column.  The steps (in order) that both scripts take otherwise are as follows:
- Imports pandas as .pd, asks for file path and file name, 1mports .csv file, and creates pandas data frame.
- Splits 'nameDest' to retain letter
- Drops unneeded columns
- Gets dummies
- Makes Feature list
- Organizes into 50 kmenas clusters
- Gets dummies
- Makes Feature list

The script that preprocesses the train.csv file is titled paysim_preprocess_train.py while the file used to preprocess is titled paysim_preprocess_test.py.  Below is the entirety of the train script for your enjoyment.

```markdown
def preprocess_train():
    """
    preprocess()

    Imports pandas as .pd, asks for file path and file name, 1mports .csv file, and creates pandas data frame.
    Splits 'nameDest' to retain letter
    Drops unneeded columns
    Gets dummies
    Makes Feature list
    Organizes into 50 kmenas clusters
    Gets dummies
    Makes Feature list

    Parameters
    ----------
    file path:
        local address of file
    file name:
        name of file with extension

    Returns
    -------
    out: data frame
        A Pandas data frame as df

    Examples
    --------
    >>> preprocess()
        What is your file path? C:\\Users\\DN\\hwkkix\\data820\\assignments\\assignment-03\\data\\
        What is your file name with extension? 1.csv
    >>> Thanks for helping store your file path and name! Enjoy your new dataframe
    """
    import pandas as pd
    path = input('What is your file path? ')
    file = input('What is your file name with extension? ')
    print('Thanks for helping store your file path and name! Enjoy your new dataframe ')
    df = pd.read_csv(path+file)
    df['Recipient'] = df['nameDest'].str[:1]
    df = df.ix[:, [0, 1, 2, 4, 5, 7, 8, 9,11]]
    df.amount = np.log1p(df.amount)
    df.oldbalanceOrg = np.log1p(df4.oldbalanceOrg)
    df.newbalanceOrig = np.log1p(df.newbalanceOrig)
    df.oldbalanceDest = np.log1p(df.oldbalanceDest)
    df.newbalanceDest = np.log1p(df.newbalanceDest)
    df.step = np.log1p(df.step)
    df = pd.get_dummies(df, prefix='was', prefix_sep='_')
    features = list(set(df.columns) - {'isFraud'})
    kmeans = KMeans(n_clusters=50, random_state=0, n_jobs=-1)
    kmeans.fit(df[features])
    predictedLabels = kmeans.predict(df[features])
    df['predictedCluster'] = predictedLabels
    df['predictedCluster'] = df.predictedCluster.astype('object')
    df = pd.get_dummies(df, prefix='was', prefix_sep='_')
    features = list(set(df.columns) - {'isFraud'})
    return df
  ```  

![script](https://cloud.githubusercontent.com/assets/22734960/26038240/bb6aa050-38d1-11e7-8373-13b7b7feb587.jpeg)

## Developing a predictive model

My efforts revolved around the development of two different model types (Random Forrest and Linear SVC), a couple of different scaling strategies, some of the available feature enginnering techniques and different cross-validation methods.  All of these tools were mixed and matched, swapped and switched, changed and exchanged for a total of 21 different submissions.  Of these 21 submissions, 9 of them score 87% or higher on both the public and private leader boards.  Though I did not end atop with of the boards with my submissions I was within .004% of the to score with my best attempt.  I also have 5 of the 10 top scores on the private leader board and so despite my slow start due to mistakes in my early code, I am very pleased with the end result.

In my file Project Final.ipynb found on the repo you will find the 115 or so code cells that document just a portion of the attempts but if you take the time to review it you will get a good idea of what transpired.  Many of those efforts were of the Random Forrest type and they provided some of the best leaderboard results.  They however did not proivde the best results.  Those were born of a Linear SVC and will be the detailed below.

I will take moment to highlight my preference of the Random Forrest model.  It not only preformed my more consistantly on the private leaderboard, it gives you much better visibility into feature importance.  Below is the feature ranking from my favorite model of them all - a random forest.  

I'm rather pleased with my simple Random Forest model utilizing k means feature engineering and a k folds cross validation strategy.  The features driving this model are (in order of importance):
 
Feature ranking:
1. 'newbalanceDest' (0.247011)
2. 'oldbalanceOrg' (0.226982)
3. 'amount' (0.155077)
4. 'step' (0.104939)
5. 'oldbalanceDest' (0.073532)
6. 'was_TRANSFER' (0.064203)
7. 'oldbalanceDest' (0.048071)
8. 'was_CASH_OUT' (0.026618)
9. 'was_CASH_IN' (0.017521)
10. 'was_11' (0.009711)
 
This model did score very high on the public leaderboard putting me at the number one spot with a 0.98416!  I noticed too that of the top models on the private board this model was the only one that had a higher rating on the private board than it had on the public board with a 0.987466.  Even when the model fell from the top spot I still took pride in this fact.

![features](https://cloud.githubusercontent.com/assets/22734960/26038504/dba5c5a6-38d7-11e7-82b4-869e7d628d68.png)

This is very useful information and head and shoulders above that which I could pull from the Linear SVC model and coefficient measurements.  I am sure there is more out there to be had and my plan is too dig in as Linear SVC provided the sinlge best score and therefore is not something to be ignored.  My SVC model (model4) was developed aas follows:

```markdown
ptr = []
p = []
for i in np.logspace(-10, 2, num=15):
    model4 = LinearSVC(C=i, random_state=0)
    model4.fit(train[features4], train.isFraud)
    print(70*"#")
    trainscore = model4.score(train[features4], train.isFraud)
    print("Model accuracy score on training data with hyperparameter of %s: %s"
          % (i, trainscore))
    testscore = model4.score(test[features4], test.isFraud)
    print("Model accuracy score on test data with hyperparameter of %s: %s"
          % (i, testscore))
    print("Number of features in the model: {}"
          .format(np.sum(model4.coef_ != 0)))
    print(70*"#", "\n")
    ptr.append(trainscore)
    p.append(testscore)
    # iterating on regularization hyperparameter
    # l2 penalty is ridge regularization
    
    ######################################################################
Model accuracy score on training data with hyperparameter of 1e-10: 0.998691083405
Model accuracy score on test data with hyperparameter of 1e-10: 0.998664073397
Number of features in the model: 63
###################################################################### 

######################################################################
Model accuracy score on training data with hyperparameter of 7.19685673001e-10: 0.998691083405
Model accuracy score on test data with hyperparameter of 7.19685673001e-10: 0.998664073397
Number of features in the model: 63
###################################################################### 

######################################################################
Model accuracy score on training data with hyperparameter of 5.17947467923e-09: 0.998691083405
Model accuracy score on test data with hyperparameter of 5.17947467923e-09: 0.998664073397
Number of features in the model: 63
###################################################################### 

######################################################################
Model accuracy score on training data with hyperparameter of 3.72759372031e-08: 0.998691083405
Model accuracy score on test data with hyperparameter of 3.72759372031e-08: 0.998664073397
Number of features in the model: 63
###################################################################### 

######################################################################
Model accuracy score on training data with hyperparameter of 2.68269579528e-07: 0.998691083405
Model accuracy score on test data with hyperparameter of 2.68269579528e-07: 0.998664073397
Number of features in the model: 63
###################################################################### 

######################################################################
Model accuracy score on training data with hyperparameter of 1.93069772888e-06: 0.998691083405
Model accuracy score on test data with hyperparameter of 1.93069772888e-06: 0.998664073397
Number of features in the model: 63
###################################################################### 

######################################################################
Model accuracy score on training data with hyperparameter of 1.38949549437e-05: 0.998691083405
Model accuracy score on test data with hyperparameter of 1.38949549437e-05: 0.998664073397
Number of features in the model: 61
###################################################################### 

######################################################################
Model accuracy score on training data with hyperparameter of 0.0001: 0.998691083405
Model accuracy score on test data with hyperparameter of 0.0001: 0.998664073397
Number of features in the model: 54
###################################################################### 

######################################################################
Model accuracy score on training data with hyperparameter of 0.000719685673001: 0.998695994912
Model accuracy score on test data with hyperparameter of 0.000719685673001: 0.998673896387
Number of features in the model: 51
###################################################################### 

######################################################################
Model accuracy score on training data with hyperparameter of 0.00517947467923: 0.999263273961
Model accuracy score on test data with hyperparameter of 0.00517947467923: 0.999223983812
Number of features in the model: 48
###################################################################### 

######################################################################
Model accuracy score on training data with hyperparameter of 0.0372759372031: 0.999280464236
Model accuracy score on test data with hyperparameter of 0.0372759372031: 0.999253452781
Number of features in the model: 45
###################################################################### 

######################################################################
Model accuracy score on training data with hyperparameter of 0.268269579528: 0.999363959853
Model accuracy score on test data with hyperparameter of 0.268269579528: 0.999322213709
Number of features in the model: 47
###################################################################### 

######################################################################
Model accuracy score on training data with hyperparameter of 1.93069772888: 0.999415530676
Model accuracy score on test data with hyperparameter of 1.93069772888: 0.999351682678
Number of features in the model: 47
###################################################################### 

######################################################################
Model accuracy score on training data with hyperparameter of 13.8949549437: 0.999363959853
Model accuracy score on test data with hyperparameter of 13.8949549437: 0.99930256773
Number of features in the model: 47
###################################################################### 

######################################################################
Model accuracy score on training data with hyperparameter of 100.0: 0.9993615041
Model accuracy score on test data with hyperparameter of 100.0: 0.99928292175
Number of features in the model: 46
###################################################################### 
```
Iterating over my hyperparameter "C" I found my best results with a C value of around 1.9.  This turned out to be the case for both the Train and Test data sets so this pleased me very much.  Even when employing a grid search did the result change very little.  Good stuff!  The final setup for model4 is found below:

```markdown
model4 = LinearSVC(C=1.93069772888, random_state=0)
model4.fit(train[features4], train.isFraud)
# resetting model to current best results

LinearSVC(C=1.93069772888, class_weight=None, dual=True, fit_intercept=True,
     intercept_scaling=1, loss='squared_hinge', max_iter=1000,
     multi_class='ovr', penalty='l2', random_state=0, tol=0.0001,
     verbose=0)
```

The resulting coefficients are visualized as such:

![coeff](https://cloud.githubusercontent.com/assets/22734960/26038558/51261154-38d9-11e7-8263-4f67a33ef5e1.png)

So this model could be titled "A Tale of Two Cities." On the one hand my public leaderboard score has improved to that of a 0.992386 - leaving only the slightest of margins for conitnued improvement and my best result yet. My private leaderboard rating however was a 0.984878 which is outperformed by the Random Forest models. Interesting. Additionally on the negative side would be a general lack of understanding of the SVM model coefficients. I can visualize them for you but I cannot - as of yet - explain them to an extent that pleases me. I do take comfort in the fact that this seems to be a common trait with us aspiring data scientists. This provides me a place to continue my learning and for now I shall move forward.
So, in the spirit of moving forward I need to figure out why my model scores less successfully on the private leaderboard which uses the AUC as it's success parameter. I'd like to ignore it but it is a more important measure and worthy of better understanding.

Using a confusion matrix and confusion report to compare the Linear SVC and Random Forest models it was clear where there is still work to do with respect to the SVC.

```markdown
confusion_matrix(df4.isFraud, model4.predict(df4[features4]))
# Recall for not Fraud = 507771/508340 = 99.9%
# Recall for is Fraud = 578/669 = 86.4%
# Precision for not Fraud = 507771/507862 = 99.9%
# Precision for is Fraud = 578/1147 = 50.3%
array([[507771,    569],
       [    91,    578]])
       
print(classification_report(df4.isFraud, model4.predict(df4[features4])))
# Whoa - not a good precision
# This could explain the poor private leaderboard score
            precision    recall  f1-score   support

          0       1.00      1.00      1.00    508340
          1       0.50      0.86      0.64       669

avg / total       1.00      1.00      1.00    509009


confusion_matrix(df1.isFraud, model.predict(df1[features1]))
# Recall for not Fraud = 508340/508340 = 100%
# Recall for is Fraud = 669/669 = 100%
# Precision for not Fraud = 508340/508340 = 100%
# Precision for is Fraud = 669/669 = 100%
array([[508340,      0],
       [     0,    669]])
 
print(classification_report(df1.isFraud, model.predict(df1[features1])))
# Wow - what a great result
# This could explain the excellent private leaderboard score 
            precision    recall  f1-score   support

          0       1.00      1.00      1.00    508340
          1       1.00      1.00      1.00       669

avg / total       1.00      1.00      1.00    509009
```

So - my issue with not performing well on the private leaderboard resides in the poor performance in my model prediticng too many false positives. There are far too many and the model needs to be further groomed to address it but it is erring on the correct side of the equation. This means we as a financial institution would perhaps make more coutesy calls verifying the appropriateness of the transactions prior to them being executed. Some people will balk at the intrusion but my preference it to err on the side of conservatism until further model improvements can be gained.
In order to compare the private leaderboard results for my final model (model 4) and that of my best model showing (model) I've calculated the confusion matrix for my model. There is not doubt in my mind now why the private leaderboard performance is so different are this exercise.
Though not perfect, this model pleases me to the point it could be productionalized for now while further enhancements are gained. Some of the methods I would use to further drive up the precision of my model4 would include: binning of my numerical features, generating polynomial features and reducing my data dimensions using PCA. Alas, the boss is here and wants to deploy sometime - asap.
Now we need to build a pipeline......

![pipe](https://cloud.githubusercontent.com/assets/22734960/26038650/28c8d9b0-38db-11e7-9736-0a5ec2a59bf2.jpeg)

## Building a pipeline 

My pipleine is a very simple version as it is my first attempt - and it shows.  Though the pipe (the name of my pipeline) scored very well on the split train/test data set and the entire train data set, when let loose into the wild and applied to the test set the scored plummmeted.  I did not go back and correct it - yet - but rather chose to celebrate that it executes and focus on my gitpage.  All in all, just creating my first pipeline at all was a win in my book.

Below is the code detailing my attempts:

```markdown
pipe = make_pipeline(LinearSVC(C=1.93069772888, random_state=0))
pipe.fit(train[features], train.isFraud)
# establishing pipeline that incorporates PCA and Linear SVM model
Pipeline(steps=[('linearsvc', LinearSVC(C=1.93069772888, class_weight=None, dual=True, fit_intercept=True,
     intercept_scaling=1, loss='squared_hinge', max_iter=1000,
     multi_class='ovr', penalty='l2', random_state=0, tol=0.0001,
     verbose=0))])
     
trainscore = pipe.score(train[features], train.isFraud)
print(70*"#")
print("Pipeline model accuracy score on training data set: %s" % (trainscore))
print(70*"#")
# Scoring the pipe on train
######################################################################
Pipeline model accuracy score on training data set: 0.999236260673
######################################################################

testscore = pipe.score(test[features], test.isFraud)
print(70*"#")
print("Pipeline model accuracy score on testing data set: %s" % (testscore))
print(70*"#")
# Scoring the pipe on test
######################################################################
Pipeline model accuracy score on testing data set: 0.999204337832
######################################################################

fullscore = pipe.score(df[features], df.isFraud)
print(70*"#")
print("Pipeline model accuracy score on the full data set: %s" % (fullscore))
print(70*"#")
# Scoring the pipe on full data set
######################################################################
Pipeline model accuracy score on the full data set: 0.999229876093
######################################################################

confusion_matrix(df.isFraud, pipe.predict(df[features]))
# Recall for not Fraud = 508325/508350 = 99.9%
# Recall for is Fraud = 292/669 = 44%
# Precision for not Fraud = 508325/508702 = 99.9%
# Precision for is Fraud = 292/307 = 95%
array([[508325,     15],
       [   377,    292]])
       
print(classification_report(df.isFraud, pipe.predict(df[features])))
           precision    recall  f1-score   support

          0       1.00      1.00      1.00    508340
          1       0.95      0.44      0.60       669

avg / total       1.00      1.00      1.00    509009
```

As you can see above my train/test/full data set results were very respectable.  However, instead of a pipeline what I developed was a pipe bomb as results, when submitted for scores were terrible.   Submitting my version 6 .csv resulted in a 0.684476 public leaderboard score and a 0.659461 private leaderboard score. These two scores would lead me back to the drawing board but all is not lost.  I've lots of pieces and parts that have provided great results.  Pre pipeline my model was scoring in a consevative manner as highlighted by the confusiion matrix and this is veyr important in the pursuit of reducing financial fraud.  We may be making a few extra coutesy calls to confirm transactions because of the false positives but in the case of people's personal finances we want to err on side of caution.  Some further tweaking to perfect a proper pipeline and we'll be in business!     

![algo](https://cloud.githubusercontent.com/assets/22734960/26038813/8801ac88-38de-11e7-99c6-12e2ee7324d5.jpeg)
