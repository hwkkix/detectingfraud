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

![code](https://cloud.githubusercontent.com/assets/22734960/26038017/573db03a-38cd-11e7-8b63-2144503ad75c.jpeg)

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
The script that preprocesses the train.csv file is titled paysim_preprocess_train.py while the file used to preprocess is titled paysim_preprocess_test.py.  Below is the entirty of the train script for your enjoyment.

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

