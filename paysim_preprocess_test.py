def preprocess_test():
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
    df = df.ix[:, [0, 1, 2, 4, 5, 7, 8, 10]]
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
