from np import *

_NLTABLE = "/normalized-linkage-table.csv"

if __name__ == '__main__':

    df = pd.read_csv(DATADIR + _NLTABLE)   # importing csvs
    df = df.drop(['Unnamed: 0'], axis = 1) # removing unamed column

    wlen = len(df)                   # number of windows
    lAvg = np.average(df)            # linkage average
    df   = np.where(df > lAvg, 1, 0) # setting edges where l(A,B) > lavg

    rowS = np.sum(df, axis = 1)        # Summing up # of edges
    rowS = (rowS - 1)/ (len(df) - 1)   # degree centrality - also note rows-1 due
                                       # to nodes not having edges to themselves

    rowS = np.where(rowS < 0, 0, rowS) # edge case where window has 0 connections
    
    sortedIndex = np.argsort(rowS)     # Sorting by index
    sortedVals  = np.sort(rowS)        # Sorting by valus 

    # Displaying results
    print("Window  Centrality")
    for i in range(wlen) :
        print("{:<7} {}".format(sortedIndex[i], sortedVals[i]))
    
    print("max centrality:", np.amax(rowS))
    print("min centrality:", np.amin(rowS))
    print("avg centrality:", np.average(rowS))


    # this is for the one thing i emailed about
    #print(df.to_numpy().sum()/( wins*wins-wins))   
    '''
    for i in range(len(df)) :
        for j in range(len(df)) :
            if i == j:
                df.iloc[i,j] = 0
    '''
