from np import *
from feature import _FEATURE_CSV

_NLTABLE = "/normalized-linkage-table.csv"
_HUB_CT  = 5

def calc_community(hubArray, df, hist1Feature, ladFeature) :

    nodes  = []

    for hub in hubArray :                 # Loop through each community hub
        c1 = np.where(df[hub, :] == 1)    # Nodes in community
        c1 = np.append(c1, hub)           # Appending hub as a node to community
        h1 = np.where(hist1Feature == 1)  # Indices of windows that have hist1
        l1 = np.where(ladFeature == 1)    # Indices of windows that have LAD
        size = len(c1)                    # Size of community

        intersectH1    = np.intersect1d(c1, h1[0])   # Hist1 & community intersect
        intersectH1Len = len(intersectH1)            # length of intersect

        intersectLad    = np.intersect1d(c1, l1[0])  # LAD feature & community intersect
        intersectLadLen = len(intersectLad)          # length of intersect

        h1Percent  = round(intersectH1Len / size, 3)  # Hist1 percentage
        ladPercent = round(intersectLadLen / size, 3) # LAD percentage

        # Display results
        print("Hub:", hub)
        print("size {}\nHist1 % {}\nLAD % {}".format(size, h1Percent, ladPercent))
        print("Nodes In Community: {}\n".format(c1))

        nodes.append(c1) # Append nodes to communities list

    return nodes

def calc_heatmap(df, nodes, hubs) :

    heatmapArray = []

    for i in range(len(hubs)) :
        tmp = np.zeros((81,81))    # initialize zero matrix
        tmp[hubs[i], nodes[i]] = 1 # set the row values
        tmp[nodes[i], hubs[i]] = 1 # set the column values

        heatmapArray.append(tmp) # append result

    return heatmapArray

def display_heatmaps(heatmapArray) :

    # Display all heatmaps
    for heatmap in heatmapArray :
        fig = plt.figure()
        heatmap = sb.heatmap(heatmap, cmap = "YlGnBu")
        fig.show()

    input()

if __name__ == '__main__':

    df = pd.read_csv(DATADIR + _NLTABLE)   # importing csvs
    df = df.drop(['Unnamed: 0'], axis = 1) # removing unamed column

    # setting linkage where node is compared to itself to 0 because nodes cant
    # have edges to themselve and the l-avg will not include these values
    for i in range(len(df)) :
        for j in range(len(df)) :
            if i == j:
                df.iloc[i,j] = 0

    wlen = len(df)                   # number of windows
    div  = wlen*wlen-wlen            # # of entries - reflecting # of entries
    lAvg = df.to_numpy().sum()/div   # linkage average
    df   = np.where(df > lAvg, 1, 0) # setting edges where l(A,B) > lavg to 1
                                     # and non edges to 0

    rowS = np.sum(df, axis = 1)      # Summing up # of edges
    rowS = (rowS)/ (wlen - 1)        # degree centrality

    sortedIndex = np.argsort(rowS)     # Sorting by index
    sortedVals  = np.sort(rowS)        # Sorting by valus 

    # Displaying results
    print("Window  Centrality")
    for i in range(wlen) :
        print("{:<7} {}".format(sortedIndex[i], sortedVals[i]))

    print("max centrality:", np.amax(rowS))
    print("min centrality:", np.amin(rowS))
    print("avg centrality:", np.average(rowS))

    features = pd.read_csv(DATADIR + _FEATURE_CSV)

    window_names = features.iloc[:, 0]

    hist1Feature = "Hist1"
    ladFeature   = "LAD"

    # Windows == Feature selection
    hist1 = features.loc[:, hist1Feature]
    lad   = features.loc[:, ladFeature]
    hubs  = sortedIndex[-_HUB_CT:]

    # Final Part
    print("Hubs:", hubs)
    nodeArray = calc_community(hubs, df, hist1, lad)
    heatmapArray = calc_heatmap(df, nodeArray, hubs)
    display_heatmaps(heatmapArray)
