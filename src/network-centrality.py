from np import *
from feature import _FEATURE_CSV

_NLTABLE = "/normalized-linkage-table.csv"
_HUB_CT  = 5

def calc_community(hubArray, df, hist1Feature, ladFeature) :
    
    result = []
    nodes  = []

    for hub in hubArray :
        #size = np.sum(df[hub, :])
        c1 = np.where(df[hub, :] == 1)
        c1 = np.append(c1, hub)
        size = len(c1)
        h1 = np.where(hist1Feature == 1)
        l1 = np.where(ladFeature == 1)
        
        intersectH1 = np.intersect1d(c1, h1[0])
        #intersectH1 = np.intersect1d(c1[0], h1[0])
        #intersectH1 = np.append(intersectH1, hub)
        intersectH1Len = len(intersectH1)

        intersectLad = np.intersect1d(c1, l1[0])
        #intersectLad = np.intersect1d(c1[0], l1[0])
        #intersectLad = np.append(intersectLad, hub)
        intersectLadLen = len(intersectLad)

        h1Percent  = round(intersectH1Len / size, 3)
        ladPercent = round(intersectLadLen / size, 3)
        
        print("Hub:", hub)
        print("size {}\nHist1 % {}\nLAD % {}".format(size, h1Percent, ladPercent))
        print("Nodes In Community: {}\n".format(c1))
        
        result.append(size)
        nodes.append(c1)

    return result, nodes

def calc_heatmap(df, nodes) :

    heatmapArray = []

    for hub in nodes :
        tmp = np.zeros((81,81))
        for node in hub :
            tmp[node, :] = df[node, :]
            tmp[:, node] = df[:, node]

        heatmapArray.append(tmp)

    return heatmapArray

def display_heatmaps(heatmapArray) :

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
    
    print("Hubs:", hubs)
    sizeArray, nodeArray = calc_community(hubs, df, hist1, lad)
    heatmapArray = calc_heatmap(df, nodeArray)
    display_heatmaps(heatmapArray)
