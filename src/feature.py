from np import *

#_ORIGINAL_CSV = "/25rm_best_clust_array_original_df.csv"
#_CLUSTERS_CSV = "/25rm_best_clust_array.csv"
#_ORIGINAL_CSV = "/16rm_clust_array_original_df.csv"
#_CLUSTERS_CSV = "/16rm_clust_array.csv"
_ORIGINAL_CSV = "/clust_array_original_df.csv"
_CLUSTERS_CSV = "/clust_array.csv"
_FEATURE_CSV  = "/gam_feature_community.csv"

_NORMALT = .70
_NORMALF = 8/9

# remove names method
def rmv_nans(clusters):

    clusters = clusters.fillna(0)
    clusters = np.array(clusters.iloc[:,1:])
    tmp      = []

    for c in clusters :
        c = c[c != 0]
        tmp.append(c)

    return tmp

# feature sum method
def feature_sum(feat, clusters, df) :

    numClusters = len(clusters)
    featureSum  = []
    featSumMax  = feat.sum()  

    #print(df.loc[feat == 1])

    for i in range(numClusters) :
        featSum = df.loc[feat == 1, clusters[i]]
        denom   = df.loc[:, clusters[i]].sum()

        featSum = featSum.sum(axis = 0)
        featSum = (featSum/denom)

        featureSum.append(featSum)

    return featureSum

# show boxplot visuals method
def show_boxplot(figData, figName) :
    
    fig = plt.figure(figName)
    img = sb.boxplot(data=figData)
    plt.ylim(0, 1)

    fig.show()

# radial possitions method
def radial_pos(posCatagory, df) :

    tmp = []

    for i in posCatagory:
        catagoryNP  = df.columns[i].tolist()
        catagorySum = np.sum(df.loc[:, catagoryNP])

        tmp.append(catagorySum)

    return tmp

# clust wind method
def clust_wind(clust, df):

    clen   = clust.shape[0]
    winSum = df.loc[:, clust].sum(axis = 1)
    avg    = round((winSum/clen), 3)

    #print(avg)

    return avg

# method to find the windows with certain percentage of occurence in NPs
def get_contenders(windows) :

    indexArray = []
    i = 0
    percent  = _NORMALF
    for win in windows :

        if i != 0 :
            percent = _NORMALT

        #print("perecent val",percent)
        minPercent = (win > percent).tolist()
        #print("minpercent\n",minPercent)

        indexVal  = np.argwhere(minPercent)
        indexVal  = indexVal.flatten()
        #print(indexVal)

        indexArray.append(indexVal.flatten())
        i += 1

    return indexArray

# Cluster bar graphs method
def cluster_bar_graphs(clusters, df) :

    
    clusterS = ["C1", "C2", "C3"]
    apical   = ["Strongly Apical",
                "Somewhat Apical",
                "Neither",
                "Somewhat Equatorial",
                "Strongly Equatorial"]

    radialPosition = []

    for i in clusters :
        radialPositionTmp = calc_percentiles(df.loc[:, i], 0, 5)
        tmp = []

        for j in radialPositionTmp :
            length = len(j)
            tmp.append(length)
        radialPosition.append(tmp)

    radialPosition = pd.DataFrame(radialPosition,
                                    columns = apical,
                                    index = clusterS)

    radialPosition = radialPosition.T

    radialPosition['apical'] = apical

    radialPosition = pd.melt(radialPosition,
                                id_vars= "apical",
                                value_vars = clusterS,
                                var_name = "clusters")

    fig = plt.figure("radial possition")
    img = sb.barplot(data = radialPosition,
                        x = "apical",
                        y = "# of NPs",
                        hue = "clusters")

    # show the figure and exit on input
    fig.show()
    input()

# Method for scatter plots 1 2 3 which was a graph represetnation
# Which is just showing the overlappig windows for each cluster.
def scatter_plots(windowsIndexes) :
    figg = plt.figure("Rip")
    scat = sb.scatterplot(data = windowsIndexes[0])
    figg.show()
    figg2 = plt.figure("2")
    scat2 = sb.scatterplot(data = windowsIndexes[1])
    figg2.show()
    figg3 = plt.figure("3")
    scat3 = sb.scatterplot(data = windowsIndexes[2])
    figg3.show()

# print statictics regarding union and intersection
def print_rip(windowsIndexes) :

    c1l = len(windowsIndexes[0])
    c2l = len(windowsIndexes[1])
    c3l = len(windowsIndexes[2])
    tl  = c1l + c2l + c3l
    rip1 = (windowsIndexes[0] + windowsIndexes[1]).unique()
    rip2 = (windowsIndexes[0] + windowsIndexes[2]).unique()
    rip3 = (windowsIndexes[1] + windowsIndexes[2]).unique()

    print("Length of Union C1 and C2: ", c1l+c2l - len(rip1) + 1)
    print("Length of Intersect C1 and C2: ", len(rip1) -1)
    print("Union Percent:", (c1l+c2l - len(rip1) + 1)/ tl)
    print("Intersect Percent:", (len(rip1) -1)/ tl, "\n")

    print("Length of Union C1 and C3:", c1l+c3l - len(rip2) + 1)
    print("Length of Intersect C1 and C3: ", len(rip2) -1)
    print("Union Percent:", (c1l+c3l - len(rip2) + 1)/ tl)
    print("Intersect Percent:", (len(rip2) -1)/ tl, "\n")

    print("Length of Union C2 and C3: ", c2l+c3l - len(rip3) + 1)
    print("Length of Intersect C2 and C3: ", len(rip3) -1)
    print("Union Percent:", (c2l+c3l - len(rip3) + 1)/ tl)
    print("Intersect Percent:", (len(rip3) -1)/ tl, "\n")

# print the sections compared to each other
def section_print(tmp) :

    result1 = pd.concat([tmp.loc[:, 'C1'], tmp.loc[:, 'C2']], axis=1, join='inner')
    result2 = pd.concat([tmp.loc[:, 'C1'], tmp.loc[:, 'C3']], axis=1, join='inner')
    result3 = pd.concat([tmp.loc[:, 'C2'], tmp.loc[:, 'C3']], axis=1, join='inner')
    print(result1)
    print(result2)
    print(result3)

if __name__ == '__main__':

    # importing the csvs.
    clusters = pd.read_csv(DATADIR + _CLUSTERS_CSV)
    clusters = rmv_nans(clusters)
    df       = pd.read_csv(DATADIR + _ORIGINAL_CSV)
    features = pd.read_csv(DATADIR + _FEATURE_CSV)

    df           = df.iloc[:, 1:]       # Remove unamed section
    window_names = features.iloc[:, 0]

    hist1Feature = "Hist1"
    ladFeature   = "LAD"

    # Windows == Feature selection
    hist1 = features.loc[:, hist1Feature]
    lad   = features.loc[:, ladFeature]

    # getting the clusters sections regarding the hist1 and lad 1 locations
    hist1Sums = feature_sum(hist1, clusters, df)
    ladSums   = feature_sum(lad, clusters, df)

    # Display box plots of the features for each cluster
    show_boxplot(hist1Sums, hist1Feature)
    show_boxplot(ladSums, ladFeature)

    # Percentage of NPs in each positional catagory
    #cluster_bar_graphs(clusters, df)

    windows = []

    # for each cluster find the windows that fall under a certain threshold
    # In this case 65-75 percent 
    for c in clusters :
        tmp = clust_wind(c, df)
        windows.append(tmp)

    # print the 3 clusters and their windows
    windowsIndexes = get_contenders(windows)
    for c in range( len(clusters ) ) :
        windowsIndexes[c] = window_names[windowsIndexes[c]]
        print("Cluster {0}\n{1}".format(c+1, windowsIndexes[c]))


    tmp = pd.DataFrame(windowsIndexes)
    scatter_plots(windowsIndexes)
    
    tmp = tmp.T
    tmp.columns = ['C1', 'C2', 'C3']

    # print the 3 clusters overtop of eachtoher
    print(tmp, "\n")
    print_rip(windowsIndexes)
    
    # print the cluster 1 overlapped with cluster 2, cluster 1 and 3, and 2 and 3.
    #section_print(tmp)
    input()
