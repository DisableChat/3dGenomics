from np import *

_ORIGINAL_CSV = "/clust_array_original_df.csv"
_FEATURE_CSV  = "/gam_feature_community.csv"
_CLUSTERS_CSV = "/clust_array.csv"

def rmv_nans(clusters):

    clusters = clusters.fillna(0)
    clusters = np.array(clusters.iloc[:,1:])
    tmp      = []

    for c in clusters :
        c = c[c != 0]
        tmp.append(c)

    return tmp

def feature_sum(feat, clusters, df) :

    numClusters = len(clusters)
    featureSum  = []
    featSumMax  = feat.sum()  

    for i in range(numClusters) :
        featSum = df.loc[feat == 1, clusters[i]]
        featSum = featSum.sum(axis = 0)
        featureSum.append(featSum)

    return featureSum

def show_boxplot(figData, figName) :
    
    fig = plt.figure(figName)
    img = sb.boxplot(data=figData)

    fig.show()

def radial_pos(posCatagory, df) :

    tmp = []

    for i in posCatagory:
        catagoryNP  = df.columns[i].tolist()
        catagorySum = np.sum(df.loc[:, catagoryNP])

        tmp.append(catagorySum)

    return tmp

if __name__ == '__main__':

    clusters = pd.read_csv(DATADIR + _CLUSTERS_CSV)
    clusters = rmv_nans(clusters)
    df       = pd.read_csv(DATADIR + _ORIGINAL_CSV)
    features = pd.read_csv(DATADIR + _FEATURE_CSV)
    
    hist1Feature = "Hist1"
    ladFeature   = "LAD"

    hist1 = features.loc[:, hist1Feature]
    lad   = features.loc[:, ladFeature]

    #df = pd.concat([df, hist1], axis = 1)
    #df = pd.concat([df, lad],   axis = 1)

    hist1Sums = feature_sum(hist1, clusters, df)
    ladSums   = feature_sum(lad, clusters, df) 

    show_boxplot(hist1Sums, hist1Feature)
    show_boxplot(ladSums, ladFeature)
    input()

    '''
    print(len(clusters))
    hist1Sums = df.loc[hist1 == 1, clusters[0]]
    print(hist1Pos.sum(axis = 0))
    '''
