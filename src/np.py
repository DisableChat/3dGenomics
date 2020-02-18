import pandas            as pd
import numpy             as np
import seaborn           as sb
import matplotlib.pyplot as plt

from   typing       import List
from   numpy.random import default_rng

# NOTE Primary reason for ignoring future warning:
# Series.nonzero() deprecation warning #900
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# Min value of NPs needed for rm_np_noise method
MIN_NP       = 20

# Number of cluster sets
_NUM_OF_SETS = 20

FDIR    = "../data/gam.txt"  # File directory for data
OUTFDIR = "../output"        # Output file directory
Vector  = List[int]          # Vector containing type int

#******************************************************************************
# Function:     windows()
# Parameters:   win - Matrix containg the target data
#               axe - Target axis
# Description:  given a matrix and axis value, 0 being the column's and 1 being
#               rows. Find the sum of each vector in the matrix, then find the
#               mean, largest and min value for these sums.
# Return val:   None
#******************************************************************************
def windows(win, axe) -> None:

    avgNP = (win == 1).sum(axis = axe)

    largest = np.amax(avgNP)
    minum   = np.amin(avgNP)
    avg     = np.mean(avgNP, axis = 0)

    mma = (largest, minum, avg)

    return mma

#******************************************************************************
# Function:     rm_chr_noise()
# Parameters:   win     - current df window
#               gw_name - names of the GWs
# Description:  Remove GW noise
# Return Val:   win     - updated df window
#               gw_name - updated gw_name values
#******************************************************************************
def rm_chr_noise(win, gw_name) :

    avgnp = (win == 1).sum(axis = 1)
    noise = avgnp[avgnp >= 100]

    win.drop((noise.index), inplace = True)

    gw_name = np.delete(gw_name, noise.index, axis = 0)

    return win, gw_name

#******************************************************************************
# Function:     rm_np_noise()
# Parameters:   win     - current df window
# Description:  remove the noise from NP section
# Return Val:   win     - updated df window
#               np_name - updated NPs names
#******************************************************************************
def rm_np_noise(win) :

    columns = win.columns
    avgnp   = (win == 1).sum(axis = 0)
    noise   = avgnp[avgnp < MIN_NP]

    win.drop((noise.index), inplace = True, axis = 1)

    np_name = win.columns

    return win, np_name

#******************************************************************************
# Function:     partition()
# Parameters:   win     - Matrix containing NP's and Chromisomes
#               axisXY  - axis on which to caluclate the percentile indices
#               PB      - Percentile begin value
#               PE      - Percentile end value
# Description:  Partition returns a section of values given percentile range
#               (i.e. percentile values 20%-40%)
# Return val:   Vector  - The partitioned section of all the ranked data where
#                         each of the values in the partitioned vector contains
#                         the value of the index location in the original list
#                         of  NP's or Chromosomes (axis 0, 1)
#******************************************************************************
def partition(win, axisXY, PB, PE) -> Vector :

    avgNP = (win == 1).sum(axis = axisXY)

    begin   = np.where(avgNP > np.percentile(avgNP, PB))
    end     = np.where(avgNP <= np.percentile(avgNP, PE))
    result  = np.intersect1d(begin, end)

    return result

#******************************************************************************
# Function:     calc_percentiles()
# Parameters:   win     - Matrix of target data
#               axis    - Axis in which to calculate percentiles on
#               divisor - Number of sections on target vector
# Description:  Determine which percentiles each NP's or Chromosomes values
#               fall into.
# Return val:   Vector  - A vector containing vectors of each percentile range
#                         based on the number of sections there are (divisor val)
#******************************************************************************
def calc_percentiles(win, axis, divisor) -> Vector :

    vals = []
    percent = round(100/divisor)

    for i in range(0, 100, percent):
        vals.append(partition(win, axis, i, i+percent))

    return vals

def display_avgs(win) -> None:

    # Nuclear profile and genomic window variables. These vars are tuples
    # containing the max, min and average for the win dataframe axis.
    npMMA = windows(win, 1)
    gwMMA = windows(win, 0)
    gws   = win.shape[0]    # Number of genomic windows
    nps   = win.shape[1]   # Number of nuclear profiles

    # Display the number of NP and GW in the current dataframe
    print("The number of Nuclear Profiles = ", nps)
    print("The number of Genomic Windows  = ", gws)

    # Display the NP and GW max, min and avg from a dataframe
    print("Nuclear Profiles: Max {} Min {} Avg {}"
            .format(npMMA[0], npMMA[1], npMMA[2]))
    print("Genomic Windows: Max {} Min {} Avg {}\n"
            .format(gwMMA[0], gwMMA[1], gwMMA[2]))

# Jaccard similarity coefficient section
def jaccard_index(s1, s2):

    sec1         = np.where(s1 == 1)            # Sec1 NP's window
    sec2         = np.where(s2 == 1)            # Sec2 NP's window

    intersect    = np.intersect1d(sec1, sec2)   # What chromosomes intersect
    union        = np.union1d(sec1, sec2)       # The union of the two sets

    intersectLen = len(intersect)               # Length of the intersect
    unionLen     = len(union)                   # Len of the Union

    # Normalizing the data - to avoid skewing results to difference in windows
    # dected by 2 Nuclear Profiles
    sec1Len        = np.size(sec1)                  # Nuclear profile sec1
    sec2Len        = np.size(sec2)                  # Nuclear Profile sec2
    minCardinality = np.minimum(sec1Len, sec2Len)   # Minimum of sec1 and sec2

    jaccardIndex = round((intersectLen / minCardinality), 4)

    return jaccardIndex

def jaccard_distance(jaccardIndex):

    jaccardDistance = 1 - jaccardIndex

    return jaccardDistance

def jaccard_heatmap(win):

    jaccardIndexY    = []
    jaccardDistanceY = []

    for i in range(win.columns.size):
        jaccardIndexX    = []
        jaccardDistanceX = []

        s1 = win.iloc[:, i]

        for j in range(win.columns.size):
            s2 = win.iloc[:,j]

            jaccardIndex   = jaccard_index(s1, s2)
            jacardDistance = jaccard_distance(jaccardIndex)

            jaccardIndexX.append(jaccardIndex)
            jaccardDistanceX.append(jacardDistance)

        jaccardIndexY.append(jaccardIndexX)
        jaccardDistanceY.append(jaccardDistanceX)

    return jaccardIndexY, jaccardDistanceY

def np_cluster_assignment(nda, numOfClusters, np_names):

    nearest = []

    for i in range(len(nda) - numOfClusters) :
        cluster = nda[i][-numOfClusters:]
        loc = np.argmin(cluster)
        nearest.append(loc)

    #print(np_names)
    nearest = pd.DataFrame(nearest,
                            index = np_names[:-numClusters],
                            columns = ['Cluster Index'])
    return nearest

def create_heatmap_visual(ndarray, ylabel, xlabel, csvName, pngName):

    ji = pd.DataFrame(ndarray, index = ylabel, columns = xlabel)
    ji.to_csv(OUTFDIR + csvName)

    heat_map = sb.heatmap(ndarray,
                            xticklabels = xlabel,
                            yticklabels = ylabel,
                            cmap="YlGnBu")

    plt.savefig(OUTFDIR + pngName)

    #plt.show()
    plt.clf()
    plt.close()

def find_medoid(npClusters, numClusters, ji, win) :

    oldclusters = win.iloc[:, -numClusters:].values.tolist()

    for i in range (numClusters) :

        clusterName = win.columns[win.shape[1] - numClusters + i]
        #print("CLUSTER ", clusterName, "\n")
        nps = npClusters.loc[npClusters['Cluster Index'] == i].index.tolist()
        #print("NPS\n", nps)

        npSimSum = ji.loc[nps, nps].sum()
        npSimSumMax = np.argmax(npSimSum)

        mediodNP = win.loc[:, nps[npSimSumMax]]

        #print("new CLust" , mediodNP.values.tolist())
        oldClust = win.loc[:, clusterName].copy()
        #print("old clust", oldClust.values.tolist())
        win.loc[:, clusterName] = mediodNP.copy()
        #print("\n old vs new clust same?", oldClust.values.tolist() == win.loc[:, clusterName].values.tolist())

    newclusters = win.iloc[:, -numClusters:].values.tolist()

    redo = oldclusters == newclusters
    #print("\n old vs new cluster sets same? ", redo)

    return win, redo

def find_variance(npClusters, numClusters, jd) :

    total_variance = 0
    cluster_nps = []

    for i in range (numClusters) :

        clusterName = win.columns[win.shape[1] - numClusters + i]
        nps = npClusters.loc[npClusters['Cluster Index'] == i].index.tolist()

        summ = jd.loc[nps, nps].sum()
        summavg = jd.loc[nps, nps].sum() / len(nps)
        sqr = (summ - summavg) * (summ - summavg)

        variance = sqr.sum()/len(nps)
        total_variance += variance

        cluster_nps.append(nps)

    return total_variance, cluster_nps

def clust_array_heatmap(gw_name, clusterNps, win) :

    xticks = gw_name[:, 1:].tolist()

    clusterOne   = win.loc[:, clusterNps[0]].T
    clusterTwo   = win.loc[:, clusterNps[1]].T
    clusterThree = win.loc[:, clusterNps[2]].T

    fig, (ax, ax2, ax3) = plt.subplots(nrows = 3, sharex = True)

    clusterOneMap   = sb.heatmap(clusterOne, ax=ax, xticklabels = xticks, yticklabels = clusterOne.index, cbar = False, cmap="YlGnBu")
    clusterTwoMap   = sb.heatmap(clusterTwo, ax=ax2, xticklabels = xticks, yticklabels = clusterTwo.index, cbar = False, cmap="YlGnBu")
    clusterThreeMap = sb.heatmap(clusterThree, ax=ax3, xticklabels = xticks, yticklabels = clusterThree.index, cbar = False, cmap="YlGnBu")

    ax.tick_params(axis  = 'y', colors = 'red')
    ax2.tick_params(axis = 'y', colors = 'blue')
    ax3.tick_params(axis = 'y', colors = 'green')
    ax.tick_params(axis  = 'x', length = .001)
    ax2.tick_params(axis = 'x', length = .001)
    ax3.tick_params(axis = 'x', labelsize = 8)

    fig.subplots_adjust(hspace=0.02, bottom = .2)
    ax.set_title("Clusters")
    fig.colorbar(ax.collections[0], ax = [ax, ax2, ax3])

    plt.show()
    plt.clf()

#******************************************************************************
# Function:     main()
# Parameters:   None
# Description:  To run the main program kappa
# Return Val:   None
#******************************************************************************
if __name__ == '__main__':
    rng = default_rng()

    # Import genome data set
    df = pd.read_csv(FDIR, sep = '\t', low_memory=False)

    win      = df.iloc[:, 3:] # numpy matrix of data (first 3 cols are not NP's)
    np_names = np.array(win.columns.values)
    gw_name  = np.array(df.iloc[:, :3])

    # Remove chromisome noise
    win, gw_name = rm_chr_noise(win, gw_name)

    # Display avgs for the dataframe
    display_avgs(win)

    # Estimated radial possition of a NP 1-5 where 1 - strongly apical,
    # 3 - neither apical or equatorial, and 5 - strongly equatorial.
    NP_percentiles = calc_percentiles(win, 0, 5)

    print("NP Radial Positioning:")
    for i in NP_percentiles:
        print(np_names[i])

    # Compaction of each genomic window. Degree of compaction rating 1-10
    # (where 10 is most condensed and 1 is least condensed
    GW_percentiles = calc_percentiles(win, 1, 10)

    print("GW compaction:")
    for i in GW_percentiles:
        print(gw_name[i])

    chromosome = np.where(gw_name[:,0] == 'chr13')
    gw_name_index = chromosome[0].tolist()
    gw_name_index = gw_name_index[0]

    chromosome = gw_name[chromosome[0], 0:]
    chromosome.tolist()

    chrIndexBegin = np.where(chromosome[:,2] >=  21700000)
    chrIndexEnd   = np.where(chromosome[:,1] <= 24100000)
    result        = np.intersect1d(chrIndexBegin, chrIndexEnd)

    gw_name = gw_name[gw_name_index+result, :]
    result  = result + gw_name_index
    win     = win.iloc[result, :]

    win, np_name = rm_np_noise(win)

    display_avgs(win)

    # K cluster section
    random_clusters = pd.DataFrame(np.random.randint(0,2,size=(81, 3)),
                                    index = win.index,
                                    columns=list(["K1", "K2", "K3"]))

    numClusters = random_clusters.shape[1]
    win = pd.concat([win, random_clusters], axis = 1)
    np_name = win.columns  # Updating win columns

    # Jaccard Section
    jaccardIndex    = []
    jaccardDistance = []

    jaccardIndex, jaccardDistance = jaccard_heatmap(win)

    create_heatmap_visual(jaccardIndex,
                            np_name,
                            np_name,
                            "/jaccard_index.csv",
                            "/jaccard_index_heat_map.png")

    create_heatmap_visual(jaccardDistance,
                            np_name,
                            np_name,
                            "/jaccard_distance.csv",
                            "/jaccard_distance_heat_map.png")

    varianceArray  = []
    clusterNpArray = []
    winArray       = []
    numberOfSets = 100

    setCount = 0
    for i in range(numberOfSets):
        redo        = False
        itter       = 1

        while (not redo) :
            print("\nFIND MEDIOD ITTERATION: ", itter, "\n")
            ji = pd.DataFrame(jaccardIndex, index = np_name, columns = np_name)
            npClusters = np_cluster_assignment(jaccardDistance, numClusters, np_name)
            win, redo = find_medoid(npClusters, numClusters, ji, win)
            jaccardIndex, jaccardDistance = jaccard_heatmap(win)

            itter += 1

        jd = pd.DataFrame(jaccardDistance, index = np_name, columns = np_name)
        variance, carray = find_variance(npClusters, numClusters, jd)
        varianceArray.append(variance)
        clusterNpArray.append(carray)
        winArray.append(win.copy())

        # Set range for random int generation without replacement to number of
        # NPs not including clusters
        randRange = win.shape[1] - numClusters
        npCopyColumns = rng.choice(randRange, size=numClusters, replace=False)

        # Setting NPs Column name based on index values randomly generated
        npCopyColumns = np_name[npCopyColumns]

        # Copy the GWs from the randomly selected NPs, to corrosponding clusters
        npCopyValues = win.loc[:, npCopyColumns].values.tolist().copy()
        win.loc[:, np_name[-3:]] = npCopyValues

        jaccardIndex, jaccardDistance = jaccard_heatmap(win)
        setCount += 1
        print("Cluster sets created:", setCount)

    print(varianceArray)
    minClusterSetIndex = np.argmin(varianceArray)
    print("Min Jaccard distance variance arg and value", minClusterSetIndex, varianceArray[minClusterSetIndex])

    # Create and display heatmap of the new clusters
    clust_array_heatmap(gw_name,
                        clusterNpArray[minClusterSetIndex],
                        winArray[minClusterSetIndex])
