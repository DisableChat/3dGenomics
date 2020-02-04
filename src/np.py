import pandas  as pd
import numpy   as np
import seaborn as sb
import matplotlib.pyplot as plt
from   typing import List

# NOTE Primary reason for ignoring future warning:
# Series.nonzero() deprecation warning #900
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


FDIR    = "../data/gam.txt"  # File directory for data
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

def rm_chr_noise(win, gw_name):

    avgnp = (win == 1).sum(axis = 1)
    noise = avgnp[avgnp >= 100]

    win.drop((noise.index), inplace = True)

    gw_name = np.delete(gw_name, noise.index, axis = 0)

    return win, gw_name

def rm_np_noise(win):

    columns = win.columns
    avgnp   = (win == 1).sum(axis = 0)
    noise   = avgnp[avgnp == 0]

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
def partition(win, axisXY, PB, PE) -> Vector:

    avgNP = (win == 1).sum(axis = axisXY)
    #print(np.argwhere([avgNP] > np.percentile(avgNP,PB)))
    begin   = np.where(avgNP > np.percentile(avgNP, PB))
    end     = np.where(avgNP <= np.percentile(avgNP, PE))
    #begin   = np.argwhere([avgNP] > np.percentile(avgNP, PB))
    #end     = np.argwhere([avgNP] <= np.percentile(avgNP, PE))
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
def calc_percentiles(win, axis, divisor) -> Vector:

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

    sec1         = np.where(s1 == 1)
    sec2         = np.where(s2 == 1)

    intersect    = np.intersect1d(sec1, sec2)
    union        = np.union1d(sec1, sec2)

    intersectLen = len(intersect)
    unionLen     = len(union)

    jaccardIndex = round((intersectLen / unionLen), 4)

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

#******************************************************************************
# Function:     main()
# Parameters:   None
# Description:  To run the main program kappa
# Return Val:   None
#******************************************************************************
if __name__ == '__main__':

    # Import genome data set
    df = pd.read_csv(FDIR, sep = '\t', low_memory=False)

    win      = df.iloc[:, 3:] # numpy matrix of data (first 3 cols are not NP's)
    np_names = np.array(win.columns.values)
    gw_name  = np.array(df.iloc[:, :3])

    # Remove chromisome noise
    win, gw_name = rm_chr_noise(win, gw_name)
    print(win)
    # Display avgs for the dataframe
    display_avgs(win)

    # Estimated radial possition of a NP 1-5 where 1 - strongly apical,
    # 3 - neither apical or equatorial, and 5 - strongly equatorial.
    NP_percentiles = calc_percentiles(win, 0, 5)

    # Compaction of each genomic window. Degree of compaction rating 1-10
    # (where 10 is most condensed and 1 is least condensed
    GW_percentiles = calc_percentiles(win, 1, 10)


    chromosome = np.where(gw_name[:,0] == 'chr13')
    gw_name_index = chromosome[0].tolist()
    gw_name_index = gw_name_index[0]
    chromosome = gw_name[chromosome[0], 0:]
    chromosome.tolist()

    chrIndexBegin = np.where(chromosome[:,2] >=  21700000)
    #chrIndexBegin = chrIndexBegin[0].tolist()
    chrIndexEnd   = np.where(chromosome[:,1] <= 24100000)
    #chrIndexEnd   = chrIndexEnd[0].tolist()
    #chrIndexEnd   = np.argwhere(gw_name[chromosome, 1 <= 24100000])
    result        = np.intersect1d(chrIndexBegin, chrIndexEnd)

    print("result",result) 
    print(gw_name)
    print("chromosome" ,chromosome)
    print("begin",chrIndexBegin)
    print("gw index", gw_name_index+ result)
    gw_name = gw_name[gw_name_index+result[1], :]

    win = win.iloc[gw_name_index + result, :]

    win, np_name = rm_np_noise(win)

    display_avgs(win)

    # Jaccard Section
    jaccardIndex    = []
    jaccardDistance = []

    #print(jaccard_index(win.iloc[1:, 0], win.iloc[1:, 1]))
    jaccardIndex, jaccardDistance = jaccard_heatmap(win)

    ji = pd.DataFrame(jaccardIndex, index = np_name, columns = np_name)
    ji.to_csv("./jaccard_index.csv")

    heat_map = sb.heatmap(jaccardIndex, xticklabels = np_name, yticklabels = np_name, cmap="YlGnBu")
    plt.savefig("jaccard_index_heat_map.png")

    ji = pd.DataFrame(jaccardDistance, index = np_name, columns = np_name)
    ji.to_csv("./jaccard_distance.csv")
    
    heat_map = sb.heatmap(jaccardDistance, xticklabels = np_name, yticklabels = np_name, cmap="YlGnBu")
    plt.savefig("jaccard_distance_heat_map.png")

    plt.show()
 
    '''
    # Displaying indicis for the NP's and GW's
    for i in NP_percentiles:
        #print(i) # Indices
        print(np_names[i])

    for i in GW_percentiles:
        #print(i) # Indices
        print(gw_name[i])
    '''
