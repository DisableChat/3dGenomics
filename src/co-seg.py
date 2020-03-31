from np import *

_ORIGINAL_CSV = "/clust_array_original_df.csv"

#******************************************************************************
# Function:     calculate_detection_frequency()
# Parameteres:  dataFrame - df that contains the windows and Nuclear profiles
# Description:  simply calculate the detection frequency of each genomic window
# Return Val:   frequency - Array of the genomic windows frequencies
#******************************************************************************
def calculate_detection_frequency(dataFrame):

    npCount   = (dataFrame == 1).sum(axis = 1)
    numOfNPS  = dataFrame.shape[1]
    frequency = npCount/numOfNPS

    return frequency

#******************************************************************************
# Function:     calculate_normalized_linkage()
# Parameteres:  dataFrame - df that contains the windows and Nuclear profiles
#               frequency - An array of all the genomic windows detection 
#                           frequencies
# Description:  The purpose of this function has 3 main parts:
#               1) calculate the linkage between two genomic windows
#               2) determine the theoretical maximum of that found linkage
#               3) calculate the normalized linkage value and append this
#                  result to the matrix of normalized linkage.
# Return Val:   nlm - normalized linkage matrix
#******************************************************************************
def calculate_normalized_linkage(dataFrame, frequency):

    windowCount   = dataFrame.shape[0] # Number of loci or windows
    numOfNPS      = dataFrame.shape[1] # Number of NPs
    nlm           = []

    for locusOne in range(windowCount) :
        rowArray = []

        for locusTwo in range(windowCount) :

            # Locus 1 and 2 variables for co-segregation calculation
            l1 = np.where(dataFrame.iloc[locusOne, :] == 1)
            l2 = np.where(dataFrame.iloc[locusTwo, :] == 1)

            coSeg = np.intersect1d(l1[0], l2[0]) # Finding the intersection vals
            coSegLen = len(coSeg)                # Determine # of shared dectections
            matchSum = coSegLen / numOfNPS       # co-segregaition /  NPs

            # Determine linkage value
            linkage  = matchSum - (frequency[locusOne] * frequency[locusTwo])

            # Normalizng linkage disequilibrium (linkage / theoretical max)
            # Where:
            # Linkage = D = fab - fafb 
            # Dmax    = { min < fa*fb, (1-fa)(1-fb)> when D < 0
            #           { min < fb*(1-fa), fa(1-fb)> when D > 0
            if linkage < 0 :
                v1   = frequency[locusOne] * frequency[locusTwo]
                v2   = (1 - frequency[locusOne]) * (1 - frequency[locusTwo])
                dmax = min(v1, v2)
            elif linkage > 0 :
                v1   = frequency[locusTwo] * (1 - frequency[locusOne])
                v2   = frequency[locusOne] * (1 - frequency[locusTwo])
                dmax = min(v1, v2)
            else :
                dmax = 1 # Prevents undefined error if linkage = 0 case.

            normalizedLinkage = linkage / dmax
            rowArray.append(normalizedLinkage)

        nlm.append(rowArray)

    return nlm

#******************************************************************************
# Function:     __main__
# Parameteres:  None
# Description:  To run the main of the program to determine the normalized
#               linkage table of loci.
# Return Val:   None
#******************************************************************************
if __name__ == '__main__':

    df = pd.read_csv(DATADIR + _ORIGINAL_CSV) # importing csvs
    df = df.iloc[:, 1:-3]                     # Remove unamed secs and clusters

    # Determine frequencies of the windows and calculate the normalized linkage table
    frequency = calculate_detection_frequency(df)
    normalizedLinkageMatrix = calculate_normalized_linkage(df, frequency)

    # Creating panda df and saving the table to csv
    normalizedLinkageMatrix = pd.DataFrame(normalizedLinkageMatrix)
    normalizedLinkageMatrix.to_csv(DATADIR + "/normalized-linkage-table.csv")
    
    # Creating a heatmap to give a visual representation
    fig     = plt.figure("Normalized Linkage")
    heatmap = sb.heatmap(normalizedLinkageMatrix, vmin = 0, vmax = 1, cmap="YlGnBu")
    plt.show()

