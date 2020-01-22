import pandas as pd
import numpy  as np
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
    print(avgNP)

    avg = np.mean(avgNP, axis = 0)

    largest = np.amax(avgNP)
    minum   = np.amin(avgNP)

    print(largest)
    print(minum)

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

    begin   = np.argwhere(avgNP > np.percentile(avgNP, PB))
    end     = np.argwhere(avgNP <= np.percentile(avgNP, PE))

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

#******************************************************************************
# Function:     main()
# Parameters:   None
# Description:  To run the main program kappa
# Return Val:   None
#******************************************************************************
if __name__ == '__main__':

    # Import genome data set
    df = pd.read_csv(FDIR, sep = '\t', low_memory=False)

    win = df.iloc[:, 3:] # numpy matrix of data (first 3 cols are not NP's)
    gws = win.shape[0]   # Number of genomic windows
    nps = win.shape[1]   # Number of nuclear profiles

    print("The number of Nuclear Profiles = ", nps)
    print("The number of Genomic Windows  = ", gws)

    # Estimated radial possition of a NP 1-5 where 1 - strongly apical,
    # 3 - neither apical or equatorial, and 5 - strongly equatorial.
    #np_percentiles = calc_percentiles(win, 0, 5)
    NP_percentiles = calc_percentiles(win, 0, 5)

    # Compaction of each genomic window. Degree of compaction rating 1-10
    # (where 10 is most condensed and 1 is least condensed
    GW_percentiles = calc_percentiles(win, 1, 10)

    # Displaying indicis for the NP's and GW's
    for i in NP_percentiles:
        print(i)

    for i in GW_percentiles:
        print(i)
