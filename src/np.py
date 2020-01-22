import pandas as pd
import numpy  as np

# NOTE Primary reason for ignoring future warning:
# Series.nonzero() deprecation warning #900
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

FDIR = "../data/gam.txt"  # File directory for data

#******************************************************************************
# Function:     windows
# Parameters:   win
#               axe
# Description: given a data frame
# Return val:
#******************************************************************************
def windows(win, axe):

    avgNP = (win == 1).sum(axis = axe)
    print(avgNP)

    avg = np.mean(avgNP, axis = 0)

    largest = np.amax(avgNP)
    minum   = np.amin(avgNP)

    print(largest)
    print(minum)

#******************************************************************************
# Function:     est_rad()
# Parameters:   win
# Description:
# Return val:
#******************************************************************************
def est_rad(win, axisXY, PB, PE):

    avgNP = (win == 1).sum(axis = axisXY)

    begin   = np.argwhere(avgNP > np.percentile(avgNP, PB))
    end     = np.argwhere(avgNP <= np.percentile(avgNP, PE))

    result  = np.intersect1d(begin, end)

    return result

#******************************************************************************
# Function:
# Parameters:
# Description:
# Return val:
#******************************************************************************
def calc_percentiles(win, axis, divisor):

    vals = []
    percent = round(100/divisor)

    for i in range(0, 100, percent):
        vals.append(est_rad(win, axis, i, i+percent))

    return vals


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
