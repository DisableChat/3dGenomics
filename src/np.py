import pandas as pd
import numpy  as np

#df = pd.read_csv("../data/gam.txt", sep = '\t', header = None)
df = pd.read_csv("../data/gam.txt", sep = '\t', low_memory=False)

gs = df.shape[0] - 1
ns = df.shape[1] - 3

#win = df.iloc[1:, 3:] # Use if using header = None
win = df.iloc[:, 3:]

def windows(win, axe):

    avgNP = (win == 1).sum(axis = axe)
    print(avgNP)

    avg = np.mean(avgNP, axis = 0)
    print(avg)

    largest = np.amax(avgNP)
    minum   = np.amin(avgNP)

    print(largest)
    print(minum)

def est_rad(win, axisXY, PB, PE):

    avgNP = (win == 1).sum(axis = axisXY)

    sec1    = np.argwhere(avgNP > np.percentile(avgNP, PB))
    sec2    = np.argwhere(avgNP <= np.percentile(avgNP, PE))

    result  = np.intersect1d(sec1, sec2)

    return result

def calc_percentiles(win, axis, divisor):

    vals = []
    percent = round(100/divisor)

    for i in range(0, 100, percent):
        vals.append(est_rad(win, axis, i, i+percent))

    return vals

if __name__ == '__main__':

    #print("The number of GW = ", gs)
    #print("The number of NP = ", ns)

    #windows(win, 1)
    #windows(win, 0)

    # Obtain the indices of the 5 percentiles in the nuclear profile's
    # *Note* older way / more repitition
    npP1 = est_rad(win, 0,  0,  20)
    npP2 = est_rad(win, 0, 20,  40)
    npP3 = est_rad(win, 0, 40,  60)
    npP4 = est_rad(win, 0, 60,  80)
    npP5 = est_rad(win, 0, 80, 100)

    NP_percentiles = calc_percentiles(win, 0, 5)
    CH_percentiles = calc_percentiles(win, 1, 10)

    for i in NP_percentiles:
        print(i)

    for i in CH_percentiles:
        print(i)
