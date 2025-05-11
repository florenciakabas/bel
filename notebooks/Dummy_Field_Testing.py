# This code is a Python version of Liwei's Matlab code for dummy field testing.

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from datetime import datetime
import argparse

def str2bool(value):
    if isinstance(value, str):
        if value.lower() in ('yes', 'true', 't', '1'):
            return True
        elif value.lower() in ('no', 'false', 'f', '0'):
            return False
    raise argparse.ArgumentTypeError('Boolean value expected.')

def main(show=False):
    # Feature Settings
    # thickness 50~200 ft
    # porosity 0.02 ~ 0.08
    # perm 0.1 ~ 0.5
    # toc 2 ~ 8
    # sw 0.3 ~ 0.7
    # dep 8000 ~ 11000
    # vlcy 0.3 ~ 0.6

    np.random.seed(1) # Seed is different in Matlab and Python

    n = 200
    nw = 500
    x = np.linspace(-94.6, -92.9, n)
    y = np.linspace(31.3, 33, n)
    X, Y = np.meshgrid(x, y)

    # Synthetic Data Generation
    # Translate the Matlab peaks function to Python
    def peaks(x, y):
        return (3 * (1 - x)**2 * np.exp(-(x**2) - (y + 1)**2)
                - 10 * (x / 5 - x**3 - y**5) * np.exp(-x**2 - y**2)
                - 1/3 * np.exp(-(x + 1)**2 - y**2))

    def peaks_linspace(x0, x1, n):
        x = np.linspace(x0, x1, n)
        y = np.linspace(x0, x1, n)
        X,Y = np.meshgrid(x, y)
        res = peaks(X, Y)
        return res

    def peaks_num(n):
        x = np.linspace(-3, 3, n)
        y = np.linspace(-3, 3, n)
        X,Y = np.meshgrid(x, y)
        Z = peaks(X, Y)
        return Z

    # Normalization function 
    def min_max_norm(x):
        return (x - np.min(x)) / (np.max(x) - np.min(x))

    # Thickness
    gthk = peaks_linspace(-2, 2, n)
    gthkn = min_max_norm(gthk)
    gthk = gthkn * 150 + 50

    # Porosity
    gphi = peaks_linspace(-1 ,1, n)
    gphin = min_max_norm(gphi)
    gphi = gphin * 0.06 + 0.02

    # Perm
    gpern = gphin.T
    gper = gpern * 0.4 + 0.1

    # TOC
    gtoc = peaks_linspace(-0.5, 0.5, n)
    gtocn = min_max_norm(gtoc)
    gtoc = gtocn * 6 + 2

    # SW
    gswn = np.fliplr(gtocn)
    gsw = gswn * 0.4 + 0.3

    # Depth
    gdep = peaks_linspace(-0.2, 0.2, n)
    gdepn = min_max_norm(gdep)
    gdep = gdepn * (-3000) + (-8000)

    # Vclay
    tmp = peaks_num(2 * n)
    gvly = tmp[50 : (n + 50), 50 : (n + 50)]
    gvlyn = min_max_norm(gvly)
    gvly = gvlyn * 0.3 + 0.3

    # Plot contours
    fig, axes = plt.subplots(2, 4, figsize = (20, 10))
    titles = ['Thickness', 'Porosity', 'Perm', 'TOC', 'SW', 'Vclay', 'Depth']
    data_list = [gthk, gphi, gper, gtoc, gsw, gvly, gdep]
    for i, ax in enumerate(axes.flat[:7]):
        cs = ax.contourf(X, Y, data_list[i], cmap='plasma')
        ax.set_title(titles[i])
        plt.colorbar(cs, ax=ax)
    plt.tight_layout()
    if show:
        plt.show()

    # Random Sampling
    xind = np.random.randint(0, n, nw)
    yind = np.random.randint(0, n, nw)
    
    samples = {
        'xx': X[xind, yind],
        'yy': Y[xind, yind],
        'wdep': gdepn[xind, yind],
        'wtoc': gtocn[xind, yind],
        'wvly': gvlyn[xind, yind],
        'wthk': gthkn[xind, yind],
        'wsw': gswn[xind, yind],
        'wper': gpern[xind, yind],
        'wphi': gphin[xind, yind],
    }

    # Synthetic Production Simulation
    np.random.seed(1)
    tt_year= np.int32(np.ceil(np.linspace(2008,2024,nw)))
    tt_month = np.int32(np.random.randint(1,13,size=(nw,)))

    t_list = []
    w_list = []
    for i in range(nw):
        tmpnm = 1.7e9 + (i+1) * 1e6  

        # Current year
        tmpm1 = np.array([mm for mm in range(tt_month[i], 13)])
        tmpy1 = np.array([tt_year[i] for _ in range(tmpm1.size)])

        tmp1_size = tmpm1.size
        tmpm1 = tmpm1.reshape((tmp1_size,1))
        tmpy1 = tmpy1.reshape((tmp1_size,1))

        tmp1 = np.concat([tmpy1, tmpm1],axis=1)

        # To 2024 
        if tt_year[i] < 2024:
            tmpy2 = np.concat([np.array([mm]*12) for mm in range(tt_year[i]+1, 2025)])
            tmpm2 = np.concat([np.array([mm for mm in range(1,13)]) for _ in range(tt_year[i]+1, 2025)])

            tmp2_size = tmpm2.size
            tmpy2 = tmpy2.reshape((tmp2_size,1))
            tmpm2 = tmpm2.reshape((tmp2_size,1))

            tmp2 = np.concat([tmpy2, tmpm2],axis=1)
        
            tmp = np.concat([tmp1, tmp2], axis=0)
        else:
            tmp = tmp1

        t_list.append(tmp)

        w_list.append(np.ones(shape=(tmp.shape[0],1)) * tmpnm)
        

    t_list = np.concat(t_list,axis=0)
    t_list = np.concat([t_list,np.ones((t_list.shape[0],1))],axis=1)
    w_list = np.concat(w_list,axis=0)
    prod = np.concat([w_list,t_list],axis=1)

    t = np.int32(t_list)
    w = w_list

    ww = np.unique(w) 

    wt = [200, 250, 400, 50, 25, 5, 1]

    df_samples = pd.DataFrame(samples)
    syn = (df_samples['wvly'].to_numpy().reshape(nw,1)*wt[0] + df_samples['wthk'].to_numpy().reshape(nw,1)*wt[1] + df_samples['wdep'].to_numpy().reshape(nw,1)*wt[2] +
        df_samples['wsw'].to_numpy().reshape(nw,1)*wt[3] + df_samples['wphi'].to_numpy().reshape(nw,1)*wt[4] + df_samples['wtoc'].to_numpy().reshape(nw,1)*wt[5] + 
        df_samples['wper'].to_numpy().reshape(nw,1)*wt[6])

    qi=syn*1e3
    b=0.85
    Di=0.01

    prod = np.concat([prod, np.zeros((prod.shape[0],1))],axis=1)

    for i in range(nw):
        
        idx = np.where(w[:,0] == ww[i])[0]

        tmpt = []
        for ii in idx:
            tmpt.append(datetime(t[ii,0],t[ii,1],1))

        tt_tmp = np.arange(1,30*idx.size+1,30)
        tt_tmp = tt_tmp.reshape((tt_tmp.size,1))
        qt = np.round(qi[i,0]/((1+b*Di*tt_tmp)**(1/b)))
        qt[qt<1] = 1
        prod[idx,4] = qt[:,0]

        plt.semilogy(tmpt,qt[:,0])
    plt.grid()
    plt.xlabel('Time')
    plt.ylabel('Monthly Gas Production (mcf)')
    plt.title('Synthetic Production Series')

    # Correlation
    df_full = pd.DataFrame({
        'x': samples['xx'],
        'y': samples['yy'],
        'Depth': samples['wdep'],
        'TOC': samples['wtoc'],
        'Vclay': samples['wvly'],
        'Thick': samples['wthk'],
        'SW': samples['wsw'],
        'Perm': samples['wper'],
        'Phi': samples['wphi'],
        'qi': qi.flatten(),
    })

    df_corr = df_full.drop(columns=['x', 'y'])

    # Plotting
    plt.figure(figsize=(10, 8))
    sns.heatmap(df_full.corr(), annot=True, cmap='coolwarm')
    plt.title('Correlation Heatmap')
    # plt.show()
    if show:
        plt.show()

    # Bubble Plot  Fixed size scatter over contour
    plt.figure(figsize=(10, 8))
    plt.contour(X, Y, gthk, cmap='plasma')
    plt.scatter(samples['xx'], samples['yy'], s=30, c='black')
    plt.title('Thickness Contours with Fixed-Size Scatter Points')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.colorbar(label='Thickness (ft)')
    # plt.show()
    if show:
        plt.show()

    # Bubble Plot - Bubble sizes proportional to qi
    plt.figure(figsize=(10, 8))
    plt.contour(X, Y, gthk, cmap='plasma')

    bubble_size = (qi - qi.min()) / (qi.max() - qi.min()) * 100 + 10
    plt.scatter(samples['xx'], samples['yy'], s=bubble_size, c='black')

    plt.title('Thickness Contours with Bubble Plot (qi-scaled)')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.colorbar(label='Thickness (ft)')
    # plt.show()
    if show:
        plt.show()

    # Full Simulation Data Output
    return df_full

if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Run the Dummy Field Testing script.")
    parser.add_argument("--show", type=str2bool, required=True, help="Display plots if set to True.")
    parsed = parser.parse_args()

    # Save the DataFrame to a CSV file
    df_full = main(show=parsed.show)