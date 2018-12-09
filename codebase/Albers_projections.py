# Authored by Shayan Amani
# @SHi-ON
# 11/21/2018

from mpl_toolkits.basemap import Basemap
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd

# in case of PROJ_LIB error related to Proj4 package, set the following Environment variable:
# PROJ_LIB = "~/anaconda3/envs/<env_name>/share/proj"

# shift offset to project data points in the actual area (US region).
OFFSET_LAT = 23
OFFSET_LON = -96


def data_prep():
    df = pd.read_csv("./datasets/barberry_sim_NewEngland.csv")
    num_sub_samples = 600
    # Pick random sub-samples
    sub_samples = df.sample(n=num_sub_samples)
    # lats = sub_samples['lat'].values / 100000 + OFFSET_LAT
    # lons = sub_samples['lon'].values / 100000 + OFFSET_LON
    lats = sub_samples['lat'].values
    lons = sub_samples['lon'].values
    alts = sub_samples['N'].values
    return lats, lons, alts


def bm_plotter(lats, lons, alts):
    # setup Albers Equal Area conic projection basemap
    # lat_1 is first standard parallel.
    # lat_2 is second standard parallel.
    # lon_0,lat_0 is central point.
    m = Basemap(width=10000000, height=8000000,
                resolution='l', projection='aea', \
                lat_1=29.5, lat_2=45.5, lat_0=23, lon_0=-96)
    m.drawcoastlines()
    m.drawcountries()
    m.fillcontinents(color='grey', lake_color='aqua')
    # draw parallels, meridians and states boundaries
    m.drawparallels(np.arange(-80., 81., 20.), labels=[1, 0, 0, 0])
    m.drawmeridians(np.arange(-180., 181., 20.), labels=[0, 0, 0, 1])
    m.drawmapboundary(fill_color='aqua')
    m.drawstates()

    ax = plt.gca()

    # format colors for elevation range
    alt_min = np.min(alts)
    alt_max = np.max(alts)
    cmap = plt.get_cmap('gist_earth')
    normalize = matplotlib.colors.Normalize(vmin=alt_min, vmax=alt_max)

    for ii in range(0, len(alts)):
        x = lons[ii] + m(OFFSET_LON, OFFSET_LAT)[0]
        y = lats[ii] + m(OFFSET_LON, OFFSET_LAT)[1]
        # x, y = m(lons[ii], lats[ii], inverse=True)
        color_interp = np.interp(alts[ii], [alt_min, alt_max], [50, 200])
        plt.plot(x, y, marker='o', alpha=0.5, color=cmap(int(color_interp)))

    # format the colorbar
    cax, _ = matplotlib.colorbar.make_axes(ax)
    cbar = matplotlib.colorbar.ColorbarBase(cax, cmap=cmap, norm=normalize, label='Presence Count')

    plt.title("Barberry Presence\nAlbers Equal Area Projection")
    plt.savefig('./figs/Albers_projection_Barberry.jpg', format='jpg', dpi=500)
    plt.show()


if __name__ == "__main__":
    latitudes, longitudes, altitudes = data_prep()
    bm_plotter(latitudes, longitudes, altitudes)
