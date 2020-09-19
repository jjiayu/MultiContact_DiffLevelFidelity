import pickle

import sys

FilePath = sys.argv[1]

from PlotResult import *

result=pickle.load(open(FilePath,"rb"))

Plot_RHP_result(NumRounds = result["StopRound"], SwingLeftFirst = result["SwingLeftFirst"], SwingRightFirst = result["SwingRightFirst"], x_fullres = result["x_fullres"], y_fullres = result["y_fullres"], z_fullres = result["z_fullres"], PL_init_fullres = result["PL_init_fullres"], PR_init_fullres = result["PR_init_fullres"], Px_fullres = result["Px_fullres"], Py_fullres = result["Py_fullres"], Pz_fullres = result["Pz_fullres"], AllSurfaces = result["TerrainModel"])