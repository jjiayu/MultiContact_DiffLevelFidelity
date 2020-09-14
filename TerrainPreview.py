import numpy as np

from PlotResult import *
from Tools import *

AllSurfaces = []

Patch1 = np.array([[1, 0.5, 0.], [-1, 0.5, 0.], [-1, -0.5, 0.], [1, -0.5, 0.]])
AllSurfaces.append(Patch1)

Patch2 = np.array([[2.2, 0, 0.], [1, 0, 0.], [1, -0.5, 0.], [2.2, -0.5, 0.]])
AllSurfaces.append(Patch2)

Patch3 = np.array([[2.2, 0.5, 0.], [1, 0.5, 0.], [1, 0, 0.], [2.2, 0, 0.]])
AllSurfaces.append(Patch3)

Patch4 = np.array([[3.4, 0, 0.], [2.2, 0, 0.], [2.2, -0.5, 0.], [3.4, -0.5, 0.]])
AllSurfaces.append(Patch4)

Patch5 = np.array([[3.4, 0.5, 0.], [2.2, 0.5, 0.], [2.2, 0, 0.], [3.4, 0, 0.]])
AllSurfaces.append(Patch5)

Patch6 = np.array([[4.6, 0, 0.], [3.4, 0, 0.], [3.4, -0.5, 0.], [4.6, -0.5, 0.]])
AllSurfaces.append(Patch6)

Patch7 = np.array([[4.6, 0.5, 0.], [3.4, 0.5, 0.], [3.4, 0, 0.], [4.6, 0, 0.]])
AllSurfaces.append(Patch7)

Patch8 = np.array([[5.8, 0, 0.], [4.6, 0, 0.], [4.6, -0.5, 0.], [5.8, -0.5, 0.]])
AllSurfaces.append(Patch8)

Patch9 = np.array([[5.8, 0.5, 0.], [4.6, 0.5, 0.], [4.6, 0, 0.], [5.8, 0, 0.]])
AllSurfaces.append(Patch9)

Patch10 = np.array([[7.0, 0, 0.], [5.8, 0, 0.], [5.8, -0.5, 0.], [7.0, -0.5, 0.]])
AllSurfaces.append(Patch10)

Patch11 = np.array([[7.0, 0.5, 0.], [5.8, 0.5, 0.], [5.8, 0, 0.], [7.0, 0, 0.]])
AllSurfaces.append(Patch11)

Patch12 = np.array([[9.0, 0.5, 0.], [7, 0.5, 0.], [7, -0.5, 0.], [9.0, -0.5, 0.]])
AllSurfaces.append(Patch12)

PlotTerrain(AllSurfaces = AllSurfaces)