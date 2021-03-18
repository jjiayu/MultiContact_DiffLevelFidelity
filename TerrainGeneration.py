from Tools import *
from PlotResult import *

def TerrainSelection(name = None, NumRounds = 1, NumContactSequence = 2):

    #Terrain Construction
    if name == "single_obstacle":
        #Terrain Definition/Surface Sequence
        #Define Patches
        #NOTE: The rectangle always stat from Top Right Corner, and the vertex move counterclockwise, it should be a list of numpy arrays
        Patch1 = np.array([[0.4, 0.5, 0.], [-0.1, 0.5, 0.], [-0.1, -0.5, 0.], [0.4, -0.5, 0.]])
        Patch1_TangentX, Patch1_TangentY, Patch1_Norm = getTerrainTagents_and_Norm(Patch1)
        #Patch1_TangentX = np.array([1,0,0])
        #Patch1_TangentY = np.array([0,1,0])
        #Patch1_Norm = np.array([0,0,1])
        Patch2 = np.array([[0.7, 0.5, 0.1], [0.4, 0.5, 0.1], [0.4, -0.5, 0.1], [0.7, -0.5, 0.1]])
        Patch2_TangentX, Patch2_TangentY, Patch2_Norm = getTerrainTagents_and_Norm(Patch2)
        #Patch2_TangentX = np.array([1,0,0])
        #Patch2_TangentY = np.array([0,1,0])
        #Patch2_Norm = np.array([0,0,1])
        Patch3 = np.array([[5, 0.5, 0.], [0.7, 0.5, 0.], [0.7, -0.5, 0.], [5, -0.5, 0.]])
        Patch3_TangentX, Patch3_TangentY, Patch3_Norm = getTerrainTagents_and_Norm(Patch3)
        #Patch3_TangentX = np.array([1,0,0])
        #Patch3_TangentY = np.array([0,1,0])
        #Patch3_Norm = np.array([0,0,1])
        #Collect all patches for the final printing of the terrain
        AllPatches = [Patch1,Patch2,Patch3]
        AllPatchesName = ["Patch1","Patch2","Patch3"]
        AllTerrainTangentsX = [Patch1_TangentX,Patch2_TangentX,Patch3_TangentX]
        AllTerrainTangentsY = [Patch1_TangentY,Patch2_TangentY,Patch3_TangentY]
        AllTerrainNorm = [Patch1_Norm,Patch2_Norm,Patch3_Norm]

    elif name == "flat":
        #Terrain Definition/Surface Sequence
        #Define Patches
        #NOTE: The rectangle always stat from Top Right Corner, and the vertex move counterclockwise, it should be a list of numpy arrays
        Patch1 = np.array([[50, 0.5, 0.], [-0.5, 0.5, 0.], [-0.5, -0.5, 0.], [50, -0.5, 0.]])
        Patch1_TangentX, Patch1_TangentY, Patch1_Norm = getTerrainTagents_and_Norm(Patch1)
        #Patch1_TangentX = np.array([1,0,0])
        #Patch1_TangentY = np.array([0,1,0])
        #Patch1_Norm = np.array([0,0,1])
        #Patch2 = np.array([[0.7, 0.5, 0.1], [0.4, 0.5, 0.1], [0.4, -0.5, 0.1], [0.7, -0.5, 0.1]])
        #Patch2_TangentX, Patch2_TangentY, Patch2_Norm = getTerrainTagents_and_Norm(Patch2)
        #Patch2_TangentX = np.array([1,0,0])
        #Patch2_TangentY = np.array([0,1,0])
        #Patch2_Norm = np.array([0,0,1])
        #Patch3 = np.array([[5, 0.5, 0.], [0.7, 0.5, 0.], [0.7, -0.5, 0.], [5, -0.5, 0.]])
        #Patch3_TangentX, Patch3_TangentY, Patch3_Norm = getTerrainTagents_and_Norm(Patch3)
        #Patch3_TangentX = np.array([1,0,0])
        #Patch3_TangentY = np.array([0,1,0])
        #Patch3_Norm = np.array([0,0,1])
        #Collect all patches for the final printing of the terrain
        AllPatches = [Patch1]
        AllPatchesName = ["Patch1"]
        AllTerrainTangentsX = [Patch1_TangentX]
        AllTerrainTangentsY = [Patch1_TangentY]
        AllTerrainNorm = [Patch1_Norm]

    elif name == "gap":
        AllPatches = []
        AllPatchesName = []
        AllTerrainTangentsX = []
        AllTerrainTangentsY = []
        AllTerrainNorm = []

        Patch1 = np.array([[2., 1, 0.], [-1, 1, 0.], [-1, -1, 0.], [2., -1, 0.]])
        Patch1_TangentX, Patch1_TangentY, Patch1_Norm = getTerrainTagents_and_Norm(Patch1)
        AllPatches.append(Patch1)
        AllPatchesName.append("Patch1")
        AllTerrainTangentsX.append(Patch1_TangentX)
        AllTerrainTangentsY.append(Patch1_TangentY)
        AllTerrainNorm.append(Patch1_Norm)

        Patch2 = np.array([[5, -0.3, 0.], [2., -0.3, 0.], [2., -1, 0.], [5, -1, 0.]])
        Patch2_TangentX, Patch2_TangentY, Patch2_Norm = getTerrainTagents_and_Norm(Patch2)
        AllPatches.append(Patch2)
        AllPatchesName.append("Patch2")
        AllTerrainTangentsX.append(Patch2_TangentX)
        AllTerrainTangentsY.append(Patch2_TangentY)
        AllTerrainNorm.append(Patch2_Norm)

        Patch3 = np.array([[12, 1, 0.], [5, 1, 0.], [5, -1, 0.], [12, -1, 0.]])
        Patch3_TangentX, Patch3_TangentY, Patch3_Norm = getTerrainTagents_and_Norm(Patch3)
        AllPatches.append(Patch3)
        AllPatchesName.append("Patch3")
        AllTerrainTangentsX.append(Patch3_TangentX)
        AllTerrainTangentsY.append(Patch3_TangentY)
        AllTerrainNorm.append(Patch3_Norm)

    elif name == "jump":
        AllPatches = []
        AllPatchesName = []
        AllTerrainTangentsX = []
        AllTerrainTangentsY = []
        AllTerrainNorm = []

        Patch1 = np.array([[1.85, 1, 0.], [-1, 1, 0.], [-1, -1, 0.], [1.85, -1, 0.]])
        Patch1_TangentX, Patch1_TangentY, Patch1_Norm = getTerrainTagents_and_Norm(Patch1)
        AllPatches.append(Patch1)
        AllPatchesName.append("Patch1")
        AllTerrainTangentsX.append(Patch1_TangentX)
        AllTerrainTangentsY.append(Patch1_TangentY)
        AllTerrainNorm.append(Patch1_Norm)

        Patch2 = np.array([[12, 1, 0.], [1.93, 1, 0.], [1.93, -1, 0.], [12, -1, 0.]])
        Patch2_TangentX, Patch2_TangentY, Patch2_Norm = getTerrainTagents_and_Norm(Patch2)
        AllPatches.append(Patch2)
        AllPatchesName.append("Patch2")
        AllTerrainTangentsX.append(Patch2_TangentX)
        AllTerrainTangentsY.append(Patch2_TangentY)
        AllTerrainNorm.append(Patch2_Norm)

        #Patch3 = np.array([[12, 1, 0.], [5, 1, 0.], [5, -1, 0.], [12, -1, 0.]])
        #Patch3_TangentX, Patch3_TangentY, Patch3_Norm = getTerrainTagents_and_Norm(Patch3)
        #AllPatches.append(Patch3)
        #AllPatchesName.append("Patch3")
        #AllTerrainTangentsX.append(Patch3_TangentX)
        #AllTerrainTangentsY.append(Patch3_TangentY)
        #AllTerrainNorm.append(Patch3_Norm)

    elif name == "antfarm_firstLevel_right_start":
        AllPatches = []
        AllPatchesName = []
        AllTerrainTangentsX = []
        AllTerrainTangentsY = []
        AllTerrainNorm = []

        HightVariation =0.07

        Patch1 = np.array([[1, 0.5, 0.], [-1, 0.5, 0.], [-1, -0.5, 0.], [1, -0.5, 0.]])
        Patch1_TangentX, Patch1_TangentY, Patch1_Norm = getTerrainTagents_and_Norm(Patch1)
        AllPatches.append(Patch1)
        AllPatchesName.append("Patch1")
        AllTerrainTangentsX.append(Patch1_TangentX)
        AllTerrainTangentsY.append(Patch1_TangentY)
        AllTerrainNorm.append(Patch1_Norm)

        Patch2 = np.array([[1.6, 0.5, HightVariation], [1, 0.5, HightVariation], [1, 0, -HightVariation], [1.6, 0, -HightVariation]])
        #Patch2 = np.array([[1.6, 0.5, 0], [1, 0.5, 0], [1, 0, 0], [1.6, 0, 0]])
        Patch2_TangentX, Patch2_TangentY, Patch2_Norm = getTerrainTagents_and_Norm(Patch2)
        AllPatches.append(Patch2)
        AllPatchesName.append("Patch2")
        AllTerrainTangentsX.append(Patch2_TangentX)
        AllTerrainTangentsY.append(Patch2_TangentY)
        AllTerrainNorm.append(Patch2_Norm)

        Patch3 = np.array([[1.6, 0, -HightVariation], [1, 0, -HightVariation], [1, -0.5, HightVariation], [1.6, -0.5, HightVariation]])
        #Patch3 = np.array([[1.6, 0, 0], [1, 0, 0], [1, -0.5, 0], [1.6, -0.5, 0]])
        Patch3_TangentX, Patch3_TangentY, Patch3_Norm = getTerrainTagents_and_Norm(Patch3)
        AllPatches.append(Patch3)
        AllPatchesName.append("Patch3")
        AllTerrainTangentsX.append(Patch3_TangentX)
        AllTerrainTangentsY.append(Patch3_TangentY)
        AllTerrainNorm.append(Patch3_Norm)

        Patch4 = np.array([[2.2, 0.5, -HightVariation], [1.6, 0.5, -HightVariation], [1.6, 0, HightVariation], [2.2, 0, HightVariation]])
        Patch4_TangentX, Patch4_TangentY, Patch4_Norm = getTerrainTagents_and_Norm(Patch4)
        AllPatches.append(Patch4)
        AllPatchesName.append("Patch4")
        AllTerrainTangentsX.append(Patch4_TangentX)
        AllTerrainTangentsY.append(Patch4_TangentY)
        AllTerrainNorm.append(Patch4_Norm)

        Patch5 = np.array([[2.2, 0, HightVariation], [1.6, 0, HightVariation], [1.6, -0.5, -HightVariation], [2.2, -0.5, -HightVariation]])
        Patch5_TangentX, Patch5_TangentY, Patch5_Norm = getTerrainTagents_and_Norm(Patch5)
        AllPatches.append(Patch5)
        AllPatchesName.append("Patch5")
        AllTerrainTangentsX.append(Patch5_TangentX)
        AllTerrainTangentsY.append(Patch5_TangentY)
        AllTerrainNorm.append(Patch5_Norm)

        Patch6 = np.array([[2.8, 0.5, HightVariation], [2.2, 0.5, HightVariation], [2.2, 0, -HightVariation], [2.8, 0, -HightVariation]])
        Patch6_TangentX, Patch6_TangentY, Patch6_Norm = getTerrainTagents_and_Norm(Patch6)
        AllPatches.append(Patch6)
        AllPatchesName.append("Patch6")
        AllTerrainTangentsX.append(Patch6_TangentX)
        AllTerrainTangentsY.append(Patch6_TangentY)
        AllTerrainNorm.append(Patch6_Norm)

        Patch7 = np.array([[2.8, 0, -HightVariation], [2.2, 0, -HightVariation], [2.2, -0.5, HightVariation], [2.8, -0.5, HightVariation]])
        Patch7_TangentX, Patch7_TangentY, Patch7_Norm = getTerrainTagents_and_Norm(Patch7)
        AllPatches.append(Patch7)
        AllPatchesName.append("Patch7")
        AllTerrainTangentsX.append(Patch7_TangentX)
        AllTerrainTangentsY.append(Patch7_TangentY)
        AllTerrainNorm.append(Patch7_Norm)

        Patch8 = np.array([[3.4, 0.5, -HightVariation], [2.8, 0.5, -HightVariation], [2.8, 0, HightVariation], [3.4, 0, HightVariation]])
        #Patch8 = np.array([[3.4, 0.5, 0], [2.8, 0.5, 0], [2.8, 0, 0], [3.4, 0, 0]])
        Patch8_TangentX, Patch8_TangentY, Patch8_Norm = getTerrainTagents_and_Norm(Patch8)
        AllPatches.append(Patch8)
        AllPatchesName.append("Patch8")
        AllTerrainTangentsX.append(Patch8_TangentX)
        AllTerrainTangentsY.append(Patch8_TangentY)
        AllTerrainNorm.append(Patch8_Norm)

        Patch9 = np.array([[3.4, 0, HightVariation], [2.8, 0, HightVariation], [2.8, -0.5, -HightVariation], [3.4, -0.5, -HightVariation]])
        #Patch9 = np.array([[3.4, 0, 0], [2.8, 0, 0], [2.8, -0.5, 0], [3.4, -0.5, 0]])
        Patch9_TangentX, Patch9_TangentY, Patch9_Norm = getTerrainTagents_and_Norm(Patch9)
        AllPatches.append(Patch9)
        AllPatchesName.append("Patch9")
        AllTerrainTangentsX.append(Patch9_TangentX)
        AllTerrainTangentsY.append(Patch9_TangentY)
        AllTerrainNorm.append(Patch9_Norm)

        Patch10 = np.array([[4, 0.5, HightVariation], [3.4, 0.5, HightVariation], [3.4, 0, -HightVariation], [4, 0, -HightVariation]])
        #Patch10 = np.array([[4, 0.5, 0], [3.4, 0.5, 0], [3.4, 0, 0], [4, 0, 0]])
        Patch10_TangentX, Patch10_TangentY, Patch10_Norm = getTerrainTagents_and_Norm(Patch10)
        AllPatches.append(Patch10)
        AllPatchesName.append("Patch10")
        AllTerrainTangentsX.append(Patch10_TangentX)
        AllTerrainTangentsY.append(Patch10_TangentY)
        AllTerrainNorm.append(Patch10_Norm)

        Patch11 = np.array([[4, 0, -HightVariation], [3.4, 0, -HightVariation], [3.4, -0.5, HightVariation], [4, -0.5, HightVariation]])
        #Patch11 = np.array([[4, 0, 0], [3.4, 0, 0], [3.4, -0.5, 0], [4, -0.5, 0]])
        Patch11_TangentX, Patch11_TangentY, Patch11_Norm = getTerrainTagents_and_Norm(Patch11)
        AllPatches.append(Patch11)
        AllPatchesName.append("Patch11")
        AllTerrainTangentsX.append(Patch11_TangentX)
        AllTerrainTangentsY.append(Patch11_TangentY)
        AllTerrainNorm.append(Patch11_Norm)

        Patch12 = np.array([[50.0, 0.5, 0.], [4, 0.5, 0.], [4, -0.5, 0.], [50.0, -0.5, 0.]])
        Patch12_TangentX, Patch12_TangentY, Patch12_Norm = getTerrainTagents_and_Norm(Patch12)
        AllPatches.append(Patch12)
        AllPatchesName.append("Patch12")
        AllTerrainTangentsX.append(Patch12_TangentX)
        AllTerrainTangentsY.append(Patch12_TangentY)
        AllTerrainNorm.append(Patch12_Norm)

        #PlotTerrain(AllSurfaces = AllPatches)

    elif name == "antfarm_firstLevel_left_start":
        AllPatches = []
        AllPatchesName = []
        AllTerrainTangentsX = []
        AllTerrainTangentsY = []
        AllTerrainNorm = []

        HightVariation =0.06

        Patch1 = np.array([[1, 0.5, 0.], [-1, 0.5, 0.], [-1, -0.5, 0.], [1, -0.5, 0.]])
        Patch1_TangentX, Patch1_TangentY, Patch1_Norm = getTerrainTagents_and_Norm(Patch1)
        AllPatches.append(Patch1)
        AllPatchesName.append("Patch1")
        AllTerrainTangentsX.append(Patch1_TangentX)
        AllTerrainTangentsY.append(Patch1_TangentY)
        AllTerrainNorm.append(Patch1_Norm)

        Patch2 = np.array([[1.6, 0, -HightVariation], [1, 0, -HightVariation], [1, -0.5, HightVariation], [1.6, -0.5, HightVariation]])
        #Patch2 = np.array([[1.6, 0, 0], [1, 0, 0], [1, -0.5, 0], [1.6, -0.5, 0]])
        Patch2_TangentX, Patch2_TangentY, Patch2_Norm = getTerrainTagents_and_Norm(Patch2)
        AllPatches.append(Patch2)
        AllPatchesName.append("Patch2")
        AllTerrainTangentsX.append(Patch2_TangentX)
        AllTerrainTangentsY.append(Patch2_TangentY)
        AllTerrainNorm.append(Patch2_Norm)

        Patch3 = np.array([[1.6, 0.5, HightVariation], [1, 0.5, HightVariation], [1, 0, -HightVariation], [1.6, 0, -HightVariation]])
        #Patch3 = np.array([[1.6, 0.5, 0], [1, 0.5, 0], [1, 0, 0], [1.6, 0, 0]])
        Patch3_TangentX, Patch3_TangentY, Patch3_Norm = getTerrainTagents_and_Norm(Patch3)
        AllPatches.append(Patch3)
        AllPatchesName.append("Patch3")
        AllTerrainTangentsX.append(Patch3_TangentX)
        AllTerrainTangentsY.append(Patch3_TangentY)
        AllTerrainNorm.append(Patch3_Norm)

        Patch4 = np.array([[2.2, 0, HightVariation], [1.6, 0, HightVariation], [1.6, -0.5, -HightVariation], [2.2, -0.5, -HightVariation]])
        Patch4_TangentX, Patch4_TangentY, Patch4_Norm = getTerrainTagents_and_Norm(Patch4)
        AllPatches.append(Patch4)
        AllPatchesName.append("Patch4")
        AllTerrainTangentsX.append(Patch4_TangentX)
        AllTerrainTangentsY.append(Patch4_TangentY)
        AllTerrainNorm.append(Patch4_Norm)

        Patch5 = np.array([[2.2, 0.5, -HightVariation], [1.6, 0.5, -HightVariation], [1.6, 0, HightVariation], [2.2, 0, HightVariation]])
        Patch5_TangentX, Patch5_TangentY, Patch5_Norm = getTerrainTagents_and_Norm(Patch5)
        AllPatches.append(Patch5)
        AllPatchesName.append("Patch5")
        AllTerrainTangentsX.append(Patch5_TangentX)
        AllTerrainTangentsY.append(Patch5_TangentY)
        AllTerrainNorm.append(Patch5_Norm)

        Patch6 = np.array([[2.8, 0, -HightVariation], [2.2, 0, -HightVariation], [2.2, -0.5, HightVariation], [2.8, -0.5, HightVariation]])
        #Patch6 = np.array([[2.8, 0, 0], [2.2, 0, 0], [2.2, -0.5, 0], [2.8, -0.5, 0]])
        Patch6_TangentX, Patch6_TangentY, Patch6_Norm = getTerrainTagents_and_Norm(Patch6)
        AllPatches.append(Patch6)
        AllPatchesName.append("Patch6")
        AllTerrainTangentsX.append(Patch6_TangentX)
        AllTerrainTangentsY.append(Patch6_TangentY)
        AllTerrainNorm.append(Patch6_Norm)

        Patch7 = np.array([[2.8, 0.5, HightVariation], [2.2, 0.5, HightVariation], [2.2, 0, -HightVariation], [2.8, 0, -HightVariation]])
        #Patch7 = np.array([[2.8, 0.5, 0], [2.2, 0.5, 0], [2.2, 0, 0], [2.8, 0, 0]])
        Patch7_TangentX, Patch7_TangentY, Patch7_Norm = getTerrainTagents_and_Norm(Patch7)
        AllPatches.append(Patch7)
        AllPatchesName.append("Patch7")
        AllTerrainTangentsX.append(Patch7_TangentX)
        AllTerrainTangentsY.append(Patch7_TangentY)
        AllTerrainNorm.append(Patch7_Norm)

        Patch8 = np.array([[3.4, 0, HightVariation], [2.8, 0, HightVariation], [2.8, -0.5, -HightVariation], [3.4, -0.5, -HightVariation]])
        #Patch8 = np.array([[3.4, 0, 0], [2.8, 0, 0], [2.8, -0.5, 0], [3.4, -0.5, 0]])
        Patch8_TangentX, Patch8_TangentY, Patch8_Norm = getTerrainTagents_and_Norm(Patch8)
        AllPatches.append(Patch8)
        AllPatchesName.append("Patch8")
        AllTerrainTangentsX.append(Patch8_TangentX)
        AllTerrainTangentsY.append(Patch8_TangentY)
        AllTerrainNorm.append(Patch8_Norm)

        Patch9 = np.array([[3.4, 0.5, -HightVariation], [2.8, 0.5, -HightVariation], [2.8, 0, HightVariation], [3.4, 0, HightVariation]])
        #Patch9 = np.array([[3.4, 0.5, 0], [2.8, 0.5, 0], [2.8, 0, 0], [3.4, 0, 0]])
        Patch9_TangentX, Patch9_TangentY, Patch9_Norm = getTerrainTagents_and_Norm(Patch9)
        AllPatches.append(Patch9)
        AllPatchesName.append("Patch9")
        AllTerrainTangentsX.append(Patch9_TangentX)
        AllTerrainTangentsY.append(Patch9_TangentY)
        AllTerrainNorm.append(Patch9_Norm)

        Patch10 = np.array([[4, 0, -HightVariation], [3.4, 0, -HightVariation], [3.4, -0.5, HightVariation], [4, -0.5, HightVariation]])
        #Patch10 = np.array([[4, 0, 0], [3.4, 0, 0], [3.4, -0.5, 0], [4, -0.5, 0]])
        Patch10_TangentX, Patch10_TangentY, Patch10_Norm = getTerrainTagents_and_Norm(Patch10)
        AllPatches.append(Patch10)
        AllPatchesName.append("Patch10")
        AllTerrainTangentsX.append(Patch10_TangentX)
        AllTerrainTangentsY.append(Patch10_TangentY)
        AllTerrainNorm.append(Patch10_Norm)

        Patch11 = np.array([[4, 0.5, HightVariation], [3.4, 0.5, HightVariation], [3.4, 0, -HightVariation], [4, 0, -HightVariation]])
        #Patch11 = np.array([[4, 0.5, 0], [3.4, 0.5, 0], [3.4, 0, 0], [4, 0, 0]])
        Patch11_TangentX, Patch11_TangentY, Patch11_Norm = getTerrainTagents_and_Norm(Patch11)
        AllPatches.append(Patch11)
        AllPatchesName.append("Patch11")
        AllTerrainTangentsX.append(Patch11_TangentX)
        AllTerrainTangentsY.append(Patch11_TangentY)
        AllTerrainNorm.append(Patch11_Norm)

        #--------------------------------------
        #Added patches
        #Patch12 = np.array([[4.6, 0, -HightVariation], [4, 0, -HightVariation], [4, -0.5, HightVariation], [4.6, -0.5, HightVariation]])
        #Patch12_TangentX, Patch12_TangentY, Patch12_Norm = getTerrainTagents_and_Norm(Patch12)
        #AllPatches.append(Patch12)
        #AllPatchesName.append("Patch12")
        #AllTerrainTangentsX.append(Patch12_TangentX)
        #AllTerrainTangentsY.append(Patch12_TangentY)
        #AllTerrainNorm.append(Patch12_Norm)

        #Patch13 = np.array([[4.6, 0.5, HightVariation], [4, 0.5, HightVariation], [4, 0, -HightVariation], [4.6, 0, -HightVariation]])
        #Patch13_TangentX, Patch13_TangentY, Patch13_Norm = getTerrainTagents_and_Norm(Patch13)
        #AllPatches.append(Patch13)
        #AllPatchesName.append("Patch13")
        #AllTerrainTangentsY.append(Patch13_TangentY)
        #AllTerrainNorm.append(Patch13_Norm)

        #Patch14 = np.array([[12.0, 0.5, 0.], [4.6, 0.5, 0.], [4.6, -0.5, 0.], [12.0, -0.5, 0.]])
        #Patch14_TangentX, Patch14_TangentY, Patch14_Norm = getTerrainTagents_and_Norm(Patch14)
        #AllPatches.append(Patch14)
        #AllPatchesName.append("Patch14")
        #AllTerrainTangentsX.append(Patch14_TangentX)
        #AllTerrainTangentsY.append(Patch14_TangentY)
        #AllTerrainNorm.append(Patch14_Norm)

        #--------------------------------------

        Patch12 = np.array([[50.0, 0.5, 0.], [4, 0.5, 0.], [4, -0.5, 0.], [50.0, -0.5, 0.]])
        Patch12_TangentX, Patch12_TangentY, Patch12_Norm = getTerrainTagents_and_Norm(Patch12)
        AllPatches.append(Patch12)
        AllPatchesName.append("Patch12")
        AllTerrainTangentsX.append(Patch12_TangentX)
        AllTerrainTangentsY.append(Patch12_TangentY)
        AllTerrainNorm.append(Patch12_Norm)

        #PlotTerrain(AllSurfaces = AllPatches)

    elif name == "antfarm_obstacle_left_start":
        AllPatches = []
        AllPatchesName = []
        AllTerrainTangentsX = []
        AllTerrainTangentsY = []
        AllTerrainNorm = []

        HightVariation =0.06

        Patch1 = np.array([[1, 0.5, 0.], [-1, 0.5, 0.], [-1, -0.5, 0.], [1, -0.5, 0.]])
        Patch1_TangentX, Patch1_TangentY, Patch1_Norm = getTerrainTagents_and_Norm(Patch1)
        AllPatches.append(Patch1)
        AllPatchesName.append("Patch1")
        AllTerrainTangentsX.append(Patch1_TangentX)
        AllTerrainTangentsY.append(Patch1_TangentY)
        AllTerrainNorm.append(Patch1_Norm)

        Patch2 = np.array([[1.6, 0, -HightVariation], [1, 0, -HightVariation], [1, -0.5, HightVariation], [1.6, -0.5, HightVariation]])
        #Patch2 = np.array([[1.6, 0, 0], [1, 0, 0], [1, -0.5, 0], [1.6, -0.5, 0]])
        Patch2_TangentX, Patch2_TangentY, Patch2_Norm = getTerrainTagents_and_Norm(Patch2)
        AllPatches.append(Patch2)
        AllPatchesName.append("Patch2")
        AllTerrainTangentsX.append(Patch2_TangentX)
        AllTerrainTangentsY.append(Patch2_TangentY)
        AllTerrainNorm.append(Patch2_Norm)

        Patch3 = np.array([[1.6, 0.5, HightVariation], [1, 0.5, HightVariation], [1, 0, -HightVariation], [1.6, 0, -HightVariation]])
        #Patch3 = np.array([[1.6, 0.5, 0], [1, 0.5, 0], [1, 0, 0], [1.6, 0, 0]])
        Patch3_TangentX, Patch3_TangentY, Patch3_Norm = getTerrainTagents_and_Norm(Patch3)
        AllPatches.append(Patch3)
        AllPatchesName.append("Patch3")
        AllTerrainTangentsX.append(Patch3_TangentX)
        AllTerrainTangentsY.append(Patch3_TangentY)
        AllTerrainNorm.append(Patch3_Norm)

        Patch4 = np.array([[2.2, 0, HightVariation], [1.6, 0, HightVariation], [1.6, -0.5, -HightVariation], [2.2, -0.5, -HightVariation]])
        Patch4_TangentX, Patch4_TangentY, Patch4_Norm = getTerrainTagents_and_Norm(Patch4)
        AllPatches.append(Patch4)
        AllPatchesName.append("Patch4")
        AllTerrainTangentsX.append(Patch4_TangentX)
        AllTerrainTangentsY.append(Patch4_TangentY)
        AllTerrainNorm.append(Patch4_Norm)

        Patch5 = np.array([[2.2, 0.5, -HightVariation], [1.6, 0.5, -HightVariation], [1.6, 0, HightVariation], [2.2, 0, HightVariation]])
        Patch5_TangentX, Patch5_TangentY, Patch5_Norm = getTerrainTagents_and_Norm(Patch5)
        AllPatches.append(Patch5)
        AllPatchesName.append("Patch5")
        AllTerrainTangentsX.append(Patch5_TangentX)
        AllTerrainTangentsY.append(Patch5_TangentY)
        AllTerrainNorm.append(Patch5_Norm)

        #making Flat
        #--------------------------
        heightlevel = -0.2
        Patch6 = np.array([[2.8, 0, heightlevel], [2.2, 0, heightlevel], [2.2, -0.5, heightlevel], [2.8, -0.5, heightlevel]])
        #Patch6 = np.array([[2.8, 0, 0], [2.2, 0, 0], [2.2, -0.5, 0], [2.8, -0.5, 0]])
        Patch6_TangentX, Patch6_TangentY, Patch6_Norm = getTerrainTagents_and_Norm(Patch6)
        AllPatches.append(Patch6)
        AllPatchesName.append("Patch6")
        AllTerrainTangentsX.append(Patch6_TangentX)
        AllTerrainTangentsY.append(Patch6_TangentY)
        AllTerrainNorm.append(Patch6_Norm)

        Patch7 = np.array([[2.8, 0.5, heightlevel], [2.2, 0.5, heightlevel], [2.2, 0, heightlevel], [2.8, 0, heightlevel]])
        #Patch7 = np.array([[2.8, 0.5, 0], [2.2, 0.5, 0], [2.2, 0, 0], [2.8, 0, 0]])
        Patch7_TangentX, Patch7_TangentY, Patch7_Norm = getTerrainTagents_and_Norm(Patch7)
        AllPatches.append(Patch7)
        AllPatchesName.append("Patch7")
        AllTerrainTangentsX.append(Patch7_TangentX)
        AllTerrainTangentsY.append(Patch7_TangentY)
        AllTerrainNorm.append(Patch7_Norm)
        #---------------------

        Patch8 = np.array([[3.4, 0, HightVariation], [2.8, 0, HightVariation], [2.8, -0.5, -HightVariation], [3.4, -0.5, -HightVariation]])
        #Patch8 = np.array([[3.4, 0, 0], [2.8, 0, 0], [2.8, -0.5, 0], [3.4, -0.5, 0]])
        Patch8_TangentX, Patch8_TangentY, Patch8_Norm = getTerrainTagents_and_Norm(Patch8)
        AllPatches.append(Patch8)
        AllPatchesName.append("Patch8")
        AllTerrainTangentsX.append(Patch8_TangentX)
        AllTerrainTangentsY.append(Patch8_TangentY)
        AllTerrainNorm.append(Patch8_Norm)

        Patch9 = np.array([[3.4, 0.5, -HightVariation], [2.8, 0.5, -HightVariation], [2.8, 0, HightVariation], [3.4, 0, HightVariation]])
        #Patch9 = np.array([[3.4, 0.5, 0], [2.8, 0.5, 0], [2.8, 0, 0], [3.4, 0, 0]])
        Patch9_TangentX, Patch9_TangentY, Patch9_Norm = getTerrainTagents_and_Norm(Patch9)
        AllPatches.append(Patch9)
        AllPatchesName.append("Patch9")
        AllTerrainTangentsX.append(Patch9_TangentX)
        AllTerrainTangentsY.append(Patch9_TangentY)
        AllTerrainNorm.append(Patch9_Norm)

        Patch10 = np.array([[4, 0, -HightVariation], [3.4, 0, -HightVariation], [3.4, -0.5, HightVariation], [4, -0.5, HightVariation]])
        #Patch10 = np.array([[4, 0, 0], [3.4, 0, 0], [3.4, -0.5, 0], [4, -0.5, 0]])
        Patch10_TangentX, Patch10_TangentY, Patch10_Norm = getTerrainTagents_and_Norm(Patch10)
        AllPatches.append(Patch10)
        AllPatchesName.append("Patch10")
        AllTerrainTangentsX.append(Patch10_TangentX)
        AllTerrainTangentsY.append(Patch10_TangentY)
        AllTerrainNorm.append(Patch10_Norm)

        Patch11 = np.array([[4, 0.5, HightVariation], [3.4, 0.5, HightVariation], [3.4, 0, -HightVariation], [4, 0, -HightVariation]])
        #Patch11 = np.array([[4, 0.5, 0], [3.4, 0.5, 0], [3.4, 0, 0], [4, 0, 0]])
        Patch11_TangentX, Patch11_TangentY, Patch11_Norm = getTerrainTagents_and_Norm(Patch11)
        AllPatches.append(Patch11)
        AllPatchesName.append("Patch11")
        AllTerrainTangentsX.append(Patch11_TangentX)
        AllTerrainTangentsY.append(Patch11_TangentY)
        AllTerrainNorm.append(Patch11_Norm)

        #--------------------------------------
        #Added patches
        #Patch12 = np.array([[4.6, 0, -HightVariation], [4, 0, -HightVariation], [4, -0.5, HightVariation], [4.6, -0.5, HightVariation]])
        #Patch12_TangentX, Patch12_TangentY, Patch12_Norm = getTerrainTagents_and_Norm(Patch12)
        #AllPatches.append(Patch12)
        #AllPatchesName.append("Patch12")
        #AllTerrainTangentsX.append(Patch12_TangentX)
        #AllTerrainTangentsY.append(Patch12_TangentY)
        #AllTerrainNorm.append(Patch12_Norm)

        #Patch13 = np.array([[4.6, 0.5, HightVariation], [4, 0.5, HightVariation], [4, 0, -HightVariation], [4.6, 0, -HightVariation]])
        #Patch13_TangentX, Patch13_TangentY, Patch13_Norm = getTerrainTagents_and_Norm(Patch13)
        #AllPatches.append(Patch13)
        #AllPatchesName.append("Patch13")
        #AllTerrainTangentsY.append(Patch13_TangentY)
        #AllTerrainNorm.append(Patch13_Norm)

        #Patch14 = np.array([[12.0, 0.5, 0.], [4.6, 0.5, 0.], [4.6, -0.5, 0.], [12.0, -0.5, 0.]])
        #Patch14_TangentX, Patch14_TangentY, Patch14_Norm = getTerrainTagents_and_Norm(Patch14)
        #AllPatches.append(Patch14)
        #AllPatchesName.append("Patch14")
        #AllTerrainTangentsX.append(Patch14_TangentX)
        #AllTerrainTangentsY.append(Patch14_TangentY)
        #AllTerrainNorm.append(Patch14_Norm)

        #--------------------------------------

        Patch12 = np.array([[50.0, 0.5, 0.], [4, 0.5, 0.], [4, -0.5, 0.], [50.0, -0.5, 0.]])
        Patch12_TangentX, Patch12_TangentY, Patch12_Norm = getTerrainTagents_and_Norm(Patch12)
        AllPatches.append(Patch12)
        AllPatchesName.append("Patch12")
        AllTerrainTangentsX.append(Patch12_TangentX)
        AllTerrainTangentsY.append(Patch12_TangentY)
        AllTerrainNorm.append(Patch12_Norm)

        #PlotTerrain(AllSurfaces = AllPatches)

    elif name == "up_and_down_left_first":
        AllPatches = []
        AllPatchesName = []
        AllTerrainTangentsX = []
        AllTerrainTangentsY = []
        AllTerrainNorm = []

        HightVariation =0.06

        Patch1 = np.array([[1, 0.5, 0.], [-1, 0.5, 0.], [-1, -0.5, 0.], [1, -0.5, 0.]])
        Patch1_TangentX, Patch1_TangentY, Patch1_Norm = getTerrainTagents_and_Norm(Patch1)
        AllPatches.append(Patch1)
        AllPatchesName.append("Patch1")
        AllTerrainTangentsX.append(Patch1_TangentX)
        AllTerrainTangentsY.append(Patch1_TangentY)
        AllTerrainNorm.append(Patch1_Norm)

        Patch2 = np.array([[1.6, 0, HightVariation], [1, 0, -HightVariation], [1, -0.5, -HightVariation], [1.6, -0.5, HightVariation]])
        Patch2_TangentX, Patch2_TangentY, Patch2_Norm = getTerrainTagents_and_Norm(Patch2)
        AllPatches.append(Patch2)
        AllPatchesName.append("Patch2")
        AllTerrainTangentsX.append(Patch2_TangentX)
        AllTerrainTangentsY.append(Patch2_TangentY)
        AllTerrainNorm.append(Patch2_Norm)

        Patch3 = np.array([[1.6, 0.5, HightVariation], [1, 0.5, -HightVariation], [1, 0, -HightVariation], [1.6, 0, HightVariation]])
        Patch3_TangentX, Patch3_TangentY, Patch3_Norm = getTerrainTagents_and_Norm(Patch3)
        AllPatches.append(Patch3)
        AllPatchesName.append("Patch3")
        AllTerrainTangentsX.append(Patch3_TangentX)
        AllTerrainTangentsY.append(Patch3_TangentY)
        AllTerrainNorm.append(Patch3_Norm)

        Patch4 = np.array([[2.2, 0, -HightVariation], [1.6, 0, HightVariation], [1.6, -0.5, HightVariation], [2.2, -0.5, -HightVariation]])
        Patch4_TangentX, Patch4_TangentY, Patch4_Norm = getTerrainTagents_and_Norm(Patch4)
        AllPatches.append(Patch4)
        AllPatchesName.append("Patch4")
        AllTerrainTangentsX.append(Patch4_TangentX)
        AllTerrainTangentsY.append(Patch4_TangentY)
        AllTerrainNorm.append(Patch4_Norm)

        Patch5 = np.array([[2.2, 0.5, -HightVariation], [1.6, 0.5, HightVariation], [1.6, 0, HightVariation], [2.2, 0, -HightVariation]])
        Patch5_TangentX, Patch5_TangentY, Patch5_Norm = getTerrainTagents_and_Norm(Patch5)
        AllPatches.append(Patch5)
        AllPatchesName.append("Patch5")
        AllTerrainTangentsX.append(Patch5_TangentX)
        AllTerrainTangentsY.append(Patch5_TangentY)
        AllTerrainNorm.append(Patch5_Norm)

        Patch6 = np.array([[2.8, 0, HightVariation], [2.2, 0, -HightVariation], [2.2, -0.5, -HightVariation], [2.8, -0.5, HightVariation]])
        Patch6_TangentX, Patch6_TangentY, Patch6_Norm = getTerrainTagents_and_Norm(Patch6)
        AllPatches.append(Patch6)
        AllPatchesName.append("Patch6")
        AllTerrainTangentsX.append(Patch6_TangentX)
        AllTerrainTangentsY.append(Patch6_TangentY)
        AllTerrainNorm.append(Patch6_Norm)

        Patch7 = np.array([[2.8, 0.5, HightVariation], [2.2, 0.5, -HightVariation], [2.2, 0, -HightVariation], [2.8, 0, HightVariation]])
        Patch7_TangentX, Patch7_TangentY, Patch7_Norm = getTerrainTagents_and_Norm(Patch7)
        AllPatches.append(Patch7)
        AllPatchesName.append("Patch7")
        AllTerrainTangentsX.append(Patch7_TangentX)
        AllTerrainTangentsY.append(Patch7_TangentY)
        AllTerrainNorm.append(Patch7_Norm)

        Patch8 = np.array([[3.4, 0, -HightVariation], [2.8, 0, HightVariation], [2.8, -0.5, HightVariation], [3.4, -0.5, -HightVariation]])
        Patch8_TangentX, Patch8_TangentY, Patch8_Norm = getTerrainTagents_and_Norm(Patch8)
        AllPatches.append(Patch8)
        AllPatchesName.append("Patch8")
        AllTerrainTangentsX.append(Patch8_TangentX)
        AllTerrainTangentsY.append(Patch8_TangentY)
        AllTerrainNorm.append(Patch8_Norm)

        Patch9 = np.array([[3.4, 0.5, -HightVariation], [2.8, 0.5, HightVariation], [2.8, 0, HightVariation], [3.4, 0, -HightVariation]])
        Patch9_TangentX, Patch9_TangentY, Patch9_Norm = getTerrainTagents_and_Norm(Patch9)
        AllPatches.append(Patch9)
        AllPatchesName.append("Patch9")
        AllTerrainTangentsX.append(Patch9_TangentX)
        AllTerrainTangentsY.append(Patch9_TangentY)
        AllTerrainNorm.append(Patch9_Norm)

        Patch10 = np.array([[4, 0, HightVariation], [3.4, 0, -HightVariation], [3.4, -0.5, -HightVariation], [4, -0.5, HightVariation]])
        Patch10_TangentX, Patch10_TangentY, Patch10_Norm = getTerrainTagents_and_Norm(Patch10)
        AllPatches.append(Patch10)
        AllPatchesName.append("Patch10")
        AllTerrainTangentsX.append(Patch10_TangentX)
        AllTerrainTangentsY.append(Patch10_TangentY)
        AllTerrainNorm.append(Patch10_Norm)

        Patch11 = np.array([[4, 0.5, HightVariation], [3.4, 0.5, -HightVariation], [3.4, 0, -HightVariation], [4, 0, HightVariation]])
        Patch11_TangentX, Patch11_TangentY, Patch11_Norm = getTerrainTagents_and_Norm(Patch11)
        AllPatches.append(Patch11)
        AllPatchesName.append("Patch11")
        AllTerrainTangentsX.append(Patch11_TangentX)
        AllTerrainTangentsY.append(Patch11_TangentY)
        AllTerrainNorm.append(Patch11_Norm)

        Patch12 = np.array([[50.0, 0.5, 0.], [4, 0.5, 0.], [4, -0.5, 0.], [50.0, -0.5, 0.]])
        Patch12_TangentX, Patch12_TangentY, Patch12_Norm = getTerrainTagents_and_Norm(Patch12)
        AllPatches.append(Patch12)
        AllPatchesName.append("Patch12")
        AllTerrainTangentsX.append(Patch12_TangentX)
        AllTerrainTangentsY.append(Patch12_TangentY)
        AllTerrainNorm.append(Patch12_Norm)

    elif name == "up_and_down_large_patch":
        AllPatches = []
        AllPatchesName = []
        AllTerrainTangentsX = []
        AllTerrainTangentsY = []
        AllTerrainNorm = []

        HightVariation =0.0725
        Length = 0.5

        Patch1 = np.array([[1, 0.5, 0.], [-1, 0.5, 0.], [-1, -0.5, 0.], [1, -0.5, 0.]])
        Patch1_TangentX, Patch1_TangentY, Patch1_Norm = getTerrainTagents_and_Norm(Patch1)
        AllPatches.append(Patch1)
        AllPatchesName.append("Patch1")
        AllTerrainTangentsX.append(Patch1_TangentX)
        AllTerrainTangentsY.append(Patch1_TangentY)
        AllTerrainNorm.append(Patch1_Norm)

        Patch2 = np.array([[1+Length, 0, HightVariation], [1, 0, -HightVariation], [1, -0.5, -HightVariation], [1+Length, -0.5, HightVariation]])
        Patch2_TangentX, Patch2_TangentY, Patch2_Norm = getTerrainTagents_and_Norm(Patch2)
        AllPatches.append(Patch2)
        AllPatchesName.append("Patch2")
        AllTerrainTangentsX.append(Patch2_TangentX)
        AllTerrainTangentsY.append(Patch2_TangentY)
        AllTerrainNorm.append(Patch2_Norm)

        Patch3 = np.array([[1+Length, 0.5, HightVariation], [1, 0.5, -HightVariation], [1, 0, -HightVariation], [1+Length, 0, HightVariation]])
        Patch3_TangentX, Patch3_TangentY, Patch3_Norm = getTerrainTagents_and_Norm(Patch3)
        AllPatches.append(Patch3)
        AllPatchesName.append("Patch3")
        AllTerrainTangentsX.append(Patch3_TangentX)
        AllTerrainTangentsY.append(Patch3_TangentY)
        AllTerrainNorm.append(Patch3_Norm)

        Patch4 = np.array([[1+2*Length, 0, -HightVariation], [1+Length, 0, HightVariation], [1+Length, -0.5, HightVariation], [1+2*Length, -0.5, -HightVariation]])
        Patch4_TangentX, Patch4_TangentY, Patch4_Norm = getTerrainTagents_and_Norm(Patch4)
        AllPatches.append(Patch4)
        AllPatchesName.append("Patch4")
        AllTerrainTangentsX.append(Patch4_TangentX)
        AllTerrainTangentsY.append(Patch4_TangentY)
        AllTerrainNorm.append(Patch4_Norm)

        Patch5 = np.array([[1+2*Length, 0.5, -HightVariation], [1+Length, 0.5, HightVariation], [1+Length, 0, HightVariation], [1+2*Length, 0, -HightVariation]])
        Patch5_TangentX, Patch5_TangentY, Patch5_Norm = getTerrainTagents_and_Norm(Patch5)
        AllPatches.append(Patch5)
        AllPatchesName.append("Patch5")
        AllTerrainTangentsX.append(Patch5_TangentX)
        AllTerrainTangentsY.append(Patch5_TangentY)
        AllTerrainNorm.append(Patch5_Norm)

        Patch6 = np.array([[1+3*Length, 0, HightVariation], [1+2*Length, 0, -HightVariation], [1+2*Length, -0.5, -HightVariation], [1+3*Length, -0.5, HightVariation]])
        Patch6_TangentX, Patch6_TangentY, Patch6_Norm = getTerrainTagents_and_Norm(Patch6)
        AllPatches.append(Patch6)
        AllPatchesName.append("Patch6")
        AllTerrainTangentsX.append(Patch6_TangentX)
        AllTerrainTangentsY.append(Patch6_TangentY)
        AllTerrainNorm.append(Patch6_Norm)

        Patch7 = np.array([[1+3*Length, 0.5, HightVariation], [1+2*Length, 0.5, -HightVariation], [1+2*Length, 0, -HightVariation], [1+3*Length, 0, HightVariation]])
        Patch7_TangentX, Patch7_TangentY, Patch7_Norm = getTerrainTagents_and_Norm(Patch7)
        AllPatches.append(Patch7)
        AllPatchesName.append("Patch7")
        AllTerrainTangentsX.append(Patch7_TangentX)
        AllTerrainTangentsY.append(Patch7_TangentY)
        AllTerrainNorm.append(Patch7_Norm)

        Patch8 = np.array([[1+4*Length, 0, -HightVariation], [1+3*Length, 0, HightVariation], [1+3*Length, -0.5, HightVariation], [1+4*Length, -0.5, -HightVariation]])
        Patch8_TangentX, Patch8_TangentY, Patch8_Norm = getTerrainTagents_and_Norm(Patch8)
        AllPatches.append(Patch8)
        AllPatchesName.append("Patch8")
        AllTerrainTangentsX.append(Patch8_TangentX)
        AllTerrainTangentsY.append(Patch8_TangentY)
        AllTerrainNorm.append(Patch8_Norm)

        Patch9 = np.array([[1+4*Length, 0.5, -HightVariation], [1+3*Length, 0.5, HightVariation], [1+3*Length, 0, HightVariation], [1+4*Length, 0, -HightVariation]])
        Patch9_TangentX, Patch9_TangentY, Patch9_Norm = getTerrainTagents_and_Norm(Patch9)
        AllPatches.append(Patch9)
        AllPatchesName.append("Patch9")
        AllTerrainTangentsX.append(Patch9_TangentX)
        AllTerrainTangentsY.append(Patch9_TangentY)
        AllTerrainNorm.append(Patch9_Norm)

        Patch10 = np.array([[1+5*Length, 0, HightVariation], [1+4*Length, 0, -HightVariation], [1+4*Length, -0.5, -HightVariation], [1+5*Length, -0.5, HightVariation]])
        Patch10_TangentX, Patch10_TangentY, Patch10_Norm = getTerrainTagents_and_Norm(Patch10)
        AllPatches.append(Patch10)
        AllPatchesName.append("Patch10")
        AllTerrainTangentsX.append(Patch10_TangentX)
        AllTerrainTangentsY.append(Patch10_TangentY)
        AllTerrainNorm.append(Patch10_Norm)

        Patch11 = np.array([[1+5*Length, 0.5, HightVariation], [1+4*Length, 0.5, -HightVariation], [1+4*Length, 0, -HightVariation], [1+5*Length, 0, HightVariation]])
        Patch11_TangentX, Patch11_TangentY, Patch11_Norm = getTerrainTagents_and_Norm(Patch11)
        AllPatches.append(Patch11)
        AllPatchesName.append("Patch11")
        AllTerrainTangentsX.append(Patch11_TangentX)
        AllTerrainTangentsY.append(Patch11_TangentY)
        AllTerrainNorm.append(Patch11_Norm)

        Patch12 = np.array([[50.0, 0.5, 0.], [1+5*Length, 0.5, 0.], [1+5*Length, -0.5, 0.], [50.0, -0.5, 0.]])
        Patch12_TangentX, Patch12_TangentY, Patch12_Norm = getTerrainTagents_and_Norm(Patch12)
        AllPatches.append(Patch12)
        AllPatchesName.append("Patch12")
        AllTerrainTangentsX.append(Patch12_TangentX)
        AllTerrainTangentsY.append(Patch12_TangentY)
        AllTerrainNorm.append(Patch12_Norm)
        #PlotTerrain(AllSurfaces = AllPatches)

    elif name == "flat_patches":
        AllPatches = []
        AllPatchesName = []
        AllTerrainTangentsX = []
        AllTerrainTangentsY = []
        AllTerrainNorm = []

        HightVariation =0.0

        Patch1 = np.array([[1, 0.5, 0.], [-1, 0.5, 0.], [-1, -0.5, 0.], [1, -0.5, 0.]])
        Patch1_TangentX, Patch1_TangentY, Patch1_Norm = getTerrainTagents_and_Norm(Patch1)
        AllPatches.append(Patch1)
        AllPatchesName.append("Patch1")
        AllTerrainTangentsX.append(Patch1_TangentX)
        AllTerrainTangentsY.append(Patch1_TangentY)
        AllTerrainNorm.append(Patch1_Norm)

        Patch2 = np.array([[1.6, 0, HightVariation], [1, 0, -HightVariation], [1, -0.5, -HightVariation], [1.6, -0.5, HightVariation]])
        Patch2_TangentX, Patch2_TangentY, Patch2_Norm = getTerrainTagents_and_Norm(Patch2)
        AllPatches.append(Patch2)
        AllPatchesName.append("Patch2")
        AllTerrainTangentsX.append(Patch2_TangentX)
        AllTerrainTangentsY.append(Patch2_TangentY)
        AllTerrainNorm.append(Patch2_Norm)

        Patch3 = np.array([[1.6, 0.5, HightVariation], [1, 0.5, -HightVariation], [1, 0, -HightVariation], [1.6, 0, HightVariation]])
        Patch3_TangentX, Patch3_TangentY, Patch3_Norm = getTerrainTagents_and_Norm(Patch3)
        AllPatches.append(Patch3)
        AllPatchesName.append("Patch3")
        AllTerrainTangentsX.append(Patch3_TangentX)
        AllTerrainTangentsY.append(Patch3_TangentY)
        AllTerrainNorm.append(Patch3_Norm)

        Patch4 = np.array([[2.2, 0, -HightVariation], [1.6, 0, HightVariation], [1.6, -0.5, HightVariation], [2.2, -0.5, -HightVariation]])
        Patch4_TangentX, Patch4_TangentY, Patch4_Norm = getTerrainTagents_and_Norm(Patch4)
        AllPatches.append(Patch4)
        AllPatchesName.append("Patch4")
        AllTerrainTangentsX.append(Patch4_TangentX)
        AllTerrainTangentsY.append(Patch4_TangentY)
        AllTerrainNorm.append(Patch4_Norm)

        Patch5 = np.array([[2.2, 0.5, -HightVariation], [1.6, 0.5, HightVariation], [1.6, 0, HightVariation], [2.2, 0, -HightVariation]])
        Patch5_TangentX, Patch5_TangentY, Patch5_Norm = getTerrainTagents_and_Norm(Patch5)
        AllPatches.append(Patch5)
        AllPatchesName.append("Patch5")
        AllTerrainTangentsX.append(Patch5_TangentX)
        AllTerrainTangentsY.append(Patch5_TangentY)
        AllTerrainNorm.append(Patch5_Norm)

        Patch6 = np.array([[2.8, 0, HightVariation], [2.2, 0, -HightVariation], [2.2, -0.5, -HightVariation], [2.8, -0.5, HightVariation]])
        Patch6_TangentX, Patch6_TangentY, Patch6_Norm = getTerrainTagents_and_Norm(Patch6)
        AllPatches.append(Patch6)
        AllPatchesName.append("Patch6")
        AllTerrainTangentsX.append(Patch6_TangentX)
        AllTerrainTangentsY.append(Patch6_TangentY)
        AllTerrainNorm.append(Patch6_Norm)

        Patch7 = np.array([[2.8, 0.5, HightVariation], [2.2, 0.5, -HightVariation], [2.2, 0, -HightVariation], [2.8, 0, HightVariation]])
        Patch7_TangentX, Patch7_TangentY, Patch7_Norm = getTerrainTagents_and_Norm(Patch7)
        AllPatches.append(Patch7)
        AllPatchesName.append("Patch7")
        AllTerrainTangentsX.append(Patch7_TangentX)
        AllTerrainTangentsY.append(Patch7_TangentY)
        AllTerrainNorm.append(Patch7_Norm)

        Patch8 = np.array([[3.4, 0, -HightVariation], [2.8, 0, HightVariation], [2.8, -0.5, HightVariation], [3.4, -0.5, -HightVariation]])
        Patch8_TangentX, Patch8_TangentY, Patch8_Norm = getTerrainTagents_and_Norm(Patch8)
        AllPatches.append(Patch8)
        AllPatchesName.append("Patch8")
        AllTerrainTangentsX.append(Patch8_TangentX)
        AllTerrainTangentsY.append(Patch8_TangentY)
        AllTerrainNorm.append(Patch8_Norm)

        Patch9 = np.array([[3.4, 0.5, -HightVariation], [2.8, 0.5, HightVariation], [2.8, 0, HightVariation], [3.4, 0, -HightVariation]])
        Patch9_TangentX, Patch9_TangentY, Patch9_Norm = getTerrainTagents_and_Norm(Patch9)
        AllPatches.append(Patch9)
        AllPatchesName.append("Patch9")
        AllTerrainTangentsX.append(Patch9_TangentX)
        AllTerrainTangentsY.append(Patch9_TangentY)
        AllTerrainNorm.append(Patch9_Norm)

        Patch10 = np.array([[4, 0, HightVariation], [3.4, 0, -HightVariation], [3.4, -0.5, -HightVariation], [4, -0.5, HightVariation]])
        Patch10_TangentX, Patch10_TangentY, Patch10_Norm = getTerrainTagents_and_Norm(Patch10)
        AllPatches.append(Patch10)
        AllPatchesName.append("Patch10")
        AllTerrainTangentsX.append(Patch10_TangentX)
        AllTerrainTangentsY.append(Patch10_TangentY)
        AllTerrainNorm.append(Patch10_Norm)

        Patch11 = np.array([[4, 0.5, HightVariation], [3.4, 0.5, -HightVariation], [3.4, 0, -HightVariation], [4, 0, HightVariation]])
        Patch11_TangentX, Patch11_TangentY, Patch11_Norm = getTerrainTagents_and_Norm(Patch11)
        AllPatches.append(Patch11)
        AllPatchesName.append("Patch11")
        AllTerrainTangentsX.append(Patch11_TangentX)
        AllTerrainTangentsY.append(Patch11_TangentY)
        AllTerrainNorm.append(Patch11_Norm)

        Patch12 = np.array([[50.0, 0.5, 0.], [4, 0.5, 0.], [4, -0.5, 0.], [50.0, -0.5, 0.]])
        Patch12_TangentX, Patch12_TangentY, Patch12_Norm = getTerrainTagents_and_Norm(Patch12)
        AllPatches.append(Patch12)
        AllPatchesName.append("Patch12")
        AllTerrainTangentsX.append(Patch12_TangentX)
        AllTerrainTangentsY.append(Patch12_TangentY)
        AllTerrainNorm.append(Patch12_Norm)

    elif name == "flat_patches_shift":
        AllPatches = []
        AllPatchesName = []
        AllTerrainTangentsX = []
        AllTerrainTangentsY = []
        AllTerrainNorm = []

        HightVariation =0.0

        Patch1 = np.array([[1, 0.5, 0.], [-1, 0.5, 0.], [-1, -0.5, 0.], [1, -0.5, 0.]])
        Patch1_TangentX, Patch1_TangentY, Patch1_Norm = getTerrainTagents_and_Norm(Patch1)
        AllPatches.append(Patch1)
        AllPatchesName.append("Patch1")
        AllTerrainTangentsX.append(Patch1_TangentX)
        AllTerrainTangentsY.append(Patch1_TangentY)
        AllTerrainNorm.append(Patch1_Norm)

        y_shift = -0.4
        Patch2 = np.array([[1.6, 0 + y_shift, HightVariation], [1, 0 + y_shift, -HightVariation], [1, -0.5 + y_shift, -HightVariation], [1.6, -0.5 + y_shift, HightVariation]])
        Patch2_TangentX, Patch2_TangentY, Patch2_Norm = getTerrainTagents_and_Norm(Patch2)
        AllPatches.append(Patch2)
        AllPatchesName.append("Patch2")
        AllTerrainTangentsX.append(Patch2_TangentX)
        AllTerrainTangentsY.append(Patch2_TangentY)
        AllTerrainNorm.append(Patch2_Norm)

        Patch3 = np.array([[1.6, 0.5, HightVariation], [1, 0.5, -HightVariation], [1, 0, -HightVariation], [1.6, 0, HightVariation]])
        Patch3_TangentX, Patch3_TangentY, Patch3_Norm = getTerrainTagents_and_Norm(Patch3)
        AllPatches.append(Patch3)
        AllPatchesName.append("Patch3")
        AllTerrainTangentsX.append(Patch3_TangentX)
        AllTerrainTangentsY.append(Patch3_TangentY)
        AllTerrainNorm.append(Patch3_Norm)

        Patch4 = np.array([[2.2, 0, -HightVariation], [1.6, 0, HightVariation], [1.6, -0.5, HightVariation], [2.2, -0.5, -HightVariation]])
        Patch4_TangentX, Patch4_TangentY, Patch4_Norm = getTerrainTagents_and_Norm(Patch4)
        AllPatches.append(Patch4)
        AllPatchesName.append("Patch4")
        AllTerrainTangentsX.append(Patch4_TangentX)
        AllTerrainTangentsY.append(Patch4_TangentY)
        AllTerrainNorm.append(Patch4_Norm)

        Patch5 = np.array([[2.2, 0.5, -HightVariation], [1.6, 0.5, HightVariation], [1.6, 0, HightVariation], [2.2, 0, -HightVariation]])
        Patch5_TangentX, Patch5_TangentY, Patch5_Norm = getTerrainTagents_and_Norm(Patch5)
        AllPatches.append(Patch5)
        AllPatchesName.append("Patch5")
        AllTerrainTangentsX.append(Patch5_TangentX)
        AllTerrainTangentsY.append(Patch5_TangentY)
        AllTerrainNorm.append(Patch5_Norm)

        Patch6 = np.array([[2.8, 0, HightVariation], [2.2, 0, -HightVariation], [2.2, -0.5, -HightVariation], [2.8, -0.5, HightVariation]])
        Patch6_TangentX, Patch6_TangentY, Patch6_Norm = getTerrainTagents_and_Norm(Patch6)
        AllPatches.append(Patch6)
        AllPatchesName.append("Patch6")
        AllTerrainTangentsX.append(Patch6_TangentX)
        AllTerrainTangentsY.append(Patch6_TangentY)
        AllTerrainNorm.append(Patch6_Norm)

        Patch7 = np.array([[2.8, 0.5, HightVariation], [2.2, 0.5, -HightVariation], [2.2, 0, -HightVariation], [2.8, 0, HightVariation]])
        Patch7_TangentX, Patch7_TangentY, Patch7_Norm = getTerrainTagents_and_Norm(Patch7)
        AllPatches.append(Patch7)
        AllPatchesName.append("Patch7")
        AllTerrainTangentsX.append(Patch7_TangentX)
        AllTerrainTangentsY.append(Patch7_TangentY)
        AllTerrainNorm.append(Patch7_Norm)

        Patch8 = np.array([[3.4, 0, -HightVariation], [2.8, 0, HightVariation], [2.8, -0.5, HightVariation], [3.4, -0.5, -HightVariation]])
        Patch8_TangentX, Patch8_TangentY, Patch8_Norm = getTerrainTagents_and_Norm(Patch8)
        AllPatches.append(Patch8)
        AllPatchesName.append("Patch8")
        AllTerrainTangentsX.append(Patch8_TangentX)
        AllTerrainTangentsY.append(Patch8_TangentY)
        AllTerrainNorm.append(Patch8_Norm)

        Patch9 = np.array([[3.4, 0.5, -HightVariation], [2.8, 0.5, HightVariation], [2.8, 0, HightVariation], [3.4, 0, -HightVariation]])
        Patch9_TangentX, Patch9_TangentY, Patch9_Norm = getTerrainTagents_and_Norm(Patch9)
        AllPatches.append(Patch9)
        AllPatchesName.append("Patch9")
        AllTerrainTangentsX.append(Patch9_TangentX)
        AllTerrainTangentsY.append(Patch9_TangentY)
        AllTerrainNorm.append(Patch9_Norm)

        Patch10 = np.array([[4, 0, HightVariation], [3.4, 0, -HightVariation], [3.4, -0.5, -HightVariation], [4, -0.5, HightVariation]])
        Patch10_TangentX, Patch10_TangentY, Patch10_Norm = getTerrainTagents_and_Norm(Patch10)
        AllPatches.append(Patch10)
        AllPatchesName.append("Patch10")
        AllTerrainTangentsX.append(Patch10_TangentX)
        AllTerrainTangentsY.append(Patch10_TangentY)
        AllTerrainNorm.append(Patch10_Norm)

        Patch11 = np.array([[4, 0.5, HightVariation], [3.4, 0.5, -HightVariation], [3.4, 0, -HightVariation], [4, 0, HightVariation]])
        Patch11_TangentX, Patch11_TangentY, Patch11_Norm = getTerrainTagents_and_Norm(Patch11)
        AllPatches.append(Patch11)
        AllPatchesName.append("Patch11")
        AllTerrainTangentsX.append(Patch11_TangentX)
        AllTerrainTangentsY.append(Patch11_TangentY)
        AllTerrainNorm.append(Patch11_Norm)

        Patch12 = np.array([[50.0, 0.5, 0.], [4, 0.5, 0.], [4, -0.5, 0.], [50.0, -0.5, 0.]])
        Patch12_TangentX, Patch12_TangentY, Patch12_Norm = getTerrainTagents_and_Norm(Patch12)
        AllPatches.append(Patch12)
        AllPatchesName.append("Patch12")
        AllTerrainTangentsX.append(Patch12_TangentX)
        AllTerrainTangentsY.append(Patch12_TangentY)
        AllTerrainNorm.append(Patch12_Norm)


    elif name == "up_and_down_right_first":
        AllPatches = []
        AllPatchesName = []
        AllTerrainTangentsX = []
        AllTerrainTangentsY = []
        AllTerrainNorm = []

        HightVariation =0.08

        Patch1 = np.array([[1, 0.5, 0.], [-1, 0.5, 0.], [-1, -0.5, 0.], [1, -0.5, 0.]])
        Patch1_TangentX, Patch1_TangentY, Patch1_Norm = getTerrainTagents_and_Norm(Patch1)
        AllPatches.append(Patch1)
        AllPatchesName.append("Patch1")
        AllTerrainTangentsX.append(Patch1_TangentX)
        AllTerrainTangentsY.append(Patch1_TangentY)
        AllTerrainNorm.append(Patch1_Norm)

        Patch2 = np.array([[1.6, 0.5, HightVariation], [1, 0.5, -HightVariation], [1, 0, -HightVariation], [1.6, 0, HightVariation]])
        Patch2_TangentX, Patch2_TangentY, Patch2_Norm = getTerrainTagents_and_Norm(Patch2)
        AllPatches.append(Patch2)
        AllPatchesName.append("Patch2")
        AllTerrainTangentsX.append(Patch2_TangentX)
        AllTerrainTangentsY.append(Patch2_TangentY)
        AllTerrainNorm.append(Patch2_Norm)

        Patch3 = np.array([[1.6, 0, HightVariation], [1, 0, -HightVariation], [1, -0.5, -HightVariation], [1.6, -0.5, HightVariation]])
        Patch3_TangentX, Patch3_TangentY, Patch3_Norm = getTerrainTagents_and_Norm(Patch3)
        AllPatches.append(Patch3)
        AllPatchesName.append("Patch3")
        AllTerrainTangentsX.append(Patch3_TangentX)
        AllTerrainTangentsY.append(Patch3_TangentY)
        AllTerrainNorm.append(Patch3_Norm)

        Patch4 = np.array([[2.2, 0.5, -HightVariation], [1.6, 0.5, HightVariation], [1.6, 0, HightVariation], [2.2, 0, -HightVariation]])
        Patch4_TangentX, Patch4_TangentY, Patch4_Norm = getTerrainTagents_and_Norm(Patch4)
        AllPatches.append(Patch4)
        AllPatchesName.append("Patch4")
        AllTerrainTangentsX.append(Patch4_TangentX)
        AllTerrainTangentsY.append(Patch4_TangentY)
        AllTerrainNorm.append(Patch4_Norm)

        Patch5 = np.array([[2.2, 0, -HightVariation], [1.6, 0, HightVariation], [1.6, -0.5, HightVariation], [2.2, -0.5, -HightVariation]])
        Patch5_TangentX, Patch5_TangentY, Patch5_Norm = getTerrainTagents_and_Norm(Patch5)
        AllPatches.append(Patch5)
        AllPatchesName.append("Patch5")
        AllTerrainTangentsX.append(Patch5_TangentX)
        AllTerrainTangentsY.append(Patch5_TangentY)
        AllTerrainNorm.append(Patch5_Norm)

        Patch6 = np.array([[2.8, 0.5, HightVariation], [2.2, 0.5, -HightVariation], [2.2, 0, -HightVariation], [2.8, 0, HightVariation]])
        Patch6_TangentX, Patch6_TangentY, Patch6_Norm = getTerrainTagents_and_Norm(Patch6)
        AllPatches.append(Patch6)
        AllPatchesName.append("Patch6")
        AllTerrainTangentsX.append(Patch6_TangentX)
        AllTerrainTangentsY.append(Patch6_TangentY)
        AllTerrainNorm.append(Patch6_Norm)

        Patch7 = np.array([[2.8, 0, HightVariation], [2.2, 0, -HightVariation], [2.2, -0.5, -HightVariation], [2.8, -0.5, HightVariation]])
        Patch7_TangentX, Patch7_TangentY, Patch7_Norm = getTerrainTagents_and_Norm(Patch7)
        AllPatches.append(Patch7)
        AllPatchesName.append("Patch7")
        AllTerrainTangentsX.append(Patch7_TangentX)
        AllTerrainTangentsY.append(Patch7_TangentY)
        AllTerrainNorm.append(Patch7_Norm)

        Patch8 = np.array([[3.4, 0.5, -HightVariation], [2.8, 0.5, HightVariation], [2.8, 0, HightVariation], [3.4, 0, -HightVariation]])
        Patch8_TangentX, Patch8_TangentY, Patch8_Norm = getTerrainTagents_and_Norm(Patch8)
        AllPatches.append(Patch8)
        AllPatchesName.append("Patch8")
        AllTerrainTangentsX.append(Patch8_TangentX)
        AllTerrainTangentsY.append(Patch8_TangentY)
        AllTerrainNorm.append(Patch8_Norm)

        Patch9 = np.array([[3.4, 0, -HightVariation], [2.8, 0, HightVariation], [2.8, -0.5, HightVariation], [3.4, -0.5, -HightVariation]])
        Patch9_TangentX, Patch9_TangentY, Patch9_Norm = getTerrainTagents_and_Norm(Patch9)
        AllPatches.append(Patch9)
        AllPatchesName.append("Patch9")
        AllTerrainTangentsX.append(Patch9_TangentX)
        AllTerrainTangentsY.append(Patch9_TangentY)
        AllTerrainNorm.append(Patch9_Norm)

        Patch10 = np.array([[4, 0.5, HightVariation], [3.4, 0.5, -HightVariation], [3.4, 0, -HightVariation], [4, 0, HightVariation]])
        Patch10_TangentX, Patch10_TangentY, Patch10_Norm = getTerrainTagents_and_Norm(Patch10)
        AllPatches.append(Patch10)
        AllPatchesName.append("Patch10")
        AllTerrainTangentsX.append(Patch10_TangentX)
        AllTerrainTangentsY.append(Patch10_TangentY)
        AllTerrainNorm.append(Patch10_Norm)

        Patch11 = np.array([[4, 0, HightVariation], [3.4, 0, -HightVariation], [3.4, -0.5, -HightVariation], [4, -0.5, HightVariation]])
        Patch11_TangentX, Patch11_TangentY, Patch11_Norm = getTerrainTagents_and_Norm(Patch11)
        AllPatches.append(Patch11)
        AllPatchesName.append("Patch11")
        AllTerrainTangentsX.append(Patch11_TangentX)
        AllTerrainTangentsY.append(Patch11_TangentY)
        AllTerrainNorm.append(Patch11_Norm)

        Patch12 = np.array([[50.0, 0.5, 0.], [4, 0.5, 0.], [4, -0.5, 0.], [50.0, -0.5, 0.]])
        Patch12_TangentX, Patch12_TangentY, Patch12_Norm = getTerrainTagents_and_Norm(Patch12)
        AllPatches.append(Patch12)
        AllPatchesName.append("Patch12")
        AllTerrainTangentsX.append(Patch12_TangentX)
        AllTerrainTangentsY.append(Patch12_TangentY)
        AllTerrainNorm.append(Patch12_Norm)

        #PlotTerrain(AllSurfaces = AllPatches)

    elif name == "darpa_like_left_first":
        AllPatches = []
        AllPatchesName = []
        AllTerrainTangentsX = []
        AllTerrainTangentsY = []
        AllTerrainNorm = []

        HightVariation =0.07
        HorizontanIncrease = 0.6

        Patch1 = np.array([[1, 0.5, 0.], [-1, 0.5, 0.], [-1, -0.5, 0.], [1, -0.5, 0.]])
        Patch1_TangentX, Patch1_TangentY, Patch1_Norm = getTerrainTagents_and_Norm(Patch1)
        AllPatches.append(Patch1)
        AllPatchesName.append("Patch1")
        AllTerrainTangentsX.append(Patch1_TangentX)
        AllTerrainTangentsY.append(Patch1_TangentY)
        AllTerrainNorm.append(Patch1_Norm)

        Patch2 = np.array([[1+HorizontanIncrease, 0, -HightVariation], [1, 0, -HightVariation], [1, -0.5, HightVariation], [1+HorizontanIncrease, -0.5, HightVariation]])
        Patch2_TangentX, Patch2_TangentY, Patch2_Norm = getTerrainTagents_and_Norm(Patch2)
        AllPatches.append(Patch2)
        AllPatchesName.append("Patch2")
        AllTerrainTangentsX.append(Patch2_TangentX)
        AllTerrainTangentsY.append(Patch2_TangentY)
        AllTerrainNorm.append(Patch2_Norm)

        Patch3 = np.array([[1+HorizontanIncrease, 0.5, -HightVariation], [1, 0.5, HightVariation], [1, 0, HightVariation], [1+HorizontanIncrease, 0, -HightVariation]])
        Patch3_TangentX, Patch3_TangentY, Patch3_Norm = getTerrainTagents_and_Norm(Patch3)
        AllPatches.append(Patch3)
        AllPatchesName.append("Patch3")
        AllTerrainTangentsX.append(Patch3_TangentX)
        AllTerrainTangentsY.append(Patch3_TangentY)
        AllTerrainNorm.append(Patch3_Norm)

        Patch4 = np.array([[1+2*HorizontanIncrease, 0, HightVariation], [1+HorizontanIncrease, 0, HightVariation], [1+HorizontanIncrease, -0.5, -HightVariation], [1+2*HorizontanIncrease, -0.5, -HightVariation]])
        Patch4_TangentX, Patch4_TangentY, Patch4_Norm = getTerrainTagents_and_Norm(Patch4)
        AllPatches.append(Patch4)
        AllPatchesName.append("Patch4")
        AllTerrainTangentsX.append(Patch4_TangentX)
        AllTerrainTangentsY.append(Patch4_TangentY)
        AllTerrainNorm.append(Patch4_Norm)

        Patch5 = np.array([[1+2*HorizontanIncrease, 0.5, HightVariation], [1+HorizontanIncrease, 0.5, -HightVariation], [1+HorizontanIncrease, 0, -HightVariation], [1+2*HorizontanIncrease, 0, HightVariation]])
        Patch5_TangentX, Patch5_TangentY, Patch5_Norm = getTerrainTagents_and_Norm(Patch5)
        AllPatches.append(Patch5)
        AllPatchesName.append("Patch5")
        AllTerrainTangentsX.append(Patch5_TangentX)
        AllTerrainTangentsY.append(Patch5_TangentY)
        AllTerrainNorm.append(Patch5_Norm)

        Patch6 = np.array([[1+3*HorizontanIncrease, 0, -HightVariation], [1+2*HorizontanIncrease, 0, HightVariation], [1+2*HorizontanIncrease, -0.5, HightVariation], [1+3*HorizontanIncrease, -0.5, -HightVariation]])
        Patch6_TangentX, Patch6_TangentY, Patch6_Norm = getTerrainTagents_and_Norm(Patch6)
        AllPatches.append(Patch6)
        AllPatchesName.append("Patch6")
        AllTerrainTangentsX.append(Patch6_TangentX)
        AllTerrainTangentsY.append(Patch6_TangentY)
        AllTerrainNorm.append(Patch6_Norm)

        Patch7 = np.array([[1+3*HorizontanIncrease, 0.5, -HightVariation], [1+2*HorizontanIncrease, 0.5, -HightVariation], [1+2*HorizontanIncrease, 0, HightVariation], [1+3*HorizontanIncrease, 0, HightVariation]])
        Patch7_TangentX, Patch7_TangentY, Patch7_Norm = getTerrainTagents_and_Norm(Patch7)
        AllPatches.append(Patch7)
        AllPatchesName.append("Patch7")
        AllTerrainTangentsX.append(Patch7_TangentX)
        AllTerrainTangentsY.append(Patch7_TangentY)
        AllTerrainNorm.append(Patch7_Norm)

        Patch8 = np.array([[1+4*HorizontanIncrease, 0, HightVariation], [1+3*HorizontanIncrease, 0, HightVariation], [1+3*HorizontanIncrease, -0.5, -HightVariation], [1+4*HorizontanIncrease, -0.5, -HightVariation]])
        Patch8_TangentX, Patch8_TangentY, Patch8_Norm = getTerrainTagents_and_Norm(Patch8)
        AllPatches.append(Patch8)
        AllPatchesName.append("Patch8")
        AllTerrainTangentsX.append(Patch8_TangentX)
        AllTerrainTangentsY.append(Patch8_TangentY)
        AllTerrainNorm.append(Patch8_Norm)

        Patch9 = np.array([[1+4*HorizontanIncrease, 0.5, HightVariation], [1+3*HorizontanIncrease, 0.5, -HightVariation], [1+3*HorizontanIncrease, 0, -HightVariation], [1+4*HorizontanIncrease, 0, HightVariation]])
        Patch9_TangentX, Patch9_TangentY, Patch9_Norm = getTerrainTagents_and_Norm(Patch9)
        AllPatches.append(Patch9)
        AllPatchesName.append("Patch9")
        AllTerrainTangentsX.append(Patch9_TangentX)
        AllTerrainTangentsY.append(Patch9_TangentY)
        AllTerrainNorm.append(Patch9_Norm)

        Patch10 = np.array([[1+5*HorizontanIncrease, 0, -HightVariation], [1+4*HorizontanIncrease, 0, -HightVariation], [1+4*HorizontanIncrease, -0.5, HightVariation], [1+5*HorizontanIncrease, -0.5, HightVariation]])
        Patch10_TangentX, Patch10_TangentY, Patch10_Norm = getTerrainTagents_and_Norm(Patch10)
        AllPatches.append(Patch10)
        AllPatchesName.append("Patch10")
        AllTerrainTangentsX.append(Patch10_TangentX)
        AllTerrainTangentsY.append(Patch10_TangentY)
        AllTerrainNorm.append(Patch10_Norm)

        Patch11 = np.array([[1+5*HorizontanIncrease, 0.5, -HightVariation], [1+4*HorizontanIncrease, 0.5, HightVariation], [1+4*HorizontanIncrease, 0, HightVariation], [1+5*HorizontanIncrease, 0, -HightVariation]])
        Patch11_TangentX, Patch11_TangentY, Patch11_Norm = getTerrainTagents_and_Norm(Patch11)
        AllPatches.append(Patch11)
        AllPatchesName.append("Patch11")
        AllTerrainTangentsX.append(Patch11_TangentX)
        AllTerrainTangentsY.append(Patch11_TangentY)
        AllTerrainNorm.append(Patch11_Norm)

        Patch12 = np.array([[50.0, 0.5, 0.], [1+5*HorizontanIncrease, 0.5, 0.], [1+5*HorizontanIncrease, -0.5, 0.], [50.0, -0.5, 0.]])
        Patch12_TangentX, Patch12_TangentY, Patch12_Norm = getTerrainTagents_and_Norm(Patch12)
        AllPatches.append(Patch12)
        AllPatchesName.append("Patch12")
        AllTerrainTangentsX.append(Patch12_TangentX)
        AllTerrainTangentsY.append(Patch12_TangentY)
        AllTerrainNorm.append(Patch12_Norm)

        #PlotTerrain(AllSurfaces = AllPatches)

    elif name == "stairs":
        AllPatches = []
        AllPatchesName = []
        AllTerrainTangentsX = []
        AllTerrainTangentsY = []
        AllTerrainNorm = []

        HightVariation =0.06
        HorizontanIncrease = 0.5

        Patch1 = np.array([[1, 0.5, 0.], [-1, 0.5, 0.], [-1, -0.5, 0.], [1, -0.5, 0.]])
        Patch1_TangentX, Patch1_TangentY, Patch1_Norm = getTerrainTagents_and_Norm(Patch1)
        AllPatches.append(Patch1)
        AllPatchesName.append("Patch1")
        AllTerrainTangentsX.append(Patch1_TangentX)
        AllTerrainTangentsY.append(Patch1_TangentY)
        AllTerrainNorm.append(Patch1_Norm)

        Patch2 = np.array([[1+HorizontanIncrease, 0, HightVariation], [1, 0, HightVariation], [1, -0.5, HightVariation], [1+HorizontanIncrease, -0.5, HightVariation]])
        Patch2_TangentX, Patch2_TangentY, Patch2_Norm = getTerrainTagents_and_Norm(Patch2)
        AllPatches.append(Patch2)
        AllPatchesName.append("Patch2")
        AllTerrainTangentsX.append(Patch2_TangentX)
        AllTerrainTangentsY.append(Patch2_TangentY)
        AllTerrainNorm.append(Patch2_Norm)

        Patch3 = np.array([[1+HorizontanIncrease, 0.5, 2*HightVariation], [1, 0.5, 2*HightVariation], [1, 0, 2*HightVariation], [1+HorizontanIncrease, 0, 2*HightVariation]])
        Patch3_TangentX, Patch3_TangentY, Patch3_Norm = getTerrainTagents_and_Norm(Patch3)
        AllPatches.append(Patch3)
        AllPatchesName.append("Patch3")
        AllTerrainTangentsX.append(Patch3_TangentX)
        AllTerrainTangentsY.append(Patch3_TangentY)
        AllTerrainNorm.append(Patch3_Norm)

        Patch4 = np.array([[1+2*HorizontanIncrease, 0, 3*HightVariation], [1+HorizontanIncrease, 0, 3*HightVariation], [1+HorizontanIncrease, -0.5, 3*HightVariation], [1+2*HorizontanIncrease, -0.5, 3*HightVariation]])
        Patch4_TangentX, Patch4_TangentY, Patch4_Norm = getTerrainTagents_and_Norm(Patch4)
        AllPatches.append(Patch4)
        AllPatchesName.append("Patch4")
        AllTerrainTangentsX.append(Patch4_TangentX)
        AllTerrainTangentsY.append(Patch4_TangentY)
        AllTerrainNorm.append(Patch4_Norm)

        Patch5 = np.array([[1+2*HorizontanIncrease, 0.5, 4*HightVariation], [1+HorizontanIncrease, 0.5, 4*HightVariation], [1+HorizontanIncrease, 0, 4*HightVariation], [1+2*HorizontanIncrease, 0, 4*HightVariation]])
        Patch5_TangentX, Patch5_TangentY, Patch5_Norm = getTerrainTagents_and_Norm(Patch5)
        AllPatches.append(Patch5)
        AllPatchesName.append("Patch5")
        AllTerrainTangentsX.append(Patch5_TangentX)
        AllTerrainTangentsY.append(Patch5_TangentY)
        AllTerrainNorm.append(Patch5_Norm)

        Patch6 = np.array([[1+3*HorizontanIncrease, 0, 5*HightVariation], [1+2*HorizontanIncrease, 0, 5*HightVariation], [1+2*HorizontanIncrease, -0.5, 5*HightVariation], [1+3*HorizontanIncrease, -0.5, 5*HightVariation]])
        Patch6_TangentX, Patch6_TangentY, Patch6_Norm = getTerrainTagents_and_Norm(Patch6)
        AllPatches.append(Patch6)
        AllPatchesName.append("Patch6")
        AllTerrainTangentsX.append(Patch6_TangentX)
        AllTerrainTangentsY.append(Patch6_TangentY)
        AllTerrainNorm.append(Patch6_Norm)

        Patch7 = np.array([[1+3*HorizontanIncrease, 0.5, 6*HightVariation], [1+2*HorizontanIncrease, 0.5, 6*HightVariation], [1+2*HorizontanIncrease, 0, 6*HightVariation], [1+3*HorizontanIncrease, 0, 6*HightVariation]])
        Patch7_TangentX, Patch7_TangentY, Patch7_Norm = getTerrainTagents_and_Norm(Patch7)
        AllPatches.append(Patch7)
        AllPatchesName.append("Patch7")
        AllTerrainTangentsX.append(Patch7_TangentX)
        AllTerrainTangentsY.append(Patch7_TangentY)
        AllTerrainNorm.append(Patch7_Norm)

        Patch8 = np.array([[1+4*HorizontanIncrease, 0, 7*HightVariation], [1+3*HorizontanIncrease, 0, 7*HightVariation], [1+3*HorizontanIncrease, -0.5, 7*HightVariation], [1+4*HorizontanIncrease, -0.5, 7*HightVariation]])
        Patch8_TangentX, Patch8_TangentY, Patch8_Norm = getTerrainTagents_and_Norm(Patch8)
        AllPatches.append(Patch8)
        AllPatchesName.append("Patch8")
        AllTerrainTangentsX.append(Patch8_TangentX)
        AllTerrainTangentsY.append(Patch8_TangentY)
        AllTerrainNorm.append(Patch8_Norm)

        Patch9 = np.array([[1+4*HorizontanIncrease, 0.5, 8*HightVariation], [1+3*HorizontanIncrease, 0.5, 8*HightVariation], [1+3*HorizontanIncrease, 0, 8*HightVariation], [1+4*HorizontanIncrease, 0, 8*HightVariation]])
        Patch9_TangentX, Patch9_TangentY, Patch9_Norm = getTerrainTagents_and_Norm(Patch9)
        AllPatches.append(Patch9)
        AllPatchesName.append("Patch9")
        AllTerrainTangentsX.append(Patch9_TangentX)
        AllTerrainTangentsY.append(Patch9_TangentY)
        AllTerrainNorm.append(Patch9_Norm)

        Patch10 = np.array([[1+5*HorizontanIncrease, 0, 9*HightVariation], [1+4*HorizontanIncrease, 0, 9*HightVariation], [1+4*HorizontanIncrease, -0.5, 9*HightVariation], [1+5*HorizontanIncrease, -0.5, 9*HightVariation]])
        Patch10_TangentX, Patch10_TangentY, Patch10_Norm = getTerrainTagents_and_Norm(Patch10)
        AllPatches.append(Patch10)
        AllPatchesName.append("Patch10")
        AllTerrainTangentsX.append(Patch10_TangentX)
        AllTerrainTangentsY.append(Patch10_TangentY)
        AllTerrainNorm.append(Patch10_Norm)

        Patch11 = np.array([[1+5*HorizontanIncrease, 0.5, 10*HightVariation], [1+4*HorizontanIncrease, 0.5, 10*HightVariation], [1+4*HorizontanIncrease, 0, 10*HightVariation], [1+5*HorizontanIncrease, 0, 10*HightVariation]])
        Patch11_TangentX, Patch11_TangentY, Patch11_Norm = getTerrainTagents_and_Norm(Patch11)
        AllPatches.append(Patch11)
        AllPatchesName.append("Patch11")
        AllTerrainTangentsX.append(Patch11_TangentX)
        AllTerrainTangentsY.append(Patch11_TangentY)
        AllTerrainNorm.append(Patch11_Norm)

        Patch12 = np.array([[50.0, 0.5, 11*HightVariation], [1+5*HorizontanIncrease, 0.5, 11*HightVariation], [1+5*HorizontanIncrease, -0.5, 11*HightVariation], [50.0, -0.5, 11*HightVariation]])
        Patch12_TangentX, Patch12_TangentY, Patch12_Norm = getTerrainTagents_and_Norm(Patch12)
        AllPatches.append(Patch12)
        AllPatchesName.append("Patch12")
        AllTerrainTangentsX.append(Patch12_TangentX)
        AllTerrainTangentsY.append(Patch12_TangentY)
        AllTerrainNorm.append(Patch12_Norm)

        #PlotTerrain(AllSurfaces = AllPatches)

    elif name == "darpa_obstacle_left_first":
        AllPatches = []
        AllPatchesName = []
        AllTerrainTangentsX = []
        AllTerrainTangentsY = []
        AllTerrainNorm = []

        HightVariation =0.07
        HorizontanIncrease = 0.6

        Patch1 = np.array([[1, 0.5, 0.], [-1, 0.5, 0.], [-1, -0.5, 0.], [1, -0.5, 0.]])
        Patch1_TangentX, Patch1_TangentY, Patch1_Norm = getTerrainTagents_and_Norm(Patch1)
        AllPatches.append(Patch1)
        AllPatchesName.append("Patch1")
        AllTerrainTangentsX.append(Patch1_TangentX)
        AllTerrainTangentsY.append(Patch1_TangentY)
        AllTerrainNorm.append(Patch1_Norm)

        Patch2 = np.array([[1+HorizontanIncrease, 0, -HightVariation], [1, 0, -HightVariation], [1, -0.5, HightVariation], [1+HorizontanIncrease, -0.5, HightVariation]])
        Patch2_TangentX, Patch2_TangentY, Patch2_Norm = getTerrainTagents_and_Norm(Patch2)
        AllPatches.append(Patch2)
        AllPatchesName.append("Patch2")
        AllTerrainTangentsX.append(Patch2_TangentX)
        AllTerrainTangentsY.append(Patch2_TangentY)
        AllTerrainNorm.append(Patch2_Norm)

        Patch3 = np.array([[1+HorizontanIncrease, 0.5, -HightVariation], [1, 0.5, HightVariation], [1, 0, HightVariation], [1+HorizontanIncrease, 0, -HightVariation]])
        Patch3_TangentX, Patch3_TangentY, Patch3_Norm = getTerrainTagents_and_Norm(Patch3)
        AllPatches.append(Patch3)
        AllPatchesName.append("Patch3")
        AllTerrainTangentsX.append(Patch3_TangentX)
        AllTerrainTangentsY.append(Patch3_TangentY)
        AllTerrainNorm.append(Patch3_Norm)

        Patch4 = np.array([[1+2*HorizontanIncrease, 0, HightVariation], [1+HorizontanIncrease, 0, HightVariation], [1+HorizontanIncrease, -0.5, -HightVariation], [1+2*HorizontanIncrease, -0.5, -HightVariation]])
        Patch4_TangentX, Patch4_TangentY, Patch4_Norm = getTerrainTagents_and_Norm(Patch4)
        AllPatches.append(Patch4)
        AllPatchesName.append("Patch4")
        AllTerrainTangentsX.append(Patch4_TangentX)
        AllTerrainTangentsY.append(Patch4_TangentY)
        AllTerrainNorm.append(Patch4_Norm)

        Patch5 = np.array([[1+2*HorizontanIncrease, 0.5, HightVariation], [1+HorizontanIncrease, 0.5, -HightVariation], [1+HorizontanIncrease, 0, -HightVariation], [1+2*HorizontanIncrease, 0, HightVariation]])
        Patch5_TangentX, Patch5_TangentY, Patch5_Norm = getTerrainTagents_and_Norm(Patch5)
        AllPatches.append(Patch5)
        AllPatchesName.append("Patch5")
        AllTerrainTangentsX.append(Patch5_TangentX)
        AllTerrainTangentsY.append(Patch5_TangentY)
        AllTerrainNorm.append(Patch5_Norm)

        #making Flat
        #--------------------------
        heightlevel = 0.1
        horizontal_shift = 0
        Patch6 = np.array([[1+3*HorizontanIncrease, 0 + horizontal_shift, -heightlevel], [1+2*HorizontanIncrease, 0 + horizontal_shift, heightlevel], [1+2*HorizontanIncrease, -0.5 + horizontal_shift, heightlevel], [1+3*HorizontanIncrease, -0.5 + horizontal_shift, -heightlevel]])
        Patch6_TangentX, Patch6_TangentY, Patch6_Norm = getTerrainTagents_and_Norm(Patch6)
        AllPatches.append(Patch6)
        AllPatchesName.append("Patch6")
        AllTerrainTangentsX.append(Patch6_TangentX)
        AllTerrainTangentsY.append(Patch6_TangentY)
        AllTerrainNorm.append(Patch6_Norm)

        Patch7 = np.array([[1+3*HorizontanIncrease, 0.5, -heightlevel], [1+2*HorizontanIncrease, 0.5, -heightlevel], [1+2*HorizontanIncrease, 0, heightlevel], [1+3*HorizontanIncrease, 0, heightlevel]])
        Patch7_TangentX, Patch7_TangentY, Patch7_Norm = getTerrainTagents_and_Norm(Patch7)
        AllPatches.append(Patch7)
        AllPatchesName.append("Patch7")
        AllTerrainTangentsX.append(Patch7_TangentX)
        AllTerrainTangentsY.append(Patch7_TangentY)
        AllTerrainNorm.append(Patch7_Norm)
        #---------------------
        Patch8 = np.array([[1+4*HorizontanIncrease, 0, HightVariation], [1+3*HorizontanIncrease, 0, HightVariation], [1+3*HorizontanIncrease, -0.5, -HightVariation], [1+4*HorizontanIncrease, -0.5, -HightVariation]])
        Patch8_TangentX, Patch8_TangentY, Patch8_Norm = getTerrainTagents_and_Norm(Patch8)
        AllPatches.append(Patch8)
        AllPatchesName.append("Patch8")
        AllTerrainTangentsX.append(Patch8_TangentX)
        AllTerrainTangentsY.append(Patch8_TangentY)
        AllTerrainNorm.append(Patch8_Norm)

        Patch9 = np.array([[1+4*HorizontanIncrease, 0.5, HightVariation], [1+3*HorizontanIncrease, 0.5, -HightVariation], [1+3*HorizontanIncrease, 0, -HightVariation], [1+4*HorizontanIncrease, 0, HightVariation]])
        Patch9_TangentX, Patch9_TangentY, Patch9_Norm = getTerrainTagents_and_Norm(Patch9)
        AllPatches.append(Patch9)
        AllPatchesName.append("Patch9")
        AllTerrainTangentsX.append(Patch9_TangentX)
        AllTerrainTangentsY.append(Patch9_TangentY)
        AllTerrainNorm.append(Patch9_Norm)

        Patch10 = np.array([[1+5*HorizontanIncrease, 0, -HightVariation], [1+4*HorizontanIncrease, 0, -HightVariation], [1+4*HorizontanIncrease, -0.5, HightVariation], [1+5*HorizontanIncrease, -0.5, HightVariation]])
        Patch10_TangentX, Patch10_TangentY, Patch10_Norm = getTerrainTagents_and_Norm(Patch10)
        AllPatches.append(Patch10)
        AllPatchesName.append("Patch10")
        AllTerrainTangentsX.append(Patch10_TangentX)
        AllTerrainTangentsY.append(Patch10_TangentY)
        AllTerrainNorm.append(Patch10_Norm)

        Patch11 = np.array([[1+5*HorizontanIncrease, 0.5, -HightVariation], [1+4*HorizontanIncrease, 0.5, HightVariation], [1+4*HorizontanIncrease, 0, HightVariation], [1+5*HorizontanIncrease, 0, -HightVariation]])
        Patch11_TangentX, Patch11_TangentY, Patch11_Norm = getTerrainTagents_and_Norm(Patch11)
        AllPatches.append(Patch11)
        AllPatchesName.append("Patch11")
        AllTerrainTangentsX.append(Patch11_TangentX)
        AllTerrainTangentsY.append(Patch11_TangentY)
        AllTerrainNorm.append(Patch11_Norm)

        Patch12 = np.array([[50.0, 0.5, 0.], [1+5*HorizontanIncrease, 0.5, 0.], [1+5*HorizontanIncrease, -0.5, 0.], [50.0, -0.5, 0.]])
        Patch12_TangentX, Patch12_TangentY, Patch12_Norm = getTerrainTagents_and_Norm(Patch12)
        AllPatches.append(Patch12)
        AllPatchesName.append("Patch12")
        AllTerrainTangentsX.append(Patch12_TangentX)
        AllTerrainTangentsY.append(Patch12_TangentY)
        AllTerrainNorm.append(Patch12_Norm)

    elif name == "darpa_like_right_first":
        AllPatches = []
        AllPatchesName = []
        AllTerrainTangentsX = []
        AllTerrainTangentsY = []
        AllTerrainNorm = []

        HightVariation =0.08
        HorizontanIncrease = 0.6

        Patch1 = np.array([[1, 0.5, 0.], [-1, 0.5, 0.], [-1, -0.5, 0.], [1, -0.5, 0.]])
        Patch1_TangentX, Patch1_TangentY, Patch1_Norm = getTerrainTagents_and_Norm(Patch1)
        AllPatches.append(Patch1)
        AllPatchesName.append("Patch1")
        AllTerrainTangentsX.append(Patch1_TangentX)
        AllTerrainTangentsY.append(Patch1_TangentY)
        AllTerrainNorm.append(Patch1_Norm)

        Patch2 = np.array([[1+HorizontanIncrease, 0.5, -HightVariation], [1, 0.5, HightVariation], [1, 0, HightVariation], [1+HorizontanIncrease, 0, -HightVariation]])
        Patch2_TangentX, Patch2_TangentY, Patch2_Norm = getTerrainTagents_and_Norm(Patch2)
        AllPatches.append(Patch2)
        AllPatchesName.append("Patch2")
        AllTerrainTangentsX.append(Patch2_TangentX)
        AllTerrainTangentsY.append(Patch2_TangentY)
        AllTerrainNorm.append(Patch2_Norm)


        Patch3 = np.array([[1+HorizontanIncrease, 0, -HightVariation], [1, 0, -HightVariation], [1, -0.5, HightVariation], [1+HorizontanIncrease, -0.5, HightVariation]])
        Patch3_TangentX, Patch3_TangentY, Patch3_Norm = getTerrainTagents_and_Norm(Patch3)
        AllPatches.append(Patch3)
        AllPatchesName.append("Patch3")
        AllTerrainTangentsX.append(Patch3_TangentX)
        AllTerrainTangentsY.append(Patch3_TangentY)
        AllTerrainNorm.append(Patch3_Norm)

        Patch4 = np.array([[1+2*HorizontanIncrease, 0.5, HightVariation], [1+HorizontanIncrease, 0.5, -HightVariation], [1+HorizontanIncrease, 0, -HightVariation], [1+2*HorizontanIncrease, 0, HightVariation]])
        Patch4_TangentX, Patch4_TangentY, Patch4_Norm = getTerrainTagents_and_Norm(Patch4)
        AllPatches.append(Patch4)
        AllPatchesName.append("Patch4")
        AllTerrainTangentsX.append(Patch4_TangentX)
        AllTerrainTangentsY.append(Patch4_TangentY)
        AllTerrainNorm.append(Patch4_Norm)

        Patch5 = np.array([[1+2*HorizontanIncrease, 0, HightVariation], [1+HorizontanIncrease, 0, HightVariation], [1+HorizontanIncrease, -0.5, -HightVariation], [1+2*HorizontanIncrease, -0.5, -HightVariation]])
        Patch5_TangentX, Patch5_TangentY, Patch5_Norm = getTerrainTagents_and_Norm(Patch5)
        AllPatches.append(Patch5)
        AllPatchesName.append("Patch5")
        AllTerrainTangentsX.append(Patch5_TangentX)
        AllTerrainTangentsY.append(Patch5_TangentY)
        AllTerrainNorm.append(Patch5_Norm)

        Patch6 = np.array([[1+3*HorizontanIncrease, 0.5, -HightVariation], [1+2*HorizontanIncrease, 0.5, -HightVariation], [1+2*HorizontanIncrease, 0, HightVariation], [1+3*HorizontanIncrease, 0, HightVariation]])
        Patch6_TangentX, Patch6_TangentY, Patch6_Norm = getTerrainTagents_and_Norm(Patch6)
        AllPatches.append(Patch6)
        AllPatchesName.append("Patch6")
        AllTerrainTangentsX.append(Patch6_TangentX)
        AllTerrainTangentsY.append(Patch6_TangentY)
        AllTerrainNorm.append(Patch6_Norm)

        #Patch7 = np.array([[1+3*HorizontanIncrease, 0., HightVariation], [1+2*HorizontanIncrease, 0., -HightVariation], [1+2*HorizontanIncrease, -0.5, -HightVariation], [1+3*HorizontanIncrease, -0.5, HightVariation]])
        Patch7 = np.array([[1+3*HorizontanIncrease, 0, -HightVariation], [1+2*HorizontanIncrease, 0, HightVariation], [1+2*HorizontanIncrease, -0.5, HightVariation], [1+3*HorizontanIncrease, -0.5, -HightVariation]])
        Patch7_TangentX, Patch7_TangentY, Patch7_Norm = getTerrainTagents_and_Norm(Patch7)
        AllPatches.append(Patch7)
        AllPatchesName.append("Patch7")
        AllTerrainTangentsX.append(Patch7_TangentX)
        AllTerrainTangentsY.append(Patch7_TangentY)
        AllTerrainNorm.append(Patch7_Norm)

        Patch8 = np.array([[1+4*HorizontanIncrease, 0.5, HightVariation], [1+3*HorizontanIncrease, 0.5, -HightVariation], [1+3*HorizontanIncrease, 0, -HightVariation], [1+4*HorizontanIncrease, 0, HightVariation]])
        Patch8_TangentX, Patch8_TangentY, Patch8_Norm = getTerrainTagents_and_Norm(Patch8)
        AllPatches.append(Patch8)
        AllPatchesName.append("Patch8")
        AllTerrainTangentsX.append(Patch8_TangentX)
        AllTerrainTangentsY.append(Patch8_TangentY)
        AllTerrainNorm.append(Patch8_Norm)

        Patch9 = np.array([[1+4*HorizontanIncrease, 0, HightVariation], [1+3*HorizontanIncrease, 0, HightVariation], [1+3*HorizontanIncrease, -0.5, -HightVariation], [1+4*HorizontanIncrease, -0.5, -HightVariation]])
        Patch9_TangentX, Patch9_TangentY, Patch9_Norm = getTerrainTagents_and_Norm(Patch9)
        AllPatches.append(Patch9)
        AllPatchesName.append("Patch9")
        AllTerrainTangentsX.append(Patch9_TangentX)
        AllTerrainTangentsY.append(Patch9_TangentY)
        AllTerrainNorm.append(Patch9_Norm)

        Patch10 = np.array([[1+5*HorizontanIncrease, 0.5, -HightVariation], [1+4*HorizontanIncrease, 0.5, HightVariation], [1+4*HorizontanIncrease, 0, HightVariation], [1+5*HorizontanIncrease, 0, -HightVariation]])
        Patch10_TangentX, Patch10_TangentY, Patch10_Norm = getTerrainTagents_and_Norm(Patch10)
        AllPatches.append(Patch10)
        AllPatchesName.append("Patch10")
        AllTerrainTangentsX.append(Patch10_TangentX)
        AllTerrainTangentsY.append(Patch10_TangentY)
        AllTerrainNorm.append(Patch10_Norm)

        Patch11 = np.array([[1+5*HorizontanIncrease, 0, -HightVariation], [1+4*HorizontanIncrease, 0, -HightVariation], [1+4*HorizontanIncrease, -0.5, HightVariation], [1+5*HorizontanIncrease, -0.5, HightVariation]])
        Patch11_TangentX, Patch11_TangentY, Patch11_Norm = getTerrainTagents_and_Norm(Patch11)
        AllPatches.append(Patch11)
        AllPatchesName.append("Patch11")
        AllTerrainTangentsX.append(Patch11_TangentX)
        AllTerrainTangentsY.append(Patch11_TangentY)
        AllTerrainNorm.append(Patch11_Norm)

        Patch14 = np.array([[50.0, 0.5, 0.], [1+5*HorizontanIncrease, 0.5, 0.], [1+5*HorizontanIncrease, -0.5, 0.], [50.0, -0.5, 0.]])
        Patch14_TangentX, Patch14_TangentY, Patch14_Norm = getTerrainTagents_and_Norm(Patch14)
        AllPatches.append(Patch14)
        AllPatchesName.append("Patch14")
        AllTerrainTangentsX.append(Patch14_TangentX)
        AllTerrainTangentsY.append(Patch14_TangentY)
        AllTerrainNorm.append(Patch14_Norm)

        #PlotTerrain(AllSurfaces = AllPatches)
    else:
        raise Exception("Undefined Terrain")

    #Build Contact sequences and tangents and norm sequences
    ContactSeqs = []
    ContactSeqNames = []
    TerrainTangentsX = []
    TerrainTangentsY = []
    TerrainNorms = []

    #Extend Patch Sequence to build contact sequence
    AllPatches_Expand = AllPatches + [AllPatches[-1]]*(NumRounds*2)
    AllPatchesName_Expand = AllPatchesName + [AllPatchesName[-1]]*(NumRounds*2)
    AllTerrainTangentsX_Expand = AllTerrainTangentsX + [AllTerrainTangentsX[-1]]*(NumRounds*2)
    AllTerrainTangentsY_Expand = AllTerrainTangentsY + [AllTerrainTangentsY[-1]]*(NumRounds*2)
    AllTerrainNorm_Expand = AllTerrainNorm + [AllTerrainNorm[-1]]*(NumRounds*2)

    if name == "gap":
        AllPatches_Expand = [Patch1,Patch1,Patch1,Patch1,Patch1,
                             Patch2,Patch2,Patch2,Patch2,Patch2,Patch2,Patch2,Patch2,Patch2,Patch2,
                             Patch3,Patch3,Patch3,Patch3,Patch3,Patch3,Patch3,Patch3,Patch3,Patch3,Patch3,Patch3,Patch3,Patch3,Patch3,Patch3,Patch3,Patch3]
        AllTerrainTangentsX_Expand = [Patch1_TangentX,Patch1_TangentX,Patch1_TangentX,Patch1_TangentX,Patch1_TangentX,
                                      Patch2_TangentX,Patch2_TangentX,Patch2_TangentX,Patch2_TangentX,Patch2_TangentX,Patch2_TangentX,Patch2_TangentX,Patch2_TangentX,Patch2_TangentX,Patch2_TangentX,
                                      Patch3_TangentX,Patch3_TangentX,Patch3_TangentX,Patch3_TangentX,Patch3_TangentX,Patch3_TangentX,Patch3_TangentX,Patch3_TangentX,Patch3_TangentX,Patch3_TangentX,Patch3_TangentX,Patch3_TangentX,Patch3_TangentX,Patch3_TangentX,Patch3_TangentX,Patch3_TangentX,Patch3_TangentX,Patch3_TangentX]
        AllTerrainTangentsY_Expand = [Patch1_TangentY,Patch1_TangentY,Patch1_TangentY,Patch1_TangentY,Patch1_TangentY,
                                      Patch2_TangentY,Patch2_TangentY,Patch2_TangentY,Patch2_TangentY,Patch2_TangentY,Patch2_TangentY,Patch2_TangentY,Patch2_TangentY,Patch2_TangentY,Patch2_TangentY,
                                      Patch3_TangentY,Patch3_TangentY,Patch3_TangentY,Patch3_TangentY,Patch3_TangentY,Patch3_TangentY,Patch3_TangentY,Patch3_TangentY,Patch3_TangentY,Patch3_TangentY,Patch3_TangentY,Patch3_TangentY,Patch3_TangentY,Patch3_TangentY,Patch3_TangentY,Patch3_TangentY,Patch3_TangentY,Patch3_TangentY]
        AllTerrainNorm_Expand = [Patch1_Norm,Patch1_Norm,Patch1_Norm,Patch1_Norm,Patch1_Norm,
                                 Patch2_Norm,Patch2_Norm,Patch2_Norm,Patch2_Norm,Patch2_Norm,Patch2_Norm,Patch2_Norm,Patch2_Norm,Patch2_Norm,Patch2_Norm,
                                 Patch3_Norm,Patch3_Norm,Patch3_Norm,Patch3_Norm,Patch3_Norm,Patch3_Norm,Patch3_Norm,Patch3_Norm,Patch3_Norm,Patch3_Norm,Patch3_Norm,Patch3_Norm,Patch3_Norm,Patch3_Norm,Patch3_Norm,Patch3_Norm,Patch3_Norm,Patch3_Norm]
    elif name == "jump":
                AllPatches_Expand = [Patch1,Patch1,Patch1,Patch1,
                                     Patch2,Patch2,Patch2,Patch2,Patch2,Patch2,Patch2,Patch2,Patch2,Patch2] + [Patch2]*100
                AllTerrainTangentsX_Expand = [Patch1_TangentX,Patch1_TangentX,Patch1_TangentX,Patch1_TangentX,
                                              Patch2_TangentX,Patch2_TangentX,Patch2_TangentX,Patch2_TangentX,Patch2_TangentX,Patch2_TangentX,Patch2_TangentX,Patch2_TangentX,Patch2_TangentX,Patch2_TangentX] + [Patch2_TangentX]*100
                AllTerrainTangentsY_Expand = [Patch1_TangentY,Patch1_TangentY,Patch1_TangentY,Patch1_TangentY,
                                              Patch2_TangentY,Patch2_TangentY,Patch2_TangentY,Patch2_TangentY,Patch2_TangentY,Patch2_TangentY,Patch2_TangentY,Patch2_TangentY,Patch2_TangentY,Patch2_TangentY] + [Patch2_TangentY]*100
                AllTerrainNorm_Expand = [Patch1_Norm,Patch1_Norm,Patch1_Norm,Patch1_Norm,
                                         Patch2_Norm,Patch2_Norm,Patch2_Norm,Patch2_Norm,Patch2_Norm,Patch2_Norm,Patch2_Norm,Patch2_Norm,Patch2_Norm,Patch2_Norm] + [Patch2_Norm]*100

    #From round 0, the contact patch starts from 0
    for roundIndex in range(NumRounds):

        ContactSeq_Temp = AllPatches_Expand[roundIndex:roundIndex + NumContactSequence-1+1]
        ContactSeqName_Temp = AllPatchesName_Expand[roundIndex:roundIndex + NumContactSequence-1+1]
        TerrainTangentsX_Temp = AllTerrainTangentsX_Expand[roundIndex:roundIndex + NumContactSequence-1+1]
        TerrainTangentsY_Temp = AllTerrainTangentsY_Expand[roundIndex:roundIndex + NumContactSequence-1+1]
        TerrainNorms_Temp = AllTerrainNorm_Expand[roundIndex:roundIndex + NumContactSequence-1+1]

        ContactSeqs.append(ContactSeq_Temp)
        ContactSeqNames.append(ContactSeqName_Temp)
        TerrainTangentsX.append(TerrainTangentsX_Temp)
        TerrainTangentsY.append(TerrainTangentsY_Temp)
        TerrainNorms.append(TerrainNorms_Temp)

    print("Tangent X: ",TerrainTangentsX)
    print("Tangent Y: ",TerrainTangentsY)
    print("Norm: ",TerrainNorms)

    return AllPatches, ContactSeqs, TerrainTangentsX, TerrainTangentsY, TerrainNorms


#def RandomTerrainGeneration(name = "random", NumRounds = 3, NumContactSequence = 4, FlatInitPatch = True):
#    #Assume the first step always swing the LEFT foot for now
#
#    TotalNumPatch = NumRounds + 2*NumContactSequence #will be always enough, and always symmetric
#
#    #Terrain Definition Container
#    AllPatches = []
#    AllPatchesName = []
#    AllTerrainTangentsX = []
#    AllTerrainTangentsY = []
#    AllTerrainNorm = []

    #Generate the Initial 






