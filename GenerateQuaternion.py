import numpy as np
from Tools import *
import pickle

AllPatches = []
AllQuat = []
AllPatchesName = []

HightVariation =0.07

Patch1 = np.array([[1, 0.5, 0.], [-1, 0.5, 0.], [-1, -0.5, 0.], [1, -0.5, 0.]])
Patch1_Quat = getQuaternion(Patch1)
AllQuat.append(Patch1_Quat)
AllPatches.append(Patch1)
AllPatchesName.append("Patch1")
print(Patch1_Quat)

Patch2 = np.array([[1.6, 0, HightVariation], [1, 0, -HightVariation], [1, -0.5, -HightVariation], [1.6, -0.5, HightVariation]])
Patch2_Quat = getQuaternion(Patch2)
AllQuat.append(Patch2_Quat)
AllPatches.append(Patch2)
AllPatchesName.append("Patch2")
print(Patch2_Quat)

Patch3 = np.array([[1.6, 0.5, HightVariation], [1, 0.5, -HightVariation], [1, 0, -HightVariation], [1.6, 0, HightVariation]])
Patch3_Quat = getQuaternion(Patch3)
AllQuat.append(Patch3_Quat)
AllPatches.append(Patch3)
AllPatchesName.append("Patch3")
print(Patch3_Quat)

Patch4 = np.array([[2.2, 0, -HightVariation], [1.6, 0, HightVariation], [1.6, -0.5, HightVariation], [2.2, -0.5, -HightVariation]])
Patch4_Quat = getQuaternion(Patch4)
AllQuat.append(Patch4_Quat)
AllPatches.append(Patch4)
AllPatchesName.append("Patch4")
print(Patch4_Quat)

Patch5 = np.array([[2.2, 0.5, -HightVariation], [1.6, 0.5, HightVariation], [1.6, 0, HightVariation], [2.2, 0, -HightVariation]])
Patch5_Quat = getQuaternion(Patch5)
AllQuat.append(Patch5_Quat)
AllPatches.append(Patch5)
AllPatchesName.append("Patch5")
print(Patch5_Quat)

Patch6 = np.array([[2.8, 0, HightVariation], [2.2, 0, -HightVariation], [2.2, -0.5, -HightVariation], [2.8, -0.5, HightVariation]])
Patch6_Quat = getQuaternion(Patch6)
AllQuat.append(Patch6_Quat)
AllPatches.append(Patch6)
AllPatchesName.append("Patch6")
print(Patch6_Quat)

Patch7 = np.array([[2.8, 0.5, HightVariation], [2.2, 0.5, -HightVariation], [2.2, 0, -HightVariation], [2.8, 0, HightVariation]])
Patch7_Quat = getQuaternion(Patch7)
AllQuat.append(Patch7_Quat)
AllPatches.append(Patch7)
AllPatchesName.append("Patch7")
print(Patch7_Quat)

Patch8 = np.array([[3.4, 0, -HightVariation], [2.8, 0, HightVariation], [2.8, -0.5, HightVariation], [3.4, -0.5, -HightVariation]])
Patch8_Quat = getQuaternion(Patch8)
AllQuat.append(Patch8_Quat)
AllPatches.append(Patch8)
AllPatchesName.append("Patch8")
print(Patch8_Quat)

Patch9 = np.array([[3.4, 0.5, -HightVariation], [2.8, 0.5, HightVariation], [2.8, 0, HightVariation], [3.4, 0, -HightVariation]])
Patch9_Quat = getQuaternion(Patch9)
AllQuat.append(Patch9_Quat)
AllPatches.append(Patch9)
AllPatchesName.append("Patch9")
print(Patch9_Quat)

Patch10 = np.array([[4, 0, HightVariation], [3.4, 0, -HightVariation], [3.4, -0.5, -HightVariation], [4, -0.5, HightVariation]])
Patch10_Quat = getQuaternion(Patch10)
AllQuat.append(Patch10_Quat)
AllPatches.append(Patch10)
AllPatchesName.append("Patch10")
print(Patch10_Quat)

Patch11 = np.array([[4, 0.5, HightVariation], [3.4, 0.5, -HightVariation], [3.4, 0, -HightVariation], [4, 0, HightVariation]])
Patch11_Quat = getQuaternion(Patch11)
AllQuat.append(Patch11_Quat)
AllPatches.append(Patch11)
AllPatchesName.append("Patch11")
print(Patch11_Quat)

Patch12 = np.array([[12.0, 0.5, 0.], [4, 0.5, 0.], [4, -0.5, 0.], [12.0, -0.5, 0.]])
Patch12_Quat = getQuaternion(Patch12)
AllQuat.append(Patch12_Quat)
AllPatches.append(Patch12)
AllPatchesName.append("Patch12")
print(Patch12_Quat)

DumpedResult = {"Quaternions":AllQuat
}
pickle.dump(DumpedResult, open('Up_and_Down_Quaternions.p', "wb"))  # save it into a file named save.p

