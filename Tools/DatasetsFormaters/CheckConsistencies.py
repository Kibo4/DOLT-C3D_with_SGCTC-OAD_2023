"""
This script is used to convert the check the consistency of annoations of the PKUMMD dataset
print an error if the end of an action is before the start
"""

import os
import numpy as np
pathData = "C:\\workspace2\\Datasets\\PKUMMDv1\\Data\\"
pathLabel = "C:\\workspace2\\Datasets\\PKUMMDv1\\Label1pers\\"
pathProtocol = "C:\\workspace2\\Datasets\\PKUMMDv1\\Split\\cross-subject-1pers.txt"
# pathProtocolOUT = "C:\\workspace2\\Datasets\\PKUMMDv1\\Split\\cross-subject-interactions-2pers.txt"

files = os.listdir(pathLabel)
for f in files:
    fi = open(pathLabel+"\\"+str(f),"r")
    lines = fi.readlines()
    fi.close()

    for line in lines:
        slpitted=  line.split(",")
        start = slpitted[1]
        end = slpitted[2]
        if int(end)-int(start) < 0:
            print("ERROR")
            print(f)
            print(line)
            continue
print("END")
