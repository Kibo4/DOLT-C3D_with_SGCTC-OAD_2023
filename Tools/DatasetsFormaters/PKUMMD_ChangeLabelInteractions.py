"""
After selecting only the sequences with 2 persons, we update the labels with this script.
"""
import os
pathLabel = "C:\\workspace2\\Datasets\\PKUMMDv1\\LabelInteractions\\"
actionOriginal = "C:\\workspace2\\Datasets\\PKUMMDv1\\Split\\Actions.csv"
newAction = "C:\\workspace2\\Datasets\\PKUMMDv1\\Split\\ActionsInteractions.csv"

file = open(newAction,"r")
lines =file.readlines()
file.close()
idNewAction = [""]*len(lines)
for l in lines:
    idAct = l.split(';')
    idNewAction[int(idAct[0])]=idAct[1].strip()

file = open(actionOriginal,"r")
lines =file.readlines()
file.close()
idOldAction = [""]*len(lines)
for l in lines:
    idAct = l.split(';')
    idOldAction[int(idAct[0])]=idAct[1].strip()


files =  os.listdir(pathLabel)


for f in files :
    file = open(pathLabel+"\\"+str(f),"r")
    lines = file.readlines()
    file.close()
    linesNew = lines
    for idLine,line in enumerate(lines):
        for idOld,actionName in enumerate(idOldAction):
            try:
                idNew = idNewAction.index(actionName)
                elems = line.split(",")
                if(elems[0]!=str(idOld)):
                    continue
                elems[0] = elems[0].replace(str(idOld),str(idNew))
                newLine = ",".join(elems)
                linesNew[idLine] = newLine
            except ValueError as e:
                print(e)
                pass
    file = open(pathLabel + "\\" + str(f), "w")
    file.writelines("".join(linesNew))
    file.close()

