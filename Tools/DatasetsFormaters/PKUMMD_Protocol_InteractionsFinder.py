"""
This cript was used to generate either the interactions or the 1-person protocoles for the PKUMMD dataset.
"""
import os
pathLabel = "C:\\workspace2\\Datasets\\PKUMMDv1\\Label\\"
pathProtocol = "C:\\workspace2\\Datasets\\PKUMMDv1\\Split\\cross-view.txt"
pathProtocolOUT = "C:\\workspace2\\Datasets\\PKUMMDv1\\Split\\cross-view-1pers.txt"

files =  os.listdir(pathLabel)


allFiles = []
for f in files :
    file = open(pathLabel+"\\"+str(f),"r")
    interact = ["12","14","16","18","21","24","26","27"]
    lines = file.readlines()
    tmp = []
    for line in lines:
        elem = line.split(",")[0]
        if elem not in tmp:
            tmp.append(elem)

    doAdd = True
    for t in tmp: # if all of the labels are in the list
        if t not in interact:
            doAdd = False
            break
                #if t not in all :
                #    print(t+"in" + f)
                #    all.append(t)
    if doAdd:# if you want to add only the interactions files
        pass
        #allFiles.append(f)
    else: # if you want to add only the 1-person files
        allFiles.append(f)

listTrain = []
listValid = []

f = open(pathProtocol)
lines = f.readlines()
f.close()

training = [s.strip()+".txt" for s in lines[1].split(',')]
valid = [s.strip()+".txt" for s in lines[3].split(',')]

print(len(training))
print(len(valid))

trainingInteraction = [s for s in training if s in allFiles]
validInteraction = [s for s in valid if s in allFiles]


print("trainingInteraction",len(trainingInteraction))
print(trainingInteraction)
print("validInteraction",len(validInteraction))
print(validInteraction)

f = open(pathProtocolOUT,"w+")
f.write("Training\n")
f.write(",".join(trainingInteraction)+"\n")
f.write("Validation\n")
f.write(",".join(validInteraction)+"\n")
f.close()
