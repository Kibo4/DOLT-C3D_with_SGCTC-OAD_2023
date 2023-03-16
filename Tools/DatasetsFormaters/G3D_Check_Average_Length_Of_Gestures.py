"""
This script is used to check the average length of gestures in G3D dataset
Category Fighting
"""

pathLabel = "C:\workspace2\Datasets\G3D\Label\\"
seqFighting = [22,23,24,43,44,45,64,65,66,85,86,87,106,107,108,127,128,129,148,149,150,190,191,192,214,215,216, 169,170,171]

summ = 0
count = 0
for f in seqFighting:
    file = open(pathLabel + str(f), "r")
    lines = file.readlines()
    file.close()

    lines = list(filter(lambda l : len(l.split(","))==4,lines))
    for l in lines:
        _,start,end,_ = l.split(",")
        summ += int(end)-int(start)+1
        print(int(end)-int(start)+1)
        count += 1

print("avg length of gesture : ",summ/count)

