"""
This script generates the train and test files for the MAD dataset.
5 folds of 4 subjects each
"""
import os
pathData = "C:\\workspace2\\Datasets\\MAD\\Data\\"
pathSplit = "C:\\workspace2\\Datasets\\MAD\\Split\\"

if not os.path.exists(pathSplit):
    print("Split file does not exist")
    os.makedirs(pathSplit)
#get the 2-number subject id
subject_ids = ["%02d"%i for i in range(1, 21)]

#shuffle the subject ids
import random
random.seed(5)
random.shuffle(subject_ids)
#split the data into 5 folds of 4 subjects each
folds = []
for i in range(5):
    folds.append(subject_ids[i*4:(i+1)*4])


print(folds)

#generate the train and test files
for i,f  in enumerate(folds):
    train = [s for s in subject_ids if s not in f]
    test = f
    train_str = ["sub"+s+"_01" for s in train] + ["sub"+s+"_02" for s in train]
    test_str = ["sub"+s+"_01" for s in test] + ["sub"+s+"_02" for s in test]

    train_str_final = ",".join(train_str)
    test_str_final = ",".join(test_str)
    with open(pathSplit+"fold%d.txt"%i, "w+") as f:
        f.write("training \n")
        f.write(train_str_final+"\n")
        f.write("testing\n")
        f.write(test_str_final)

