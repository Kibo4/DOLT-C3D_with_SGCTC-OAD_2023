"""
This file contains the class PostureToFile which is used to save a list of postures to a file
"""
from typing import List, Tuple

from Tools.Gesture.Posture import Posture


class PostureToFile:

    @classmethod
    def toFile(cls,postures : List[Posture],path:str):
        fileLines = []
        for pos in postures:
            line = " ".join(map(lambda j : " ".join(map(lambda x:str(x),j.position)), pos.joints))+"\n"
            fileLines.append(line)

        f = open(path,"w+")
        f.writelines(fileLines)
        f.close()


    @classmethod
    def toFile2Sq(cls,postures : List[Posture],postures2 : List[Posture],path:str):
        fileLines = []
        assert len(postures)==len(postures2)
        for id,pos in enumerate(postures):
            line = " ".join(map(lambda j: " ".join(map(lambda x: str(x), j.position)), pos.joints))
            line += " "+" ".join(map(lambda j: " ".join(map(lambda x: str(x), j.position)), postures2[id].joints)) + "\n"
            fileLines.append(line)

        f = open(path, "w+")
        f.writelines(fileLines)
        f.close()


    @classmethod
    def toFilePerJoint(cls,postures : List[Posture],path:str):
        fileLines = [[] for _ in range(len(postures[0].joints))]
        for pos in postures:
            for i,join in enumerate(pos.joints):
                fileLines[i].append(join.position)
        text = ""
        for lineJoint in fileLines:
            text += ", ".join(map(" ".join,lineJoint ))+"\n"

        f = open(path,"w+")
        f.writelines(text)
        f.close()

class LabelsToFile:
    @classmethod
    def toFile(cls,labels : List[Tuple[any,int,int]], path:str):
        fileLines = []
        for lab in labels:
            line = ",".join(list(map(str,lab)))+"\n"
            fileLines.append(line)

        f = open(path,"w+")
        f.writelines(fileLines)
        f.close()