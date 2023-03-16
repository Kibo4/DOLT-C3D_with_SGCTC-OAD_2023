class Label:
    def __init__(self, classe, classId, beginPostureId, endPostureId, fileName="", actionPoint = -1):
        self.fileName = fileName
        self.classId = classId
        self.endPostureId = endPostureId
        self.beginPostureId = beginPostureId
        self.classe = classe
        self.actionPoint = actionPoint

    def __repr__(self) -> str:
        return str(self)

    def __str__(self) -> str:
        return f"(fileName : {self.fileName};classe : {self.classe},classid : {self.classId}, begin : {self.beginPostureId}, end : {self.endPostureId})"
