class JointType:
    def __init__(self,id,name=""):
        self.id=id
        self.name=name

    def __repr__(self):
        return str(self)

    def __str__(self):
        return "("+str(self.id)+","+self.name+")"

    def __eq__(self, other):
        return self.id==other.id

