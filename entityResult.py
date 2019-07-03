

class entityResult():

    def __init__(self, fileName):
        self.fileName = fileName
        self.precision = []
        self.recall = []
        self.F1 = []
        self.readResult()

    def readResult(self):
        with open(self.fileName,'r') as f:
            i = 0
            for line in f:
                i += 1
                if i == 10:
                    line = line.strip()
                    ls = line.split(' ')
                    self.precision.append(float(ls[4]))
                    self.recall.append(float(ls[5]))
                    self.F1.append(float(ls[6]))
                    i = 0

    def getPrecision(self):
        return self.precision

    def getRecall(self):
        return self.recall

    def getF1(self):
        return self.F1
