import os



class Dataset:

    def __init__(self, name, path):
        self.name = name
        self.path = path



class FejkDataset(Dataset):

    def __init__(self, name, path):
        super().__init__(name, path)
        self.images = []


    def findImages(self):
        self.images = os.listdir(self.path)

#ds = FejkDataset("FejkovyDataset", "D:/bakalarkaaaa/MOJDATASET/")
#ds.findImages()
#print(ds.images[0])
