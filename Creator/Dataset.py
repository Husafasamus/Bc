import os



class Dataset:

    def __init__(self, name, path):
        self.name = name
        self.path = path

    def find_images(self):
        print(f"{self.name}'s function findImages() has not been defined!")


class FejkDataset(Dataset):

    def __init__(self, name, path):
        super().__init__(name, path)
        self.images = []


    def find_images(self): #  Tato funkcia mi najde vsetky obrazky v danom priecinku pre dany dataset,
        self.images = os.listdir(self.path)
        for x in range(len(self.images)):
            self.images[x] = f"{self.path}{self.images[x]}"



