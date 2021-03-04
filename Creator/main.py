from Creator import FejkDataset as ds
from Creator import Reader as rd
import os

fejkDs = ds.FejkDataset("FejkDataset", "D:/bakalarkaaaa/MOJDATASET/")
manager = rd.FileManager();

fejkDs.findImages()
print(os.getcwd())
manager.createDir("../NEWDATASET")

index = 1
for item in fejkDs.images:
    manager.copyAndMoveFile(fr"D:\bakalarkaaaa\MOJDATASET\{item}", fr"D:\bakalarkaaaa\NEWDATASET\{index}.jpg")
    index+=1
