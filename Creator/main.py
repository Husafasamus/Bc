from Creator import Handler as hn
from Creator import Dataset as ds
from Creator.Reader import FileManager as fm
import os
import json



if os.path.isdir("D:/bakalarkaaaa/MOJDATASET/"):
    pass
if os.path.isdir("D:/bakalarkaaaa/MOJDATASET2/"):
    pass




fejkDs1 = ds.FejkDataset("FejkDataset1", "D:/bakalarkaaaa/MOJDATASET/")
fejkDs2 = ds.FejkDataset("FejkDataset2", "D:/bakalarkaaaa/MOJDATASET2/")



#fejkDs.find_images()
#print(os.getcwd())
#rd.FileManager.create_dir("../NEWDATASET")




#index = 1
#for item in fejkDs.images:
#    rd.FileManager.copy_move(fr"{item}", fr"D:\bakalarkaaaa\NEWDATASET\{index}.jpg")
#    index+=1

#rd.FileManager.rename(r"D:\bakalarkaaaa\NEWDATASET\1.jpg", r"D:\bakalarkaaaa\NEWDATASET\x.jpg")

handler = hn.Handler("../NEWDATASET", fejkDs1, fejkDs2, "x")
handler.merge()

#fm.create_file("../new.json")



#json.loads(data)









