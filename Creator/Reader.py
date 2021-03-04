# Reader bude objekt ktory bude nacitavat datasety a vyuzivat jeho funkcie

import os
import shutil


class FileManager:

    def __init__(self):
        pass

    def renameFile(self, src, newName):
        os.rename(src, rf'D:\bakalarkaaaa\MOJDATASET\{newName}' )

    def copyAndMoveFile(self, src, dst):
        shutil.copyfile(src, dst)

    def createDir(self, path):
        try:
            os.mkdir(path)
        except FileExistsError:
            print("I cant make this dir, because it is already created or the path is wrong!")

#fm = FileManager()
#fm.renameFile(r'D:\bakalarkaaaa\MOJDATASET\12.jpg','14.jpg')
#fm.moveFile('D:/bakalarkaaaa/MOJDATASET/14.jpg','try.jpg')