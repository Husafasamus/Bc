import os
import shutil


class FileManager:

    @classmethod
    def rename(self, src, newName):
        os.rename(src, newName)

    @classmethod
    def copy_move(self, src, dst):
        shutil.copyfile(src, dst)

    @classmethod
    def create_dir(self, path):
        try:
            os.mkdir(path)
        except FileExistsError:
            print("I cant make this dir, because it is already created or the path is wrong!")

    @classmethod
    def create_file(self, path):
        try:
            with open(path, 'w'):
                pass
        except FileExistsError:
            print("I cant make this nod, because it is already created or the path is wrong!")
#fm = FileManager()
#fm.renameFile(r'D:\bakalarkaaaa\MOJDATASET\12.jpg','14.jpg')
#fm.moveFile('D:/bakalarkaaaa/MOJDATASET/14.jpg','try.jpg')