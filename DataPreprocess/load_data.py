
from Decompress_Data import Decompress
import os

class LoadData:
    def __init__(self , data_path , type_comp):
        self.data_path = data_path
        self.type_comp = type_comp
        self.check_compress_type()

    def check_compress_type(self):
        if self.type_comp == ".zip":
            self.process_file(".zip" , self.de_zip)
        elif self.type_comp == ".tar":
            self.process_file(".tar" , self.de_tar)
        else :
            print("no thing compress type using 'zip' , or 'tar'" )
    
    def process_file(self , extention , decompress_method):
        if not os.path.exists(self.data_path):
            print("the path not found")
            return []
        compress_file = []
        for root , dir , files in os.walk(self.data_path):
            for file in files :
                if file.endswith(extention):
                    compress_file.append(os.path.join(root,file))
        if not compress_file :
            print(f"no extention with {extention} in this path : {self.data_path}")
            return None
        for file in compress_file:
            # (os.path.splitext) -> return two items the pathe+the name without extention and the secand item is extention 
            path_unzip_file = os.path.splitext(file)[0] 
            decompress_method(file ,path_unzip_file )

    def de_zip(self , file_path , output_path):

            dezip_train = Decompress(file_path ,output_path)
            dezip_train.dacompressZIP()
            print(f"Decompressing zip: {file_path} -> {output_path}")
            
    
    def de_tar(self , file_path , output_path):
            detar_train = Decompress(file_path ,output_path )
            detar_train.dacompressZIP()
            print(f"Decompressing TAR: {file_path} -> {output_path}")




    
    


        