import face_recognition as fr
import os
import shutil
from pathlib import Path

pltfrm_pth_sep=os.path.sep
absolute_path="D:"+pltfrm_pth_sep+"PythonProjects"+pltfrm_pth_sep+"ComputerVision"

# Reference images for the subjects to be classified
aadish_image = fr.load_image_file("Aadish.jpg")
anisha_image = fr.load_image_file("Anisha.jpg")

aadish_face_encoding = fr.face_encodings(aadish_image)[0]
anisha_face_encoding = fr.face_encodings(anisha_image)[0]

known_faces = [
    aadish_face_encoding,
    anisha_face_encoding
    ]

#Grab a folder containing the pictures to classify
data="data"+pltfrm_pth_sep+"datapictures"
aadish_pictures=data+pltfrm_pth_sep+"aadish_pictures"
anisha_pictures=data+pltfrm_pth_sep+"anisha_pictures"


#create folders
if not os.path.exists(aadish_pictures):
    os.makedirs(aadish_pictures)
if not os.path.exists(anisha_pictures):
    os.makedirs(anisha_pictures)

#input folder
directory = data+pltfrm_pth_sep+"input"
files = Path(directory).glob('*')

#meat of the logic
def process_files(file_name):
    unknown_image=fr.load_image_file(directory+pltfrm_pth_sep+file_name)
    unknown_face_encodings = fr.face_encodings(unknown_image)
    print("Detected {} faces in the given input image".format(len(unknown_face_encodings)))
    for face_encodings in unknown_face_encodings:
        results = fr.compare_faces(known_faces, face_encodings)
        if results[0]:
            print("Aadish found in the picture")
            shutil.copy(directory+pltfrm_pth_sep+file_name,aadish_pictures)
        elif results[1]:
            print("Anisha found in the picture")
            shutil.copy(directory + pltfrm_pth_sep + file_name, anisha_pictures)
        else:
            print("Unknown people are present in the picture")

for file in files:
    process_files(file.name)


