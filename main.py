from imutils import face_utils
import imutils
import cv2
import dlib
import os
import PIL.Image 
import PIL.ImageTk
import numpy as np
from tkinter import *
from tkinter import messagebox

################################################## MAIN WINDOW DISPLAY ##################################################

window = Tk()
window.title("FaceRecognition 1.0")
window.geometry('260x400')

################################################## DISPLAY COMPANY LOGO ##################################################

image = PIL.Image.open("logo.png").resize((200, 200), PIL.Image.ANTIALIAS)
photo = PIL.ImageTk.PhotoImage(image)

label = Label(image=photo)
label.image = photo # keep a reference!
label.grid(column = 0, row = 0)

################################################## LABELS ##################################################

lbl1 = Label(window, text = "Please enter your ID number", font = ("System", 20))
lbl1.grid(column = 0, row = 30)

lbl2 = Label(window, text = "v.1.0 | Athiya Deviyani", font = ("System", 10))
lbl2.grid(column = 0, row = 300)

################################################## TEXTBOX ##################################################

idn = Entry(window, width = 10, font = ("System", 14))
idn.grid(column = 0, row = 60)

################################################## FACE CAPTURE ##################################################

face_id = 0 # initialize face_id = 0 if no input!

def input():
    face_id = idn.get()
    detector = dlib.get_frontal_face_detector() 
    predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

    camera = cv2.VideoCapture(0)

    count = 0 

    while(True): 
        ret, image = camera.read()
        image = cv2.flip(image, 1) # mirror the image (flip vertically)
        image = imutils.resize(image, width = 800, height = 600) # (set the size of the frame)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = detector(gray, 0)

        for face in faces:
            faceshape = predictor(image, face)
            faceshape = face_utils.shape_to_np(faceshape)
            for (x,y) in faceshape:
                cv2.circle(image, (x,y), 1, (255, 255, 255), -1)

        cv2.putText(image,"PRESS 'ESC' TO EXIT", (150, 460), cv2.FONT_HERSHEY_DUPLEX, 1, (0,255,0), 2)
        
        for face in faces:
            (x, y, w, h) = face_utils.rect_to_bb(face)
            cv2.rectangle(image, (x,y), (x+w, y+h), (255,0,0), 2)
            count+=1
            cv2.imwrite("facedata/User." + str(face_id) + '.' + str(count) + ".jpg", gray[y:y+h,x:x+w])
        
        cv2.imshow('FACE CAPTURE', image)
        k = cv2.waitKey(100) & 0xff
        if k == 27:
            break
        elif count >= 30: # Take 30 face samples for each ID and stop video automatically
            break
        
    print("\n FACE CAPTURE DONE") # prints to console
    camera.release()
    cv2.destroyAllWindows()

################################################## LEARNING PHASE ##################################################

def learn():
    path = 'facedata'
    recognizer = cv2.face.LBPHFaceRecognizer_create(); 
    detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

    def getImagesAndLabels(path):
        imagePaths = [os.path.join(path,f) for f in os.listdir(path)]     
        faceSamples=[]
        ids = []
        for imagePath in imagePaths:
            if imagePath == 'facedata/.DS_Store':
                continue # ignores the .DS_Store file (hidden by default)
            PIL_img = PIL.Image.open(imagePath).convert('L') 
            img_numpy = np.array(PIL_img,'uint8')

            id = int(os.path.split(imagePath)[-1].split(".")[1])
            faces = detector.detectMultiScale(img_numpy)
            for (x,y,w,h) in faces:
                faceSamples.append(img_numpy[y:y+h,x:x+w])
                ids.append(id)

        return faceSamples,ids
    
    print("\n TRAINING BEGUN") # prints to console when training starts

    faces, ids = getImagesAndLabels(path)
    recognizer.train(faces, np.array(ids))

    recognizer.write('trainer/newtrainer.yml')
    print("\n TRAINING DONE. {0} faces trained.".format(len(np.unique(ids)))) # prints to console the number of faces trained
    
    # displays a dialog box alerting the user that the training was successful, complete with the number of faces in the dataset that was learned.
    messagebox.showinfo('TRAINING DONE', 'All {0} faces in the dataset have been learned.'.format(len(np.unique(ids))))

################################################## REAL TIME FACIAL RECOGNITION ##################################################

def recognize():
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read('trainer/newtrainer.yml')
    faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml');
    font = cv2.FONT_HERSHEY_DUPLEX

    log = [line.split(';') for line in open('members.csv')]
    names = []
    namecount = 1

    for name in range(len(log) - 1):
        if "\n" in log[namecount][1]:
            n = log[namecount][1][:len(log[namecount][1]) - 1]
        else:
            n = log[namecount][1]
        names.append(n)
        namecount+=1

    id = 0

    cam = cv2.VideoCapture(0)
    cam.set(3, 640) # set video width
    cam.set(4, 480) # set video height 

    # Define min. window size to be recognized as a face
    minW = 0.1*cam.get(3)
    minH = 0.1*cam.get(4)

    while True:
        ret, img = cam.read() 
        img = cv2.flip(img, 1) # NO NEED TO FLIP!!!
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # convert to grayscale of course, to increase contrast

        faces = faceCascade.detectMultiScale(gray, scaleFactor = 1.25, minNeighbors = 8, minSize = (int(minW), int(minH)),)
        for (x,y,w,h) in faces: 
            cv2.rectangle(img, (x,y), (x+w, y+h), (0,255,0), 5)

            id, confidence = recognizer.predict(gray[y:y+h, x:x+w])
        
            probability = round(100 - confidence)
            
            # match has to be at least 40% to be recognized as a unique individual.
            if (probability >= 40):
                name = names[id - 1]
                confidence = "  {0}%".format(probability)
            elif (probability < 40):
                id = "-"
                name = "unknown"
                confidence = " "
        
            cv2.putText(img, "ID: " + str(id) + " Name: " + str(name), (x+5, y-5), font, 0.5, (255,255,255), 1)
            cv2.putText(img, str(confidence), (x+5, y+h-5), font, 1, (255,255,0), 1)
        
        cv2.putText(img,"PRESS 'ESC' TO EXIT", (150, 460), cv2.FONT_HERSHEY_DUPLEX, 1, (0,255,0), 2)

        cv2.imshow('RECOGNIZER', img)
        k = cv2.waitKey(10) & 0xff 
        if k == 27:
            break

    cam.release() 
    cv2.destroyAllWindows()

################################################## BUTTONS ##################################################

ok = Button(window, text = "ENTER", bg = "black", fg = "black", command = input, font = ("Arial",20))
ok.grid(column = 0, row = 90)

learn = Button(window, text = "REFRESH", bg = "black", fg = "black", command = learn, font = ("Arial",20))
learn.grid(column = 0, row = 120)

recognize = Button(window, text = "RECOGNIZE", bg = "black", fg = "black", command = recognize, font = ("Arial",20))
recognize.grid(column = 0, row = 150)

################################################## MOST IMPORTANT PART ##################################################

window.mainloop()
