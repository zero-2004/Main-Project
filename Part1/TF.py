import pyautogui
import time
from keras.models import load_model 
import cv2 
import numpy as np

np.set_printoptions(suppress=True)

model = load_model("Part1\\keras_model.h5", compile=False)

class_names = open("Part1\\labels.txt", "r").readlines()

camera = cv2.VideoCapture(0)

#
labels=["A","B","C","D","F","G","H","I","J","M","S","O","V",]

while True:
    ret, image = camera.read()
    
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)

    cv2.imshow("Webcam Image", image)

    image = np.asarray(image, dtype=np.float32).reshape(1, 224, 224, 3)

    image = (image / 127.5) - 1

    prediction = model.predict(image)
    index = np.argmax(prediction)
    class_name = class_names[index]
    confidence_score = prediction[0][index]

    print("Class:", class_name[2:], end="")
    print("Confidence Score:", str(np.round(confidence_score * 100))[:-2], "%")
#    
    if labels[index] == 'A':
        print("A pressed")
        pyautogui.keyDown('A')
        time.sleep(0.4)
        pyautogui.keyUp('A')
    elif labels[index]=='B':
            print("B pressed")
            pyautogui.keyDown('B')
            time.sleep(0.4)
            pyautogui.keyUp('B')
    elif labels[index]=='C':
            print("C pressed")
            pyautogui.keyDown('C')
            time.sleep(0.4)
            pyautogui.keyUp('C')
    elif labels[index]=='D':
            print("D pressed")
            pyautogui.keyDown('D')
            time.sleep(0.4)
            pyautogui.keyUp('D')
    elif labels[index]=='F':
            print("F pressed")
            pyautogui.keyDown('F')
            time.sleep(0.4)
            pyautogui.keyUp('F')
    elif labels[index]=='G':
            print("G pressed")
            pyautogui.keyDown('G')
            time.sleep(0.4)
            pyautogui.keyUp('G')
    elif labels[index]=='H':
            print("H pressed")
            pyautogui.keyDown('H')
            time.sleep(0.4)
            pyautogui.keyUp('H')
    elif labels[index]=='I':
            print("I pressed")
            pyautogui.keyDown('I')
            time.sleep(0.4)
            pyautogui.keyUp('I')
    elif labels[index]=='J':
            print("J pressed")
            pyautogui.keyDown('J')
            time.sleep(0.4)
            pyautogui.keyUp('J')
    elif labels[index]=='M':
            print("M pressed")
            pyautogui.keyDown('M')
            time.sleep(0.4)
            pyautogui.keyUp('M')
    elif labels[index]=='S':
            print("S pressed")
            pyautogui.keyDown('S')
            time.sleep(0.4)
            pyautogui.keyUp('S')
    elif labels[index]=='O':
            print("O pressed")
            pyautogui.keyDown('O')
            time.sleep(0.4)
            pyautogui.keyUp('O')
    elif labels[index]=='V':
            print("V pressed")
            pyautogui.keyDown('V')
            time.sleep(0.4)
            pyautogui.keyUp('V')
#
    keyboard_input = cv2.waitKey(1)

    if keyboard_input == 27:
        break

camera.release()
cv2.destroyAllWindows()
