import pyautogui
import time
from keras.models import load_model 
import cv2 
import numpy as np

np.set_printoptions(suppress=True)

model = load_model("Part2\\keras_model.h5", compile=False)

class_names = open("Part2\\labels.txt", "r").readlines()

camera = cv2.VideoCapture(0)

#
labels=["Z","Y","X","W","P","N","U","L","K","E","R","T","Q",]

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
    if labels[index] == 'Z':
        print("Z pressed")
        pyautogui.keyDown('Z')
        time.sleep(0.4)
        pyautogui.keyUp('Z')
    elif labels[index]=='Y':
            print("Y pressed")
            pyautogui.keyDown('Y')
            time.sleep(0.4)
            pyautogui.keyUp('Y')
    elif labels[index]=='X':
            print("X pressed")
            pyautogui.keyDown('X')
            time.sleep(0.4)
            pyautogui.keyUp('X')
    elif labels[index]=='W':
            print("W pressed")
            pyautogui.keyDown('W')
            time.sleep(0.4)
            pyautogui.keyUp('W')
    elif labels[index]=='P':
            print("P pressed")
            pyautogui.keyDown('P')
            time.sleep(0.4)
            pyautogui.keyUp('P')
    elif labels[index]=='N':
            print("N pressed")
            pyautogui.keyDown('N')
            time.sleep(0.4)
            pyautogui.keyUp('N')
    elif labels[index]=='U':
            print("U pressed")
            pyautogui.keyDown('U')
            time.sleep(0.4)
            pyautogui.keyUp('U')
    elif labels[index]=='L':
            print("L pressed")
            pyautogui.keyDown('L')
            time.sleep(0.4)
            pyautogui.keyUp('L')
    elif labels[index]=='K':
            print("K pressed")
            pyautogui.keyDown('K')
            time.sleep(0.4)
            pyautogui.keyUp('K')
    elif labels[index]=='E':
            print("E pressed")
            pyautogui.keyDown('E')
            time.sleep(0.4)
            pyautogui.keyUp('E')
    elif labels[index]=='R':
            print("R pressed")
            pyautogui.keyDown('R')
            time.sleep(0.4)
            pyautogui.keyUp('R')
    elif labels[index]=='T':
            print("T pressed")
            pyautogui.keyDown('T')
            time.sleep(0.4)
            pyautogui.keyUp('T')
    elif labels[index]=='Q':
            print("Q pressed")
            pyautogui.keyDown('Q')
            time.sleep(0.4)
            pyautogui.keyUp('Q')
#
    keyboard_input = cv2.waitKey(1)

    if keyboard_input == 27:
        break

camera.release()
cv2.destroyAllWindows()
