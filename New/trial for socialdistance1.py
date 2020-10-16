import numpy as np
import cv2
import time
import winsound
def Setup(yolo):
    global net,LABELS
    weights = r'C:\Users\hp\Desktop\New\yolov4-tiny.weights'
    config = r'C:\Users\hp\Desktop\New\yolov4-tiny.cfg'
    labelsPath = r'C:\Users\hp\Desktop\New\coco.names'
    LABELS = open(labelsPath).read().strip().split("\n")
    net = cv2.dnn.readNetFromDarknet(config,weights)#load the saved weights,configuration file into the network.
    #print(net.getLayerNames(),len(net.getLayerNames()))#it prints all the layers in networks (78 layers)
def ImageProcess(image):
    global processedImg,true
    (H, W)=image.shape[:2]
    blob = cv2.dnn.blobFromImage(image,1/255.0,(416,416),(0,0,0), swapRB=True, crop=False)
    net.setInput(blob)
    starttime = time.time()
    layerOutputs = net.forward(net.getUnconnectedOutLayersNames())
    #print(layerOutputs)
    stoptime = time.time()
    print("Video is Getting Processed at {:.5f}seconds per frame ".format((stoptime-starttime))) 
    confidences = []
    outline = []
    for output in layerOutputs:
        for detection in output:
            scores = detection[5:]
            maxi_class = np.argmax(scores)
            confidence = scores[maxi_class]
            if LABELS[maxi_class] == "person":
                if confidence > 0.5:
                    box = detection[0:4] * np.array([W, H, W, H])
                    (centerX, centerY, width, height) = box.astype("int")
                    x = int(centerX - (width / 2))
                    y = int(centerY - (height / 2))
                    outline.append([x, y, int(width), int(height)])
                    confidences.append(float(confidence))
                    #print(confidences)
    scorethreshold=0.7
    iouthreshold=0.3
    box_line= cv2.dnn.NMSBoxes(outline,confidences,scorethreshold,iouthreshold)
    #print(box_line)
    if len(box_line) > 0:
        flat_box = box_line.flatten()
        pairs = []
        center = []
        status = []
        true=[]
        for i in flat_box:
            (x, y) = (outline[i][0], outline[i][1])
            (w, h) = (outline[i][2], outline[i][3])
            center.append([int(x + w / 2), int(y + h / 2)])
            status.append(False)
            #print(len(center))
        for i in range(len(center)):
            for j in range(len(center)):
                close = Check(center[i], center[j])
                if close:
                    pairs.append([center[i], center[j]])
                    #print(pairs)
                    status[i] = True
                    status[j] = True
        index = 0
        
        for i in flat_box:
            k=0
            (x, y) = (outline[i][0], outline[i][1])
            (w, h) = (outline[i][2], outline[i][3])
            if status[index] == True:
                #print(status[index])
                a=status[index]
                true.append(a)
                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 150), 2)
                k=k+1
                text = "person{}".format(k)
                cv2.putText(image, text, (x +w, y + h), cv2.FONT_HERSHEY_SIMPLEX,0.5, (0, 0, 150), 2)
                
                if(len(true)%2)==0:
                  print('social distance is not maintained between person'+str(k-1)+' '+'and person'+str(k))
                  print('sound')
                
            elif status[index] == False:
                cv2.rectangle(image,(x, y), (x + w, y + h), (0, 255, 0), 2)
            index += 1
        print(k)    
        for h in pairs:
            cv2.line(image, tuple(h[0]), tuple(h[1]), (0, 0, 255), 2)
        #print(true)
        h_letters = [ letter for letter in range(len(true))]
        print(len(true))
        if(len(true))==4:
            print('MESSAGE SENT')

        #print(h_letters)
        #s print(len(h_letters))
        test=i+len(h_letters)
        #print(test)
        #if len(h_letters)>test:
        # winsound.Beep(freq, duration)
        #print(test)

        
            
    processedImg = image.copy()    
def Check(a,  b):
    dist = ((a[0] - b[0]) ** 2 + 550 / ((a[1] + b[1]) / 2) * (a[1] - b[1]) ** 2) ** 0.5
    calibration = (a[1] + b[1]) / 2       
    if 0 < dist < 0.3 * calibration:
        return True
    else:
        return False
#duration = 200
#freq = 400
frameno = 0
yolo = "yo"
filename= r'C:\Users\hp\Desktop\New\test_video.mp4'
cap = cv2.VideoCapture(filename)
#cap=cv2.VideoCapture(0)'
time1 = time.time()
while(True):
    ret,frame = cap.read()
    if ret==False:
        break
    current_img = frame.copy()
    current_img = cv2.resize(current_img,(800,600))
    Setup(yolo)
    frameno += 1
    if(frameno%2 == 0 or frameno == 1):
       ImageProcess(current_img)
       Frame = processedImg
       text="no of violators : "+str(len(true))
       cv2.putText(Frame, text, (50,50), cv2.FONT_HERSHEY_SIMPLEX,0.9, (0, 0,0), 3)
       cv2.imshow('image',Frame)
       #print(len(true))
    if cv2.waitKey(1) & 0xFF == ord('s'):
        break
time2 = time.time()
print("Completed. Total Time Taken: {} minutes".format((time2-time1)/60))

cap.release()
cv2.destroyAllWindows()
