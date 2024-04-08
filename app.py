# TF Stuff
import tensorflow as tf
import cv2
import time
import numpy as np
import serial
from matplotlib import pyplot as plt


ser = serial.Serial('/dev/cu.usbmodem14201', baudrate=9600)
THRESHOLD = 0.80


labels = ['Plastic Bottle', 'Metal Can', 'Reject']
h1 = plt.bar(labels, [1, 1, 1])

interpreter = tf.lite.Interpreter(model_path='./model.tflite')

interpreter.allocate_tensors()

# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print(input_details)

# Camera Stream -> TF-Lite Interpreter -> HTTPS Request to ESP3 -> Motion

vid = cv2.VideoCapture(0) 

ret, frame = vid.read()
crop_y1, crop_y2 = 248, 472
crop_x1, crop_x2 = 528, 752

plt.ion()
plt.show()

while(True): 
    while ser.in_waiting:
        print(ser.readline())
    # Capture the video frame 
    # by frame 
    ret, frame = vid.read() 
    crop = cv2.resize(frame, (224, 224)) # frame[crop_y1:crop_y2, crop_x1:crop_x2, :]
  
    # Display the resulting frame 
    cv2.imshow('frame', crop) 

    interpreter.set_tensor(input_details[0]['index'], np.asarray([crop]))
    interpreter.invoke()
    res = interpreter.get_tensor(output_details[0]['index'])
    res = np.ndarray.tolist(res)[0]

    for v, l in zip(h1, [x / 255.0 for x in res]):
        v.set_height(l)
    
    plt.draw()
    plt.pause(0.001)
    # print(list(zip(labels, [x / 255.0 for x in res]))) 

    if cv2.waitKey(1) & 0xFF == ord('g'):
        # GO!
        # check to see which result is majority confidence
        m_i, m_x = 0, 0
        for i, x in enumerate(res):
            x = x / 255
            if x > m_x:
                m_i, m_x = i, x
        print(m_i, m_x)
        if m_x >= THRESHOLD and m_i <= 1:
            cmd = 'P' if m_i == 0 else 'M'
            ser.write(cmd.encode())
            time.sleep(10) # wait 10s before doing anything
        elif m_i == 2:
            print('Reject Detected')


    # time.sleep(0.5) # Give the system some rest
  
# After the loop release the cap object 
vid.release() 
# Destroy all the windows 
cv2.destroyAllWindows() 
