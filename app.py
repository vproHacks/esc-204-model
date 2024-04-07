# TF Stuff
import tensorflow as tf
import cv2
import time
import numpy as np

labels = ['Plastic Bottle', 'Metal Can', 'Reject']

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

while(True): 
      
    # Capture the video frame 
    # by frame 
    ret, frame = vid.read() 
    crop = frame[crop_y1:crop_y2, crop_x1:crop_x2, :]
  
    # Display the resulting frame 
    cv2.imshow('frame', crop) 
      
    # the 'q' button is set as the 
    # quitting button you may use any 
    # desired button of your choice 
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    interpreter.set_tensor(input_details[0]['index'], np.asarray([crop]))
    interpreter.invoke()
    res = interpreter.get_tensor(output_details[0]['index'])
    res = np.ndarray.tolist(res)[0]
    print(list(zip(labels, [x / 255.0 for x in res]))) 

    time.sleep(0.5) # Give the system some rest
  
# After the loop release the cap object 
vid.release() 
# Destroy all the windows 
cv2.destroyAllWindows() 
