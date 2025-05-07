import cv2
import numpy as np
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
image = cv2.imread('image.jpg')
image_resized = cv2.resize(image, (416, 416))
blob = cv2.dnn.blobFromImage(image_resized, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
net.setInput(blob)
outputs = net.forward(output_layers)
height, width, _ = image.shape
boxes = []
confidences = []
class_ids = []
for output in outputs:
    for detection in output:
        
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]
        
        if confidence > 0.5:  
            
            center_x = int(detection[0] * width)
            center_y = int(detection[1] * height)
            w = int(detection[2] * width)
            h = int(detection[3] * height)

            
            x = int(center_x - w / 2)
            y = int(center_y - h / 2)

            
            boxes.append([x, y, w, h])
            confidences.append(float(confidence))
            class_ids.append(class_id)


indices = cv2.dnn.NMSBoxes(boxes, confidences, score_threshold=0.5, nms_threshold=0.4)


for i in range(len(boxes)):
    if i in indices:
        x, y, w, h = boxes[i]
        label = str(class_ids[i])  
        confidence = confidences[i]
        
        
        color = (0, 255, 0) 
        cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)

       
        cv2.putText(image, f'{label} {confidence:.2f}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)


cv2.imshow('Object Detection', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite('output_image.jpg', image)
