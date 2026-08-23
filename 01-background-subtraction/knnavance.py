# -*- coding: utf-8 -*-
"""
Created on Wed Apr  1 10:45:59 2026

@author: jawad_el_motabit
"""


import cv2

bg_substractor = cv2.createBackgroundSubtractorKNN(history=500, dist2Threshold=400, detectShadows=True)

cap = cv2.VideoCapture(0)

while True:
    ret, frame =cap.read()
    if not ret:
        break
    fg_mask = bg_substractor.apply(frame)
    contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for contour in contours :
        if cv2.contourArea(contour) > 1000:
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(frame, (x, y), (x+w, y+h),(0,255,0), 2)
           
    
    #kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    #fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
    
   
    
    cv2.imshow("frame", frame)
    cv2.imshow("foeground mask", fg_mask)
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break 
cap.release()
cv2.destroyAllWindows()