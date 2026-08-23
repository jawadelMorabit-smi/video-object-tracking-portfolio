# -*- coding: utf-8 -*-
"""
Created on Wed Apr  1 10:11:40 2026

@author: jawad_el_motabit
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Apr  1 10:01:40 2026

@author: jawad_el_motabit
"""

import cv2
import numpy as np

cap = cv2.VideoCapture(0)

mog = cv2.bgsegm.createBackgroundSubtractorMOG()
mog2 = cv2.createBackgroundSubtractorMOG2(detectShadows=True)
gmg = cv2.createBackgroundSubtractorGMG()


while True:
    ret, frame =cap.read()
    if not ret:
        break
    frame = cv2.resize(frame, (480,270))
    
    
    #applique les trois methodes 
    mask_mog = mog.apply(frame)
    mask_mog2 = mog2.apply(frame)
    mask_gmg = gmg.apply(frame)
    
    
    
    #convertir gmg por eviter leserrers 8 bits
    mask_gmg = cv2.normalize(mask_gmg, None, 0, 255, cv2.NORM_MINMAX)
    mask_gmg = np.uint8(mask_gmg)
    
    
    combined = np.hstack((mask_mog,mask_mog2,mask_gmg))
    cv2.imshow("MOG2", combineopencd)
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break 
cap.release()
cv2.destroyAllWindows()