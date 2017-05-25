 
import cv2
import numpy as np
ori = cv2.imread('carro6.jpg')
img = cv2.GaussianBlur(ori,(15,15),0)
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype("float64")
maxcolor = np.amax(img_gray)
img_gray*= 255.0/maxcolor
img_gray = img_gray.astype('uint8')
maxcolor = np.amax(img_gray)
mask = cv2.inRange(img_gray, 225,255)
cv2.imshow("mask", mask)
contours, hierarchy = cv2.findContours(mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
#cv2.drawContours(ori, contours, -1, (0,0,255), 3) 
filtered_contours = [i for i in contours if(20<cv2.moments(i)["m00"] <150)]

for c in filtered_contours:
        # compute the center of the contour
        M = cv2.moments(c)
        try :
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                print("cx = {}, cy={}".format(cX,cY))
         
                # draw the contour and center of the shape on the image
                # cv2.drawContours(ori, [c], -1, (0, 255, 0), 2)
                cv2.circle(ori, (cX, cY), 7, (0, 0, 255), -1)
        except:
                 pass
         
cv2.imshow("dsm", ori)
cv2.waitKey(0)
cv2.destroyAllWindows()
