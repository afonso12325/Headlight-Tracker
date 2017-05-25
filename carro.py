 
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
cars = []
for c in filtered_contours:
        # compute the center of the contour
        M = cv2.moments(c)
        try :
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                print("cx = {}, cy={}".format(cX,cY))
                for i in filtered_contours:
                        M2=cv2.moments(i)
                        try:        
                                iX = int(M2["m10"] / M2["m00"])
                                iY = int(M2["m01"] / M2["m00"])
                                if(90<abs(cX-iX)<115 and abs(iY-cY)<20):
                                        cars.append((c,i,cX,iX,cY,iY))
                                        del i
                                        break
                        except:
                                pass
                # draw the contour and center of the shape on the image
                # cv2.drawContours(ori, [c], -1, (0, 255, 0), 2)
                #cv2.circle(ori, (cX, cY), 7, (0, 0, 255), -1)
        except:
                 pass
print(cars)
for car in cars:
        cv2.rectangle(ori, (min(car[2],car[3])-10, max(car[4],car[5])+10),(max(car[2],car[3])+10, min(car[4],car[5])-10), (0,255,0),7)
         
cv2.imshow("dsm", ori)
cv2.waitKey(0)
cv2.destroyAllWindows()
