 
import cv2
import numpy as np
global minDistX = 50, maxDistX = 90, maxDistY = 20, minCarDist = 20
class car:
    prev_positions = []
    def __init__(self, X0, Y0, X1, Y1):

        X0 = int(M0["m10"] / M0["m00"])
        Y0 = int(M0["m01"] / M0["m00"])

        X1 = int(M1["m10"] / M1["m00"])
        Y1 = int(M1["m01"] / M1["m00"])

        self.prev_positions.append(((X0+X1)/2, (Y0+Y1)/2))

def append_car(cars, X0, Y0, X1, Y1):
    center = ((X0+X1)/2, (Y0+Y1)/2))
    for i in range(0, len(cars)):
        if(((cars[i].prev_positions[-1][0]-center[0])**2 + (cars[i].prev_positions[-1][1] - center[1])**2)**0.5 < minCarDist):
            car[i].prev_position.append(center)
            return
    cars.append(car(X0, Y0, X1, Y1)) #restricted append to a minimum distance
    
            
def find_car(cars, original_img, masked_img,min_headlight_area = 20, max_headlight_area = 150):        
    _, cnts, _ = cv2.findContours(masked_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    filt_cnts = [i for i in cnts if(min_headlight_area<cv2.moments(i)["m00"] <max_headlight_area)]
    for cnt0 in filt_cnts:
        M0 = cv2.moments(cnt0)

        try:
                X0 = int(M0["m10"] / M0["m00"])
                Y0 = int(M0["m01"] / M0["m00"])

                for cnt1 in filt_cnts:
                    M1 = cv2.moments(cnt1)

                    try:
                        X1 = int(M1["m10"] / M1["m00"])
                        Y1 = int(M1["m01"] / M1["m00"])
                        
                        if(minDistX<abs(cX-iX)<maxDistX and abs(iY-cY)<maxDistY):
                            append_car(cars, X0, Y0, X1, Y1)
                            del i
                            break
                    except:
                        pass
        except:
            pass
        
def roi(img, vertices): #creates a region of interest based on the vertices given
    mask = np.zeros_like(img)
    cv2.fillPoly(mask, vertices, 255)
    masked = cv2.bitwise_and(img, mask)
    return masked

def main() : 
    video = cv2.VideoCapture('vid.mp4')
    cars = []
    while True:
            _, original_img = video.read()
            #ori = cv2.imread('carro6.jpg')
            img_blur = cv2.GaussianBlur(original_img,(15,15),0)
            img_gray = cv2.cvtColor(img_blur, cv2.COLOR_BGR2GRAY).astype("float64")
            roi_gray = roi(img_gray,[np.array([[100,720],[200,500],[1080,500],[1180,720]])])


            maxcolor = np.amax(roi_gray)
            if maxcolor != 0:
                roi_gray*= 255.0/maxcolor
            roi_gray = roi_gray.astype('uint8')
            masked_img = cv2.inRange(roi_gray, 225,255)


            find_car(cars, original_img, masked_img)
            
            for car in cars:
                    cv2.rectangle(ori, (car.prev_positions[-1]-((maxDistX + minDistX)/4,(maxDistY)/4),car.prev_positions[-1] +((maxDistX + minDistX)/4,(maxDistY)/4)),(max(car[2],car[3])+10, min(car[4],car[5])-10), (0,255,0),7)      
            if cv2.waitKey(1) & 0xFF == ord('q') and not ret:
                    break

            cv2.imshow('ori',ori)
    video.release()
    cv2.destroyAllWindows()
