import cv2
import numpy as np
import sys
minDistX = 50
maxDistX = 90
maxDistY = 20
minCarDist = 20
cars = []
class car:
    
    def __init__(self, X0, Y0, X1, Y1):

        self.prev_positions = [((X0+X1)/2, (Y0+Y1)/2)]

def append_car(X0, Y0, X1, Y1):
    
    global cars
    flag = True
    center = ((X0+X1)/2, (Y0+Y1)/2)
    print(len(cars))
    for i in range(len(cars)):
        if ((cars[i].prev_positions[-1][0]-center[0])**2 + (cars[i].prev_positions[-1][1] - center[1])**2)**0.5 < minCarDist :
            car[i].prev_position.append(center)
            flag = False
            break
    if flag :
        new_car = car(X0, Y0, X1, Y1)
        cars.append(new_car) #restricted append to a minimum distance
    
            
def find_car(original_img, masked_img,min_headlight_area = 20, max_headlight_area = 150):
    global cars
    _, cnts, _ = cv2.findContours(masked_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    filt_cnts = [i for i in cnts if(min_headlight_area<cv2.moments(i)["m00"] <max_headlight_area)]
    for cnt0 in filt_cnts:
        M0 = cv2.moments(cnt0)

        try:
                X0 = int(M0["m10"] / M0["m00"])
                Y0 = int(M0["m01"] / M0["m00"])
                #print('x0 = {} , y0 = {}'.format(X0,Y0))
                
                for cnt1 in filt_cnts:
                    M1 = cv2.moments(cnt1)

                    try:
                        X1 = int(M1["m10"] / M1["m00"])
                        Y1 = int(M1["m01"] / M1["m00"])
                        #print('x1 = {} , y1 = {}'.format(X1,Y1))    
                        if minDistX<abs(X0-X1)<maxDistX and abs(Y0-Y1)<maxDistY :
                            append_car(X0, Y0, X1, Y1)
                            print('oi')
                            sys.exit()
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
        global cars
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

                find_car( original_img, masked_img)

                for car in cars:
                        quinas = (car.prev_positions[-1][0] - (maxDistX + minDistX)/4,car.prev_positions[-1][1] - (maxDistY)/4)
                        quinai = (car.prev_positions[-1][0] + (maxDistX + minDistX)/4,car.prev_positions[-1][1] + (maxDistY)/4)
                        cv2.rectangle(original_img,quinas, quinai, (0,255,0),7)      
                if cv2.waitKey(1) & 0xFF == ord('q') and not ret:
                        break
        
                cv2.imshow('ori',original_img)
        video.release()
        cv2.destroyAllWindows()
main()

