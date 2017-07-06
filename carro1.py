import cv2
import numpy as np
import sys
from scipy.cluster.vq import vq, kmeans
minDistX = 50
maxDistX = 90
maxDistY = 20
minCarDist = 20
cars = []
nOfPos = 10
frameGap = 10
minHArea=20
maxHArea = 150
video_fps = 30
meters_per_pixel = 0.11
def maskify_HL(original_img, low_white = 225, roi_vertexes=[[0,720],[0,400],[1280,400],[1280,720]], blur=(15,15)):
    img_blur = cv2.GaussianBlur(original_img, blur,0)
    img_gray = cv2.cvtColor(img_blur, cv2.COLOR_BGR2GRAY).astype("float64")
    roi_gray = roi(img_gray,[np.array(roi_vertexes)]) if roi_vertexes else img_gray
    maxcolor = np.amax(roi_gray)
    if maxcolor != 0:
        roi_gray*= 255.0/maxcolor
    roi_gray = roi_gray.astype('uint8')
    masked_img = cv2.inRange(roi_gray, low_white,255)
    return masked_img
def roi(img, vertices): #creates a region of interest based on the vertices given
    mask = np.zeros_like(img)
    cv2.fillPoly(mask, vertices, 255)
    masked = cv2.bitwise_and(img, mask)
    return masked
class car:
    def __init__(self, X0, Y0, X1, Y1, frame):
        self.last_frame = frame
        self.first_frame = frame
        self.prev_positions = [((X0+X1)/2, (Y0+Y1)/2)]
        self.lane = 0
def append_car(X0, Y0, X1, Y1, actual_frame):
    global cars
    flag = True
    center = ((X0+X1)/2, (Y0+Y1)/2)
    for i in range(len(cars)):
        if ((cars[i].prev_positions[-1][0]-center[0])**2 + (cars[i].prev_positions[-1][1] - center[1])**2)**0.5 < minCarDist and actual_frame - cars[i].last_frame<= frameGap :
            cars[i].prev_positions.append(center)
            cars[i].last_frame = actual_frame
            flag = False
            break
    if flag :
        new_car = car(X0, Y0, X1, Y1, actual_frame)
        cars.append(new_car) #restricted append to a minimum distance
def find_cars(original_img, masked_img,actual_frame,min_headlight_area = 20, max_headlight_area = 150):
    global cars
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
                        if minDistX<abs(X0-X1)<maxDistX and abs(Y0-Y1)<maxDistY :
                            append_car(X0, Y0, X1, Y1, actual_frame)
                            del cnt1
                            break
                    except:
                        pass
        except:
            pass
        

def print_cars(original_img, colors = [(0,255,0),(255,0,0),(255,255,0),(255,0,255)]):
    global cars
    for car in cars:
        if len(car.prev_positions) >= nOfPos :
            cv2.circle(original_img, (car.prev_positions[-1][0], car.prev_positions[-1][1]), 3, colors[car.lane], 3)           
def find_lanes(original_img, color = (0,0,255)):
    global cars
    actual_pos=[]
    try:
        for i in range(len(cars)):
            if len(cars[i].prev_positions) >= nOfPos :
                 actual_pos.append(([float(cars[i].prev_positions[-1][0]),float(cars[i].prev_positions[-1][1])], i))
        means = kmeans([ac[0] for ac in actual_pos],4)
        
        lmean =[list(m) for m in means[0]]
        lmean.sort(key = lambda x: x[0])
        
        try:
            lane = vq([ac[0] for ac in actual_pos],np.array(lmean))
            for j in range(len(lane[0])):
              cars[actual_pos[j][1]].lane = lane[0][j]
        except:
            pass
        for j in means[0]:
            cv2.circle(original_img, (int(j[0]), int(j[1])), 3, color, 3)
    except:
        pass
def number_of_cars():
    global cars
    return len([car for car in cars if len(car.prev_positions)>=nOfPos])
def relative_car_flux(frame_number,time_interval = 60):
    global video_fps
    frame_interval = time_interval*video_fps
    relative_number_of_cars = len([car for car in cars if len(car.prev_positions)>=nOfPos and car.last_frame > frame_number - frame_interval])
    return float(relative_number_of_cars)/time_interval
def total_car_flux(frame_number):
    global video_fps
    return float(video_fps)*number_of_cars()/frame_number
def average_velocity(car):
    global cars
    global video_fps
    try:
        return (3.6*meters_per_pixel*((car.prev_positions[-1][0]-car.prev_positions[0][0])**2 + (car.prev_positions[-1][1]-car.prev_positions[0][1])**2)**(0.5))*video_fps/(car.last_frame-car.first_frame)
    except:
        return 0
def inst_velocity(car):
    global cars
    global video_fps
    try:
    	return (3.6*meters_per_pixel*((car.prev_positions[-1][0]-car.prev_positions[-2][0])**2 + (car.prev_positions[-1][1]-car.prev_positions[-2][1])**2)**(0.5))*video_fps
    except:
      return 0
def car_id(car):
    try:
        return [c for c in cars if len(c.prev_positions)>=nOfPos].index(car)
    except:
        return None
def main() :
        global cars
        video = cv2.VideoCapture('vid.mp4')
        frame_number = 0
        while True:
                frame_number+=1
                _, original_img = video.read()
                masked_img = maskify_HL(original_img, roi_vertexes = [[100,720],[200,500],[1080,500],[1180,720]])
                find_cars( original_img, masked_img, frame_number,minHArea,maxHArea)
                find_lanes(original_img)
                print_cars(original_img)
                print [average_velocity(car) for car in cars]
                cv2.putText(original_img, '{}'.format(frame_number), (200,200), cv2.FONT_HERSHEY_SIMPLEX, 5, (0,255,0))
                if frame_number == 30:
                    cv2.imwrite('frame30.jpg', original_img)
                
                if cv2.waitKey(1) & 0xFF == ord('q') and not ret:
                        break
                cv2.imshow('ori',original_img)
        video.release()
        cv2.destroyAllWindows()

main()
