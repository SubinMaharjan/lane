import cv2
import math
import numpy as np
import matplotlib.pyplot as plt
import os
import csv
i = 0

def make_coordinates(image,line_parameters):
    try:
        slope,intercept = line_parameters
    except:
        slope,intercept = 0.00001,0
    #print(image.shape)

    y1 = image.shape[0]
    y2 = int(y1*(3/5))
    x1 = int((y1 - intercept)/slope)
    x2 = int((y2 - intercept) / slope)
    return np.array([x1,y1,x2,y2])

def mid_line_calc(left_point,right_point):
    try:
        x1, y1, x2, y2 = left_point
    except:
        x1, y1, x2, y2 = 0,0,0,0
    mid_fit = []
    x1_mid = ((x1 + x2) / 2)
    y1_mid = ((y1 + y2) / 2)
    try:
        x1, y1, x2, y2 = right_point
    except:
        x1, y1, x2, y2 = 0, 0, 0, 0
    x2_mid = ((x1 + x2) / 2)
    y2_mid = ((y1 + y2) / 2)
    #print(y2_mid-y1_mid)
    #change required
    if((x2_mid-x1_mid) !=  0):
        steering_angle = math.atan((y2_mid - y1_mid) / (x2_mid - x1_mid))
        #x_change = 1
    #elif(x2_mid-x1_mid <0):
        #steering_angle = math.atan((y2_mid - y1_mid) / (x2_mid - x1_mid))
        #x_change = -1
    else:
        steering_angle = 0;
    print(steering_angle)
    #print([x1_mid,y1_mid,x2_mid,y2_mid])
    with open('test.csv', 'a', newline='') as file:
        write = csv.writer(file)
        if i == 0:
            write.writerow(['image', 'steering angle'])
        write.writerow(['image_' + str(i), steering_angle])


def average_slope_intercept(image, lines):
    left_fit = []
    right_fit = []
    left_point = []
    right_point = []
    if lines is not None:
        for line in lines:
            #print(line)
            x1,y1,x2,y2 = line.reshape(4)
            parameters = np.polyfit((x1, x2), (y1, y2), 1)
            slope = parameters[0]
            intercept = parameters[1]

            #print(parameters)
            if slope<0:
                left_point = np.array([x1, y1, x2, y2])
                #print(left_point)
                left_fit.append((slope,intercept))
            else:
                right_point = np.array([x1, y1, x2, y2])
                right_fit.append((slope,intercept))
        left_fit_average = np.average(left_fit, axis = 0)
        right_fit_average = np.average(right_fit, axis = 0)
        #print(left_fit_average)
        mid_line = mid_line_calc(left_point,right_point)
        left_line = make_coordinates(image,left_fit_average)
        right_line = make_coordinates(image, right_fit_average)
        return np.array([left_line,right_line])


def canny(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    canny = cv2.Canny(blur, 50, 150)
    return canny

def region_of_interest(image):
    #print(x_component)
    height = image.shape[0]
    polygons =np.array([
        [(150, height), (1700,height), (1500,650),(250,700) ]
                       ])
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, polygons, 255)
    masked_image = cv2.bitwise_and(image,mask)
    #plt.imshow(masked_image)
    #plt.show()
    return masked_image


def display_lines(image,lines):
    line_image = np.zeros_like(image)
    if lines is not None:
        for x1,y1,x2,y2  in lines:
            try:
                cv2.line(image, (x1, y1), (x2, y2), (250,255,100), 10)
            except OverflowError:
                return line_image
        return line_image

'''
image = cv2.imread('C:\\Users\\HP\\Desktop\\lane.jpg')
lane_image = np.copy(image)

canny_image =canny(lane_image)
#plt.imshow(canny_image)
#plt.show()
region = region_of_interest(canny_image)
lines = cv2.HoughLinesP(region, 50, np.pi/180, 50, np.array([]), minLineLength = 40, maxLineGap=2)
average_lines = average_slope_intercept(lane_image,lines)
line_image = display_lines(lane_image, average_lines)
combo_image = cv2.addWeighted(lane_image,0.8, line_image,1,1)
cv2.imshow("result",combo_image)
cv2.waitKey(0)

'''
cap=cv2.VideoCapture("C:\\Users\\HP\Downloads\\car_2020-01-28_14-24-1.mkv")
directory = 'C:\\Users\\HP\\Desktop\\Images'
os.chdir(directory)
while(cap.isOpened()):
    _,frame = cap.read()
    frame_1 = cv2.resize(frame, (500, 250))
    img = cv2.imread(image_path)
    cv2.imwrite('image_' + str(i) + '.jpg', frame_1)
    canny_image = canny(frame)
    # plt.imshow(canny)
    # plt.show()
    #print(x_component)
    region = region_of_interest(canny_image)
    lines = cv2.HoughLinesP(region, 2, np.pi / 180, 100, np.array([]), minLineLength=40, maxLineGap=5)
    average_lines = average_slope_intercept(frame, lines)
    line_image = display_lines(frame, average_lines)

    #line_image = display_lines()
    if line_image is not None:
        combo_image = cv2.addWeighted(frame, 0.8, line_image, 1, 1)
    else:
        combo_image = frame
    cv2.imshow("result", combo_image)
    if cv2.waitKey(200) & 0xFF == ord('q'):
        break
    i = i+1
cap.release()
cv2.destroyAllWindows()



