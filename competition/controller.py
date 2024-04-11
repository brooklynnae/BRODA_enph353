#! /usr/bin/env python3

import rospy
import cv2
import numpy as np
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from std_msgs.msg import String

import sign_reader

class Driver():
    def __init__(self):
        rospy.init_node('robot_pid_er')

        self.bridge = CvBridge()
        rospy.Subscriber("/R1/pi_camera/image_raw", Image, self.callback)
        self.vel_pub = rospy.Publisher('/R1/cmd_vel', Twist, queue_size=1)
        self.score_pub = rospy.Publisher('/score_tracker', String, queue_size=1)

        self.state = 'init' # init, road, ped, truck, desert, yoda, tunnel, mountain
        
        # image variables
        self.img = None
        self.img_height = 0
        self.img_width = 0
        
        self.cycle_count = 0

        # PID controller variables
        self.move = Twist()
        self.lin_speed = 0.5 # defualt PID linear speed of robot
        self.sign_lin_speed = 0.25 # slower PID linear speed for sign reading
        self.rot_speed = 1.0 # base PID angular speed of robot
        self.sign_rot_speed = 1.1 # slower PID angular speed for sign reading

        self.road_line_width = 150
        self.road_min_white_val = 250
        self.road_max_white_val = 255
        
        self.kp = 11 # proportional gain for PID controller
        self.kd = 0.5
        self.prev_error = 0
        self.last_time = 0
        self.dt = 0
        self.road_buffer = 200 # pixels above bottom of image to find road centre
        self.speed_buffer = 1.3 # buffer for gradual speed increase/decrease

        self.accel_rate = 0.1 # velocity to increase by with each loop
        self.decel_rate = 0.1 # velocity to decrease by with each loop
        self.accel_freq = 50 # frequency of loop when increasing/decreasing speed
        
        # Pedestraian detection variables
        self.reached_crosswalk = False
        self.red_line_min_area = 1000 # minimum contour area for red line
        self.red_line_approach_lin_vel = 0.4
        self.red_line_approach_rot_vel = 0.3
        self.red_line_min_angle = 1.0
        self.red_line_max_angle = 89.0
        self.red_line_stop_y = 400
       
        self.bg_sub = cv2.createBackgroundSubtractorMOG2()
        self.ped_crop_x_min = 400 # values for cropping image to crosswalk and pedestrian
        self.ped_crop_x_max = 920
        self.ped_crop_y_min = 320
        self.ped_crop_y_max = 440
        
        self.ped_left_buffer = 60 # lateral pixel buffers for pedestrian detection
        self.ped_right_buffer = 80
        self.ped_min_area = 400 # minimum contour area for detecting the pedestrian
                
        self.ped_safe_count = 0
        self.ped_safe_count_buffer = 5
        
        self.ped_lin_speed = 2.5 # linear speed of robot when crossing crosswalk
        self.ped_ang_speed = 0 # angular speed of robot when crossing crosswalk
        self.ped_sleep_time = 0.01 # time to sleep when crossing crosswalk

        # Truck detection variables
        self.reached_truck = False
        self.truck_init_cycle = 0
        self.truck_cycle_buffer = 13

        self.truck_left_area = 580 # goes right if truck area is bigger than this
        self.truck_wait_area = 7000
        
        self.truck_min_area = 5000 # used in function, but never actually called, check later
        
        self.truck_turn_dir = ''
        self.truck_left_turn_amplifier = 1.5 # amplifies error when find no road to turn left
        self.truck_right_kp = 12
        self.truck_right_lin_speed = 0.7

        self.truck_to_desert_sleep = 0.3 # time to go straight when transitioning from truck to desert states

        # Desert detection variables
        self.desert_min_arc_length = 750 
        self.desert_line_cnt_min_height = 125
        
        self.desert_min_magenta_area = 5000
        self.desert_past_magenta_line_area = 1000
        
        self.desert_road_buffer = 250

        self.magneta_min_angle = 0.5
        self.magneta_max_angle = 89.5
        self.magenta_angle_lin_speed = 0.2
        self.magenta_angle_rot_speed = 0.2

        # Yoda detection variables
        self.reached_yoda = False
        
        self.cactus_min_area = 640
        self.cactus_max_area = 825
        self.cactus_lin_speed = 0.6
        
        self.tunnel_turn_speed = 4.0
        self.tunnel_min_area = 30
        self.tunnel_mid_x = 500
        
        self.over_hill = False
        self.hill_lin_speed = 0.5
        self.hill_rot_speed = -0.1

        self.yoda_mag_min_area_for_y = 5000 # get the y value of the magenta line only when the contour area is greater than this
        self.yoda_find_mag_min_area = 100 # minimum contour area of the magenta line to say that it detects the line
        self.yoda_mag_x_mid = 550 # x coord of the magenta line to pid to
        self.yoda_mag_y_exit = 700 # starts going to tunnel state when y value of mageneta line is greater than this

        # Tunnel detection variables
        self.tunnel_pid_height = 400

        # Mountain detection variables
        self.found_mountain_lines = False
        self.boost = False
        self.mountain_start_cycle = 0
        self.boost_count = 0
        self.boost_cycle = 0


    # callback function for camera subscriber
    def callback(self, msg):
        self.img = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        self.img_height, self.img_width = self.img.shape[:2]
        self.cycle_count += 1
        self.dt = rospy.Time.now().to_sec() - self.last_time
        self.last_time = rospy.Time.now().to_sec()
        if self.cycle_count > 1700:
            for i in range(my_bot.num_signs):
                message = String()
                prediction = my_bot.read_sign(my_bot.signs[i])
                message.data = "Broda,adorb,"+str(i+1)+","+prediction
                self.score_pub.publish(message)
            end_timer = String()
            end_timer.data = "Broda,adorb,-1,NA"
            self.score_pub.publish(end_timer)

    def find_road_centre(self, img, y, width, height, ret_sides=False):
        left_index = right_index = -1
        for i in range(width):
            if img[height - y, i] == 255 and left_index == -1:
                left_index = i
            elif img[height - y, i] == 255 and left_index != -1:
                right_index = i

        if ret_sides:
            return left_index, right_index

        # print(f'index difference: {right_index - left_index}')

        road_centre = -1
        if left_index != -1 and right_index != -1:
            if right_index - left_index > self.road_line_width:
                road_centre = (left_index + right_index) // 2
            elif left_index < width // 2: # and self.state != 'mountain':
                road_centre = (left_index + width) // 2
            elif left_index >= width // 2: # and self.state != 'mountain':
                road_centre = right_index // 2
            # elif self.state == 'mountain':
            #     if right_index - left_index < 200:
            #         road_centre = -1
            #     elif right_index > width // 2 and self.state == 'mountain':
            #         road_centre = right_index // 2
            #     else:
            #         road_centre = (left_index + width) // 2
        else:
            # print('no road at this y level')
            road_centre = -1

        # if road_centre != -1:
        #     cv2.imshow('camera feed', cv2.circle(img, (road_centre, height - y), 5, (0, 0, 255), -1))
        #     cv2.waitKey(1)

        # for y, row in enumerate(img):
        #     white_pixels = np.where(row == 255)[0]
        #     if white_pixels.__len__() > 0:
        #         if y > 600 and self.state == 'mountain':
        #             road_centre = -1
        #             print('road too low')
        #         break

        return road_centre
    
    # returns the error between the centre of the road and the centre of a thresholded image
    # for either a road or desert image, default is road
    # returns error of 0 if no road lines are found on either side
    # enters the truck state if no road is detected and have reached the crosswalk
    def get_error(self, img):
        if self.state == 'road' or self.state == 'truck':
            gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            mask = cv2.inRange(gray_img, self.road_min_white_val, self.road_max_white_val)
        elif self.state == 'desert':
            mask = cv2.cvtColor(self.thresh_desert(img), cv2.COLOR_BGR2GRAY)
            self.road_buffer = self.desert_road_buffer
            # cv2.imshow('desert mask', cv2.resize(mask, (self.img_width // 2, self.img_height // 2)))
            # cv2.waitKey(1)
        elif self.state == 'tunnel':
            mask = self.find_tunnel(img, ret_mask=True)
            self.road_buffer = self.tunnel_pid_height
            self.road_line_width = 350
        elif self.state == 'mountain':
            mask = cv2.cvtColor(self.thresh_desert(img), cv2.COLOR_BGR2GRAY)
            self.road_buffer = 215
            self.road_line_width = 450
            # cv2.imshow('mountain mask', cv2.resize(mask, (self.img_width // 2, self.img_height // 2)))
            # cv2.waitKey(1)
            uh_road = 37; us_road = 255; uv_road = 255
            lh_road = 0; ls_road = 27; lv_road = 110
            lower_hsv_road = np.array([lh_road, ls_road, lv_road])
            upper_hsv_road = np.array([uh_road, us_road, uv_road])
            road_mask = cv2.inRange(cv2.cvtColor(img, cv2.COLOR_BGR2HSV), lower_hsv_road, upper_hsv_road)

        road_centre = self.find_road_centre(mask, self.road_buffer, self.img_width, self.img_height)
        # mask_image = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        # cv2.circle(mask_image, (road_centre, self.img_height - self.road_buffer), 5, (0, 0, 255), -1)
        # cv2.imshow('mask', cv2.resize(mask_image, (self.img_width // 2, self.img_height // 2)))
        # cv2.waitKey(1)

        if road_centre != -1:
            error = ((self.img_width // 2) - road_centre) / (self.img_width // 2)
        elif self.reached_crosswalk and not self.reached_truck and my_bot.num_signs == 2:
            error = 0
            print('no road detected, going to truck state')
            self.state = 'truck'
            self.truck_init_cycle = self.cycle_count
        elif self.truck_turn_dir == 'left' and self.state == 'truck':
            error = self.truck_left_turn_amplifier * ((self.img_width // 2) - (self.img_width // 4)) / (self.img_width // 2)
        elif self.truck_turn_dir == 'right' and self.state == 'truck':
            error = ((self.img_width // 2) - (3 * self.img_width // 4)) / (self.img_width // 2)
        else:
            error = 0
        
        if self.state == 'mountain' and road_mask[self.img_height - 215, road_centre] != 255: # and self.boost_count < 10:
            # print('road centre not on the road')
            error = 0 #1.4 * ((self.img_width // 2) - (self.img_width // 4)) / (self.img_width // 2)
            self.boost = True
        return error
    
    def check_red(self, img, ret_angle=False, ret_y=False):
        uh_red = 255; us_red = 255; uv_red = 255
        lh_red = 90; ls_red = 50; lv_red = 230
        lower_hsv_red = np.array([lh_red, ls_red, lv_red])
        upper_hsv_red = np.array([uh_red, us_red, uv_red])
        
        hsv_img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        red_mask = cv2.inRange(hsv_img, lower_hsv_red, upper_hsv_red)

        contours, _ = cv2.findContours(red_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if contours.__len__() == 0:
            return False

        largest_contour = max(contours, key=cv2.contourArea)
        rect = cv2.minAreaRect(largest_contour)

        if not ret_angle and not ret_y:
            if cv2.contourArea(largest_contour) < self.red_line_min_area:
                return False
            else:
                return True
        elif ret_angle:
            return rect[2]
        elif ret_y:
            return rect[0][1]

    # return true if the pedestrian is on the cross walk or within the 
    def check_pedestrian(self, img):
        cropped_img = img[self.ped_crop_y_min:self.ped_crop_y_max, self.ped_crop_x_min:self.ped_crop_x_max]
        height, width = cropped_img.shape[:2]
        fg_mask = self.bg_sub.apply(cropped_img)

        contours, _ = cv2.findContours(fg_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if contours.__len__() == 0:
            return False
        largest_contour = max(contours, key=cv2.contourArea)

        if cv2.contourArea(largest_contour) < self.ped_min_area:
            return False

        x, y, w, h = cv2.boundingRect(largest_contour)

        gray_img = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2GRAY)
        white_mask = cv2.inRange(gray_img, self.road_min_white_val, self.road_max_white_val)
        ped_height_from_bottom = height - (y + h - 1)
        road_left, road_right = self.find_road_centre(white_mask, ped_height_from_bottom, width, height, ret_sides=True)

        if road_left == -1 and road_right == -1:
            return True

        # cv2.rectangle(cropped_img, (x, y), (x+w, y+h), (0, 255, 0), 2)
        # cv2.circle(cropped_img, (x + w//2, y + h), 5, (0, 0, 255), -1)
        # cv2.circle(cropped_img, (road_left, y+h), 5, (0, 0, 255), -1)
        # cv2.circle(cropped_img, (road_right, y+h), 5, (0, 0, 255), -1)
        
        # cv2.imshow('camera feed', np.hstack((cropped_img, cv2.cvtColor(fg_mask, cv2.COLOR_GRAY2BGR))))
        # cv2.waitKey(1)
        
        if road_left - self.ped_left_buffer < (x + w//2) < road_right + self.ped_right_buffer:
            return True
        else:
            return False
        
    def drive_robot(self, linear, angular):
        if linear >  self.move.linear.x + self.speed_buffer:
            rate = rospy.Rate(self.accel_freq)
            vel = self.move.linear.x
            while vel < linear:
                self.move.linear.x = vel
                self.move.angular.z = 0
                self.vel_pub.publish(self.move)
                vel += self.accel_rate
                rate.sleep()

        elif linear < self.move.linear.x - self.speed_buffer:
            vel = self.move.linear.x
            rate = rospy.Rate(self.accel_freq)
            while vel > linear:
                self.move.linear.x = vel
                self.move.angular.z = 0
                self.vel_pub.publish(self.move)
                vel -= self.decel_rate
                rate.sleep()
        # elif my_bot.num_signs == 2:
        #     self.move.linear.x = linear - 0.2
        #     self.move.angular.z = angular
        #     self.vel_pub.publish(self.move) 
        else:
            self.move.linear.x  = linear
            self.move.angular.z = angular
            self.vel_pub.publish(self.move)

    # returns true if it detects that the truck is big, if at intersection, returns contour area and mid x point
    def check_truck(self, img, at_intersection=False):
        fg_mask = self.bg_sub.apply(img)
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours.__len__() == 0:
            return 0, 0 if at_intersection else True
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)

        # cv2.imshow('fg mask', fg_mask)
        # cv2.waitKey(1)

        if at_intersection:
            return cv2.contourArea(largest_contour), x + w // 2
        
        # TODO: check if this is used, don't think it is
        elif cv2.contourArea(largest_contour) > self.truck_min_area:
            print('testing code not used for truck, here')
            return True
        else:
            print('also testing code not used for truck, here')
            return False
    
    # returns true if there is magenta at or below the point where we detect for road lines
    def check_magenta(self, img, ret_angle=False, ret_y=False, ret_midx=False):
        uh_mag = 175; us_mag = 255; uv_mag = 255
        lh_mag = 150; ls_mag = 90; lv_mag = 110
        lower_hsv_mag = np.array([lh_mag, ls_mag, lv_mag])
        upper_hsv_mag = np.array([uh_mag, us_mag, uv_mag])

        hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        magenta_mask = cv2.inRange(hsv_img, lower_hsv_mag, upper_hsv_mag)

        contours, _ = cv2.findContours(magenta_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if contours.__len__() == 0:
            if not ret_angle and not ret_y and not ret_midx:
                return False
            elif ret_angle:
                return 0
            elif ret_y:
                return self.img_height - 1 if self.state == 'desert' else 0
            elif ret_midx:
                return 0
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        rect = cv2.minAreaRect(largest_contour)

        if self.state == 'truck':
            if y >= self.img_height - self.road_buffer:
                return True
            else: 
                return False
        
        elif self.state == 'desert':
            if ret_angle:
                return rect[2]
            elif ret_y:
                if cv2.contourArea(largest_contour) < self.desert_past_magenta_line_area:
                    return self.img_height -1
                else:
                    return rect[0][1]
            elif cv2.contourArea(largest_contour) > self.desert_min_magenta_area:
                return True
            else:
                return False
        
        elif self.state == 'yoda':
            if ret_midx:
                return x + w // 2
            elif ret_y:
                return y + h // 2 #if cv2.contourArea(largest_contour) > self.yoda_mag_min_area_for_y else 0
            elif ret_angle:
                return rect[2]
            else: 
                return True if cv2.contourArea(largest_contour) > self.yoda_find_mag_min_area else False
            
    def thresh_desert(self, img):
        uh = 37; us = 98; uv = 255
        lh = 13; ls = 35; lv = 179
        lower_hsv = np.array([lh, ls, lv])
        upper_hsv = np.array([uh, us, uv])

        hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv_img, lower_hsv, upper_hsv)

        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=lambda contour: cv2.arcLength(contour, True), reverse=True) # don't think this line is necessary
        contours = [cnt for cnt in contours if cv2.arcLength(cnt, True) > self.desert_min_arc_length 
                    and cv2.boundingRect(cnt)[3] > self.desert_line_cnt_min_height ]
        if len(contours) == 0:
                    return np.zeros_like(img)
        epsilon = 0.01 * cv2.arcLength(contours[0], True)
        approx_cnts = [cv2.approxPolyDP(cnt, epsilon, True) for cnt in contours]

        blank_img = np.zeros_like(img)

        if self.state == 'desert':
            return cv2.fillPoly(blank_img, approx_cnts, (255, 255, 255))
        elif self.state == 'mountain':
            road_1 = cv2.fillPoly(blank_img, approx_cnts, (255, 255, 255))
            lv2 = 173
            lower_hsv2 = np.array([lh, ls, lv2])
            mask2 = cv2.inRange(hsv_img, lower_hsv2, upper_hsv)
            contours2, _ = cv2.findContours(mask2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            contours2 = sorted(contours2, key=lambda contour: cv2.arcLength(contour, True), reverse=True)
            contours2 = [cnt for cnt in contours2 if cv2.arcLength(cnt, True) > self.desert_min_arc_length
                        and cv2.boundingRect(cnt)[3] > self.desert_line_cnt_min_height]
            if len(contours2) == 0:
                return road_1
            epsilon2 = 0.01 * cv2.arcLength(contours2[0], True)
            approx_cnts2 = [cv2.approxPolyDP(cnt, epsilon2, True) for cnt in contours2]
            blank_img2 = np.zeros_like(img)
            road_2 = cv2.fillPoly(blank_img2, approx_cnts2, (255, 255, 255))

            total_road = cv2.bitwise_or(road_1, road_2)
            # cv2.imshow('mountain mask', cv2.resize(total_road, (self.img_width // 2, self.img_height // 2)))
            # cv2.waitKey(1)
            return total_road
        return blank_img
    
    def check_yoda(self, img):
        uh = 68; us = 255; uv = 255
        lh = 57; ls = 96; lv = 89
        lower_hsv = np.array([lh, ls, lv])
        upper_hsv = np.array([uh, us, uv])

        hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv_img, lower_hsv, upper_hsv)

        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if contours.__len__() == 0:
            return False
        largest_contour = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest_contour) > 800:
            return True
        else:
            return False

    # returns true if cactus contour area within range
    def check_cactus(self, img):
        uh_cactus = 66; us_cactus = 255; uv_cactus = 255
        lh_cactus = 56; ls_cactus = 86; lv_cactus = 63
        lower_hsv_cactus = np.array([lh_cactus, ls_cactus, lv_cactus])
        upper_hsv_cactus = np.array([uh_cactus, us_cactus, uv_cactus])

        uh_yoda = 68; us_yoda = 255; uv_yoda = 255
        lh_yoda = 57; ls_yoda = 96; lv_yoda = 89
        lower_hsv_yoda = np.array([lh_yoda, ls_yoda, lv_yoda])
        upper_hsv_yoda = np.array([uh_yoda, us_yoda, uv_yoda])

        hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        cactus_mask = cv2.inRange(hsv_img, lower_hsv_cactus, upper_hsv_cactus)
        yoda_mask = cv2.inRange(hsv_img, lower_hsv_yoda, upper_hsv_yoda)
        yoda_mask = cv2.bitwise_not(yoda_mask)

        mask = cv2.bitwise_and(cactus_mask, yoda_mask)

        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if contours.__len__() == 0:
            return False
        largest_contour = max(contours, key=cv2.contourArea)
        if self.cactus_min_area < cv2.contourArea(largest_contour) < self.cactus_max_area:
            return True
        else:
            return False
        
    def check_hill_stop(self, img):
        fg_mask = self.bg_sub.apply(img)
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) 
        if len(contours) == 0:
            return True
        else:
            largest_contour = max(contours, key=cv2.contourArea)
            return True if cv2.contourArea(largest_contour) < 30 else False
    
    # returns the centre point of the bounding rectangle of the tunnel, img width if no tunnel found by default
    # can also return the contour area and the mask image
    def find_tunnel(self, img, ret_area=False, ret_mask=False):
        uh = 9; us = 255; uv = 255
        lh = 0; ls = 106; lv = 66
        lower_hsv= np.array([lh, ls, lv])
        upper_hsv= np.array([uh, us, uv])

        hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv_img, lower_hsv, upper_hsv)

        if ret_mask:
            return mask

        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours = [cnt for cnt in contours if cv2.contourArea(cnt) > self.tunnel_min_area]
        if len(contours) == 0:
            return -1 if not ret_area else 0
        combined_contour = np.concatenate(contours)
        x, y, w, h = cv2.boundingRect(combined_contour)
        return x + w // 2 if not ret_area else cv2.contourArea(combined_contour)

    # finds middle x value of sign at top of mountain
    def find_mountain_sign(self, img, check_area=False):
        lower_hsv = (5,20,0)
        upper_hsv = (150,255,255)
        hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        blue_mask = cv2.inRange(hsv_img, lower_hsv, upper_hsv)

        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        white_mask = cv2.inRange(gray_img, 95, 105)

        blue_mask_not = cv2.bitwise_not(blue_mask)
        combined_mask = cv2.bitwise_and(white_mask, blue_mask_not)

        # cv2.imshow('sign mask', cv2.resize(combined_mask, (self.img_width // 2, self.img_height // 2)))
        # cv2.waitKey(1)

        contours, _ = cv2.findContours(combined_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if contours.__len__() == 0:
            return -1
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        if check_area:
            return cv2.contourArea(largest_contour) > 25000
        else:
            return x + w // 2 if cv2.contourArea(largest_contour) > 5000 else -1

    
    # placeholder for start function
    def start(self):
        # start the timer
        print('starting timer, entering road pid state')
        start_timer = String()
        start_timer.data = "Broda,adorb,0,NA"
        self.score_pub.publish(start_timer)
        self.state = 'road'
        # self.state = 'desert'
    
    # main loop for the driver
    def run(self):
        while not rospy.is_shutdown():
            if self.img is None:
                continue

            if self.cycle_count > 1700:
                self.state = 'clue submission'
            
            # --------------- initialization state ---------------
            elif self.state == 'init':
                self.start()

            # -------------------- road state --------------------
            elif self.state == 'road':
                cropped_img = my_bot.check_if_sign(self.img) # returns None if no sign detected
                if cropped_img is not None:
                    my_bot.compare_sign(cropped_img) # changes self.sign_img if new sign is larger
                if self.reached_crosswalk == False and self.check_red(self.img):
                    print('red detected, going to ped state')
                    self.state = 'ped'
                # elif self.reached_truck and self.check_magenta(self.img):
                #     print('magenta detected, going to desert state')
                #     self.state = 'desert'
                else:
                    error = self.kp * self.get_error(self.img)
                    # print(error)
                    if my_bot.num_signs == 2:
                        self.drive_robot(self.sign_lin_speed, self.sign_rot_speed * error)
                    if cropped_img is not None:
                        self.drive_robot(self.sign_lin_speed, self.sign_rot_speed * error)
                    else:
                        self.drive_robot(self.lin_speed, self.rot_speed * error)

            # ----------------- pedestrian state -----------------
            elif self.state == 'ped':
                # angle to be straight on with crosswalk
                while self.red_line_min_angle < self.check_red(self.img, ret_angle=True) < self.red_line_max_angle:
                    angle = self.check_red(self.img, ret_angle=True)
                    if angle < 45:
                        self.drive_robot(self.red_line_approach_lin_vel, -1 * angle * self.red_line_approach_rot_vel)
                    else:
                        self.drive_robot(self.red_line_approach_lin_vel, (90 - angle) * self.red_line_approach_rot_vel)
                
                # get close to crosswalk
                while self.check_red(self.img, ret_y=True) < self.red_line_stop_y:
                    self.drive_robot(self.lin_speed, 0)

                self.drive_robot(0, 0)

                if self.check_pedestrian(self.img):
                    # print('pedestrian detected, waiting...')
                    self.ped_safe_count = 0
                else:
                    self.ped_safe_count += 1
                    if self.ped_safe_count > self.ped_safe_count_buffer:
                        # print('no pedestrian, going!')
                        self.drive_robot(self.ped_lin_speed, self.ped_ang_speed)
                        # rospy.sleep(self.ped_sleep_time)
                        print('crossing crosswalk, going back to road pid state')
                        self.state = 'road'
                        self.reached_crosswalk = True
            
            # ------------------- truck state --------------------
            elif self.state == 'truck':
                cropped_img = my_bot.check_if_sign(self.img) # returns None if no sign detected
                if cropped_img is not None:
                    my_bot.compare_sign(cropped_img) # changes self.sign_img if new sign is larger
                if not self.reached_truck:
                    self.drive_robot(0, 0)
                    truck_area, truck_mid = self.check_truck(self.img, at_intersection=True)
                    if self.cycle_count < self.truck_init_cycle + self.truck_cycle_buffer:
                        # print('too early to tell')
                        pass
                    elif truck_mid < self.img_width // 2 and truck_area > self.truck_left_area:
                        print('truck close but on left, going right')
                        self.truck_turn_dir = 'right'
                        self.reached_truck = True
                    elif truck_area > self.truck_wait_area:
                        print('truck is close, waiting...')
                        self.drive_robot(0, 0)
                        self.truck_turn_dir = 'wait'
                    else:
                        print('going left')
                        self.truck_turn_dir = 'left'
                        self.reached_truck = True
                elif self.truck_turn_dir == 'right':
                    error = self.truck_right_kp * self.get_error(self.img)
                    self.drive_robot(self.truck_right_lin_speed, self.rot_speed * error)
                else:
                    error = self.kp * self.get_error(self.img)
                    if cropped_img is not None:
                        self.drive_robot(self.sign_lin_speed, self.sign_rot_speed * error)
                    else:
                        self.drive_robot(self.lin_speed, self.rot_speed * error)

                if self.check_magenta(self.img):
                    print('magenta detected, going to desert state')
                    self.state = 'desert'
                    self.drive_robot(self.lin_speed, 0)
                    rospy.sleep(self.truck_to_desert_sleep)

            # ------------------ desert state --------------------
            elif self.state == 'desert':
                cropped_img = my_bot.check_if_sign(my_bot.img) # returns None if no sign detected
                if cropped_img is not None:
                    my_bot.compare_sign(cropped_img) # changes self.sign_img if new sign is larger
                if self.check_magenta(self.img):
                    self.drive_robot(0, 0)
                    print('detected magenta')
                    while self.magneta_min_angle < self.check_magenta(self.img, ret_angle=True) < self.magneta_max_angle:
                        angle = self.check_magenta(self.img, ret_angle=True)
                        if angle < 45:
                            self.drive_robot(self.magenta_angle_lin_speed, -1 * angle * self.magenta_angle_rot_speed)
                        else:
                            self.drive_robot(self.magenta_angle_lin_speed, (90 - angle) * self.magenta_angle_rot_speed)

                    print('done angling, moving closer')
                    while self.check_magenta(self.img, ret_y=True) < self.img_height - 10:
                        self.drive_robot(self.magenta_angle_lin_speed, 0) 
                    self.drive_robot(0, 0)
                    print('going to yoda state')
                    self.state = 'yoda'
                else:
                    error = self.kp * self.get_error(self.img)
                    if cropped_img is not None:
                        self.drive_robot(self.sign_lin_speed, self.sign_rot_speed * error)
                    else:
                        self.drive_robot(self.lin_speed, self.rot_speed * error)

            # -------------------- yoda state --------------------
            elif self.state == 'yoda':
                cropped_img = my_bot.check_if_sign(my_bot.img) # returns None if no sign detected
                if cropped_img is not None:
                    my_bot.compare_sign(cropped_img) # changes self.sign_img if new sign is larger
                if not self.reached_yoda:
                    while self.check_yoda(self.img) and self.cycle_count < 1500:
                        print('detecting yoda')
                        self.drive_robot(0, 0)
                    print('getting close to cactus')
                    while not self.check_cactus(self.img) and self.cycle_count < 1500:
                        self.drive_robot(self.cactus_lin_speed, 0)
                    print('turning to see tunnel')
                    while self.find_tunnel(self.img) < self.tunnel_mid_x and self.cycle_count < 1500:
                        self.drive_robot(0, self.tunnel_turn_speed)
                    print('all good, ready to go over the hill')
                    self.drive_robot(0, 0)
                    self.reached_yoda = True
                else:
                    if not self.over_hill:
                        tunnel_mid = self.find_tunnel(self.img)
                        if tunnel_mid == -1:
                            while not self.check_magenta(self.img) and self.cycle_count < 1500:
                                self.drive_robot(self.hill_lin_speed, self.hill_rot_speed)
                            print('over the hill now, checking for magenta')
                            self.over_hill = True
                        else:
                            error = self.kp * (self.tunnel_mid_x - tunnel_mid) / self.tunnel_mid_x
                            self.drive_robot(self.lin_speed, self.rot_speed * error)
                            if self.check_yoda(self.img):
                                self.drive_robot(0, 0)
                                rospy.sleep(0.3)
                    else:
                        while self.check_magenta(self.img, ret_y=True) < 408 and self.cycle_count < 1500: 
                            if self.check_hill_stop(self.img):
                                print('stalled on hill')
                                self.drive_robot(0, 0)
                                rospy.sleep(0.5)
                            mag_x = self.check_magenta(self.img, ret_midx=True)
                            error = self.kp * (self.yoda_mag_x_mid - mag_x) / self.yoda_mag_x_mid
                            self.drive_robot(0.6, self.rot_speed * error)
                            # print('y: ', self.check_magenta(self.img, ret_y=True))
                            if self.check_yoda(self.img):
                                self.drive_robot(0, 0)
                                rospy.sleep(0.3)
                        print('going straight now')
                        while self.check_magenta(self.img, ret_y=True) < 590 and self.cycle_count < 1500:
                            self.drive_robot(0.5, 0)
                            # print('y: ', self.check_magenta(self.img, ret_y=True))
                        print('close to magenta, angling to be straight')
                        while 0.5 < self.check_magenta(self.img, ret_angle=True) < 89.5 and self.cycle_count < 1500:
                            angle = self.check_magenta(self.img, ret_angle=True)
                            # print('angle: ', angle)
                            if angle < 45:
                                self.drive_robot(0, -1 * angle * 0.05)
                            else:
                                self.drive_robot(0, (90 - angle) * 0.05)
                        self.drive_robot(0.4, 0)
                        print('going to tunnel state')
                        # self.drive_robot(self.lin_speed, -0.8)
                        rospy.sleep(0.3)
                        self.state = 'tunnel'

            # ------------------ tunnel state ------------------
            elif self.state == 'tunnel':
                cropped_img = my_bot.check_if_sign(my_bot.img) # returns None if no sign detected
                if cropped_img is not None:
                    my_bot.compare_sign(cropped_img) # changes self.sign_img if new sign is larger
                if self.find_tunnel(self.img, ret_area=True) > 10000:
                    self.drive_robot(1.2, 0)
                else:
                    print('tunnel contour too small, going to mountain state')
                    self.state = 'mountain'
                    rospy.sleep(0.5)

            # ----------------- mountain state -----------------
            elif self.state == 'mountain':
                if not self.found_mountain_lines:
                    while not np.any(self.thresh_desert(self.img)) and self.cycle_count < 1500:
                        # print('no lines found')
                        self.drive_robot(0.5, 0)
                    self.found_mountain_lines = True
                    print('found mountain lines, going to pid')
                    rospy.sleep(0.4)
                    self.mountain_start_cycle = self.cycle_count
                else:
                    while self.find_mountain_sign(self.img) == -1 and self.cycle_count < 1500:
                        error = self.get_error(self.img)
                        derivative = (error - self.prev_error) / self.dt
                        self.prev_error = error
                        rot_amp = 8 * error + self.kd * derivative
                        if self.boost and 150 < self.cycle_count - self.mountain_start_cycle and self.cycle_count > self.boost_cycle + 1:
                            self.drive_robot(0.5, 0.9)
                            self.boost = False
                            self.boost_cycle = self.cycle_count
                            print('boosting!!!')
                            # self.boost_count += 1
                        else:
                            rot_speed = 1.2 * rot_amp
                            if rot_speed < -1.5:
                                rot_speed = -1.5
                            self.drive_robot(0.3, rot_speed)
                    print('found sign, going to sign state')
                    self.state = 'mountain top'

            # ---------------- mountain top state ---------------
            elif self.state == 'mountain top':
                cropped_img = my_bot.check_if_sign(my_bot.img) # returns None if no sign detected
                if cropped_img is not None:
                    my_bot.compare_sign(cropped_img) # changes self.sign_img if new sign is larger
                if not self.find_mountain_sign(self.img, check_area=True) and self.cycle_count < 1500:
                    sign_mid_x = self.find_mountain_sign(self.img)
                    print('pid ing to sign')
                    if sign_mid_x == -1:
                        self.drive_robot(0.3, 0)
                    else:
                        error = 9 * (self.img_width // 2 - sign_mid_x) / (self.img_width // 2)
                        self.drive_robot(0.3, self.rot_speed * error)
                else:
                    print('close to sign, stopping')
                    self.state = 'clue submission'
                    self.drive_robot(0, 0)
                    print('final cycle', self.cycle_count)

            # ----------------- clue submission state -----------------
            elif self.state == "clue submission":
                for i in range(my_bot.num_signs):
                    message = String()
                    prediction = my_bot.read_sign(my_bot.signs[i])
                    message.data = "Broda,adorb,"+str(i+1)+","+prediction
                    self.score_pub.publish(message)
                end_timer = String()
                end_timer.data = "Broda,adorb,-1,NA"
                self.score_pub.publish(end_timer)
                self.state = 'finished'

            # ----------------- finished state -----------------
            elif self.state == "finished":
                self.drive_robot(0, 0)
                break
            

            if my_bot.sign_img is not None:
                # display the sign image
                #cv2.imshow('feed', my_bot.img)
                #cv2.waitKey(1)
                # check if enough time has elapsed to read the sign
                current_time = rospy.Time.now()
                elapsed_time = current_time - my_bot.firstSignTime
                if elapsed_time > my_bot.durationBetweenSigns:
                    my_bot.signs.append(my_bot.sign_img)
                    my_bot.num_signs += 1
                    my_bot.sign_img = None
        
                
        # rospy.sleep(0.1)

if __name__ == '__main__':
    try:
        my_driver = Driver()
        my_bot = sign_reader.SignReader()
        rospy.sleep(1)
        my_driver.run()
    except rospy.ROSInterruptException:
        pass
                