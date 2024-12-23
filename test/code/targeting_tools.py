import time
import threading
import queue
import math
import numpy as np
import cv2

class Camera_Thread:

    # camera setup
    camera_source = 1
    camera_width = 1280
    camera_height = 800
    camera_frame_rate = 60
    # buffer setup
    buffer_length = 5
    buffer_all = False

    # ------------------------------
    # System Variables
    # ------------------------------

    # camera
    camera = None
    camera_init = 0.5

    # buffer
    buffer = None

    # control states
    frame_grab_run = False
    frame_grab_on = False

    # counts and amounts
    frame_count = 0
    frames_returned = 0
    current_frame_rate = 0
    loop_start_time = 0

    # ------------------------------
    # Functions
    # ------------------------------

    def start(self):

        # buffer
        if self.buffer_all:
            self.buffer = queue.Queue(self.buffer_length)
        else:
            # last frame only
            self.buffer = queue.Queue(1)

        # camera setup
        self.camera = cv2.VideoCapture(self.camera_source)
        # self.camera.set(3, 1748)
        self.camera.set(3, 1748)
        self.camera.set(5,self.camera_frame_rate)
        time.sleep(self.camera_init)

        self.camera_area = self.camera_width*self.camera_height

        # black frame (filler)
        self.black_frame = np.zeros((self.camera_height,self.camera_width,3),np.uint8)

        # set run state
        self.frame_grab_run = True
        
        # start thread
        self.thread = threading.Thread(target=self.loop)
        self.thread.start()

    def stop(self):

        # set loop kill state
        self.frame_grab_run = False
        
        # let loop stop
        while self.frame_grab_on:
            time.sleep(0.1)

        # stop camera if not already stopped
        if self.camera:
            try:
                self.camera.release()
            except:
                pass
        self.camera = None

        # drop buffer
        self.buffer = None

    def loop(self):

        # load start frame
        frame = self.black_frame
        if not self.buffer.full():
            self.buffer.put(frame,False)

        # status
        self.frame_grab_on = True
        self.loop_start_time = time.time()

        # frame rate
        fc = 0
        t1 = time.time()

        # loop
        while 1:

            # external shut down
            if not self.frame_grab_run:
                break

            # true buffered mode (for files, no loss)
            if self.buffer_all:

                # buffer is full, pause and loop
                if self.buffer.full():
                    time.sleep(1/self.camera_frame_rate)

                # or load buffer with next frame
                else:
                    
                    grabbed,frame = self.camera.read()

                    if not grabbed:
                        break

                    self.buffer.put(frame,False)
                    self.frame_count += 1
                    fc += 1

            # false buffered mode (for camera, loss allowed)
            else:

                grabbed,frame = self.camera.read()
                if not grabbed:
                    break

                # open a spot in the buffer
                if self.buffer.full():
                    self.buffer.get()

                self.buffer.put(frame,False)
                self.frame_count += 1
                fc += 1

            # update frame read rate
            if fc >= 10:
                self.current_frame_rate = round(fc/(time.time()-t1),2)
                fc = 0
                t1 = time.time()

        # shut down
        self.loop_start_time = 0
        self.frame_grab_on = False
        self.stop()

    def get_views(self, frame):
        if self.camera_width == 1280:
            left = frame[:, 48:48 + 1280]
            right = frame[:, 1328:1328 + 1280]
        elif self.camera_width == 640:
            left = frame[:, 48:48 + 640]
            right = frame[:, 688:688 + 637]
        else:
            left = frame[0:int(self.camera_height/2), 48:int(self.camera_width/2)]
            right = frame[0:int(self.camera_height/2), int(self.camera_width/2):self.camera_width]
        return (left, right)

    def next(self,black=True,wait=0):

        # black frame default
        if black:
            frame = self.black_frame

        # no frame default
        else:
            frame = None

        # get from buffer (fail if empty)
        try:
            frame = self.buffer.get(timeout=wait)
            left, right = self.get_views(frame)
            # mirror right
            # right = cv2.flip(right,1)
            self.frames_returned += 1
        except queue.Empty:
            #print('Queue Empty!')
            #import traceback
            #print(traceback.format_exc())
            pass

        # done
        return (left, right)

# ------------------------------
# Motion Detection
# ------------------------------

class Frame_Motion:

    # ------------------------------
    # User Instructions
    # ------------------------------

    # ------------------------------
    # User Variables
    # ------------------------------

    # blur (must be positive and odd)
    gaussian_blur = 15

    # threshold
    threshold = 15

    # dilation
    dilation_value = 6
    dilation_iterations = 2
    dilation_kernel = np.ones((dilation_value,dilation_value),np.uint8)

    # contour size
    contour_min_area = 1  # percent of frame area
    contour_max_area = 80 # percent of frame area

    # target select
    targets_max = 4 # max targets returned
    target_on_contour = True # else use box size
    target_return_box  = False # True = return (x,y,bx,by,bw,bh), else check target_return_size
    target_return_size = False # True = return (x,y,percent_frame_size), else just (x,y)

    # display contour
    contour_draw  = True
    contour_line  = 1 # border width
    contour_point = 5 # centroid point radius
    contour_pline = -1 # centroid point line width
    contour_color = (0,255,255) # BGR color
        
    # display contour box
    contour_box_draw  = True
    contour_box_line  = 1 # border width
    contour_box_point = 4 # centroid point radius
    contour_box_pline = -1 # centroid point line width
    contour_box_color = (0,255,0) # BGR color

    # display targets
    targets_draw  = True
    targets_point = 4 # centroid radius
    targets_pline = -1 # border width
    targets_color = (0,0,255) # BGR color

    # ------------------------------
    # System Variables
    # ------------------------------

    last_frame = None

    # ------------------------------
    # Functions
    # ------------------------------

    def targets2(self, frame):

        # Frame dimensions
        width, height, depth = np.shape(frame)
        area = width * height

        # Convert frame to HSV color space
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Define HSV range for turquoise color
        lower_bound = np.array([0, 0, 0])
        upper_bound = np.array([180, 255, 30])

        # Create a mask for the turquoise color
        mask = cv2.inRange(hsv_frame, lower_bound, upper_bound)

        # Apply morphological operations to remove noise and improve detection
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        # Find contours in the mask
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Targets list
        targets = []
        for c in contours:

            # Basic contour data
            ca = cv2.contourArea(c)
            bx, by, bw, bh = cv2.boundingRect(c)
            ba = bw * bh

            # Target on contour
            if self.target_on_contour:
                p = 100 * ca / area
                if (p >= self.contour_min_area) and (p <= self.contour_max_area):
                    M = cv2.moments(c)
                    if M['m00'] != 0:
                        tx = int(M['m10'] / M['m00'])
                        ty = int(M['m01'] / M['m00'])
                    else:
                        tx, ty = 0, 0
                    targets.append((p, tx, ty, bx, by, bw, bh, c))

            # Target on contour box
            else:
                p = 100 * ba / area
                if (p >= self.contour_min_area) and (p <= self.contour_max_area):
                    tx = bx + int(bw / 2)
                    ty = by + int(bh / 2)
                    targets.append((p, tx, ty, bx, by, bw, bh, c))

        # Select targets
        targets.sort(reverse=True)
        targets = targets[:self.targets_max]

        # Add contours to frame
        if self.contour_draw:
            for size, x, y, bx, by, bw, bh, c in targets:
                cv2.drawContours(frame, [c], 0, self.contour_color, self.contour_line)
                cv2.circle(frame, (x, y), self.contour_point, self.contour_color, self.contour_pline)

        # Add contour boxes to frame
        if self.contour_box_draw:
            for size, x, y, bx, by, bw, bh, c in targets:
                cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), self.contour_box_color, self.contour_box_line)
                cv2.circle(frame, (bx + int(bw / 2), by + int(bh / 2)), self.contour_box_point, self.contour_box_color, self.contour_box_pline)

        # Add targets to frame
        if self.targets_draw:
            for size, x, y, bx, by, bw, bh, c in targets:
                cv2.circle(frame, (x, y), self.targets_point, self.targets_color, self.targets_pline)

        # Return target coordinates
        if self.target_return_box:
            return [(x, y, bx, by, bw, bh) for (size, x, y, bx, by, bw, bh, c) in targets]
        elif self.target_return_size:
            return [(x, y, size) for (size, x, y, bx, by, bw, bh, c) in targets]
        else:
            return [(x, y) for (size, x, y, bx, by, bw, bh, c) in targets]
    
    def targets(self,frame):

        # frame dimensions
        width,height,depth = np.shape(frame)
        area = width*height

        # save frame
        # grayscale
        frame2 = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        # blur
        frame2 = cv2.GaussianBlur(frame2,(self.gaussian_blur,self.gaussian_blur),0)
        # initialize compare frame
        if self.last_frame is None:
            self.last_frame = frame2
            return []

        # delta
        frame3 = cv2.absdiff(self.last_frame,frame2)
        # threshold
        frame3 = cv2.threshold(frame3,self.threshold,255,cv2.THRESH_BINARY)[1]
        # dilation
        frame3 = cv2.dilate(frame3,self.dilation_kernel,iterations=self.dilation_iterations)
        # get contours
        # older opencv: frame3,contours,hierarchy = cv2.findContours(frame3,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        contours,hierarchy = cv2.findContours(frame3,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

        # targets
        targets = []
        for c in contours:
                    
            # basic contour data
            ca = cv2.contourArea(c)
            bx,by,bw,bh = cv2.boundingRect(c)
            ba = bw*bh

            # target on contour
            if self.target_on_contour:
                p = 100*ca/area
                if (p >= self.contour_min_area) and (p <= self.contour_max_area):
                    M = cv2.moments(c)#;print( M )
                    tx = int(M['m10']/M['m00'])
                    ty = int(M['m01']/M['m00'])
                    targets.append((p,tx,ty,bx,by,bw,bh,c))

            # target on contour box
            else:
                p = 100*ba/area
                if (p >= self.contour_min_area) and (p <= self.contour_max_area):
                    tx = bx+int(bw/2)
                    ty = by+int(bh/2)
                    targets.append((p,tx,ty,bx,by,bw,bh,c))

        # select targets
        targets.sort()
        targets.reverse()
        targets = targets[:self.targets_max +4]

        # add contours to frame
        if self.contour_draw:
            for size,x,y,bx,by,bw,bh,c in targets:
                cv2.drawContours(frame,[c],0,self.contour_color,self.contour_line)
                cv2.circle(frame,(x,y),self.contour_point,self.contour_color,self.contour_pline)

        # add contour boxes to frame
        if self.contour_box_draw:
            for size,x,y,bx,by,bw,bh,c in targets:
                cv2.rectangle(frame,(bx,by),(bx+bw,by+bh),self.contour_box_color,self.contour_box_line)
                cv2.circle(frame,(bx+int(bw/2),by+int(bh/2)),self.contour_box_point,self.contour_box_color,self.contour_box_pline)

        # add targets to frame
        if self.targets_draw:
            for size,x,y,bx,by,bw,bh,c in targets:
                cv2.circle(frame,(x,y),self.targets_point,self.targets_color,self.targets_pline)

        # reset last frame
        self.last_frame = frame2

        # return target x,y
        if self.target_return_box:
            return [(x,y,bx,by,bw,bh) for (size,x,y,bx,by,bw,bh,c) in targets]
        elif self.target_return_size:
            return [(x,y,size) for (size,x,y,bx,by,bw,bh,c) in targets]
        else:
            return [(x,y) for (size,x,y,bx,by,bw,bh,c) in targets]

    def frame_add_crosshairs(self,frame,x,y,r=20,lc=(0,0,255),cc=(0,0,255),lw=1,cw=1):

        x = int(round(x,0))
        y = int(round(y,0))
        r = int(round(r,0))

        cv2.line(frame,(x,y-r*2),(x,y+r*2),lc,lw)
        cv2.line(frame,(x-r*2,y),(x+r*2,y),lc,lw)

        cv2.circle(frame,(x,y),r,cc,cw)


# ------------------------------
# Frame Angles and Distance
# ------------------------------

class Frame_Angles:

    # ------------------------------
    # User Instructions
    # ------------------------------

    # Set the pixel width and height.
    # Set the angle width (and angle height if it is disproportional).
    # These can be set during init, or afterwards.

    # Run build_frame.

    # Use angles_from_center(self,x,y,top_left=True,degrees=True) to get x,y angles from center.
    # If top_left is True, input x,y pixels are measured from the top left of frame.
    # If top_left is False, input x,y pixels are measured from the center of the frame.
    # If degrees is True, returned angles are in degrees, otherwise radians.
    # The returned x,y angles are always from the frame center, negative is left,down and positive is right,up.

    # Use pixels_from_center(self,x,y,degrees=True) to convert angle x,y to pixel x,y (always from center).
    # This is the reverse of angles_from_center.
    # If degrees is True, input x,y should be in degrees, otherwise radians.

    # Use frame_add_crosshairs(frame) to add crosshairs to a frame.
    # Use frame_add_degrees(frame) to add 10 degree lines to a frame (matches target).
    # Use frame_make_target(openfile=True) to make an SVG image target and open it (matches frame with degrees).

    # ------------------------------
    # User Variables
    # ------------------------------

    pixel_width = 640
    pixel_height = 480

    angle_width = 60
    angle_height = None
    
    # ------------------------------
    # System Variables
    # ------------------------------

    x_origin = None
    y_origin = None

    x_adjacent = None
    x_adjacent = None

    # ------------------------------
    # Init Functions
    # ------------------------------

    def __init__(self,pixel_width=None,pixel_height=None,angle_width=None,angle_height=None):

        # full frame dimensions in pixels
        if type(pixel_width) in (int,float):
            self.pixel_width = int(pixel_width)
        if type(pixel_height) in (int,float):
            self.pixel_height = int(pixel_height)

        # full frame dimensions in degrees
        if type(angle_width) in (int,float):
            self.angle_width = float(angle_width)
        if type(angle_height) in (int,float):
            self.angle_height = float(angle_height)

        # do initial setup
        self.build_frame()

    def build_frame(self):

        # this assumes correct values for pixel_width, pixel_height, and angle_width

        # fix angle height
        if not self.angle_height:
            self.angle_height = self.angle_width*(self.pixel_height/self.pixel_width)

        # center point (also max pixel distance from origin)
        self.x_origin = int(self.pixel_width/2)
        self.y_origin = int(self.pixel_height/2)

        # theoretical distance in pixels from camera to frame
        # this is the adjacent-side length in tangent calculations
        # the pixel x,y inputs is the opposite-side lengths
        self.x_adjacent = self.x_origin / math.tan(math.radians(self.angle_width/2))
        self.y_adjacent = self.y_origin / math.tan(math.radians(self.angle_height/2))

    # ------------------------------
    # Pixels-to-Angles Functions
    # ------------------------------

    def angles(self,x,y):

        return self.angles_from_center(x,y)

    def angles_from_center(self,x,y,top_left=True,degrees=True):

        # x = pixels right from left edge of frame
        # y = pixels down from top edge of frame
        # if not top_left, assume x,y are from frame center
        # if not degrees, return radians

        if top_left:
            x = x - self.x_origin
            y = self.y_origin - y

        xtan = x/self.x_adjacent
        ytan = y/self.y_adjacent

        xrad = math.atan(xtan)
        yrad = math.atan(ytan)

        if not degrees:
            return xrad,yrad

        return math.degrees(xrad),math.degrees(yrad)

    def pixels_from_center(self,x,y,degrees=True):

        # this is the reverse of angles_from_center

        # x = horizontal angle from center
        # y = vertical angle from center
        # if not degrees, angles are radians

        if degrees:
            x = math.radians(x)
            y = math.radians(y)

        return int(self.x_adjacent*math.tan(x)),int(self.y_adjacent*math.tan(y))

    # ------------------------------
    # 3D Functions
    # ------------------------------

    def distance(self,*coordinates):
        return self.distance_from_origin(*coordinates)

    def distance_from_origin(self,*coordinates):
        return math.sqrt(sum([x**2 for x in coordinates]))
    
    def intersection(self,pdistance,langle,rangle,degrees=False):

        # return (X,Y) of target from left-camera-center

        # pdistance is the measure from left-camera-center to right-camera-center (point-to-point, or point distance)
        # langle is the left-camera  angle to object measured from center frame (up/right positive)
        # rangle is the right-camera angle to object measured from center frame (up/right positive)
        # left-camera-center is origin (0,0) for return (X,Y)
        # X is measured along the baseline from left-camera-center to right-camera-center
        # Y is measured from the baseline

        # fix degrees
        if degrees:
            langle = math.radians(langle)
            rangle = math.radians(rangle)

        # fix angle orientation (from center frame)
        # here langle is measured from right baseline
        # here rangle is measured from left  baseline
        langle = math.pi/2 - langle
        rangle = math.pi/2 + rangle

        # all calculations using tangent
        ltan = math.tan(langle)
        rtan = math.tan(rangle)

        # get Y value
        # use the idea that pdistance = ( Y/ltan + Y/rtan )
        Y = pdistance / ( 1/ltan + 1/rtan )

        # get X measure from left-camera-center using Y
        X = Y/ltan

        # done
        return X,Y

    def location(self,pdistance,lcamera,rcamera,center=False,degrees=True):

        # return (X,Y,Z,D) of target from left-camera-center (or baseline midpoint if center-True)

        # pdistance is the measure from left-camera-center to right-camera-center (point-to-point, or point distance)
        # lcamera = left-camera-center (Xangle-to-target,Yangle-to-target)
        # rcamera = right-camera-center (Xangle-to-target,Yangle-to-target)
        # left-camera-center is origin (0,0) for return (X,Y)
        # X is measured along the baseline from left-camera-center to right-camera-center
        # Y is measured from the baseline
        # Z is measured vertically from left-camera-center (should be same as right-camera-center)
        # D is distance from left-camera-center (based on pdistance units)

        # separate values
        lxangle,lyangle = lcamera
        rxangle,ryangle = rcamera

        # yangle should be the same for both cameras (if aligned correctly)
        yangle = (lyangle+ryangle)/2

        # fix degrees
        if degrees:
            lxangle = math.radians(lxangle)
            rxangle = math.radians(rxangle)
            yangle  = math.radians( yangle)

        # get X,Z (remember Y for the intersection is Z frame)
        X,Z = self.intersection(pdistance,lxangle,rxangle,degrees=False)

        # get Y
        # using yangle and 2D distance to target
        Y = math.tan(yangle) * self.distance_from_origin(X,Z)

        # baseline-center instead of left-camera-center
        if center:
            X -= pdistance/2

        # get 3D distance
        D = self.distance_from_origin(X,Y,Z)

        # done
        return X,Y,Z,D
    
    # ------------------------------
    # Tertiary Functions
    # ------------------------------

    def frame_add_crosshairs(self,frame):

        # add crosshairs to frame to aid in aligning

        cv2.line(frame,(0,self.y_origin),(self.pixel_width,self.y_origin),(0,255,0),1)
        cv2.line(frame,(self.x_origin,0),(self.x_origin,self.pixel_height),(0,255,0),1)

        cv2.circle(frame,(self.x_origin,self.y_origin),int(round(self.y_origin/8,0)),(0,255,0),1)

    def frame_add_degrees(self,frame):

        # add lines to frame every 10 degrees (horizontally and vertically)
        # use this to test that your angle values are set up properly

        for angle in range(10,95,10):

            # calculate pixel offsets
            x,y = self.pixels_from_center(angle,angle)

            # draw verticals
            if x <= self.x_origin:
                cv2.line(frame,(self.x_origin-x,0),(self.x_origin-x,self.pixel_height),(255,0,255),1)
                cv2.line(frame,(self.x_origin+x,0),(self.x_origin+x,self.pixel_height),(255,0,255),1)

            # draw horizontals
            if y <= self.y_origin:
                cv2.line(frame,(0,self.y_origin-y),(self.pixel_width,self.y_origin-y),(255,0,255),1)
                cv2.line(frame,(0,self.y_origin+y),(self.pixel_width,self.y_origin+y),(255,0,255),1)

    def frame_make_target(self,outfilename='targeting_angles_frame_target.svg',openfile=False):

        # this will make a printable target that matches the frame_add_degrees output
        # use this to test that your angle values are set up properly
        
        # svg size
        ratio = self.pixel_height/self.pixel_width
        width = 1600
        height = 1600 * ratio

        #svg frame locations
        x_origin = width/2
        y_origin = height/2
        distance = width*0.5

        # start svg
        svg  = '<svg xmlns="http://www.w3.org/2000/svg"\n'
        svg += 'xmlns:xlink="http://www.w3.org/1999/xlink"\n'
        svg += 'width="{}px"\n'.format(width)
        svg += 'height="{}px">\n'.format(height)

        # crosshairs
        svg += '<line x1="{}" x2="{}" y1="{}" y2="{}" stroke-width="1" stroke="green"/>\n'.format(0,width,y_origin,y_origin)
        svg += '<line x1="{}" x2="{}" y1="{}" y2="{}" stroke-width="1" stroke="green"/>\n'.format(x_origin,x_origin,0,height)

        # center circle
        svg += '<circle cx="{}" cy="{}" r="{}" stroke="green" stroke-width="1" fill="none"/>'.format(x_origin,y_origin,y_origin/8)

        # distance from screen line
        svg += '<line x1="{0}" x2="{1}" y1="{2}" y2="{2}" stroke-width="1" stroke="red"/>\n'.format(x_origin-distance/2,x_origin+distance/2,y_origin-y_origin/8)
        svg += '<line x1="{0}" x2="{0}" y1="{1}" y2="{2}" stroke-width="1" stroke="red"/>\n'.format(x_origin-distance/2,y_origin-y_origin/16,y_origin-y_origin/8)
        svg += '<line x1="{0}" x2="{0}" y1="{1}" y2="{2}" stroke-width="1" stroke="red"/>\n'.format(x_origin+distance/2,y_origin-y_origin/16,y_origin-y_origin/8)

        # add degree lines
        for angle in range(10,95,10):
            pixels = distance * math.tan(math.radians(angle))

            # draw verticals
            if pixels <= x_origin:
                svg += '<line x1="{0}" x2="{0}" y1="0" y2="{1}" stroke-width="1" stroke="black"/>\n'.format(x_origin-pixels,height)
                svg += '<line x1="{0}" x2="{0}" y1="0" y2="{1}" stroke-width="1" stroke="black"/>\n'.format(x_origin+pixels,height)

            # draw horizontals
            if pixels <= y_origin:
                svg += '<line x1="0" x2="{0}" y1="{1}" y2="{1}" stroke-width="1" stroke="black"/>\n'.format(width,y_origin-pixels)
                svg += '<line x1="0" x2="{0}" y1="{1}" y2="{1}" stroke-width="1" stroke="black"/>\n'.format(width,y_origin+pixels)

        # end svg
        svg += '</svg>'

        # write file
        outfile = open(outfilename,'w')
        outfile.write(svg)
        outfile.close()

        # open file
        if openfile:
            import os
            import webbrowser
            webbrowser.open(os.path.abspath(outfilename))

# ------------------------------
# end
# ------------------------------
    
