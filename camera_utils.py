import cv2
import time

class MotionDetector:
    def __init__(self):
        self.prev_frame = None
        self.motion_threshold = 20
        self.min_contour_area = 400
        self.body_regions = {
            'left_hand': {'region': (0.0, 0.3, 0.5, 0.7), 'brain_area': 'C4', 'description': 'Right side of brain (C4)'},
            'right_hand': {'region': (0.5, 0.3, 1.0, 0.7), 'brain_area': 'C3', 'description': 'Left side of brain (C3)'},
            'head': {'region': (0.25, 0.0, 0.75, 0.3), 'brain_area': 'Cz', 'description': 'Middle of brain (Cz)'}
        }
        self.active_regions = {region: False for region in self.body_regions}
        self.cooldown = {region: 0 for region in self.body_regions}
        self.cooldown_time = 0.5
    
    def start_camera(self):
        
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise Exception("Could not open video device")
        
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        return self.cap.isOpened()
    
    def stop_camera(self):
        
        if hasattr(self, 'cap') and self.cap.isOpened():
            self.cap.release()
    
    def get_frame(self):
        
        if not hasattr(self, 'cap') or not self.cap.isOpened():
            return None
        
        ret, frame = self.cap.read()
        if not ret:
            return None
        
        
        frame = cv2.flip(frame, 1)
        return frame
    
    def detect_motion(self, frame):
        
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        gray = cv2.GaussianBlur(gray, (21, 21), 0)
        
        
        if self.prev_frame is None:
            self.prev_frame = gray
            return frame, {}
        
        
        frame_delta = cv2.absdiff(self.prev_frame, gray)
        
        thresh = cv2.threshold(frame_delta, self.motion_threshold, 255, cv2.THRESH_BINARY)[1]
        
        thresh = cv2.dilate(thresh, None, iterations=2)
        
        
        contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        
        motion_regions = {}
        current_time = time.time()
        
        
        for region in self.active_regions:
            
            if self.cooldown[region] < current_time:
                self.active_regions[region] = False
        
        
        height, width = frame.shape[:2]
        
        for contour in contours:
            if cv2.contourArea(contour) < self.min_contour_area:
                continue
                
            (x, y, w, h) = cv2.boundingRect(contour)
            

            for region_name, region_info in self.body_regions.items():
                rx1, ry1, rx2, ry2 = region_info['region']
                region_x1, region_y1 = int(rx1 * width), int(ry1 * height)
                region_x2, region_y2 = int(rx2 * width), int(ry2 * height)
                
                
                cv2.rectangle(frame, (region_x1, region_y1), (region_x2, region_y2), (0, 255, 0), 1)
                
                
                contour_center_x = x + w//2
                contour_center_y = y + h//2
                
                if (region_x1 <= contour_center_x <= region_x2 and 
                    region_y1 <= contour_center_y <= region_y2):
                    
                    self.active_regions[region_name] = True
                    self.cooldown[region_name] = current_time + self.cooldown_time
                    
                    
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
        
        
        self.prev_frame = gray
        
        
        return frame, self.active_regions
    
    def get_active_brain_regions(self):
        
        active_brain_regions = {}
        for region_name, is_active in self.active_regions.items():
            if is_active:
                brain_area = self.body_regions[region_name]['brain_area']
                description = self.body_regions[region_name]['description']
                active_brain_regions[brain_area] = {
                    'active': True,
                    'body_part': region_name,
                    'description': description
                }
        
        
        all_brain_regions = {'C3', 'C4', 'Cz'}
        for region in all_brain_regions:
            if region not in active_brain_regions:
                active_brain_regions[region] = {
                    'active': False,
                    'body_part': None,
                    'description': f"{'Left' if region == 'C3' else 'Right' if region == 'C4' else 'Middle'} side of brain ({region})"
                }
                
        return active_brain_regions
        
    def generate_real_time_signal(self, region, base_signal, num_points=100):
        
        import numpy as np
        
        
        active_regions = self.get_active_brain_regions()
        is_active = active_regions[region]['active']
        
        
        if len(base_signal) > num_points:
            start_idx = np.random.randint(0, len(base_signal) - num_points)
            signal_segment = base_signal[start_idx:start_idx + num_points].copy()
        else:
            
            signal_segment = base_signal.copy()
            
        
        if is_active:
            
            amplitude_factor = 1.5 + 0.5 * np.random.random()
            signal_segment = signal_segment * amplitude_factor
            
            
            t = np.linspace(0, 2*np.pi, len(signal_segment))
            high_freq = 0.3 * np.sin(10*t) + 0.2 * np.sin(15*t)
            signal_segment += high_freq[:len(signal_segment)]
        else:
            
            amplitude_factor = 0.7 + 0.3 * np.random.random()
            signal_segment = signal_segment * amplitude_factor
            
            
            t = np.linspace(0, 2*np.pi, len(signal_segment))
            low_freq = 0.2 * np.sin(2*t)
            signal_segment += low_freq[:len(signal_segment)]
        
        return signal_segment