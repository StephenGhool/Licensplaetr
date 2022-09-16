import cv2
# import nanocamera as nano

# # Create the Camera instance for 640 by 480
# camera = nano.Camera(camera_type = 0)
# frame = camera.read()
# print(frame)

gstreamer_str = "nvarguscamerasrc sensor_id=0 ! \
   'video/x-raw(memory:NVMM),width=1920, height=1080, framerate=30/1' ! \
   nvvidconv flip-method=0 ! 'video/x-raw,width=960, height=540' ! \
   nvvidconv ! nvegltransform ! nveglglessink -e"

# function would returnt he gstream pipeline
def gstreamer_pipeline(
    sensor_id=0,
    capture_width=1920,
    capture_height=1080,
    display_width=960,
    display_height=540,
    framerate=30,
    flip_method=0,
):
    return (
        "nvarguscamerasrc sensor-id=%d !"
        "video/x-raw(memory:NVMM), width=(int)%d, height=(int)%d, framerate=(fraction)%d/1 ! "
        "nvvidconv flip-method=%d ! "
        "video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=(string)BGR ! appsink"
        % (
            sensor_id,
            capture_width,
            capture_height,
            framerate,
            flip_method,
            display_width,
            display_height,
        )
    )
print("Reading camera ")
print(gstreamer_pipeline(flip_method=0), cv2.CAP_GSTREAMER)
cap = cv2.VideoCapture(gstreamer_pipeline(flip_method=0), cv2.CAP_GSTREAMER)
# cap = cap.open()
print(cap.isOpened())
while(cap.isOpened()):
    print('Get frame')
    ret, frame = cap.read()
    print(frame,ret)
 
    cv2.imshow("Input via Gstreamer", frame)
     
cap.release()
cv2.destroyAllWindows()