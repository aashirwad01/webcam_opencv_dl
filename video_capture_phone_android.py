# import the opencv library 
import cv2 


# define a video capture object
url = "http://192.168.43.6:8080" 
vid = cv2.VideoCapture(url+"/video")

while(True): 
	
	# Capture the video frame 
	# by frame 
	ret, frame = vid.read() 
	 # resize = cv2.resize(frame, (176, 144)) 
	# Display the resulting frame 
	cv2.imshow('frame', frame) 
	
	# the 'q' button is set as the 
	# quitting button you may use any 
	# desired button of your choice 
	if cv2.waitKey(1) & 0xFF == ord('q'): 
		break

# After the loop release the cap object 
vid.release() 
# Destroy all the windows 
cv2.destroyAllWindows() 

