
import cv2
import numpy as np
import glob
import os
from PIL import Image



def create_image_sequence(folder_path,x,y,i,transparency=0):
    
    files = os.listdir(folder_path)
    image_files = [file for file in files if file.lower().endswith(('.png', '.jpg', '.jpeg'))]

    if not image_files:
        print("No image files found in the folder.")
        return
    
    if i==0:
        #origin_x=x
        #origin_y=y
        new_position = (0,0)
        background_path = os.path.join(folder_path, image_files[0])
        background = Image.open(background_path)
        #print(origin_x)
        #print(origin_y)
        
    else:

        #print(origin_x)
        #print(origin_y)
        new_position = (150-x,72-y)
        background_path = f"output.png"
        background = Image.open(background_path)


    overlap_path = os.path.join(folder_path, image_files[i])
    overlap = Image.open(overlap_path)

    result = background.copy()    
    result.paste(overlap, new_position, overlap)

    if transparency < 1:
        result = Image.blend(background, result, transparency)

    output_path = f"output.png"
    result.save(output_path)

    print(f"Image sequence created: {output_path}")

if __name__ == "__main__":
    
    file = 'Red_dot_position.txt'
    if os.path.isfile(file):
        os.remove("Red_dot_position.txt")

    image_path = glob.glob("Source/*.jpg") 
    print(image_path)

    # Transparency level (0 to 1, where 0 is fully transparent and 1 is fully opaque)
    transparency = 1
    i=0

    for image in image_path:

        image = cv2.imread(image, cv2.IMREAD_UNCHANGED)
        

        # Convert the image from BGR to HSV color space
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # Define a range of red color in HSV
        lower_red = np.array([0, 100, 100])
        upper_red = np.array([10, 255, 255])

        # Create a mask to extract the red region
        mask = cv2.inRange(hsv_image, lower_red, upper_red)

        # Find contours in the mask
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Iterate through contours and find the position of the red dot
        for contour in contours:
            # You may add more conditions based on the contour properties to filter out noise
            area = cv2.contourArea(contour)
            if area > 0:  # Adjust the threshold based on your image
                # Get the centroid of the contour
                M = cv2.moments(contour)
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])

                # Draw a circle around the red dot
                cv2.circle(image, (cx, cy), 10, (0, 255, 0), -1)  # Adjust the circle properties

                # Print the position of the red dot
                print("Red dot position: ({}, {})".format(cx, cy))


        with open(file, 'a') as f:
            f.write("Red dot position: ({}, {})\n".format(cx, cy))

        create_image_sequence("Result",cx,cy,i,transparency)
        #overlap_images(background_path, overlay_path, output_path, position, transparency)     

        i=i+1
        # Display the result
        cv2.imshow("Result", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

