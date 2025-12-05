import cv2
import numpy as np

# function to delete green spots
def delete_green_spot(img_path, mask_path):
    img_original = cv2.imread(img_path)

    hsv = cv2.cvtColor(img_original, cv2.COLOR_BGR2HSV)
    lower_green = np.array([35, 50, 50])
    upper_green = np.array([90, 255, 255])
    mask_with_holes = cv2.inRange(hsv, lower_green, upper_green)

    contours, _ = cv2.findContours(mask_with_holes, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    mask_solid = np.zeros_like(mask_with_holes)

    cv2.drawContours(mask_solid, contours, -1, (255), thickness=cv2.FILLED)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask_solid_final = cv2.dilate(mask_solid, kernel, iterations=1)

    final_result  = cv2.inpaint(img_original, mask_solid_final, 3, cv2.INPAINT_TELEA)

    mask = cv2.imread(mask_path, 0)
    mask[mask_solid_final == 255] = 0
    
    return final_result, mask