# ------------------------------------------------------------------------------
# 0. Import Libraries
# ------------------------------------------------------------------------------
import numpy as np
import math
import time
import cv2
from TFLite_Preprocessor import Yolov8TFLite
from config_tflite import ConfigTFLite
from utils_tflite import UtilsTFLite
import argparse

# ------------------------------------------------------------------------------
# 1. Instantiate Class. + Global Variables
# ------------------------------------------------------------------------------
config_tflite = ConfigTFLite()

# Extract models yaml file
rotate_yaml = config_tflite.rotate_yaml
syringe_yaml = config_tflite.syringe_yaml
rubber_yaml = config_tflite.rubber_yaml
line_yaml = config_tflite.line_yaml

# Extract the trained models
rotation_model = config_tflite.rotate_detection_model
syringe_model = config_tflite.syringe_detection_model
rubber_model = config_tflite.rubber_detection_model
line_models = config_tflite.configuration["line-models"]
threshold_distance_config = config_tflite.configuration["threshold-distance"]
multiplier = config_tflite.configuration["multiplier"]

# Extract the utility functions
convert_xywh_to_xyxy = UtilsTFLite.convert_xywh_to_xyxy
crop_image = UtilsTFLite.crop_image
load_line_model = UtilsTFLite.load_line_model
plot_bounding_boxes = UtilsTFLite.plot_bounding_boxes
center_of_left_border = UtilsTFLite.center_of_left_border
calculate_distance = UtilsTFLite.calculate_distance
remove_duplicate_boxes = UtilsTFLite.remove_duplicate_boxes
find_anomalies = UtilsTFLite.find_anomalies
create_multiple_artificial_boxes = UtilsTFLite.create_multiple_artificial_boxes
check_for_missing_boxes_after_last = UtilsTFLite.check_for_missing_boxes_after_last
generate_artificial_boxes_after_last = UtilsTFLite.generate_artificial_boxes_after_last
calculate_iqr_threshold = UtilsTFLite.calculate_iqr_threshold

conf_thresh = 0.5
iou_thresh = 0.5
inference_times = []

def print_inference_times(inference_times):
    if len(inference_times) != 4:
        print("The inference_times list must contain exactly 4 values")
        return

    print(f"Rotate Detection Inference Time: {inference_times[0]} seconds")
    print(f"Syringe Detection Inference Time: {inference_times[1]} seconds")
    print(f"Rubber Detection Inference Time: {inference_times[2]} seconds")
    print(f"Line Detection Inference Time: {inference_times[3]} seconds")

# ------------------------------------------------------------------------------
# 2. Set Up Arguments Parser.
# ------------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--image", type=str, help="Path to the image")
parser.add_argument("--width", type=int, default=1920, help="Width of the window")
parser.add_argument("--height", type=int, default=1080, help="Height of the window")
args = parser.parse_args()

image = args.image
window_width = args.width
window_height = args.height

# ------------------------------------------------------------------------------
# 3. Execution Logic.
# ------------------------------------------------------------------------------
def main():
    ####################
    # Rotation Detection
    ####################
    
    # Create an instance of the Yolov8TFLite class with the specified arguments
    detection = Yolov8TFLite(rotation_model, image, conf_thresh, iou_thresh, rotate_yaml)

    # Perform inference
    start_inference_time = time.time()
    output = detection.main()
    end_inference_time = time.time()
    inference_times.append(end_inference_time - start_inference_time)

    _, output_prediction, input_image = output

    if len(output_prediction) == 2:
        down_box = output_prediction[0]['box']
        up_box = output_prediction[1]['box']

        # Convert YOLO coordinates to pixel coordinates
        up_x, up_y = up_box[0] + up_box[2] / 2, up_box[1] + up_box[3] / 2
        down_x, down_y = down_box[0] + down_box[2] / 2, down_box[1] + down_box[3] / 2

        # Calculate the angle of rotation using atan2 (in radians)
        theta_radians = math.atan2(down_x - up_x, down_y - up_y)   

        # Convert the angle to degrees
        theta_degrees = math.degrees(theta_radians)

        angle = theta_degrees

        # Calculate the rotation matrix
        (h, w) = input_image.shape[:2]
        center = (w // 2, h // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, -angle, 1.0)

        # Perform the rotation
        rotated_image = cv2.warpAffine(input_image, rotation_matrix, (w, h))
    
    ###################
    # Syringe Detection
    ###################

    # Create an instance of the Yolov8TFLite class with the specified arguments
    detection = Yolov8TFLite(syringe_model, rotated_image, conf_thresh, iou_thresh, syringe_yaml)
    
    # Perform inference
    start_inference_time = time.time()
    output = detection.main()
    end_inference_time = time.time()
    inference_times.append(end_inference_time - start_inference_time)


    _, output_prediction, input_image = output
    predicted_class = output_prediction[0]['class_id']
    # print(f"Predicted Syringe Class: {predicted_class}")

    # Ensure boxes are in the correct format (list of tuples/lists)
    boxes = output_prediction[0]['box']
    if isinstance(boxes[0], (int, float)):
        boxes = [boxes]  # Convert single box to a list of one box
    
    # Convert YOLO coordinates to pixel coordinates
    syringe_image = crop_image(rotated_image, boxes)

    ###################
    # Rubber Detection
    ###################
    
    # Create an instance of the Yolov8TFLite class with the specified arguments
    detection = Yolov8TFLite(rubber_model, syringe_image[0], conf_thresh, iou_thresh, rubber_yaml)

    # Perform inference
    start_inference_time = time.time()
    output = detection.main()
    end_inference_time = time.time()
    inference_times.append(end_inference_time - start_inference_time)


    _, output_prediction, input_image = output

    # Ensure boxes are in the correct format (list of tuples/lists)
    boxes = output_prediction[0]['box']
    boxes_xyxy = convert_xywh_to_xyxy(boxes)

    rubber_coordinates = boxes_xyxy

    y_min = rubber_coordinates[1]
    y_max = rubber_coordinates[3]

    rubber_image = input_image[:y_min, :]
    rubber_image_plotting = input_image[:y_max, :]

    ################
    # Line Detection 
    ################
    boxes = np.array([]).reshape(0, 4)  # For storing all detected boxes
    artificial_boxes = []  # For storing all artificial boxes
    centers = np.array([]).reshape(0, 2)  # For storing the centers of the boxes
    distances = np.array([])  # For storing distances between centers
    anomaly_indices = []  # For storing indices of detected anomalies

    model = load_line_model(predicted_class, line_models)

    # Create an instance of the Yolov8TFLite class with the specified arguments
    start_inference_time = time.time()
    detection = Yolov8TFLite(model, rubber_image, conf_thresh, iou_thresh, line_yaml)
    end_inference_time = time.time()
    inference_times.append(end_inference_time - start_inference_time)

    # Perform inference and get the output image
    output = detection.main()
    _, output_prediction, input_image = output

    # Initialize an empty list to collect all bounding boxes
    box_list = []


    # Extract bounding boxes from output_prediction and convert to xyxy format
    for prediction in output_prediction:
        box = prediction['box']
        xyxy_box = convert_xywh_to_xyxy(box)
        box_list.append(xyxy_box)

    # Convert the list of boxes to a NumPy array
    boxes = np.array(box_list)

    # Sort boxes by the y-coordinate (2nd element in each box)
    sorted_indices = boxes[:, 1].argsort()
    sorted_boxes = boxes[sorted_indices]

    ##########################################################################
    # Correction Detection: Artificial Box Insertion in Between Detected Boxes
    ##########################################################################

    # Remove overlapping detected boxes
    threshold_distance = threshold_distance_config[predicted_class]
    unique_boxes = remove_duplicate_boxes(sorted_boxes, threshold_distance)

    # Find anomalies in the detected boxes
    centers = np.array([center_of_left_border(box) for box in unique_boxes])

    distances = np.array([
        calculate_distance(centers[i], centers[i + 1])
        for i in range(len(centers) - 1)
    ])

    # Calculate the IQR-based anomaly threshold
    anomaly_threshold, Q1 = calculate_iqr_threshold(distances)

    anomaly_indices = find_anomalies(distances, anomaly_threshold)

    # Initialize a counter for the number of inserted artificial boxes
    inserted_artificial_boxes = 0

    for index in anomaly_indices:
        # gap_dist = int(distances[index] - minimum_distance)
        gap_dist = distances[index] - Q1

        # if gap_dist >= minimum_distance:
        if gap_dist >= Q1:
            adjusted_index = index + inserted_artificial_boxes
            
            original_box = unique_boxes[adjusted_index]
            new_artificial_boxes = create_multiple_artificial_boxes(
                original_box, gap_dist, Q1
            )

            for new_box in new_artificial_boxes:
                # Adjust the index to account for previously inserted artificial boxes
                adjusted_insert_index = adjusted_index + 1
                unique_boxes = np.insert(
                    unique_boxes, adjusted_insert_index, new_box, axis=0
                )
                artificial_boxes.append(new_box)  # Add the artificial box to the list
                inserted_artificial_boxes += 1  # Increment the counter for inserted boxes
                adjusted_index += 1  # Update the adjusted index to insert subsequent boxes correctly

    #################################################################
    # Correction Detection: Artificial Box Insertion in Before Rubber
    #################################################################
    rubber_y_min = rubber_coordinates[1]

    missing_boxes_detected = check_for_missing_boxes_after_last(unique_boxes, rubber_y_min, Q1)

    if missing_boxes_detected:
        new_artificial_boxes = generate_artificial_boxes_after_last(unique_boxes, rubber_y_min, Q1)
        if len(new_artificial_boxes) > 0:
            unique_boxes = np.concatenate((unique_boxes, new_artificial_boxes), axis=0)

    # Plot boxes on the image
    for i, box in enumerate(unique_boxes):
        x1, y1, x2, y2 = box

        if predicted_class == 0:
            # Calculate the center and radius for the cricle
            center_y = (y_min + y_max) // 2
            radius = 1

            # Draw the circle
            cv2.circle(rubber_image_plotting, (x1, center_y), radius, (0, 255, 0), -1)
        else:
            # Draw the bounding box
            cv2.rectangle(rubber_image_plotting, (x1, y1), (x2, y2), (0, 255, 0), 2)
    
    # Text Settings
    text = f"Lines Detected: {len(unique_boxes)}"
    height, width = rubber_image_plotting.shape[:2]
    aspect_ratio = width / height

    # Font settings
    font_scale = 2 * aspect_ratio
    position = (10, int(50 * font_scale))
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_color = (255, 255, 255)
    line_type = 2

    cv2.putText(rubber_image_plotting, text, position, font, font_scale, font_color, line_type)

    # Text settings for inference times
    print("Inference Times:")
    print_inference_times(inference_times)
    text_time = f"Inference Times: {', '.join(f'{t:.2f} sec' for t in inference_times)}"
    position_time = (10, int(100 * font_scale))  # Adjust position based on font scale
    cv2.putText(
        rubber_image_plotting,
        text_time,
        position_time,
        font,
        font_scale,
        font_color,
        line_type,
    )

    # Resize the window
    cv2.namedWindow("Syringe Detection", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Syringe Detection", window_width, window_height)

    # Display the image using OpenCV
    cv2.imshow("Syringe Detection", rubber_image_plotting)

    # Wait for a key press to close the window
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()