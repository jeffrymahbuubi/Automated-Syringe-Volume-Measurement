import numpy as np
import cv2

class UtilsTFLite:
    @staticmethod
    def convert_xywh_to_xyxy(box):
        x, y, w, h = box
        return list(map(int, [x, y, x + w, y + h]))

    @staticmethod
    def crop_image(image, boxes):
        cropped_images = []
        for box in boxes:
            x1, y1, x2, y2 = UtilsTFLite.convert_xywh_to_xyxy(box)
            cropped_images.append(image[y1:y2, x1:x2])
        return np.array(cropped_images)

    @staticmethod
    def load_line_model(predicted_class, models):
        if predicted_class in models:
            model = models[predicted_class]
            return model
        else:
            raise ValueError("Model not found for the predicted class")

    @staticmethod
    def plot_bounding_boxes(image, boxes):
        for box in boxes:
            # Convert xywh to xyxy
            x1, y1, x2, y2 = UtilsTFLite.convert_xywh_to_xyxy(box['box'])
            # Draw rectangle
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

    @staticmethod
    def center_of_left_border(box):
        """Calculate the center of the left border of a box."""
        x_min, y_min, x_max, y_max = box
        return x_min, (y_min + y_max) / 2

    @staticmethod
    def calculate_distance(center1, center2):
        """Calculate Euclidean distance between two points."""
        return np.sqrt((center1[0] - center2[0]) ** 2 + (center1[1] - center2[1]) ** 2)

    @staticmethod
    def remove_duplicate_boxes(boxes, threshold):
        unique_boxes = []
        centers = [UtilsTFLite.center_of_left_border(box) for box in boxes]

        for i, box in enumerate(boxes):
            is_duplicate = False
            for unique_box in unique_boxes:
                if UtilsTFLite.calculate_distance(centers[i], UtilsTFLite.center_of_left_border(unique_box)) < threshold:
                    is_duplicate = True
                    break
            if not is_duplicate:
                unique_boxes.append(box)

        return np.array(unique_boxes)

    @staticmethod
    def find_anomalies(distances, anomaly_threshold):
        """Find indices in the distances array where anomalies (distances significantly larger than the minimum) occur."""
        return np.where(distances >= anomaly_threshold)[0]

    @staticmethod
    def create_multiple_artificial_boxes(original_box, gap_dist, min_dist):
        """
        Create multiple artificial boxes based on the original box, gap distance, and minimum distance.

        Args:
            original_box (list): The original box [x_min, y_min, x_max, y_max].
            gap_dist (float): The gap distance where boxes are missing.
            min_dist (float): The minimum distance between boxes.

        Returns:
            list: A list of new artificial boxes.
        """
        boxes = []
        num_boxes = int(gap_dist // min_dist)

        x_min, y_min, x_max, y_max = original_box
        box_height = y_max - y_min

        for i in range(num_boxes):
            y_increment = (i + 1) * min_dist  # Only using min_dist for gap between boxes
            new_y_min = y_min + y_increment
            new_y_max = new_y_min + box_height

            new_box = [x_min, new_y_min, x_max, new_y_max]
            boxes.append(new_box)

        return boxes

    @staticmethod
    def check_for_missing_boxes_after_last(unique_boxes, rubber_y_min, min_dist):
        """Check if there are missing boxes after the last detected box."""
        if len(unique_boxes) == 0:
            return False
        last_box = unique_boxes[-1]
        _, last_box_y_max, _, _ = last_box
        distance_to_rubber = rubber_y_min - last_box_y_max
        return distance_to_rubber >= min_dist

    @staticmethod
    def generate_artificial_boxes_after_last(unique_boxes, rubber_y_min, min_dist):
        """Generate artificial boxes after the last detected box."""
        artificial_boxes = []
        if len(unique_boxes) > 0:
            last_box = unique_boxes[-1]
            _, last_box_y_max, _, _ = last_box
            current_y = last_box_y_max + min_dist
            while current_y + (last_box[3] - last_box[1]) <= rubber_y_min:
                new_box = last_box.copy()
                new_box[1] = current_y
                new_box[3] = current_y + (last_box[3] - last_box[1])
                artificial_boxes.append(new_box)
                current_y += (new_box[3] - new_box[1]) + min_dist
        return np.array(artificial_boxes)

    @staticmethod
    def calculate_iqr_threshold(distances):
        """Calculate the IQR-based anomaly threshold."""
        Q1 = np.percentile(distances, 25)
        Q3 = np.percentile(distances, 75)
        IQR = Q3 - Q1
        anomaly_threshold = Q3 + 1.5 * IQR

        # print("Q1: ", Q1)
        # print("Q3: ", Q3)
        # print("IQR: ", IQR)
        # print("Anomaly Threshold: ", anomaly_threshold)

        return anomaly_threshold, Q1
