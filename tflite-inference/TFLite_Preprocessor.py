from pathlib import Path
import numpy as np
import yaml
import copy
import cv2
import re

from tflite_runtime import interpreter as tflite

# Global variables
img_width = 640
img_height = 640


def yaml_load(file="data.yaml", append_filename=False):
    """
    Load YAML data from a file.

    Args:
        file (str, optional): File name. Default is 'data.yaml'.
        append_filename (bool): Add the YAML filename to the YAML dictionary. Default is False.

    Returns:
        (dict): YAML data and file name.
    """
    assert Path(file).suffix in {".yaml", ".yml"}, f"Attempting to load non-YAML file {file} with yaml_load()"
    with open(file, errors="ignore", encoding="utf-8") as f:
        s = f.read()  # string

        # Remove special characters
        if not s.isprintable():
            s = re.sub(r"[^\x09\x0A\x0D\x20-\x7E\x85\xA0-\uD7FF\uE000-\uFFFD\U00010000-\U0010ffff]+", "", s)

        # Add YAML filename to dict and return
        data = yaml.safe_load(s) or {}  # always return a dict (yaml.safe_load() may return None for empty files)
        if append_filename:
            data["yaml_file"] = str(file)
        return data
    

class LetterBox:
    def __init__(
        self, new_shape=(img_width, img_height), auto=False, scaleFill=False, scaleup=True, center=True, stride=32
    ):
        self.new_shape = new_shape
        self.auto = auto
        self.scaleFill = scaleFill
        self.scaleup = scaleup
        self.stride = stride
        self.center = center  # Put the image in the middle or top-left

    def __call__(self, labels=None, image=None):
        """Return updated labels and image with added border."""

        if labels is None:
            labels = {}
        img = labels.get("img") if image is None else image
        shape = img.shape[:2]  # current shape [height, width]
        new_shape = labels.pop("rect_shape", self.new_shape)
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)

        # Scale ratio (new / old)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        if not self.scaleup:  # only scale down, do not scale up (for better val mAP)
            r = min(r, 1.0)

        # Compute padding
        ratio = r, r  # width, height ratios
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
        if self.auto:  # minimum rectangle
            dw, dh = np.mod(dw, self.stride), np.mod(dh, self.stride)  # wh padding
        elif self.scaleFill:  # stretch
            dw, dh = 0.0, 0.0
            new_unpad = (new_shape[1], new_shape[0])
            ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

        if self.center:
            dw /= 2  # divide padding into 2 sides
            dh /= 2

        if shape[::-1] != new_unpad:  # resize
            img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)) if self.center else 0, int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)) if self.center else 0, int(round(dw + 0.1))
        img = cv2.copyMakeBorder(
            img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114)
        )  # add border
        if labels.get("ratio_pad"):
            labels["ratio_pad"] = (labels["ratio_pad"], (left, top))  # for evaluation

        if len(labels):
            labels = self._update_labels(labels, ratio, dw, dh)
            labels["img"] = img
            labels["resized_shape"] = new_shape
            return labels
        else:
            return img

    def _update_labels(self, labels, ratio, padw, padh):
        """Update labels."""

        labels["instances"].convert_bbox(format="xyxy")
        labels["instances"].denormalize(*labels["img"].shape[:2][::-1])
        labels["instances"].scale(*ratio)
        labels["instances"].add_padding(padw, padh)
        return labels
    
class Yolov8TFLite:
    def __init__(self, tflite_model, input_image, confidence_thres, iou_thres, yaml_file):
        """
        Initializes an instance of the Yolov8TFLite class.

        Args:
            tflite_model: Path to the TFLite model.
            input_image: Path to the input image.
            confidence_thres: Confidence threshold for filtering detections.
            iou_thres: IoU (Intersection over Union) threshold for non-maximum suppression.
        """

        self.tflite_model = tflite_model
        self.input_image = input_image
        self.confidence_thres = confidence_thres
        self.iou_thres = iou_thres
        self.yaml_file = yaml_file

        # Load the class names from the Custom dataset
        self.classes = yaml_load((yaml_file))["names"]

        # Generate a color palette for the classes
        self.color_palette = np.random.uniform(0, 255, size=(len(self.classes), 3))

    def draw_detections(self, img, box, score, class_id):
        """
        Draws bounding boxes and labels on the input image based on the detected objects.

        Args:
            img: The input image to draw detections on.
            box: Detected bounding box.
            score: Corresponding detection score.
            class_id: Class ID for the detected object.

        Returns:
            None
        """

        # Extract the coordinates of the bounding box
        x1, y1, w, h = box

        # Retrieve the color for the class ID
        color = self.color_palette[class_id]

        # Draw the bounding box on the image
        cv2.rectangle(img, (int(x1), int(y1)), (int(x1 + w), int(y1 + h)), color, 2)

        # Create the label text with class name and score
        label = f"{self.classes[class_id]}: {score:.2f}"

        # Calculate the dimensions of the label text
        (label_width, label_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

        # Calculate the position of the label text
        label_x = x1
        label_y = y1 - 10 if y1 - 10 > label_height else y1 + 10

        # Draw a filled rectangle as the background for the label text
        cv2.rectangle(
            img,
            (int(label_x), int(label_y - label_height)),
            (int(label_x + label_width), int(label_y + label_height)),
            color,
            cv2.FILLED,
        )

        # Draw the label text on the image
        cv2.putText(img, label, (int(label_x), int(label_y)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

    def preprocess(self):
        """
        Preprocesses the input image before performing inference.

        Returns:
            image_data: Preprocessed image data ready for inference.
        """
         # Check if input is a path or an image matrix
        if isinstance(self.input_image, str):
            # Read the input image using OpenCV
            self.img = cv2.imread(self.input_image)
        else:
            # Use the provided image matrix directly
            self.img = self.input_image

        # print("image before", self.img)
        # print("image before shape", self.img.shape)
        # Get the height and width of the input image
        self.img_height, self.img_width = self.img.shape[:2]

        letterbox = LetterBox(new_shape=[img_width, img_height], auto=False, stride=32)
        image = letterbox(image=self.img)
        image = [image]
        image = np.stack(image)
        image = image[..., ::-1].transpose((0, 3, 1, 2))
        img = np.ascontiguousarray(image)
        # print("image after", img)

        # n, h, w, c
        image = img.astype(np.float32)
        return image / 255

    def postprocess(self, input_image, output):
        """
        Performs post-processing on the model's output to extract bounding boxes, scores, and class IDs.

        Args:
            input_image (numpy.ndarray): The input image.
            output (numpy.ndarray): The output of the model.

        Returns:
            numpy.ndarray: The input image with detections drawn on it.
        """

        postprocess_image = copy.deepcopy(input_image)

        boxes = []
        scores = []
        class_ids = []
        detections = []
        for pred in output:
            pred = np.transpose(pred)
            for box in pred:
                x, y, w, h = box[:4]
                x1 = x - w / 2
                y1 = y - h / 2
                boxes.append([x1, y1, w, h])
                idx = np.argmax(box[4:])
                scores.append(box[idx + 4])
                class_ids.append(idx)

        indices = cv2.dnn.NMSBoxes(boxes, scores, self.confidence_thres, self.iou_thres)

        for i in indices:
            # Get the box, score, and class ID corresponding to the index
            box = boxes[i]
            gain = min(img_width / self.img_width, img_height / self.img_height)
            pad = (
                round((img_width - self.img_width * gain) / 2 - 0.1),
                round((img_height - self.img_height * gain) / 2 - 0.1),
            )
            box[0] = (box[0] - pad[0]) / gain
            box[1] = (box[1] - pad[1]) / gain
            box[2] = box[2] / gain
            box[3] = box[3] / gain
            score = scores[i]
            class_id = class_ids[i]
            if score > 0.25:
                # print("After Non-Max Suppression: ")
                # print(box, score, class_id)
                # Draw the detection on the input image
                self.draw_detections(postprocess_image, box, score, class_id)
                detections.append({
                    "box": box,
                    "score": score, 
                    "class_id": class_id
                })

        return postprocess_image, detections, input_image

    def main(self):
        """
        Performs inference using a TFLite model and returns the output image with drawn detections.

        Returns:
            output_img: The output image with drawn detections.
        """

        # Create an interpreter for the TFLite model
        interpreter = tflite.Interpreter(model_path=self.tflite_model)
        self.model = interpreter
        interpreter.allocate_tensors()

        # Get the model inputs
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        # Store the shape of the input for later use
        input_shape = input_details[0]["shape"]
        self.input_width = input_shape[1]
        self.input_height = input_shape[2]

        # Preprocess the image data
        img_data = self.preprocess()
        img_data = img_data
        # img_data = img_data.cpu().numpy()
        # Set the input tensor to the interpreter
        # print(input_details[0]["index"])
        # print(img_data.shape)
        img_data = img_data.transpose((0, 2, 3, 1))

        scale, zero_point = input_details[0]["quantization"]
        img_data_int8 = (img_data / scale + zero_point).astype(np.int8)
        interpreter.set_tensor(input_details[0]["index"], img_data_int8)

        # Run inference
        interpreter.invoke()

        # Get the output tensor from the interpreter
        output = interpreter.get_tensor(output_details[0]["index"])
        scale, zero_point = output_details[0]["quantization"]
        output = (output.astype(np.float32) - zero_point) * scale

        output[:, [0, 2]] *= img_width
        output[:, [1, 3]] *= img_height

        # Perform post-processing on the outputs to obtain output image.
        return self.postprocess(self.img, output)