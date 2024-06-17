class ConfigTFLite:
    def __init__(self):
        self.rotate_yaml = "./yaml/rotate-detection.yaml"
        self.syringe_yaml = "./yaml/syringe-detection.yaml"
        self.rubber_yaml = "./yaml/rubber-detection.yaml"
        self.line_yaml = "./yaml/line-detection.yaml"

        self.rotate_detection_model = './exported_models_tflite/rotate-detection-tflite/best_full_integer_quant.tflite'
        self.syringe_detection_model = './exported_models_tflite/syringe-detection-tflite/best_full_integer_quant.tflite'
        self.rubber_detection_model = './exported_models_tflite/rubber-detection-tflite/best_full_integer_quant.tflite'
        
        # Dictionary mapping predicted classes to model paths
        self.configuration = {
            "line-models": {
                0: './exported_models_tflite/line-detection-tflite/line-detection-tflite-1cc/best_full_integer_quant.tflite',
                1: './exported_models_tflite/line-detection-tflite/line-detection-tflite-3cc/best_full_integer_quant.tflite',
                2: './exported_models_tflite/line-detection-tflite/line-detection-tflite-10cc/best_full_integer_quant.tflite',
                3: './exported_models_tflite/line-detection-tflite/line-detection-tflite-30cc/best_full_integer_quant.tflite',
                4: './exported_models_tflite/line-detection-tflite/line-detection-tflite-50cc/best_full_integer_quant.tflite'
            },
            "threshold-distance": {
                0: 2,
                1: 10,
                2: 10,
                3: 10,
                4: 10,
            },
            "multiplier": {
                0: 0.01,
                1: 0.1,
                2: 0.2,
                3: 1,
                4: 1,
            },
        }
