import numpy as np
import json
import triton_python_backend_utils as pb_utils # type: ignore
import cv2
import cProfile
import pstats
import io
import time


class TritonPythonModel:

    def initialize(self, args):
        """`initialize` is called only once when the model is being loaded.
        Implementing `initialize` function is optional. This function allows
        the model to intialize any state associated with this model.
        Parameters
        ----------
        args : dict
          Both keys and values are strings. The dictionary keys and values are:
          * model_config: A JSON string containing the model configuration
          * model_instance_kind: A string containing model instance kind
          * model_instance_device_id: A string containing model instance device ID
          * model_repository: Model repository path
          * model_version: Model version
          * model_name: Model name
        """

        # You must parse model_config. JSON string is not parsed here
        self.model_config = model_config = json.loads(args['model_config'])

        # Get OUTPUT0 configuration
        detection_boxes_config = pb_utils.get_output_config_by_name(
            model_config, "detection_boxes")

        detection_scores_config = pb_utils.get_output_config_by_name(
            model_config, "detection_scores")

        detection_classes_config = pb_utils.get_output_config_by_name(
            model_config, "detection_classes")

        # Convert Triton types to numpy types
        self.detection_boxes_dtype = pb_utils.triton_string_to_numpy(
            detection_boxes_config['data_type'])

        self.detection_scores_dtype = pb_utils.triton_string_to_numpy(
            detection_scores_config['data_type'])

        self.detection_classes_dtype = pb_utils.triton_string_to_numpy(
            detection_classes_config['data_type'])

        self.score_threshold = 0.25
        self.nms_threshold = 0.45
        self.profile = True

    def calculate_score(self, box1, box2):
        """
        Calculate the score two bounding boxes.
        currently, score = intersection

        Parameters:
        box1, box2 : tuple or list
            Bounding boxes in the format (x, y, w, h).
            x, y: top-left corner coordinates.
            w: width of the bounding box.
            h: height of the bounding box.

        Returns:
        score : float
           0 being the lowest score and 1 being the highest.`
        """

        # Extract the coordinates and dimensions
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2

        # Calculate the coordinates of the intersection rectangle
        x_intersection = max(x1, x2)
        y_intersection = max(y1, y2)
        w_intersection = min(x1 + w1, x2 + w2) - x_intersection
        h_intersection = min(y1 + h1, y2 + h2) - y_intersection

        # Ensure the width and height of the intersection are positive
        if w_intersection > 0 and h_intersection > 0:
            intersection_area = w_intersection * h_intersection
        else:
            intersection_area = 0

        return intersection_area


    def handle_request(self, request):
        # Get INPUT0
        in_0 = pb_utils.get_input_tensor_by_name(request, "INPUT_0")

        # Get the output arrays from the results
        outputs = in_0.as_numpy()

        outputs = np.array([cv2.transpose(outputs[0])])
        rows = outputs.shape[1]

        # filtering out the boxes with lower confidence
        boxes = []
        scores = []
        class_ids = []
        for i in range(rows):
            classes_scores = outputs[0][i][4:]
            (minScore, maxScore, minClassLoc, (x, maxClassIndex)
                ) = cv2.minMaxLoc(classes_scores)
            if maxScore >= self.score_threshold:
                box = [outputs[0][i][0] -
                        (0.5 *
                        outputs[0][i][2]), outputs[0][i][1] -
                        (0.5 *
                        outputs[0][i][3]), outputs[0][i][2], outputs[0][i][3]]
                boxes.append(box)
                scores.append(maxScore)
                class_ids.append(maxClassIndex)
                
        # NMS to reduce the number of boxes
        nms_boxes = cv2.dnn.NMSBoxes(boxes, scores,
                                        self.score_threshold,
                                        self.nms_threshold,
                                        0.5)
        
        filtered_boxes = []
        # ============================= MANUAL FILTERING =============================
        # parameters
        class1_id = 37 # gun
        class2_id = 0 # person
        threshold_score = 0.0

        # filter out required boxes
        class1_boxes = [box_ind for box_ind in nms_boxes if class_ids[box_ind] == class1_id]
        class2_boxes = [box_ind for box_ind in nms_boxes if class_ids[box_ind] == class2_id]

        # brute-force loop to check for max score pair
        filtered_boxes = set()
        for box1_ind in class1_boxes:
            box1 = boxes[box1_ind]
            max_score = 0.0
            max_score_box2 = -1

            for box2_ind in class2_boxes:
                box2 = boxes[box2_ind]
                
                score = self.calculate_score(box1, box2)
                if score >= max_score:
                    max_score = score
                    max_score_box2 = box2_ind 

            # only add box1 and box2 pair if the score is higher than threshold
            if max_score >= threshold_score:
                filtered_boxes.add(box1_ind)
                filtered_boxes.add(max_score_box2)
        
        filtered_boxes = list(filtered_boxes)
        # ============================================================================
        
        # preparing output
        output_boxes = []
        output_scores = []
        output_classids = []
        output_filtered_boxes = []

        for i in range(len(filtered_boxes)):
            index = filtered_boxes[i]

            output_boxes.append(boxes[index])
            output_scores.append(scores[index])
            output_classids.append(class_ids[index])

        # formatting the output with correct types
        detection_boxes = pb_utils.Tensor(
            "detection_boxes", np.array(output_boxes).astype(self.detection_boxes_dtype))

        detection_scores = pb_utils.Tensor(
            "detection_scores", np.array(output_scores).astype(self.detection_scores_dtype))
        
        detection_classes = pb_utils.Tensor(
            "detection_classes", np.array(output_classids).astype(self.detection_classes_dtype))
        
        # setting the output
        inference_response = pb_utils.InferenceResponse(
            output_tensors=[
                detection_boxes,
                detection_scores,
                detection_classes,
            ])

        return inference_response

    def execute(self, requests):
        """`execute` MUST be implemented in every Python model. `execute`
        function receives a list of pb_utils.InferenceRequest as the only
        argument. This function is called when an inference request is made
        for this model. Depending on the batching configuration (e.g. Dynamic
        Batching) used, `requests` may contain multiple requests. Every
        Python model, must create one pb_utils.InferenceResponse for every
        pb_utils.InferenceRequest in `requests`. If there is an error, you can
        set the error argument when creating a pb_utils.InferenceResponse
        Parameters
        ----------
        requests : list
          A list of pb_utils.InferenceRequest
        Returns
        -------
        list
          A list of pb_utils.InferenceResponse. The length of this list must
          be the same as `requests`
        """

        logger = pb_utils.Logger

        responses = []

        # Every Python backend must iterate over everyone of the requests
        # and create a pb_utils.InferenceResponse for each of them.
        for request in requests:
            # Start profiling
            profiler = cProfile.Profile()
            profiler.enable()

            start_time = time.perf_counter()

            # Call your function and capture the return value
            result = self.handle_request(request)

            end_time = time.perf_counter()

            # Stop profiling
            profiler.disable()

            # Create a StringIO buffer to capture the profiling output
            s = io.StringIO()
            ps = pstats.Stats(profiler, stream=s).sort_stats('cumulative')
            ps.print_stats()

            # Get the profiling output as a string
            profile_output = s.getvalue()
            if self.profile:
                logger.log_info(profile_output)
                logger.log_info(f'Total Time: {end_time - start_time}')

            responses.append(result)

        return responses

    def finalize(self):
        """`finalize` is called only once when the model is being unloaded.
        Implementing `finalize` function is OPTIONAL. This function allows
        the model to perform any necessary clean ups before exit.
        """
        pass
