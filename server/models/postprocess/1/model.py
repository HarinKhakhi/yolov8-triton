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

        # parsing model config
        self.model_config = model_config = json.loads(args['model_config'])

        # loading output field data types
        OUTPUT_FIELD_NAMES = ["detection_boxes", "detection_scores", "detection_classes"]
        self.output_dtypes = {}
        
        for OUTPUT_FILED_NAME in OUTPUT_FIELD_NAMES:
            field_config = pb_utils.get_output_config_by_name(model_config, OUTPUT_FILED_NAME)
            self.output_dtypes[OUTPUT_FILED_NAME] = pb_utils.triton_string_to_numpy(field_config['data_type'])

        # parameters
        self.score_threshold = 0.25
        self.nms_threshold = 0.45
        self.profile = True

    def handle_request(self, request):
        # get INPUT
        in_0 = pb_utils.get_input_tensor_by_name(request, "INPUT_0")

        # get OUTPUT
        outputs = in_0.as_numpy()
        outputs = np.array([cv2.transpose(outputs[0])])
        rows = outputs.shape[1]

        # filtering out the boxes with lower confidence
        # and setup box format as per cv2.dnn.NMSBoxes requirement
        boxes = []
        scores = []
        class_ids = []
        for i in range(rows):
            classes_scores = outputs[0][i][4:]
            (_, maxScore, _, (_, maxClassIndex)) = cv2.minMaxLoc(classes_scores)

            if maxScore >= self.score_threshold:
                box = [
                    outputs[0][i][0] - (0.5 * outputs[0][i][2]), 
                    outputs[0][i][1] - (0.5 * outputs[0][i][3]), 
                    outputs[0][i][2], 
                    outputs[0][i][3]
                ]

                boxes.append(box)
                scores.append(maxScore)
                class_ids.append(maxClassIndex)
                
        # NMS to reduce the number of boxes
        result_boxes = cv2.dnn.NMSBoxes(
            boxes, scores,
            self.score_threshold, self.nms_threshold,
            0.5
        )
        
        # preparing output
        output_boxes = []
        output_scores = []
        output_classids = []

        for i in range(len(result_boxes)):
            index = result_boxes[i]

            output_boxes.append(boxes[index])
            output_scores.append(scores[index])
            output_classids.append(class_ids[index])

        # formatting the output with correct types
        detection_boxes = pb_utils.Tensor(
            "detection_boxes", np.array(output_boxes, dtype=self.output_dtypes["detection_boxes"]))

        detection_scores = pb_utils.Tensor(
            "detection_scores", np.array(output_scores, dtype=self.output_dtypes["detection_scores"]))
        
        detection_classes = pb_utils.Tensor(
            "detection_classes", np.array(output_classids, dtype=self.output_dtypes["detection_classes"]))
        
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
