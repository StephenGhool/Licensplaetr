import torch
import cv2
import os
import torch.nn.functional as F
import io
from PIL import Image
import base64
from ts.torch_handler.base_handler import BaseHandler
from torchvision import transforms
from yolov5 import YOLOv5

# supposed to help spead up inference time when using cpu
torch.set_num_threads(1)


# class ModelHandler(BaseHandler):
#     """
#     A custom model handler implementation.
#     """
#
#     def __init__(self):
#         self.manifest = None
#         self._context = None
#         self.initialized = False
#         self.model = None
#         self.device = None
#
#     def initialize(self, context):
#         """
#         Invoke by torchserve for loading a model
#         :param context: context contains model server system properties
#         :return:
#         """
#
#         #  load the model
#         self.manifest = context.manifest
#
#         properties = context.system_properties
#         model_dir = properties.get("model_dir")
#         self.device = torch.device("cuda:" + str(properties.get("gpu_id")) if torch.cuda.is_available() else "cpu")
#
#         # Read model serialize/pt file
#         serialized_file = self.manifest['model']['serializedFile']
#         model_pt_path = os.path.join(model_dir, serialized_file)
#         if not os.path.isfile(model_pt_path):
#             raise RuntimeError("Missing the model.pt file")
#
#         self.model = torch.jit.load(model_pt_path)
#
#         self.initialized = True
#
#     def handle(self, data, context):
#         """
#         Invoke by TorchServe for prediction request.
#         Do pre-processing of data, prediction using model and postprocessing of prediciton output
#         :param data: Input data for prediction
#         :param context: Initial context contains model server system properties.
#         :return: prediction output
#         """
#         pred_out = self.model.forward(data)
#         return pred_out
#

class MyHandler(BaseHandler):
    """
    Custom handler for pytorch serve. This handler supports batch requests.
    For a deep description of all method check out the doc:
    https://pytorch.org/serve/custom_service.html
    """

    # def __init__(self, *args, **kwargs):
    #     super().__init__()
    #     self.device = "cpu"
    #     self.path = "best.pt"
    #     self.model = self.load_model()

    def initialize(self, context):
        super().__init__()
        self.device = "cpu"
        self.path = "best.pt"
        self.model = self.load_model()

    def preprocess_one_image(self, req):
        """
        Process one single image.
        """
        # get image from the request
        req = req[0]
        image = req.get("data")
        if image is None:
            image = req.get("body")

        image = Image.open(io.BytesIO(image))
        return image

    def load_model(self):
        # init yolov5 model
        model = YOLOv5(self.path, self.device)
        return model

    def inference(self, x):
        """
        Given the data from .preprocess, perform inference using the model.
        We return the predicted label for each image.
        """
        # x = cv2.imread(x)
        print("This is input to infer", x)
        outs = self.model.predict(x)
        print("This is output", outs)
        predictions = outs.pred[0]
        boxes = predictions[:, :4]  # x1, y1, x2, y2
        scores = predictions[:, 4]
        categories = predictions[:, 5]
        return [{"predictions": predictions.numpy().tolist()}]


_service = MyHandler()


def handle(data, context):
    print("This is context", context)
    print("This is context sys properties", context.system_properties)
    if not _service.initialized:
        _service.initialize(context)

    if data is None:
        return None
    print("This is data type", type(data))
    data = _service.preprocess_one_image(data)
    data = _service.inference(data)
    return data

# class ObjectDetectionHandler(BaseHandler):
#     """
#     Accepts two types for requests:
#         Base64 String
#         Image Blob(ByteStream)
#     """
#
#     def __init__(self, *args, **kwargs):
#         super().__init__()
#         self.transform = transforms.Compose(
#             [
#                 transforms.Resize((640, 640)),
#                 transforms.ToTensor(),
#             ]
#         )
#
#     def preprocess_one_image(self, req):
#         """
#         Process one single image.
#         """
#
#         # get image from the request
#         image = req.get("data")
#
#         if image is None:
#             image = req.get("body")
#
#         # create a stream from the encoded image
#         if isinstance(image, str):
#             image = base64.b64decode(image)
#
#         byte_img = io.BytesIO(image)
#         byte_img.seek(0)
#         image = Image.open(byte_img)
#
#         image_size = image.size
#         image = self.transform(image)
#
#         # add batch dim
#         image = image.unsqueeze(0)
#
#         return [image, image_size]
#
#     def preprocess(self, requests):
#
#         images = []
#         image_sizes = []
#         for req in requests:
#             image, image_size = self.preprocess_one_image(req)
#             images.append(image)
#             image_sizes.append(image_size)
#
#         images = torch.cat(images)
#         if len(image_sizes) == 1:
#             image_sizes = image_sizes[0]
#
#         return {"images": images, "sizes": image_sizes}
#
#     def inference(self, inp):
#
#         x = inp["images"]
#         x_size = inp["sizes"]
#
#         preds = self.model.forward(x)
#
#         return preds
#
#     def postprocess(self, preds):
#         return [{"predictions": preds}]
#
#
# _service = ObjectDetectionHandler()
#
#
# def handle(data, context):
#     if not _service.initialized:
#         _service.initialize(context)
#
#     if data is None:
#         return None
#
#     data = _service.preprocess(data)
#     data = _service.inference(data)
#     data = _service.postprocess(data)
#
#     return data

# %%
