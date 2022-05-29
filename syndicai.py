import torch
from yolov5 import YOLOv5


class PythonPredictor ():
    def __init__(self):
        """
        Initalization of the handler.
        """
        super(PythonPredictor, self).__init__()
        self.model = self.load_model()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def load_model(self):
        """
        Function loads the yolo5 model .
        """
        model_path = "best.pt"
        device = self.device # or "cpu"

        # init model model
        model = YOLOv5(model_path, device)

        return model

    def Predict(self, requests):
        """
        Perform inference on the given input data.
        """
        image = requests.get("data")
        if image is None:
            image = requests.get("body")
        output = self.model.predict(image)
        return output


#%%
