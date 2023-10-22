from ultralytics import YOLO

class Detector:

    def __init__():
        pass

    def train(weights, dataset, epochs, batch_size, device):

        model = YOLO(weights)
        model.train(data=dataset, epochs=epochs, batch=batch_size, device=device)

        return model

    def inference():
        pass

    def get_metrics(model):
        # evaluate model performance on the validation set
        return model.val()
