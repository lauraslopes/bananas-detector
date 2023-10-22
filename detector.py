from ultralytics import YOLO

class Detector:

    def __init__():
        pass

    def train(weights, dataset, epochs, batch_size, device):

        model = YOLO(weights)
        model.train(data=dataset, epochs=epochs, batch=batch_size, device=device, degrees=45, flipud=1, mixup=1)

        return model

    def inference(weights, source, conf, device):
        model = YOLO(weights)
        # predict on an image
        return model.predict(source, conf=conf, device=device, save=True, save_txt=True, save_conf=True)

    def get_metrics(model):
        # evaluate model performance on the validation set
        return model.val()

