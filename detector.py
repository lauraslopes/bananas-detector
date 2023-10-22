from ultralytics import YOLO


class Detector:
    def __init__(self, weights, device):
        self.model = YOLO(weights)
        self.device = device

    def train(self, dataset, epochs, batch_size):
        self.model.train(
            data=dataset,
            epochs=epochs,
            batch=batch_size,
            device=self.device,
            degrees=45,
            flipud=1,
            mixup=1,
        )

    def inference(self, source, conf):
        # predict on an image
        return self.model.predict(
            source,
            conf=conf,
            device=self.device,
            save=True,
            save_txt=True,
            save_conf=True,
        )

    def get_metrics(self):
        # evaluate model performance on the validation set
        return self.model.val()
