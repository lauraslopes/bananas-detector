from detector import Detector
import cv2
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='yolov8m.pt', help='model.pt path(s)')
    parser.add_argument('--dataset', type=str, default='data/data.yaml', help='dataset.yaml path(s)')
    parser.add_argument('--epochs', type=int, default=100, help='number of traning epochs')
    parser.add_argument('--batch-size', type=int, default=16, help='batch size')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--train', action='store_true', help='whether is training or not')
    parser.add_argument('--show-results', action='store_true', help='show results images with predictions')
    parser.add_argument('--source', type=str, default='', help='source for inference (path to an image or directory of images)')
    parser.add_argument('--conf', type=float, default=0.25, help='object confidence threshold for detection')
    opt = parser.parse_args()
    print(opt)

    if opt.train:
        model = Detector.train(opt.weights, opt.dataset, opt.epochs, opt.batch_size, opt.device)
        print(Detector.get_metrics(model))
    else:
        results = Detector.inference(opt.weights, opt.source, opt.conf, opt.device)

        if opt.show_results:
            # Show the results
            for i, r in enumerate(results):
                im_array = r.plot()  # plot a BGR numpy array of predictions
                cv2.imshow('Result', im_array)
                cv2.waitKey(0) 
