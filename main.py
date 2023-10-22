from detector import Detector
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='yolov8m.pt', help='model.pt path(s)')
    parser.add_argument('--dataset', type=str, default='data/data.yaml', help='dataset.yaml path(s)')
    parser.add_argument('--epochs', type=int, default=100, help='number of traning epochs')
    parser.add_argument('--batch-size', type=int, default=16, help='batch size')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--train', action='store_true', help='whether is training or not')
    opt = parser.parse_args()
    print(opt)

    if opt.train:
        model = Detector.train(opt.weights, opt.dataset, opt.epochs, opt.batch_size, opt.device)
        print(Detector.get_metrics(model))
