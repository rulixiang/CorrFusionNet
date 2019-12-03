
import argparse

def argparser():
    parser = argparse.ArgumentParser()

    parser.add_argument('-g', '--gpu', help='gpu device id', default='1')
    parser.add_argument('-b', '--batch_size', help='batch size', type=int, default=32)
    parser.add_argument('-e', '--epoches', help='max epoches', type=int, default=100)
    parser.add_argument('-n', '--num_classes', help='num classes', type=int, default=14)

    parser.add_argument('-tb', '--use_tfboard', help='use tensorboard', type=bool, default=False)
    parser.add_argument('-sm', '--save_model', help='save best model', type=bool, default=False)
    parser.add_argument('-log', '--save_log', help='save training log', type=bool, default=False)

    parser.add_argument('-trn', '--trn_dir', help='training file dir', default='./data_small/trn/')
    parser.add_argument('-tst', '--tst_dir', help='testing file dir', default='./data_small/tst/')
    parser.add_argument('-val', '--val_dir', help='validation file dir', default='./data_small/val/')

    parser.add_argument('-lpath', '--log_path', help='log file path', default='./log/')
    parser.add_argument('-mpath', '--model_path', help='model file path', default='./model/')
    parser.add_argument('-tbpath', '--tb_path', help='tfboard file path', default='./tfboard/')
    parser.add_argument('-rpath', '--result_path', help='validation file path', default='./result/')
    
    args = parser.parse_args()
    return args
