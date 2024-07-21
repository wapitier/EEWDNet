import argparse
from work_4.Net.Net_main import EEWDNet
from ptflops import get_model_complexity_info



if __name__ == '__main__':
    parser = argparse.ArgumentParser('Print Info')
    parser.add_argument('--train_mode', default=False)
    args = parser.parse_args()

    model = EEWDNet(args)

    # 计算并打印模型的 FLOPs 和参数数量
    flops, params = get_model_complexity_info(model, (3, 224, 224), as_strings=True, print_per_layer_stat=True, cal_type=1)
    print()
    print(f'GFLOPs: {flops}')
    print(f'Params: {params}')


    