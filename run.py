import gc
import logging
import os

# 设置 hugging face 代理路径
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from config import get_config_regression
from data_loader import MMDataLoader
from trains import ATIO
from utils import assign_gpu, setup_seed
from trains.singleTask.model import dmd
from trains.singleTask.distillnets import get_distillation_kernel, get_distillation_kernel_homo
from trains.singleTask.misc import softmax
import sys
# 确保 GPU 编号在不同运行环境中一致
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:2"
logger = logging.getLogger('MMSA')

def _set_logger(log_dir, model_name, dataset_name, verbose_level):

    # base logger
    log_file_path = Path(log_dir) / f"{model_name}-{dataset_name}.log"
    logger = logging.getLogger('MMSA')
    logger.setLevel(logging.DEBUG)

    # file handler
    fh = logging.FileHandler(log_file_path)
    fh_formatter = logging.Formatter('%(asctime)s - %(name)s [%(levelname)s] - %(message)s')
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fh_formatter)
    logger.addHandler(fh)

    # stream handler
    stream_level = {0: logging.ERROR, 1: logging.INFO, 2: logging.DEBUG}
    ch = logging.StreamHandler()
    ch.setLevel(stream_level[verbose_level])
    ch_formatter = logging.Formatter('%(name)s - %(message)s')
    ch.setFormatter(ch_formatter)
    logger.addHandler(ch)

    return logger


def DMD_run(
    model_name, dataset_name, config=None, config_file="", seeds=[], is_tune=False,
    tune_times=500, feature_T="", feature_A="", feature_V="",
    model_save_dir="", res_save_dir="", log_dir="",
    gpu_ids=[0], num_workers=4, verbose_level=1, mode = '', is_distill = False
):
    # Initialization
    model_name = model_name.lower()
    dataset_name = dataset_name.lower()
    # 读取配置
    if config_file != "":
        config_file = Path(config_file)
    else: # use default config files
        config_file = Path(__file__).parent / "config" / "config.json"
    if not config_file.is_file():
        raise ValueError(f"Config file {str(config_file)} not found.")
    # 保存模型路径
    if model_save_dir == "":
        model_save_dir = Path.home() / "MMSA" / "saved_models"
    Path(model_save_dir).mkdir(parents=True, exist_ok=True)
    if res_save_dir == "":
        res_save_dir = Path.home() / "MMSA" / "results"
    Path(res_save_dir).mkdir(parents=True, exist_ok=True)
    if log_dir == "":
        log_dir = Path.home() / "MMSA" / "logs"
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    seeds = seeds if seeds != [] else [1111, 1112, 1113, 1114, 1115]
    logger = _set_logger(log_dir, model_name, dataset_name, verbose_level)
    
    # 从配置文件加载模型参数
    args = get_config_regression(model_name, dataset_name, config_file)
    args.is_distill = is_distill  # use or not use distill, train use, test not use
    args.mode = mode # train or test
    args['model_save_path'] = Path(model_save_dir) / f"{args['model_name']}-{args['dataset_name']}.pth"
    args['device'] = assign_gpu(gpu_ids)
    args['train_mode'] = 'regression'
    args['feature_T'] = feature_T
    args['feature_A'] = feature_A
    args['feature_V'] = feature_V
    if config:
        args.update(config)

    # 保存结果
    res_save_dir = Path(res_save_dir) / "normal"
    res_save_dir.mkdir(parents=True, exist_ok=True)
    # 多种子集实验循环
    model_results = []
    for i, seed in enumerate(seeds):
        setup_seed(seed)
        args['cur_seed'] = i + 1
        # ! 这里真正运行模型
        result = _run(args, num_workers, is_tune)
        model_results.append(result)
    # 启动知识蒸馏时
    if args.is_distill:
        criterions = list(model_results[0].keys())
        # save result to csv
        csv_file = res_save_dir / f"{dataset_name}.csv"
        if csv_file.is_file():
            df = pd.read_csv(csv_file)
        else:
            df = pd.DataFrame(columns=["Model"] + criterions)
        # save results
        res = [model_name]
        for c in criterions:
            values = [r[c] for r in model_results]
            mean = round(np.mean(values)*100, 2)
            std = round(np.std(values)*100, 2)
            res.append((mean, std))
        df.loc[len(df)] = res
        df.to_csv(csv_file, index=None)
        logger.info(f"Results saved to {csv_file}.")


def _run(args, num_workers=4, is_tune=False, from_sena=False):
    # 准备 dataloader
    dataloader = MMDataLoader(args, num_workers)
    if args.is_distill:
        print("training for DMD")

        # param of homogeneous graph distillation
        args.gd_size_low = 64  # hidden size of graph distillation
        args.w_losses_low = [1, 10]  # weights for losses: [logit, repr]
        args.metric_low = 'l1'  # distance metric for distillation loss

        # param of heterogeneous graph distillation
        args.gd_size_high = 32  # hidden size of graph distillation
        args.w_losses_high = [1, 10]  # weights for losses: [logit, repr]
        args.metric_high = 'l1'  # distance metric for distillation loss

        to_idx = [0, 1, 2]  # all modalities can be distilled from each other simultaneously
        from_idx = [0, 1, 2]  # all modalities can be distilled from each other simultaneously
        assert len(from_idx) >= 1

        model = []
        # from trains.singleTask.model import dmd 来自这里
        # getattr(dmd, 'DMD') 获取 DMD类本身，用 args 来调用类的构造函数创建类对象 model_dmd
        model_dmd = getattr(dmd, 'DMD')(args)
        # 共享模态蒸馏模型
        model_distill_homo = getattr(get_distillation_kernel_homo, 'DistillationKernel')(n_classes=1,
                                                                               hidden_size=
                                                                               args.dst_feature_dim_nheads[0],
                                                                               gd_size=args.gd_size_low,
                                                                               to_idx=to_idx, from_idx=from_idx,
                                                                               gd_prior=softmax([0, 0, 1, 0, 1, 0], 0.25),
                                                                               gd_reg=10,
                                                                               w_losses=args.w_losses_low,
                                                                               metric=args.metric_low,
                                                                               alpha=1 / 8,
                                                                               hyp_params=args)
        # 私有模态蒸馏模型
        model_distill_hetero = getattr(get_distillation_kernel, 'DistillationKernel')(n_classes=1,
                                                                                   hidden_size=
                                                                                   args.dst_feature_dim_nheads[0] * 2,
                                                                                   gd_size=args.gd_size_high,
                                                                                   to_idx=to_idx, from_idx=from_idx,
                                                                                   gd_prior=softmax([0, 0, 1, 0, 1, 1], 0.25),
                                                                                   gd_reg=10,
                                                                                   w_losses=args.w_losses_high,
                                                                                   metric=args.metric_high,
                                                                                   alpha=1 / 8,
                                                                                   hyp_params=args)

        model_dmd, model_distill_homo, model_distill_hetero = model_dmd.cuda(), model_distill_homo.cuda(), model_distill_hetero.cuda()

        model = [model_dmd, model_distill_homo, model_distill_hetero]
    else:
        print("testing phase for DMD")
        model = getattr(dmd, 'DMD')(args)
        model = model.cuda()
    # 动态获取一个训练器实例，ATIO 是一个工厂类
    trainer = ATIO().getTrain(args)


    if args.mode == 'test':  # 进行测试
        model.load_state_dict(torch.load('pt/mosi-aligned.pth'))
        results = trainer.do_test(model, dataloader['test'], mode="TEST")
        sys.stdout.flush()
        input('[Press Any Key to start another run]')
    # TODO: 要改就在这里改，先预热，预热结束后，训练一次，执行调制一次
    else:  # 进行训练
        epoch_results = trainer.do_train(model, dataloader, return_epoch_results=from_sena)
        model[0].load_state_dict(torch.load('pt/dmd.pth'))

        results = trainer.do_test(model[0], dataloader['test'], mode="TEST")

        del model
        torch.cuda.empty_cache()
        gc.collect()
        time.sleep(1)
    return results