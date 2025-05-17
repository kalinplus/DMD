import os
import logging
import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

from data_loader import MMDataset, MMDataset_modality_level, MMDataLoader_modality_level
from ..utils import MetricsTop, dict_to_str
from .HingeLoss import HingeLoss

logger = logging.getLogger('MMSA')

# 所以这三个变量有啥用？
sample_nums = []
part_t = []
part_a = []
part_v = []

class MSE(nn.Module):
    def __init__(self):
        super(MSE, self).__init__()

    def forward(self, pred, real):
        diffs = torch.add(real, -pred)
        n = torch.numel(diffs.data)
        mse = torch.sum(diffs.pow(2)) / n
        return mse

class DMD():
    def __init__(self, args):
        self.args = args
        self.criterion = nn.L1Loss()
        self.cosine = nn.CosineEmbeddingLoss()
        self.metrics = MetricsTop(args.train_mode).getMetics(args.dataset_name)
        self.MSE = MSE()
        self.sim_loss = HingeLoss()

    def execute_modulation(self, args, model, device, dataloader, log_name, epoch):
        """
        估计贡献参考 VEMC 中的 execute_modulation, 前向传播/计算精读参考下方 do_test
        Args:
            args: 传入的参数，参考 VEMC 项目
            model: 主模型，也就是 3 个模型中的 model[0]
            device: 运行的设备，cpu/cuda
            dataloader: 传入 data_laoder.py 中 dataloder 类返回的字典，需要指定用哪个，应该是训练集
            log_name: 日志文件位置
            epoch: 这是第几个 epoch

        Returns:
            cont, conv, cona, 三个模态的贡献
            dataloader  包含 3个 dataloader 的列表，其中 dataloader['train'] 为空（预热）或 重采样过的

        """
        train_dataloader = None
        n_classes = args.n_classes

        contribution = {}
        softmax = nn.Softmax(dim=1)
        cont = 0.0
        cona = 0.0
        conv = 0.0
        model = model.to(device)

        with torch.no_grad():
            model.eval()
            for step, batch in tqdm(enumerate(dataloader['train'])):
                # raw_text = batch['raw_text']

                text = batch['text'].to(device)  # (B, T_t, D_t)
                vision = batch['vision'].to(device)  # (B, T_v, D_v)
                audio = batch['audio'].to(device)  # (B, T_a, D_a)
                text_zero = torch.zeros_like(text)
                vision_zero = torch.zeros_like(vision)
                audio_zero = torch.zeros_like(audio)

                index = batch['index']
                ids = batch['id']
                labels = batch['labels']['M'].to(device)  # 假设只用 'M' 标签
                remain = batch['remain'].to(device)  # (B,)

                # 各种输入情况下的输出
                tva_output = model(text, vision, audio, is_distill=True)
                tv_output = model(text, vision, audio_zero, is_distill=True)
                ta_output = model(text, vision_zero, audio, is_distill=True)
                va_output = model(text_zero, vision, audio, is_distill=True)
                # 单模态就没必要使用蒸馏了
                t_output = model(text, vision_zero, audio_zero)
                v_output = model(text_zero, vision, audio_zero)
                a_output = model(text_zero, vision_zero, audio)

                # ! 原论文结果中，7类准确率结果普遍偏低，故而使用它作为分类指标
                bins = [-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0]
                # 一个一个样本处理
                for i, item in enumerate(labels):
                    # 依次算出各种输入情况下的计算结果
                    index_tva = np.digitize(tva_output[i].cpu().data.numpy(), bins)
                    index_tv = np.digitize(tv_output[i].cpu().data.numpy(), bins)
                    index_ta = np.digitize(ta_output[i].cpu().data.numpy(), bins)
                    index_va = np.digitize(va_output[i].cpu().data.numpy(), bins)
                    index_t = np.digitize(t_output[i].cpu().data.numpy(), bins)
                    index_v = np.digitize(v_output[i].cpu().data.numpy(), bins)
                    index_a = np.digitize(a_output[i].cpu().data.numpy(), bins)

                    index_label = np.digitize(labels[i].cpu().data.numpy(), bins)

                    # 根据公式计算贡献值
                    value_tva = 0.0
                    value_tv = 0.0
                    value_ta = 0.0
                    value_va = 0.0
                    value_t = 0.0
                    value_v = 0.0
                    value_a = 0.0
                    if index_tva == index_label:
                        value_tva = 3.0
                    if index_tv == index_label:
                        value_tv = 2.0
                    if index_ta == index_label:
                        value_ta = 2.0
                    if index_va == index_label:
                        value_va = 2.0
                    if index_t == index_label:
                        value_t = 1.0
                    if index_v == index_label:
                        value_v = 1.0
                    if index_a == index_label:
                        value_a = 1.0

                    contrib_t = (1.0 / 3.0) * (value_t + value_tva - value_va) + (1.0 / 6.0) * (value_ta + value_tv - value_v - value_a)
                    contrib_v = (1.0 / 3.0) * (value_v + value_tva - value_ta) + (1.0 / 6.0) * (value_tv + value_va - value_t - value_a)
                    contrib_a = (1.0 / 3.0) * (value_a + value_tva - value_tv) + (1.0 / 6.0) * (value_ta + value_va - value_t - value_v)
                    contrib_t = max(0, contrib_t)
                    contrib_v = max(0, contrib_v)
                    contrib_a = max(0, contrib_a)
                    cont += contrib_t
                    conv += contrib_v
                    cona += contrib_a

                    contribution[int(index[i])] = (contrib_t, contrib_v, contrib_a)

        cont /= len(dataloader['train'].dataset)
        conv /= len(dataloader['train'].dataset)
        cona /= len(dataloader['train'].dataset)

        # 打印日志
        if not os.path.exists(os.path.join(args.log_path, log_name)):
            os.mkdir(os.path.join(args.log_path, log_name))
        if not os.path.exists(os.path.join(args.log_path, log_name, "contribution")):
            os.mkdir(os.path.join(args.log_path, log_name, "contribution"))
        np.save(
            os.path.join(args.log_path, log_name, "contribution", str(epoch) + ".npy"),
            contribution,
        )
        print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
        print("now train epoch, cont, conv and cona: ", cont, conv, cona)
        print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
        # ! 下面是重采样部分
        if epoch >= args.warmup - 1:  # 超出预热阶段才开始重采样
            part_cont = 0.0
            part_cona = 0.0
            part_conv = 0.0
            num = int(len(dataloader['train'].dataset) * args.part_radio)  # 重采样次数
            choice = np.random.choice(len(dataloader['train'].dataset), num)
            print(f"length of choice: {len(choice)}")
            for i in choice:
                contri_t, contri_v, contri_a = contribution[i]
                part_cont += contri_t
                part_conv += contri_v
                part_cona += contri_a
            part_cont /= num
            part_conv /= num
            part_cona /= num
            # ! part_x 是全局变量，因为模态级的重采样，就是要根据历史所有样本，计算该模态的平均贡献
            part_t.append(part_cont)
            part_v.append(part_conv)
            part_a.append(part_cona)
            gap_t = 1.0 - part_cont
            gap_v = 1.0 - part_conv
            gap_a = 1.0 - part_cona
            # 现在有 3 个模态了，给绝对值取个平均
            part_difference = (abs(gap_t - gap_v) + abs(gap_t - gap_a) + abs(gap_a - gap_v)) / 3.0 / 3 * 2 * args.alpha
            print("part_p: ", part_difference)
            # 执行重采样
            train_dataset = MMDataset_modality_level(
                args=args,
                contribution_t=part_t,
                contribution_a=part_a,
                contribution_v=part_v,
                alpha=args.alpha
            )
            dataloader = MMDataLoader_modality_level(
                args=args,
                num_workers=4,
                train_dataset=train_dataset
            )
        return cont, cona, conv, dataloader

    def do_train(self, model, dataloader, return_epoch_results=False):
        # 定义需要的变量
        cont_all = []
        conv_all = []
        cona_all = []

        # 0: DMD model, 1: Homo GD, 2: Hetero GD
        params = list(model[0].parameters()) + \
                 list(model[1].parameters()) + \
                 list(model[2].parameters())

        optimizer = optim.Adam(params, lr=self.args.learning_rate)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, verbose=True, patience=self.args.patience)

        epochs, best_epoch = 0, 0
        if return_epoch_results:
            epoch_results = {
                'train': [],
                'valid': [],
                'test': []
            }
        min_or_max = 'min' if self.args.KeyEval in ['Loss'] else 'max'
        best_valid = 1e8 if min_or_max == 'min' else 0

        net = []
        net_dmd = model[0]
        net_distill_homo = model[1]
        net_distill_hetero = model[2]
        net.append(net_dmd)
        net.append(net_distill_homo)
        net.append(net_distill_hetero)
        model = net

        # TODO: 要改就在这里改，先预热，预热结束后，训练一次，执行调制一次
        # ! 训练会一致进行，直到因为没有优化而早停为止
        while True:
            epochs += 1
            y_pred, y_true = [], []
            for mod in model:
                mod.train()

            train_loss = 0.0
            left_epochs = self.args.update_epochs  # 默认值为 10，使得梯度累计 10 次才更新一次
            with tqdm(dataloader['train']) as td:
                for batch_data in td:
                    # 在小批量训练中模拟大批量训练的效果，节省显存
                    # 这里的 epoch 其实是训练 一个 batch 的意思
                    if left_epochs == self.args.update_epochs:
                        optimizer.zero_grad()
                    left_epochs -= 1
                    # 准备输入，都移动到指定设备
                    vision = batch_data['vision'].to(self.args.device)
                    audio = batch_data['audio'].to(self.args.device)
                    text = batch_data['text'].to(self.args.device)
                    labels = batch_data['labels']['M'].to(self.args.device)
                    labels = labels.view(-1, 1)

                    logits_homo, reprs_homo, logits_hetero, reprs_hetero = [], [], [], []
                    # DMD 模型前向传播
                    output = model[0](text, audio, vision, is_distill=True)

                    # logits for homo GD
                    logits_homo.append(output['logits_l_homo'])
                    logits_homo.append(output['logits_v_homo'])
                    logits_homo.append(output['logits_a_homo'])

                    # reprs for homo GD
                    reprs_homo.append(output['repr_l_homo'])
                    reprs_homo.append(output['repr_v_homo'])
                    reprs_homo.append(output['repr_a_homo'])

                    # logits for hetero GD
                    logits_hetero.append(output['logits_l_hetero'])
                    logits_hetero.append(output['logits_v_hetero'])
                    logits_hetero.append(output['logits_a_hetero'])

                    # reprs for hetero GD
                    reprs_hetero.append(output['repr_l_hetero'])
                    reprs_hetero.append(output['repr_v_hetero'])
                    reprs_hetero.append(output['repr_a_hetero'])

                    logits_homo = torch.stack(logits_homo)
                    reprs_homo = torch.stack(reprs_homo)

                    logits_hetero = torch.stack(logits_hetero)
                    reprs_hetero = torch.stack(reprs_hetero)

                    # edges for homo distill
                    edges_homo, edges_origin_homo = model[1](logits_homo, reprs_homo)

                    # edges for hetero distill
                    edges_hetero, edges_origin_hetero = model[2](logits_hetero, reprs_hetero)

                    # task loss
                    loss_task_all = self.criterion(output['output_logit'], labels)
                    loss_task_l_homo = self.criterion(output['logits_l_homo'], labels)
                    loss_task_v_homo = self.criterion(output['logits_v_homo'], labels)
                    loss_task_a_homo = self.criterion(output['logits_a_homo'], labels)
                    loss_task_l_hetero = self.criterion(output['logits_l_hetero'], labels)
                    loss_task_v_hetero = self.criterion(output['logits_v_hetero'], labels)
                    loss_task_a_hetero = self.criterion(output['logits_a_hetero'], labels)
                    loss_task_c = self.criterion(output['logits_c'], labels)
                    loss_task = loss_task_all + loss_task_l_homo + loss_task_v_homo + loss_task_a_homo + loss_task_l_hetero + loss_task_v_hetero + loss_task_a_hetero + loss_task_c

                    # reconstruction loss
                    loss_recon_l = self.MSE(output['recon_l'], output['origin_l'])
                    loss_recon_v = self.MSE(output['recon_v'], output['origin_v'])
                    loss_recon_a = self.MSE(output['recon_a'], output['origin_a'])
                    loss_recon = loss_recon_l + loss_recon_v + loss_recon_a

                    # cycle consistency loss between s_x and s_x_r
                    loss_sl_slr = self.MSE(output['s_l'].permute(1, 2, 0), output['s_l_r'])
                    loss_sv_slv = self.MSE(output['s_v'].permute(1, 2, 0), output['s_v_r'])
                    loss_sa_sla = self.MSE(output['s_a'].permute(1, 2, 0), output['s_a_r'])
                    loss_s_sr = loss_sl_slr + loss_sv_slv + loss_sa_sla

                    # ort loss
                    # print("output['s_l']", output['s_l'].shape)
                    # print("output['c_l']", output['c_l'].shape)
                    cosine_similarity_s_c_l = self.cosine(output['s_l'].transpose(0,1).contiguous().view(labels.size(0),-1), output['c_l'].transpose(0,1).contiguous().view(labels.size(0),-1),
                                                          torch.tensor([-1]).cuda()).mean(0)
                    cosine_similarity_s_c_v = self.cosine(output['s_v'].transpose(0,1).contiguous().view(labels.size(0),-1), output['c_v'].transpose(0,1).contiguous().view(labels.size(0),-1),
                                                          torch.tensor([-1]).cuda()).mean(0)
                    cosine_similarity_s_c_a = self.cosine(output['s_a'].transpose(0,1).contiguous().view(labels.size(0),-1), output['c_a'].transpose(0,1).contiguous().view(labels.size(0),-1),
                                                          torch.tensor([-1]).cuda()).mean(0)
                    loss_ort = cosine_similarity_s_c_l + cosine_similarity_s_c_v + cosine_similarity_s_c_a

                    # margin loss
                    c_l, c_v, c_a = output['c_l_sim'], output['c_v_sim'], output['c_a_sim']
                    ids, feats = [], []
                    for i in range(labels.size(0)):
                        feats.append(c_l[i].view(1, -1))
                        feats.append(c_v[i].view(1, -1))
                        feats.append(c_a[i].view(1, -1))
                        ids.append(labels[i].view(1, -1))
                        ids.append(labels[i].view(1, -1))
                        ids.append(labels[i].view(1, -1))
                    feats = torch.cat(feats, dim=0)
                    ids = torch.cat(ids, dim=0)
                    loss_sim = self.sim_loss(ids, feats)

                    # homo GD loss
                    loss_reg_homo, loss_logit_homo, loss_repr_homo = \
                        model[1].distillation_loss(logits_homo, reprs_homo, edges_homo)
                    graph_distill_loss_homo = 0.05 * (loss_logit_homo + loss_reg_homo)

                    # hetero GD loss
                    loss_reg_hetero, loss_logit_hetero, loss_repr_hetero = \
                        model[2].distillation_loss(logits_hetero, reprs_hetero, edges_hetero)
                    graph_distill_loss_hetero = 0.05 * (loss_logit_hetero + loss_repr_hetero + loss_reg_hetero)

                    combined_loss = loss_task + \
                                    graph_distill_loss_homo + graph_distill_loss_hetero + \
                                    (loss_s_sr + loss_recon + (loss_sim+loss_ort) * 0.1) * 0.1

                    combined_loss.backward()

                    # 梯度 clipping
                    if self.args.grad_clip != -1.0:
                        params = list(model[0].parameters()) + \
                                 list(model[1].parameters()) + \
                                 list(model[2].parameters())
                        nn.utils.clip_grad_value_(params, self.args.grad_clip)
                    # 累计 loss
                    train_loss += combined_loss.item()

                    y_pred.append(output['output_logit'].cpu())
                    y_true.append(labels.cpu())
                    if not left_epochs:
                        optimizer.step()
                        left_epochs = self.args.update_epochs
                if not left_epochs:
                    # update
                    optimizer.step()

            train_loss = train_loss / len(dataloader['train'])
            pred, true = torch.cat(y_pred), torch.cat(y_true)
            train_results = self.metrics(pred, true)
            logger.info(
                f">> Epoch: {epochs} "
                f"TRAIN-({self.args.model_name}) [{epochs - best_epoch}/{epochs}/{self.args.cur_seed}] "
                f">> total_loss: {round(train_loss, 4)} "
                f"{dict_to_str(train_results)}"
            )
            # validation
            val_results = self.do_test(model[0], dataloader['valid'], mode="VAL")
            test_results = self.do_test(model[0], dataloader['test'], mode="TEST")
            cur_valid = val_results[self.args.KeyEval]
            scheduler.step(val_results['Loss'])
            # save each epoch model
            torch.save(model[0].state_dict(), './pt/' + str(epochs) + '.pth')
            # save best model
            isBetter = cur_valid <= (best_valid - 1e-6) if min_or_max == 'min' else cur_valid >= (best_valid + 1e-6)
            if isBetter:
                best_valid, best_epoch = cur_valid, epochs
                # save model
                model_save_path = './pt/dmd.pth'
                torch.save(model[0].state_dict(), model_save_path)

            if return_epoch_results:
                train_results["Loss"] = train_loss
                epoch_results['train'].append(train_results)
                epoch_results['valid'].append(val_results)
                test_results = self.do_test(model, dataloader['test'], mode="TEST")
                epoch_results['test'].append(test_results)
            # early stop
            if epochs - best_epoch >= self.args.early_stop:
                return epoch_results if return_epoch_results else None

    def do_test(self, model, dataloader, mode="VAL", return_sample_results=False):

        model.eval()
        y_pred, y_true = [], []

        eval_loss = 0.0
        if return_sample_results:
            ids, sample_results = [], []
            all_labels = []
            features = {
                "Feature_t": [],
                "Feature_a": [],
                "Feature_v": [],
                "Feature_f": [],
            }

        with torch.no_grad():
            with tqdm(dataloader) as td:
                for batch_data in td:
                    vision = batch_data['vision'].to(self.args.device)
                    audio = batch_data['audio'].to(self.args.device)
                    text = batch_data['text'].to(self.args.device)
                    labels = batch_data['labels']['M'].to(self.args.device)
                    labels = labels.view(-1, 1)
                    output = model(text, audio, vision, is_distill=True)
                    loss = self.criterion(output['output_logit'], labels)
                    eval_loss += loss.item()
                    y_pred.append(output['output_logit'].cpu())
                    y_true.append(labels.cpu())

        eval_loss = eval_loss / len(dataloader)
        pred, true = torch.cat(y_pred), torch.cat(y_true)

        eval_results = self.metrics(pred, true)
        eval_results["Loss"] = round(eval_loss, 4)
        logger.info(f"{mode}-({self.args.model_name}) >> {dict_to_str(eval_results)}")

        if return_sample_results:
            eval_results["Ids"] = ids
            eval_results["SResults"] = sample_results
            for k in features.keys():
                features[k] = np.concatenate(features[k], axis=0)
            eval_results['Features'] = features
            eval_results['Labels'] = all_labels

        return eval_results