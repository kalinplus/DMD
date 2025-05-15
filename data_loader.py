import logging
import pickle
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
__all__ = ['MMDataLoader']
logger = logging.getLogger('MMSA')

class MMDataset(Dataset):
    def __init__(self, args, mode='train'):
        self.mode = mode
        self.args = args
        DATASET_MAP = {
            'mosi': self.__init_mosi,
            'mosei': self.__init_mosei,
        }
        # 根据数据集调用对应函数，参数来自 train.py
        DATASET_MAP[args['dataset_name']]()
    # __init_mosi 会被调用 3 次，分别加载训练、验证、测试集
    def __init_mosi(self):
        # 加载并初始化默认的 数据文本、视觉和语音模态的特征
        '''
        特征维度应该是：
        第 0 维：样本数量
        第 1 维：序列长度/token数量（文本），或者 时间步数/帧数（视觉、音频）
        第 2 维：每个 token 或 帧 的特征维度
        '''
        with open(self.args['featurePath'], 'rb') as f:
            data = pickle.load(f)
        if 'use_bert' in self.args and self.args['use_bert']:
            # text: ndarray(1284, 3, 50)
            self.text = data[self.mode]['text_bert'].astype(np.float32)  # BERT feature
        else:
            self.text = data[self.mode]['text'].astype(np.float32)  # GLOVE feature
        # vision: ndarray(1284, 50, 20)
        self.vision = data[self.mode]['vision'].astype(np.float32)
        # audio: ndarray(1284, 50 5)
        self.audio = data[self.mode]['audio'].astype(np.float32)
        # raw_text: ndarray(1284,)
        self.raw_text = data[self.mode]['raw_text']
        # id 对应的应该是有点杂乱无章的那些文件名
        self.ids = data[self.mode]['id']

        # 打印数据维度
        print(f"text: {self.text.shape}")
        print(f"vision: {self.vision.shape}")
        print(f"audio: {self.audio.shape}")

        # 如果用户指定了额外的各个模态的路径，则用这些特征替换对应的模态特征 (run.py 中默认为空)
        if self.args['feature_T'] != "":
            with open(self.args['feature_T'], 'rb') as f:
                data_T = pickle.load(f)
            # 相同的数据处理流程
            if 'use_bert' in self.args and self.args['use_bert']:
                self.text = data_T[self.mode]['text_bert'].astype(np.float32)
                self.args['feature_dims'][0] = 768
            else:
                self.text = data_T[self.mode]['text'].astype(np.float32)
                self.args['feature_dims'][0] = self.text.shape[2]
        if self.args['feature_A'] != "":
            with open(self.args['feature_A'], 'rb') as f:
                data_A = pickle.load(f)
            self.audio = data_A[self.mode]['audio'].astype(np.float32)
            self.args['feature_dims'][1] = self.audio.shape[2]
        if self.args['feature_V'] != "":
            with open(self.args['feature_V'], 'rb') as f:
                data_V = pickle.load(f)
            self.vision = data_V[self.mode]['vision'].astype(np.float32)
            self.args['feature_dims'][2] = self.vision.shape[2]
        # 加载当前数据集（如 MOSI、MOSEI）在指定模式（训练、验证或测试）下的回归任务标签
        # 这是多模态融合后的主任务标签（MultiModal），用于模型输出层的监督学习
        # labels['M']: ndarray(1284,)
        self.labels = {
            'M': np.array(data[self.mode]['regression_labels']).astype(np.float32)
        }

        print(f"labels: {self.labels['M'].shape}")

        logger.info(f"{self.mode} samples: {self.labels['M'].shape}")
        # 判断数据是否需要对齐并处理之
        if not self.args['need_data_aligned']:
            if self.args['feature_A'] != "":
                self.audio_lengths = list(data_A[self.mode]['audio_lengths'])
            else:
                self.audio_lengths = data[self.mode]['audio_lengths']
            if self.args['feature_V'] != "":
                self.vision_lengths = list(data_V[self.mode]['vision_lengths'])
            else:
                self.vision_lengths = data[self.mode]['vision_lengths']
        # 音频模态异常值处理
        self.audio[self.audio == -np.inf] = 0
        # 是否正则化
        if 'need_normalized' in self.args and self.args['need_normalized']:
            self.__normalize()
    
    def __init_mosei(self):
        return self.__init_mosi()

    def __init_sims(self):
        return self.__init_mosi()
    '''
    虽然写的很复杂，但是代码中根本没有用到这个函数...
    '''
    def __truncate(self):
        '''
        示例：
        假设 modal_features.shape = (3, 50, 20)，即 3 个样本，每个样本有 50 个时间步，每个时间步有 20 维特征。
        对于第 1 个样本：
        如果它的前几个时间步是 padding（全 0），则一直往后找；
        找到第一个非 padding 时间步后，就从这个位置开始取 20 个时间步作为该样本的新表示；
        如果整个样本的前面都是 padding，则从最后面取 20 个时间步。
        '''
        def do_truncate(modal_features, length):
            # 若当前模态特征长度等于目标长度，无需截断，直接返回
            if length == modal_features.shape[1]:
                return modal_features
            truncated_feature = []
            # 一个（每个）特征的长度
            padding = np.array([0 for i in range(modal_features.shape[2])])
            for instance in modal_features:  # 对于每一个样本
                for index in range(modal_features.shape[1]):  # 对于每个 token/帧
                    # 在没有有效内容时，尽量从接近结尾的地方取一段 padding 数据
                    if((instance[index] == padding).all()):  # 如果这帧全是 0，即为 padding
                        if(index + length >= modal_features.shape[1]):
                            truncated_feature.append(instance[index:index+20])
                            break
                    # 优先保留有效信息的起始部分
                    else:                        
                        truncated_feature.append(instance[index:index+20])
                        break
            truncated_feature = np.array(truncated_feature)
            return truncated_feature
        
        text_length, audio_length, video_length = self.args['seq_lens']
        self.vision = do_truncate(self.vision, video_length)
        self.text = do_truncate(self.text, text_length)
        self.audio = do_truncate(self.audio, audio_length)

    def __normalize(self):
        # (N, T, D) -> (T, N, D) 时间为第 0 维
        self.vision = np.transpose(self.vision, (1, 0, 2))
        self.audio = np.transpose(self.audio, (1, 0, 2))
        # (T, N, D) -> (1, N, D) 每个样本的信息都变为了时序上的平均值
        self.vision = np.mean(self.vision, axis=0, keepdims=True)
        self.audio = np.mean(self.audio, axis=0, keepdims=True)
        # NaN != NaN 返回 True，将所有 NaN 替换为 0
        self.vision[self.vision != self.vision] = 0
        self.audio[self.audio != self.audio] = 0
        # (1, N, D) -> (N, 1, D) 时间维度被压缩为 1
        self.vision = np.transpose(self.vision, (1, 0, 2))
        self.audio = np.transpose(self.audio, (1, 0, 2))

    def __len__(self):
        return len(self.labels['M'])

    def get_seq_len(self):
        if 'use_bert' in self.args and self.args['use_bert']:
            # 这里， self.text.shape[2] 代表每个 token 的特征维度， 而 a/v.shape[1] 则是有多少帧，二者含义是相同的 ?
            # FIXME: AI 说这里返回的有问题，应该是 self.text.shape[1]，除非 bert 处理后还有特殊设置
            return (self.text.shape[2], self.audio.shape[1], self.vision.shape[1])
        else:
            return (self.text.shape[1], self.audio.shape[1], self.vision.shape[1])
    '''
    代码中根本没有用到这个函数...
    '''
    def get_feature_dim(self):
        return self.text.shape[2], self.audio.shape[2], self.vision.shape[2]

    def __getitem__(self, index):
        sample = {
            'raw_text': self.raw_text[index],
            'text': torch.Tensor(self.text[index]), 
            'audio': torch.Tensor(self.audio[index]),
            'vision': torch.Tensor(self.vision[index]),
            'index': index,
            'id': self.ids[index],
            # 'M': 展品为一维的数据（张量）
            'labels': {k: torch.Tensor(v[index].reshape(-1)) for k, v in self.labels.items()}
        }
        if not self.args['need_data_aligned']:  # 如果原始未对齐，则只传递音频和视觉模态的实际有效长度，默认为 True（config.json)
            sample['audio_lengths'] = self.audio_lengths[index]
            sample['vision_lengths'] = self.vision_lengths[index]
        return sample
'''
返回一个字典，里面包含 3 个 dataloader
'''
# TODO: 改这里，动态重采样
def MMDataLoader(args, num_workers):

    datasets = {
        'train': MMDataset(args, mode='train'),
        'valid': MMDataset(args, mode='valid'),
        'test': MMDataset(args, mode='test')
    }

    if 'seq_lens' in args:
        # 序列维度
        args['seq_lens'] = datasets['train'].get_seq_len() 

    dataLoader = {
        ds: DataLoader(datasets[ds],
                       batch_size=args['batch_size'],
                       num_workers=num_workers,
                       shuffle=True)
        for ds in datasets.keys()  # train, valid, test
    }
    
    return dataLoader
