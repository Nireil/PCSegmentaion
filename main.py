import os, sys, logging, argparse, random, tempfile, time
import numpy as np
from os.path import join, exists
from tqdm import tqdm

# os.environ["CUDA_VISIBLE_DEVICES"]='2'              # 指定可见的显卡需要在import torch 之前
# os.environ["CUDA_VISIBLE_DEVICES"]='0, 1, 2, 3'


import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader # 单卡dataloader创建
from torch.multiprocessing import Process
import torch.multiprocessing as mp # 分布式启动方式 或 toch.distributed.launch
from torch.utils.data.distributed import DistributedSampler  # 负责分布式dataloader创建
from torch.utils.tensorboard import SummaryWriter


# from apex.parallel import DistributedDataParallel
import torch.distributed as dist
from distributed_utils import cleanup, reduce_value

from input import inputpoints
from metric import SemSegMetric
from builder import build_dataset, build_model, build_loss, build_optimizer, build_scheduler
from loss import filter_valid_label

from torchpack.utils.config import configs
from datasets.utils.cloud_sampler import get_sampler
from open3d._ml3d.torch.dataloaders import default_batcher, concat_batcher


log = logging.getLogger(__name__)

def get_delta_time(aTime, bTime):
    a_second = int(aTime[-2:]) + 60*int(aTime[-5:-3]) + 60*60*int(aTime[-8:-6]) + 24*60*60*int(aTime[8:10])
    b_second = int(bTime[-2:]) + 60*int(bTime[-5:-3]) + 60*60*int(bTime[-8:-6]) + 24*60*60*int(bTime[8:10])
    gap = b_second - a_second
    day, gap = divmod(gap, 24*3600)
    hour, gap = divmod(gap, 3600)
    minute, second = divmod(gap, 60)

    if day>0:
        gapTime = "%02d天%02d时%02d分%02d秒"%(day, hour, minute, second)
    if day==0 and hour>0:
        gapTime = "%02d时%02d分%02d秒"%(hour, minute, second)
    if day==0 and hour==0 and minute>0:
        gapTime = "%02d分%02d秒"%(minute, second)
    if day==0 and hour==0 and minute==0 and second>0:
        gapTime = "%02d秒"%(second)
    
    return(gapTime)

class PointCloudSeg:

    def __init__(self, args) -> None:
        self.args = args
        # self.rank = rank
        # 固定随机种子
        self.seed = 42
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)
        # 获取配置文件
        configs.load(args.cfg_path)
        # configs.load('./debug.yml')
        self.cfg_name = args.cfg_path.split('/')[-1].split('.')[-2]
        # self.cfg_name = 'debug'
        self.cfg = configs
        self.device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
        # self.device = torch.device('cpu')
        # 初始化各进程环境 start
        os.environ["MASTER_ADDR"] = "localhost"
        # os.environ["MASTER_PORT"] = "12345"

        self.save_results_dir = join(self.cfg.pipeline.results_root, self.cfg.model.name + '_' + self.cfg.dataset.name + '_' + self.cfg_name)
        if not exists(self.save_results_dir):
             os.makedirs(self.save_results_dir)
             os.makedirs(join(self.save_results_dir, 'logs'))
             os.makedirs(join(self.save_results_dir, 'tensorboard_summary'))
             os.makedirs(join(self.save_results_dir, 'checkpoints'))
             os.makedirs(join(self.save_results_dir, 'predict_results'))
        self.train_log_path = join(self.save_results_dir, 'logs', 'train.txt')
        self.test_log_path = join(self.save_results_dir, 'logs', 'test.txt')
        self.tensorboard_dir = join(self.save_results_dir, 'tensorboard_summary')
        self.best_train_ckpt = join(self.save_results_dir, 'checkpoints', 'train_best.pth')
        self.best_valid_ckpt = join(self.save_results_dir, 'checkpoints', 'val_best.pth')
        self.last_ckpt = join(self.save_results_dir, 'checkpoints', 'last_epoch.pth')
        self.test_predict_dir = join(self.save_results_dir, 'predict_results')

        self.model = build_model(self.cfg)
        self.dataset = build_dataset(self.cfg) 

        self.Loss = build_loss(self.cfg.pipeline)
        ## 计算数据集各类别点数后手动写到cfg, 只算有效类别
        # self.num_per_class = self.dataset.get_pointnum() 
        self.num_per_class = self.cfg.dataset.num_per_class
        self.weights = self.get_class_weights(self.num_per_class)

        self.cfg.pipeline.optimizer.lr *= args.world_size 
        self.optimizer = build_optimizer(self.model, self.cfg.pipeline)
        self.scheduler = build_scheduler(self.optimizer, self.cfg.pipeline)

        self.metric_train = SemSegMetric()
        self.metric_valid = SemSegMetric()
        self.metric_test = SemSegMetric()
        self.save_iou = 0.7  # miou大于0.7的开始保存为最优模型
    
    def get_class_weights(self, num_per_class):
        num_per_class = np.array(num_per_class)
        frequency = num_per_class / float(sum(num_per_class))
        if self.cfg.pipeline.loss.name == 'sqrt' or \
           self.cfg.pipeline.loss.name == 'lovas':
            ce_label_weight = 1 / np.sqrt(frequency)
        elif self.cfg.pipeline.loss.name == 'wce' or \
            self.cfg.pipeline.loss.name == 'focal':
            ce_label_weight = 1 / (frequency + 0.02)
        else:
            raise ValueError('Only support sqrt and wce')
        # num_per_class已经去除了忽略类别
        # ce_label_weight = np.delete(ce_label_weight, self.cfg.dataset.ignored_label_inds) 
        return np.expand_dims(ce_label_weight, axis=0)

    def get_batcher(self, device):
        """Get the batcher to be used based on the modelname and device"""
        batcher_name = getattr(self.cfg.model, 'batcher')

        if batcher_name == 'DefaultBatcher':
            batcher = default_batcher.DefaultBatcher()
        elif batcher_name == 'ConcatBatcher':
            batcher = concat_batcher.ConcatBatcher(device, self.cfg.model.name)
        else:
            batcher = None
        return batcher

    def train(self, rank):
        args = self.args
        cfg = self.cfg
        model = self.model
        dataset = self.dataset
        device = self.device
        model.to(device)

        pretrain_path = args.pretrain_path
        checkpoint_path = ""

        batcher = self.get_batcher(device)
        weights = torch.tensor(self.weights, dtype=torch.float, device=device)

        if rank == 0:
            log.info("训练日志保存于 : {}".format(self.train_log_path))
            log.addHandler(logging.FileHandler(self.train_log_path))
            writer = SummaryWriter(self.tensorboard_dir)
            log.info("Tensorboard训练日志保存于 {}.".format(self.tensorboard_dir))

        train_split = dataset.get_split('train')
        valid_split = dataset.get_split('valid')
        train_sampler = train_split.sampler
        valid_sampler = valid_split.sampler
        train_dataset = inputpoints(cfg=cfg.pipeline,
                                    dataset=train_split,
                                    preprocess=train_split.preprocess,
                                    use_cache=cfg.dataset.use_cache,
                                    transform=model.transform)
        if cfg.dataset.valid_files == []:
            valid_dataset = None
        else:
            valid_dataset = inputpoints(cfg=cfg.pipeline,
                                        dataset=valid_split,
                                        preprocess=train_split.preprocess,
                                        use_cache=cfg.dataset.use_cache,
                                        transform=model.transform)
        if args.distributed:
            print('| distributed init (rank {}): {}'.format(rank, 'env://'), flush=True)
            torch.cuda.set_device(rank)
            dist.init_process_group(backend='nccl', init_method='env://', world_size=args.world_size, rank=rank)
            dist.barrier()
            # device = torch.device('cuda') # 考虑能否把单卡和多卡的device, model.to(device)写在外面
            
            distributed_train_sampler = DistributedSampler(train_dataset)
            distributed_valid_sampler = DistributedSampler(valid_dataset)
            train_batch_sampler = torch.utils.data.BatchSampler(distributed_train_sampler, self.cfg.dataset.train_batch_size, drop_last=True)
            
            train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                                       batch_sampler=train_batch_sampler, 
                                                       num_workers=cfg.get('num_workers', 2),
                                                       pin_memory=cfg.get('pin_memory', True))
            if valid_dataset is not None:                             
                valid_loader = torch.utils.data.DataLoader(dataset=valid_dataset,
                                                        batch_size=self.cfg.dataset.valid_batch_size,
                                                        sampler=distributed_valid_sampler,
                                                        num_workers=cfg.get('num_workers', 2),
                                                        pin_memory=cfg.get('pin_memory', True))
            
            if os.path.exists(pretrain_path):
                weights_dict = torch.load(pretrain_path, map_location=device)
                load_pretrain_dict = {k: v for k, v in weights_dict.items()
                                     if model.state_dict()[k].numel() == v.numel()}
                model.load_state_dict(load_pretrain_dict, strict=False)
            else:
                # 如果不存在预训练权重，需要将第一个进程中的权重保存，然后其他进程载入，保持初始化权重一致
                checkpoint_path = os.path.join(tempfile.gettempdir(), "initial_weights.pt")
                if rank == 0:
                    torch.save(model.state_dict(), checkpoint_path)
                dist.barrier()
                # 这里注意，一定要指定map_location参数，否则会导致第一块GPU占用更多资源
                model.load_state_dict(torch.load(checkpoint_path, map_location=device))
            
            # 待查看
            # 是否冻结权重
            if args.freeze_layers:
                for name, para in model.named_parameters():
                    # 除最后的全连接层外，其他权重全部冻结
                    if "fc" not in name:
                        para.requires_grad_(False)
            else:
                # 只有训练带有BN结构的网络时使用SyncBatchNorm才用意义
                if args.syncBN:
                    # 使用SyncBatchNorm后训练会更耗时
                    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).to(device)
            # 转为DDP模型
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[rank])
            
        else:
            # batcher = self.get_batcher(device) # 考虑多卡训练是否需要这个，把和device相关的写在外面是否影响多卡
            # weights = torch.tensor(self.weights, dtype=torch.float, device=device)

            train_loader = DataLoader(dataset=train_dataset,
                                      batch_size=cfg.pipeline.train_batch_size,
                                      sampler=get_sampler(train_sampler),
                                      num_workers=cfg.get('num_workers', 2),
                                      pin_memory=cfg.get('pin_memory', True),
                                      collate_fn=batcher.collate_fn,
                                      worker_init_fn=lambda x: np.random.seed(x + np.uint32(torch.utils.data.get_worker_info().seed)))
            if valid_dataset is not None:
                valid_loader = DataLoader(dataset=valid_dataset,
                                        batch_size=self.cfg.pipeline.valid_batch_size,
                                        sampler=get_sampler(valid_sampler),
                                        num_workers=cfg.get('num_workers', 2),
                                        pin_memory=cfg.get('pin_memory', True),
                                        collate_fn=batcher.collate_fn,
                                        worker_init_fn=lambda x: np.random.seed(x + np.uint32(torch.utils.data.get_worker_info().seed)))

        start_train_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
        log.info(f"{start_train_time} 开始训练") if rank == 0 else None
        save_train_iou = self.save_iou
        if valid_dataset is not None:
            save_val_iou = self.save_iou
        for epoch in range(self.cfg.pipeline.num_epochs):
            log.info(f'======== EPOCH {epoch:d}/{(self.cfg.pipeline.num_epochs-1):d} ========') if rank == 0 else None
            if args.distributed:
                distributed_train_sampler.set_epoch(epoch)

            mean_train_loss, train_metric = self.train_one_epoch(model=model, 
                                                                  train_loader=train_loader,
                                                                  device=device,
                                                                  weights=weights,
                                                                  rank=rank)
            self.scheduler.step()
            if valid_dataset is not None:
                mean_valid_loss, valid_metric = self.evaluate(model=model,
                                                            valid_loader=valid_loader,
                                                            device=device,
                                                            weights=weights,
                                                            rank=rank)

            if rank == 0:
                writer.add_scalar('Training loss', mean_train_loss, epoch)
                for key, val in train_metric.items():                # 最后一个为平均精度和mIoU
                    writer.add_scalar("Training {}".format(key), val, epoch)
                log.info(f"Train loss: {mean_train_loss:.3f}"
                         f" OA: {train_metric['OA']:.3f}"
                         f" mIoU: {train_metric['mIoU']:.3f}")
                # 保存训练最优，验证最优和最后一个epoch的模型（最优: mIoU最高）
                if train_metric['mIoU'] >= save_train_iou:
                    save_train_iou = train_metric['mIoU']
                    torch.save( dict(epoch=epoch,
                                    model_state_dict=self.model.state_dict(),
                                    optimizer_state_dict=self.optimizer.state_dict(),
                                    scheduler_state_dict=self.scheduler.state_dict()),
                                self.best_train_ckpt )
                    log.info(f'Epoch {epoch:3d}: save best_train_iou ckpt as {self.best_train_ckpt:s}')

                if valid_dataset is not None:
                    writer.add_scalar('Validation loss', mean_valid_loss, epoch)
                    for key, val in valid_metric.items():            
                        writer.add_scalar("Validation {}".format(key), val, epoch)     
                    log.info(f"Val loss: {mean_valid_loss:.3f} "
                            f" OA: {valid_metric['OA']:.3f}"
                            f" mIoU: {valid_metric['mIoU']:.3f}")
                    if valid_metric['mIoU'] >= save_val_iou:
                        save_val_iou = valid_metric['mIoU']
                        torch.save(dict(epoch=epoch,
                                        model_state_dict=self.model.state_dict(),
                                        optimizer_state_dict=self.optimizer.state_dict(),
                                        scheduler_state_dict=self.scheduler.state_dict()),
                                    self.best_valid_ckpt)
                        log.info(f'Epoch {epoch:3d}: save best_valid_iou ckpt to {self.best_valid_ckpt:s}')

                if epoch == self.cfg.pipeline.num_epochs-1:
                    torch.save(dict(epoch=epoch,
                                    model_state_dict=self.model.state_dict(),
                                    optimizer_state_dict=self.optimizer.state_dict(),
                                    scheduler_state_dict=self.scheduler.state_dict()),
                                    self.last_ckpt)
                    log.info(f'Epoch {epoch:3d}: save last ckpt to {self.last_ckpt:s}')

        end_train_time = time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))
        log.info(f"{end_train_time} 训练完成")
        delta_time = get_delta_time(start_train_time, end_train_time)
        log.info(f"训练用时：{delta_time}")
        # 多卡训练删除临时缓存文件
        if args.distributed and rank == 0:
            if os.path.exists(checkpoint_path) is True:
                os.remove(checkpoint_path)
            cleanup() 

    def train_one_epoch(self, model, train_loader, device, weights, rank):
        # B, N = self.cfg.pipeline.train_batch_size, self.cfg.dataset.get_input['parameter']['num_points']
        device = self.device
        model.train()
        mean_train_loss = torch.zeros(1).to(device)
        self.metric_train.reset()
  
        if rank == 0:
            train_loader = tqdm(train_loader, desc='training')
        
        for step, inputs in enumerate(train_loader):
            data = inputs['data']
            # if 'features' not in data:
            #     data['features'] = torch.zeros([B,N,1])
            data = {key: ([attr.cuda() for attr in data[key]] if isinstance(data[key], list) 
                           else data[key].cuda()) for key in data}
            self.optimizer.zero_grad()
            results = model(data)
            predict_scores, gt_labels = filter_valid_label(results, 
                                                           data['labels'],
                                                           model.cfg.num_classes,
                                                           self.cfg.dataset.ignored_label_inds,
                                                           device)
            # loss = self.Loss(predict_scores, gt_labels, weight=weights)
            loss = self.Loss(predict_scores, gt_labels, weight=None)
            if predict_scores.size()[-1] == 0:
                continue
            
            loss.backward()

            loss = reduce_value(loss, average=True)
            mean_train_loss = (mean_train_loss * step + loss.detach()) / (step + 1)
            
            if not torch.isfinite(loss):
                print('WARNING: non-finite loss, ending training ', loss)
                sys.exit(1)
            # 是否进行梯度裁剪
            if self.cfg.get('grad_clip_norm', -1) > 0:
                torch.nn.utils.clip_grad_value_(model.parameters(), self.cfg.grad_clip_norm)

            self.optimizer.step()

            self.metric_train.update(predict_scores, gt_labels)
            oa = self.metric_train.tp()/data['labels'].numel()
            oa = reduce_value(torch.tensor(oa).to(device), average=True)
            mIoU = self.metric_train.iou()[-1]
            mIoU = reduce_value(torch.tensor(mIoU).to(device), average=True)
            train_metric = {'OA': oa, 'mIoU': mIoU}

        # 等待所有进程计算完毕
        if device != torch.device("cpu"):
            torch.cuda.synchronize(device)

        return mean_train_loss.item(), train_metric 
    
    def evaluate(self, model, valid_loader, device, weights, rank):
        B, N = self.cfg.pipeline.valid_batch_size, self.cfg.dataset.get_input['parameter']['num_points']
        model.eval()
        mean_valid_loss = torch.zeros(1).to(device)
        self.metric_valid.reset()

        if rank == 0:
            valid_loader = tqdm(valid_loader, desc='validation')
        with torch.no_grad():
            for step, inputs in enumerate(valid_loader):
                data = inputs['data']
                # if 'features' not in data:
                #     data['features'] = torch.zeros([B,N,1])
                data = {key: ([attr.cuda() for attr in data[key]] if isinstance(data[key], list) 
                            else data[key].cuda()) for key in data}
                results = model(data)
                predict_scores, gt_labels = filter_valid_label(results, 
                                                              data['labels'],
                                                              model.cfg.num_classes,
                                                              self.cfg.dataset.ignored_label_inds,
                                                              device)
                # loss = self.Loss(predict_scores, gt_labels, weights)
                loss = self.Loss(predict_scores, gt_labels, weight=None)
                if predict_scores.size()[-1] == 0:
                    continue

                loss = reduce_value(loss, average=True)
                mean_valid_loss = (mean_valid_loss * step + loss.detach()) / (step + 1)
                self.metric_valid.update(predict_scores, gt_labels)
                oa = self.metric_valid.tp()/data['labels'].numel()
                oa = reduce_value(torch.tensor(oa).to(device), average=True)
                mIoU = self.metric_valid.iou()[-1]
                mIoU = reduce_value(torch.tensor(mIoU).to(device), average=True)
                valid_metric = {'OA': oa, 'mIoU': mIoU}
            # 等待所有进程计算完毕
            if device != torch.device("cpu"):
                torch.cuda.synchronize(device)

        return mean_valid_loss.item(), valid_metric
    
    def load_ckpt(self, ckpt_path):
        log.info(f'Loading checkpoint {ckpt_path}')
        ckpt = torch.load(ckpt_path, map_location=self.device)
        if not exists(ckpt_path):
            raise FileNotFoundError(f'ckpt {self.cfg.model.pretrain_path} not found')
        self.model.load_state_dict(ckpt['model_state_dict'])
        total_model = sum([param.nelement() for param in self.model.parameters()])
        log.info(f'{self.cfg.model.name} has {"%.2f"%(total_model/1e6)}M params')

    def test(self):
        cfg = self.cfg
        # B, N = cfg.pipeline.test_batch_size, cfg.dataset.get_input['parameter']['num_points']
        model = self.model
        dataset = self.dataset
        device = self.device
        model.to(device)
        model.eval()
        log.info("DEVICE : {}".format(device))
        log.info("Logging in file : {}".format(self.test_log_path))
        log.addHandler(logging.FileHandler(self.test_log_path))
        self.load_ckpt(self.cfg.model.ckpt_path)
        # self.log_tests()
        batcher = self.get_batcher(device)
        test_split = dataset.get_split('test')
        self.test_split = test_split  # update_tests时调用preprocess方法

        test_sampler = test_split.sampler
        test_dataset = inputpoints(cfg=cfg.pipeline,
                                   dataset=test_split,
                                   preprocess=test_split.preprocess,
                                   use_cache=cfg.dataset.use_cache,
                                   transform=model.transform)
        test_loader = DataLoader(dataset=test_dataset, 
                                 batch_size=cfg.pipeline.test_batch_size, 
                                 sampler=get_sampler(test_sampler),
                                 collate_fn=batcher.collate_fn)
 
        self.curr_cloud_id = -1
        self.test_probs = []
        self.test_labels = []
        self.ori_test_probs = []
        self.ori_test_labels = []
        start_test_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
        log.info(f"{start_test_time} 开始测试")
        with torch.no_grad():
            for step, inputs in enumerate(test_loader):
                data, attr = inputs['data'], inputs['attr']
                # if 'features' not in data:
                #     data['features'] = torch.zeros([B,N,1])
                data = {key: ([attr.to(device) for attr in data[key]] 
                                       if isinstance(data[key], list) 
                                       else data[key].to(device)) 
                                  for key in data}
                results = self.model(data)
                self.update_tests(test_sampler, data, results, device)
                if self.complete_infer:
                    inference_result = {'predict_labels': self.ori_test_labels.pop(),
                                        'predict_scores': self.ori_test_probs.pop()}
                    gt_labels = test_split.get_data(test_sampler.cloud_id)['labels']
                    gt_labels = torch.tensor(gt_labels).to(device)
                    self.total_number = gt_labels.numel()
                    if (gt_labels > 0).any():
                        valid_scores, valid_labels = filter_valid_label(inference_result['predict_scores'],
                                                                        gt_labels,
                                                                        model.cfg.num_classes,
                                                                        dataset.cfg.ignored_label_inds,
                                                                        device)
                        self.metric_test.update(valid_scores, valid_labels)
                        # log.info(f"Accuracy : {self.metric_test.precision()}")
                        # log.info(f"IoU : {self.metric_test.iou()}")
            end_test_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
            log.info(f"{end_test_time} 结束测试")
            delta_time = get_delta_time(start_test_time, end_test_time)
            log.info(f"测试用时：{delta_time}") 
            which_ckpt = (cfg.model.ckpt_path).split('/')[-1].split('.')[-2]
            self.dataset.save_test_result(join(self.test_predict_dir, which_ckpt), inference_result['predict_labels'], attr)
            log.info(f"保存预测结果npy文件到{self.test_predict_dir}") 
            np.save(join(self.test_predict_dir, which_ckpt, 'confusion_matrix'), self.metric_test.confusion_matrix)
            self.log_tests3()

    def update_tests(self, sampler, data, results, device):
        end_threshold = 0.5
        if self.curr_cloud_id != sampler.cloud_id:
            self.curr_cloud_id = sampler.cloud_id
            num_points = sampler.possibilities[sampler.cloud_id].shape[0]
            self.pbar = tqdm(total=num_points, desc="test {}/{}".format(self.curr_cloud_id, len(self.test_split)))
            self.pbar_update = 0
            self.test_probs.append(torch.zeros([num_points, self.model.cfg.num_classes], dtype=torch.float16).to(device))
            self.test_labels.append(torch.zeros([num_points], dtype=torch.int16).to(device))
            self.complete_infer = False
        this_possiblility = sampler.possibilities[self.curr_cloud_id]
        self.pbar.update(this_possiblility[this_possiblility > end_threshold].shape[0] - self.pbar_update)
        self.pbar_update = this_possiblility[this_possiblility > end_threshold].shape[0]
        self.test_probs[self.curr_cloud_id], self.test_labels[self.curr_cloud_id] = self.update_probs(data['point_inds'], results, 
                                                                                                      self.test_probs[self.curr_cloud_id],
                                                                                                      self.test_labels[self.curr_cloud_id])
        if this_possiblility[this_possiblility > end_threshold].shape[0] == this_possiblility.shape[0]:
            proj_inds = data['proj_inds'].type(torch.long)
            if proj_inds is None:
                proj_inds = np.arange(self.test_probs[self.curr_cloud_id].shape[0])
            self.ori_test_probs.append(self.test_probs[self.curr_cloud_id][proj_inds])
            self.ori_test_labels.append(self.test_labels[self.curr_cloud_id][proj_inds])
            self.complete_infer = True
        
    def update_probs(self, point_inds, results, test_probs, test_labels):
        self.test_smooth = 0.95
        for b in range(results.size()[0]):
            result = torch.reshape(results[b], (-1, self.cfg.model.num_classes))
            probs = F.softmax(result, dim=-1).type(torch.float16)
            labels = torch.argmax(probs, 1).type(torch.int16)
            inds = point_inds[b].type(torch.long)
            test_probs[inds] = self.test_smooth * test_probs[inds] + (1 - self.test_smooth) * probs
            test_labels[inds] = labels
        return test_probs, test_labels

    def log_tests1(self):
        cfg = self.cfg
        label_to_names = self.dataset.label_to_names
        ignored_label_inds = cfg.dataset.ignored_label_inds
        for i in ignored_label_inds:
            del label_to_names[i]
        label_values = label_to_names.values()
        idx_to_names = {i: n for i, n in enumerate(label_values)}
                    
        metrics = self.metric_test
        model_name = cfg.model.name
        dataset_name = cfg.dataset.name
        max_length = 0

        for key, value in idx_to_names.items():
            if len(value)>max_length:
                max_length=len(value)
        a = '            +{}+-----------+-----------+\n'.format('-'*max_length)
        b = '            |Class{}|precision (%)|   IoU (%) |\n'.format(' '*(max_length-5))
        c = a
        for key, value in idx_to_names.items():
            precision = round(metrics.precision()[key]*100, 2)
            iou = round(metrics.iou()[key]*100, 2)
            m = '            |{}|{}{:.2f}   |{}{:.2f}   |\n'\
                .format(value+' '*(max_length-len(value)), ' '*(5-len(str(int(precision)))), precision,  ' '*(5-len(str(int(iou)))), iou) # 小数末尾是0时不计入字符串，故使用整数部分计算空格长度
            c += m+a
        out = a + b + c
        log.info(('{}对{}中各类别分割指标：\n'+out).format(model_name, dataset_name))

        mprecision = round(metrics.precision()[-1]*100, 2)
        miou = round(metrics.iou()[-1]*100, 2)
        log.info('{}在{}上的平均分割指标：\n \
                +-----------+-----------+\n \
                |mprecisi(%)|  mIoU(%)  |\n \
                +-----------+-----------+\n \
                |{}{:.2f}   |{}{:.2f}   |\n \
                +-----------+-----------+'.format(model_name, dataset_name, ' '*(5-len(str(int(mprecision)))), mprecision, ' '*(5-len(str(int(miou)))), miou))
    def log_tests2(self):
        cfg = self.cfg
        label_to_names = self.dataset.label_to_names
        ignored_label_inds = cfg.dataset.ignored_label_inds
        for i in ignored_label_inds:
            del label_to_names[i]
        label_values = label_to_names.values()
        idx_to_names = {i: n for i, n in enumerate(label_values)}
                    
        metrics = self.metric_test
        model_name = cfg.model.name
        dataset_name = cfg.dataset.name

        k=3
        # mprecision = round(metrics.precision()[-1]*100, 2)
        mprecision = metrics.precision()[-1]*100
        mprecision = ('%.2f'%mprecision)
        lmprecision = len(str(mprecision))
        # lmprecision = 3+len(str(int(mprecision))) # 按上面的赋值方法直接str可以保留末尾的0
        # miou = round(metrics.iou()[-1]*100, 2)
        miou = metrics.iou()[-1]*100
        miou = ('%.2f'%miou)
        lmiou = len(str(miou))
        # lmiou = 3+len(str(int(miou)))
        a = '+{}+{}+'.format('-'*(2*k+6),'-'*10)
        b = '|{}Metric{}|   Mean   |'.format(' '*k, ' '*k)
        c = '|{}precision(%){}|{}{}{}|'.format(' '*k, ' '*k, ' '*((10-lmprecision)//2), mprecision, ' '*(10-lmprecision-((10-lmprecision)//2)))
        d = '|{}IoU(%){}|{}{}{}|'.format(' '*k, ' '*k, ' '*((10-lmiou)//2), miou, ' '*(10-lmiou-((10-lmiou)//2)))
        for key, value in idx_to_names.items():
            precision = ('%.2f'%(metrics.precision()[key]*100))
            iou = ('%.2f'%(metrics.iou()[key]*100))
            kvk = 2*k+len(value)
            lprecision = len(precision)
            liou = len(iou)
            a += '{}+'.format('-'*kvk)
            b += '{}{}{}|'.format(' '*k, value, ' '*k)
            c += '{}{}{}|'.format(' '*((kvk-lprecision)//2), precision, ' '*(kvk-lprecision-((kvk-lprecision)//2)))
            d += '{}{}{}|'.format(' '*((kvk-liou)//2), iou, ' '*(kvk-liou-((kvk-liou)//2)))
        a += '\n'
        b += '\n'+a
        c += '\n'+a
        d += '\n'+a
        out = a+b+c+d
        log.info(('{}在{}的平均及各类别分割指标：\n'+out).format(model_name, dataset_name))
    def log_tests3(self):
        cfg = self.cfg
        label_to_names = self.dataset.label_to_names
        # ignored_label_inds = cfg.dataset.ignored_label_inds
        # for i in ignored_label_inds:
        #     del label_to_names[i]
        label_values = label_to_names.values()
        idx_to_names = {i: n for i, n in enumerate(label_values)}
                    
        metrics = self.metric_test
        model_name = cfg.model.name
        dataset_name = cfg.dataset.name

        k=3
        # tp = metrics.tp()
        # oa = (tp/self.total_number)*100
        oa = metrics.overall_accuracy()*100
        oa = ('%.2f'%oa)
        loa = len(str(oa))
        miou = metrics.iou()[-1]*100
        miou = ('%.2f'%miou)
        lmiou = len(str(miou))
        a = '+{}+{}+'.format('-'*(10),'-'*10)
        b = '|    OA    |   mIoU   |'.format(' '*k, ' '*k)
        c = '|{}{}{}|{}{}{}|'.format(' '*((10-loa)//2), oa, ' '*(10-loa-((10-loa)//2))\
                                    ,' '*((10-lmiou)//2), miou, ' '*(10-lmiou-((10-lmiou)//2)))
        for key, value in idx_to_names.items():
            iou = ('%.2f'%(metrics.iou()[key]*100))
            kvk = 2*k+len(value)
            liou = len(iou)
            a += '{}+'.format('-'*kvk)
            b += '{}{}{}|'.format(' '*k, value, ' '*k)
            c += '{}{}{}|'.format(' '*((kvk-liou)//2), iou, ' '*(kvk-liou-((kvk-liou)//2)))
        a += '\n'
        b += '\n'+a
        c += '\n'+a
        out = a+b+c
        log.info(('{}在{}的分割结果（百分数）：\n'+out).format(model_name, dataset_name))
if __name__ == '__main__':
    args = argparse.ArgumentParser(description='训练或测试模型')
    args.add_argument('--cfg_path', help='配置文件路径', default='/data/panjieyun/PCSemanticSegmentation/xyz.yml')
    args.add_argument('--split', help='train or test'
                        , default='train'
                        # , default='test'
                        )
    args.add_argument('--ckpt_path', help='待测试ckpt路径'
    # , default='/dyao ata/panjieyun/mypointcloudseg/all_results/RandLA-Net_Toronto3D_open3d_randlanet/checkpoints/train_best.pth'
    )
    args.add_argument('--device', default='cuda', help='device id (i.e. 0 or 0,1 or cpu)')
    args.add_argument('--distributed', action='store_true', help='是否采用分布式训练（单机多卡）')  # 不输入时默认为false, 输入后为true
    args.add_argument('--world_size', help='显卡数量', nargs='?', type=int, default=1)
    args.add_argument('--pretrain_path', type=str, default='pretrain.pth',help='initial weights path')
    args.add_argument('--syncBN', type=bool, default=False)
    args.add_argument('--freeze-layers', type=bool, default=False)
    args = args.parse_args()
    # import pdb;pdb.set_trace()
    if args.split == 'train': 
        if args.distributed:
            mp.set_start_method('spawn')
            world_size = args.world_size
            processes = []
            for rank in range(world_size):
                p = Process(target=PointCloudSeg(args).train, args = (rank,))
                p.start()
                processes.append(p)
            for p in processes:
                p.join()
        else:
            PointCloudSeg(args).train(rank=0)
    elif args.split == 'test':
        PointCloudSeg(args).test()