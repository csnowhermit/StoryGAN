# StoryGAN

官方README.md见：ORIGIN-README.md

官方版只支持GPU多卡训练方式，该版本改为CPU版训练方式。

其余用法与官方相同。

## Details:

### 1、配置文件

​	cfg/clevr.yml文件

``` python
GPU_ID: ''    # 用cpu直接留空即可
WORKERS: 0    # 对应torch.utils.data.DataLoader中num_workers参数，Windows下直接写0即可。
	# 注意：config.py里也有__C.WORKERS字段，最终以clevr.yml文件为准。
VIDEO_LEN: 4    # 视频的长度，图像文件标号1~4，这里应该写4，而不是5。
```

​	miscc/config.py文件

``` python
__C.GPU_ID = ''
__C.CUDA = False    # 用cpu训练的话以上留空，该项写为False
__C.WORKERS = 0    # windows下直接写0。
```

### 2、trainer.py：GanTrainer类

#### 2.1、__init__()方法：分别处理cpu和gpu的参数初始化

``` python
def __init__(self, output_dir, ratio = 1.0, test_dir = None):
    self.model_dir = os.path.join(output_dir, 'Model')
    self.image_dir = os.path.join(output_dir, 'Image')
    self.log_dir = os.path.join(output_dir, 'Log')
    mkdir_p(self.model_dir)
    mkdir_p(self.image_dir)
    mkdir_p(self.log_dir)
    self.video_len = cfg.VIDEO_LEN
    self.max_epoch = cfg.TRAIN.MAX_EPOCH
    self.snapshot_interval = cfg.TRAIN.SNAPSHOT_INTERVAL
    self.test_dir = test_dir

    # 以下分别处理cpu和gpu的情况
    if cfg.CUDA is False:    # cfg.CUDA False，说明用cpu
        self.gpus = -1
        self.num_gpus = 0

        self.imbatch_size = cfg.TRAIN.IM_BATCH_SIZE
        self.stbatch_size = cfg.TRAIN.ST_BATCH_SIZE
        self.ratio = ratio
    else:
        s_gpus = cfg.GPU_ID.split(',')
        self.gpus = [int(ix) for ix in s_gpus]
        self.num_gpus = len(self.gpus)
        self.imbatch_size = cfg.TRAIN.IM_BATCH_SIZE * self.num_gpus
        self.stbatch_size = cfg.TRAIN.ST_BATCH_SIZE * self.num_gpus
        self.ratio = ratio
        torch.cuda.set_device(self.gpus[0])
        cudnn.benchmark = True
```

#### 2.2、train()方法：对损失的计算

``` python
# D模块
if cfg.CUDA:
    im_errD, im_errD_real, im_errD_wrong, im_errD_fake, accD = compute_discriminator_loss(netD_im, im_real_imgs, im_fake,
                                   im_real_labels, im_fake_labels, im_catelabel,
                                   im_mu, self.gpus)

    st_errD, st_errD_real, st_errD_wrong, st_errD_fake, _ = compute_discriminator_loss(netD_st, st_real_imgs, st_fake,
                                   st_real_labels, st_fake_labels, st_catelabel,
                                   c_mu, self.gpus)
else:
    im_errD, im_errD_real, im_errD_wrong, im_errD_fake, accD = compute_discriminator_loss_cpu(netD_im, im_real_imgs, im_fake,
                                   im_real_labels, im_fake_labels, im_catelabel,
                                   im_mu)

    st_errD, st_errD_real, st_errD_wrong, st_errD_fake, _ = compute_discriminator_loss_cpu(netD_st, st_real_imgs, st_fake,
                                   st_real_labels, st_fake_labels, st_catelabel,
                                   c_mu)

# G模块
if cfg.CUDA:
    im_errG, accG = compute_generator_loss(netD_im, im_fake, im_real_labels, im_catelabel, im_mu, self.gpus)
    st_errG, _ = compute_generator_loss(netD_st, st_fake, st_real_labels, st_catelabel, c_mu, self.gpus)
else:
    im_errG, accG = compute_generator_loss_cpu(netD_im, im_fake, im_real_labels, im_catelabel, im_mu)
    st_errG, _ = compute_generator_loss_cpu(netD_st, st_fake, st_real_labels, st_catelabel, c_mu)
```

### 3、miscc/utils.py：

#### 3.1、多卡并行和CPU计算的区别

``` python
# 以下用于GPU多卡情况下计算
fake_features = nn.parallel.data_parallel(netD, (fake_imgs), gpus)

# cpu采用如下方式计算：
fake_features = netD(fake_imgs)
```

​	补充点：

PyTorch多卡训练有以下三种方式：

（1）DataParallel：pytorch自带的多卡训练方法，并不是完全的并行计算。只是数据在两张卡上并行计算，模型的保存和loss计算仍集中在一张卡上，算完后同步给其他卡。这也导致了用这种方法两张卡的现存占用不一致；

（2）Pytorch-Encoding：第三方包，解决了loss计算不并行的问题，除此之外还有其他好用的办法。链接：https://github.com/zhanghang1989/PyTorch-Encoding

（3）distributedDataparallel：真正的多卡并行计算。每个GPU都会对自己分配到的数据进行求导计算，然后将结果传递给下一个GPU。这与DataParallel将所有数据汇聚到一个GPU求导，计算loss和更新参数不同。

#### 3.2、G、D模块损失计算改造

``` python
'''
	G模块GPU计算损失
'''
def compute_generator_loss(netD, fake_imgs, real_labels, fake_catelabels, conditions, gpus):
    ratio = 0.4
    criterion = nn.BCELoss()
    cate_criterion =nn.MultiLabelSoftMarginLoss()
    cond = conditions.detach()

    fake_features = nn.parallel.data_parallel(netD, (fake_imgs), gpus)
    # fake pairs
    inputs = (fake_features, cond)
    fake_logits = nn.parallel.data_parallel(netD.get_cond_logits, inputs, gpus)
    errD_fake = criterion(fake_logits, real_labels)
    if netD.get_uncond_logits is not None:
        fake_logits = \
            nn.parallel.data_parallel(netD.get_uncond_logits,
                                      (fake_features), gpus)
        uncond_errD_fake = criterion(fake_logits, real_labels)
        errD_fake += uncond_errD_fake
    acc = 0
    if netD.cate_classify is not None:
        cate_logits = nn.parallel.data_parallel(netD.cate_classify, fake_features, gpus)
        cate_logits = cate_logits.squeeze()
        errD_fake = errD_fake + ratio * cate_criterion(cate_logits, fake_catelabels)
        acc = accuracy_score(fake_catelabels.cpu().data.numpy().astype('int32'), 
            (cate_logits.cpu().data.numpy() > 0.5).astype('int32'))
    return errD_fake, acc

'''
	G模块CPU计算损失
'''
def compute_generator_loss_cpu(netD, fake_imgs, real_labels, fake_catelabels, conditions):
    ratio = 0.4
    criterion = nn.BCELoss()
    cate_criterion =nn.MultiLabelSoftMarginLoss()
    cond = conditions.detach()

    # fake_features = nn.parallel.data_parallel(netD, (fake_imgs))
    fake_features = netD(fake_imgs)

    # fake pairs
    inputs = (fake_features, cond)
    # fake_logits = nn.parallel.data_parallel(netD.get_cond_logits, inputs)
    fake_logits = netD.get_cond_logits(fake_features, cond)

    errD_fake = criterion(fake_logits, real_labels)
    if netD.get_uncond_logits is not None:
        fake_logits = netD.get_uncond_logits(fake_features)
        uncond_errD_fake = criterion(fake_logits, real_labels)
        errD_fake += uncond_errD_fake
    acc = 0
    if netD.cate_classify is not None:
        cate_logits = netD.cate_classify(fake_features)
        cate_logits = cate_logits.squeeze()
        errD_fake = errD_fake + ratio * cate_criterion(cate_logits, fake_catelabels)
        acc = accuracy_score(fake_catelabels.cpu().data.numpy().astype('int32'),
            (cate_logits.cpu().data.numpy() > 0.5).astype('int32'))
    return errD_fake, acc
```

​	D模块类似。



