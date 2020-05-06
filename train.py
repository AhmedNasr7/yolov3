import argparse

import torch.distributed as dist
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler

import test  # import test.py to get mAP after each epoch
from models import *
from utils.datasets import *
from utils.utils import *

mixed_precision = True
try:  # Mixed precision training https://github.com/NVIDIA/apex
    from apex import amp
except:
    mixed_precision = False  # not installed

wdir = 'weights' + os.sep  # weights dir
last = wdir + 'last.pt'
best = wdir + 'best.pt'
results_file = 'results.txt'

# Hyperparameters (results68: 59.9 mAP@0.5 yolov3-spp-416) https://github.com/ultralytics/yolov3/issues/310

hyp = {'giou': 3.54,  # giou loss gain
       'cls': 37.4,  # cls loss gain
       'cls_pw': 1.0,  # cls BCELoss positive_weight
       'obj': 49.5,  # obj loss gain (*=img_size/320 if img_size != 320)
       'obj_pw': 1.0,  # obj BCELoss positive_weight
       'iou_t': 0.225,  # iou training threshold
       'lr0': 0.00579,  # initial learning rate (SGD=5E-3, Adam=5E-4)
       'lrf': -4.,  # final LambdaLR learning rate = lr0 * (10 ** lrf)
       'momentum': 0.937,  # SGD momentum
       'weight_decay': 0.000484,  # optimizer weight decay
       'fl_gamma': 0.5,  # focal loss gamma
       'hsv_h': 0.0138,  # image HSV-Hue augmentation (fraction)
       'hsv_s': 0.678,  # image HSV-Saturation augmentation (fraction)
       'hsv_v': 0.36,  # image HSV-Value augmentation (fraction)
       'degrees': 1.98,  # image rotation (+/- deg)
       'translate': 0.05,  # image translation (+/- fraction)
       'scale': 0.05,  # image scale (+/- gain)
       'shear': 0.641}  # image shear (+/- deg)

# Overwrite hyp with hyp*.txt (optional)
f = glob.glob('hyp*.txt')
if f:
    print('Using %s' % f[0])
    for k, v in zip(hyp.keys(), np.loadtxt(f[0])):
        hyp[k] = v


def train():
    cfg = opt.cfg
    data = opt.data
    img_size, img_size_test = opt.img_size if len(opt.img_size) == 2 else opt.img_size * 2  # train, test sizes
    epochs = opt.epochs  # 500200 batches at bs 64, 117263 images = 273 epochs
    batch_size = opt.batch_size
    accumulate = opt.accumulate  # effective bs = batch_size * accumulate = 16 * 4 = 64
    weights = opt.weights  # initial training weights

    # Initialize
    init_seeds()
    if opt.multi_scale:
        img_sz_min = round(img_size / 32 / 1.5)
        img_sz_max = round(img_size / 32 * 1.5)
        img_size = img_sz_max * 32  # initiate with maximum multi_scale size
        print('Using multi-scale %g - %g' % (img_sz_min * 32, img_size))

    # Configure run
    data_dict = parse_data_cfg(data)
    train_path = data_dict['train']
    test_path = data_dict['valid']
    nc = 1 if opt.single_cls else int(data_dict['classes'])  # number of classes

    # Remove previous results
    for f in glob.glob('*_batch*.jpg') + glob.glob(results_file):
        os.remove(f)

    # Initialize model
    model = Darknet(cfg, arc=opt.arc).to(device)

    # Optimizer
    pg0, pg1, pg2 = [], [], []  # optimizer parameter groups
    for k, v in dict(model.named_parameters()).items():
        if '.bias' in k:
            pg2 += [v]  # biases
        elif 'Conv2d.weight' in k:
            pg1 += [v]  # apply weight_decay
        else:
            pg0 += [v]  # all else

    if opt.adam:
        # hyp['lr0'] *= 0.1  # reduce lr (i.e. SGD=5E-3, Adam=5E-4)
        optimizer = optim.Adam(pg0, lr=opt.lr)
        # optimizer = AdaBound(pg0, lr=hyp['lr0'], final_lr=0.1)
    else:
        optimizer = optim.SGD(pg0, lr=opt.lr, momentum=opt.momentum, nesterov=True)
    optimizer.add_param_group({'params': pg1, 'weight_decay': opt.decay})  # add pg1 with weight_decay
    optimizer.add_param_group({'params': pg2})  # add pg2 (biases)
    del pg0, pg1, pg2

    # https://github.com/alphadl/lookahead.pytorch
    # optimizer = torch_utils.Lookahead(optimizer, k=5, alpha=0.5)

    start_epoch = 0
    best_fitness = float('inf')
    attempt_download(weights)
    if weights.endswith('.pt'):  # pytorch format
        # possible weights are '*.pt', 'yolov3-spp.pt', 'yolov3-tiny.pt' etc.
        chkpt = torch.load(weights, map_location=device)

        # load model
        try:
            chkpt['model'] = {k: v for k, v in chkpt['model'].items() if model.state_dict()[k].numel() == v.numel()}
            model.load_state_dict(chkpt['model'], strict=False)
        except KeyError as e:
            s = "%s is not compatible with %s. Specify --weights '' or specify a --cfg compatible with %s. " \
                "See https://github.com/ultralytics/yolov3/issues/657" % (opt.weights, opt.cfg, opt.weights)
            raise KeyError(s) from e

        # load optimizer
        if chkpt['optimizer'] is not None:
            optimizer.load_state_dict(chkpt['optimizer'])
            best_fitness = chkpt['best_fitness']

        # load results
        if chkpt.get('training_results') is not None:
            with open(results_file, 'w') as file:
                file.write(chkpt['training_results'])  # write results.txt

        start_epoch = chkpt['epoch'] + 1
        del chkpt

    elif len(weights) > 0:  # darknet format
        # possible weights are '*.weights', 'yolov3-tiny.conv.15',  'darknet53.conv.74' etc.
        load_darknet_weights(model, weights)

    # Scheduler https://github.com/ultralytics/yolov3/issues/238
    # lf = lambda x: 1 - x / epochs  # linear ramp to zero
    # lf = lambda x: 10 ** (hyp['lrf'] * x / epochs)  # exp ramp
    # lf = lambda x: 1 - 10 ** (hyp['lrf'] * (1 - x / epochs))  # inverse exp ramp
    # scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
    # scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=range(59, 70, 1), gamma=0.8)  # gradual fall to 0.1*lr0
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[round(opt.epochs * x) for x in [0.8, 0.9]], gamma=0.1)
    scheduler.last_epoch = start_epoch - 1

    # # Plot lr schedule
    # y = []
    # for _ in range(epochs):
    #     scheduler.step()
    #     y.append(optimizer.param_groups[0]['lr'])
    # plt.plot(y, label='LambdaLR')
    # plt.xlabel('epoch')
    # plt.ylabel('LR')
    # plt.tight_layout()
    # plt.savefig('LR.png', dpi=300)

    # Mixed precision training https://github.com/NVIDIA/apex
    if mixed_precision:
        model, optimizer = amp.initialize(model, optimizer, opt_level='O1', verbosity=0)

    # Initialize distributed training
    if device.type != 'cpu' and torch.cuda.device_count() > 1:
        dist.init_process_group(backend='nccl',  # 'distributed backend'
                                init_method='tcp://127.0.0.1:9999',  # distributed training init method
                                world_size=1,  # number of nodes for distributed training
                                rank=0)  # distributed training node rank
        model = torch.nn.parallel.DistributedDataParallel(model, find_unused_parameters=True)
        model.yolo_layers = model.module.yolo_layers  # move yolo layer indices to top level

    # Dataset
    dataset = LoadImagesAndLabels(train_path, img_size, batch_size,
                                  augment=True,
                                  hyp=hyp,  # augmentation hyperparameters
                                  rect=opt.rect,  # rectangular training
                                  cache_labels=True,
                                  cache_images=opt.cache_images,
                                  single_cls=opt.single_cls)

    # Dataloader
    batch_size = min(batch_size, len(dataset))
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=batch_size,
                                             num_workers=nw,
                                             shuffle=not opt.rect,  # Shuffle=True unless rectangular training is used
                                             pin_memory=True,
                                             collate_fn=dataset.collate_fn)

    # Testloader
    testloader = torch.utils.data.DataLoader(LoadImagesAndLabels(test_path, img_size_test, batch_size * 2,
                                                                 hyp=hyp,
                                                                 rect=True,
                                                                 cache_labels=True,
                                                                 cache_images=opt.cache_images,
                                                                 single_cls=opt.single_cls),
                                             batch_size=batch_size * 2,
                                             num_workers=nw,
                                             pin_memory=True,
                                             collate_fn=dataset.collate_fn)

    # Start training
    nb = len(dataloader)
    prebias = start_epoch == 0
    model.nc = nc  # attach number of classes to model
    model.arc = opt.arc  # attach yolo architecture
    model.hyp = hyp  # attach hyperparameters to model
    model.class_weights = labels_to_class_weights(dataset.labels, nc).to(device)  # attach class weights
    maps = np.zeros(nc)  # mAP per class
###################################################################################################
    names = load_classes(data_dict['names'])  # class names
    iouv = torch.linspace(0.5, 0.95, 10).to(device)  # iou vector for mAP@0.5:0.95
    iouv = iouv[0].view(1)  # comment for mAP@0.5:0.95
    niou = iouv.numel()

###################################################################################################
    # torch.autograd.set_detect_anomaly(True)
    results = (0, 0, 0, 0, 0, 0, 0)  # 'P', 'R', 'mAP', 'F1', 'val GIoU', 'val Objectness', 'val Classification'
    t0 = time.time()
    torch_utils.model_info(model, report='summary')  # 'full' or 'summary'
    print('Using %g dataloader workers' % nw)
    print('Starting training for %g epochs...' % epochs)
    for epoch in range(start_epoch, epochs):  # epoch ------------------------------
        model.train()
        seen = 0
        # Prebias
        if prebias:
            if epoch < 3:  # prebias
                ps = 0.1, 0.9  # prebias settings (lr=0.1, momentum=0.9)
            else:  # normal training
                ps = hyp['lr0'], hyp['momentum']  # normal training settings
                print_model_biases(model)
                prebias = False

            # Bias optimizer settings
            optimizer.param_groups[2]['lr'] = opt.lr
            if optimizer.param_groups[2].get('momentum') is not None:  # for SGD but not Adam
                optimizer.param_groups[2]['momentum'] = opt.momentum

        # Update image weights (optional)
        if dataset.image_weights:
            w = model.class_weights.cpu().numpy() * (1 - maps) ** 2  # class weights
            image_weights = labels_to_image_weights(dataset.labels, nc=nc, class_weights=w)
            dataset.indices = random.choices(range(dataset.n), weights=image_weights, k=dataset.n)  # rand weighted idx

        mloss = torch.zeros(4).to(device)  # mean losses
        pbar = tqdm(enumerate(dataloader), total=nb)  # progress bar
###############################################################################################

        s = ('%20s' + '%10s' * 6) % ('Class', 'Images', 'Targets', 'P', 'R', 'mAP@0.5', 'F1')
        p, r, f1, mp, mr, map, mf1 = 0., 0., 0., 0., 0., 0., 0.
        detailed_loss = torch.zeros(3)
        stats, ap, ap_class = [], [], []

###############################################################################################
        
        print(('\n' + '%10s' * 8) % ('Epoch', 'gpu_mem', 'GIoU', 'obj', 'cls', 'total', 'targets', 'img_size'))
        for i, (imgs, targets, paths, _) in pbar:  # batch -------------------------------------------------------------
            ni = i + nb * epoch  # number integrated batches (since train start)
            imgs = imgs.to(device).float() / 255.0  # uint8 to float32, 0 - 255 to 0.0 - 1.0
            targets = targets.to(device)
            _, _, height, width = imgs.shape  # batch size, channels, height, width

            # Multi-Scale training
            if opt.multi_scale:
                if ni / accumulate % 10 == 0:  #  adjust (67% - 150%) every 10 batches
                    img_size = random.randrange(img_sz_min, img_sz_max + 1) * 32
                sf = img_size / max(imgs.shape[2:])  # scale factor
                if sf != 1:
                    ns = [math.ceil(x * sf / 32.) * 32 for x in imgs.shape[2:]]  # new shape (stretched to 32-multiple)
                    imgs = F.interpolate(imgs, size=ns, mode='bilinear', align_corners=False)
                _, _, height, width = imgs.shape  # batch size, channels, height, width

            # Plot images with bounding boxes
            if ni == 0:
                fname = 'train_batch%g.jpg' % i
                plot_images(imgs=imgs, targets=targets, paths=paths, fname=fname)
                if tb_writer:
                    tb_writer.add_image(fname, cv2.imread(fname)[:, :, ::-1], dataformats='HWC')

            # Hyperparameter burn-in
            # n_burn = nb - 1  # min(nb // 5 + 1, 1000)  # number of burn-in batches
            # if ni <= n_burn:
            #     for m in model.named_modules():
            #         if m[0].endswith('BatchNorm2d'):
            #             m[1].momentum = 1 - i / n_burn * 0.99  # BatchNorm2d momentum falls from 1 - 0.01
            #     g = (i / n_burn) ** 4  # gain rises from 0 - 1
            #     for x in optimizer.param_groups:
            #         x['lr'] = hyp['lr0'] * g
            #         x['weight_decay'] = hyp['weight_decay'] * g

            # Run model
            inf_out,pred = model(imgs)

            # Compute loss
            loss, loss_items = compute_loss(pred, targets, model, not prebias)
            if not torch.isfinite(loss):
                print('WARNING: non-finite loss, ending training ', loss_items)
                return results

            # Scale loss by nominal batch_size of 64
            loss *= batch_size / 64

            # Compute gradient
            if mixed_precision:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            # Accumulate gradient for x batches before optimizing
            if ni % accumulate == 0:
                optimizer.step()
                optimizer.zero_grad()


            # Print batch results
            mloss = (mloss * i + loss_items) / (i + 1)  # update mean losses
            mem = torch.cuda.memory_cached() / 1E9 if torch.cuda.is_available() else 0  # (GB)
            s = ('%10s' * 2 + '%10.3g' * 6) % (
                '%g/%g' % (epoch, epochs - 1), '%.3gG' % mem, *mloss, len(targets), img_size)
            
            if(ni%10==9):
                pbar.set_description(s)
    #################################################################################################
                ########    PRINT F1 METRICS DURING TRAINING        #########

            with torch.no_grad():
                output = non_max_suppression(inf_out,conf_thres=0.1,iou_thres=0.6)
                for si, predi in enumerate(output):
                    labels = targets[targets[:, 0] == si, 1:]
                    nl = len(labels)
                    tcls = labels[:, 0].tolist() if nl else []  # target class
                    seen += 1

                    if predi is None:        
                        if nl:
                            stats.append((torch.zeros(0, niou, dtype=torch.bool), torch.Tensor(), torch.Tensor(), tcls))
                        continue
                    # Append to text file
                    # with open('test.txt', 'a') as file:
                    #    [file.write('%11.5g' * 7 % tuple(x) + '\n') for x in pred]

                    # Clip boxes to image bounds
                    clip_coords(predi, (height, width))


                    # Assign all predictions as incorrect
                    correct = torch.zeros(len(predi), niou, dtype=torch.bool)
                    if nl:
                        detected = []  # target indices
                        tcls_tensor = labels[:, 0]
                        # target boxes
                        tbox = xywh2xyxy(labels[:, 1:5]) * torch.Tensor([width, height, width, height]).to(device)

                        # Per target class
                        for cls in torch.unique(tcls_tensor):
                            ti = (cls == tcls_tensor).nonzero().view(-1)  # prediction indices
                            pi = (cls == predi[:, 5]).nonzero().view(-1)  # target indices
                            # Search for detections
                            if len(pi):
                                # Prediction to target ious
                                ious, ind = box_iou(predi[pi, :4], tbox[ti]).max(1)  # best ious, indices

                                # Append detections
                                for j in (ious > iouv[0]).nonzero():
                                    d = ti[ind[j]]  # detected target
                                    if d not in detected:
                                        detected.append(d)
                                        correct[pi[j]] = (ious[j] > iouv).cpu()  # iou_thres is 1xn
                                        if len(detected) == nl:  # all targets already located in image
                                            break

                    # Append statistics (correct, conf, pcls, tcls)
                    stats.append((correct, predi[:, 4].cpu(), predi[:, 5].cpu(), tcls))


        stats = [np.concatenate(x, 0) for x in zip(*stats)]  # to numpy
        if len(stats):
            p, r, ap, f1, ap_class = ap_per_class(*stats)
            if niou > 1:
                p, r, ap, f1 = p[:, 0], r[:, 0], ap.mean(1), ap[:, 0]  # [P, R, AP@0.5:0.95, AP@0.5]
            mp, mr, map, mf1 = p.mean(), r.mean(), ap.mean(), f1.mean()
            nt = np.bincount(stats[3].astype(np.int64), minlength=nc)  # number of targets per class
        else:
            nt = torch.zeros(1)
        # Print results
        pf = '%20s' + '%10.3g' * 6  # print format
        print(('%20s' + '%10s' * 6) % ('Class', 'Images', 'Targets', 'P', 'R', 'mAP@0.5', 'F1'))
        print(pf % ('all', seen, nt.sum(), mp, mr, map, mf1))

        # Print results per class
        for i, c in enumerate(ap_class):
            print(pf % (names[c], seen, nt[c], p[i], r[i], ap[i], f1[i]))

#################################################################################################

        
        # Process epoch results
        final_epoch = epoch + 1 == epochs
        if epoch%opt.testevery == (opt.testevery -1) or final_epoch:  # Calculate mAP
            is_coco = any([x in data for x in ['coco.data', 'coco2014.data', 'coco2017.data']]) and model.nc == 80
            results, maps = test.test(cfg,
                                      data,
                                      batch_size=batch_size * 2,
                                      img_size=img_size_test,
                                      model=model,
                                      conf_thres=0.001 if final_epoch and is_coco else 0.1,  # 0.1 for speed
                                      iou_thres=0.6,
                                      save_json=final_epoch and is_coco,
                                      single_cls=opt.single_cls,
                                      dataloader=testloader)
        scheduler.step()

        # Write epoch results
        with open(results_file, 'a') as f:
            f.write(s + '%10.3g' * 7 % results + '\n')  # P, R, mAP, F1, test_losses=(GIoU, obj, cls)
        if len(opt.name) and opt.bucket:
            os.system('gsutil cp results.txt gs://%s/results/results%s.txt' % (opt.bucket, opt.name))

        # Write Tensorboard results
        if tb_writer:
            x = list(mloss) + list(results)
            titles = ['GIoU', 'Objectness', 'Classification', 'Train loss',
                      'Precision', 'Recall', 'mAP', 'F1', 'val GIoU', 'val Objectness', 'val Classification']
            for xi, title in zip(x, titles):
                tb_writer.add_scalar(title, xi, epoch)

        # Update best mAP
        fitness = sum(results[4:])  # total loss
        if fitness < best_fitness:
            best_fitness = fitness

        # Save training results
        save = (not opt.nosave) or (final_epoch and not opt.evolve)
        if save:
            with open(results_file, 'r') as f:
                # Create checkpoint
                chkpt = {'epoch': epoch,
                         'best_fitness': best_fitness,
                         'training_results': f.read(),
                         'model': model.module.state_dict() if type(
                             model) is nn.parallel.DistributedDataParallel else model.state_dict(),
                         'optimizer': None if final_epoch else optimizer.state_dict()}

            # Save last checkpoint
            torch.save(chkpt, last)

            # Save best checkpoint
            if best_fitness == fitness:
                torch.save(chkpt, best)

            # Save backup every 10 epochs (optional)
            # if epoch > 0 and epoch % 10 == 0:
            #     torch.save(chkpt, wdir + 'backup%g.pt' % epoch)

            # Delete checkpoint
            del chkpt

        # end epoch ----------------------------------------------------------------------------------------------------

    # end training
    n = opt.name
    if len(n):
        n = '_' + n if not n.isnumeric() else n
        fresults, flast, fbest = 'results%s.txt' % n, 'last%s.pt' % n, 'best%s.pt' % n
        os.rename('results.txt', fresults)
        os.rename(wdir + 'last.pt', wdir + flast) if os.path.exists(wdir + 'last.pt') else None
        os.rename(wdir + 'best.pt', wdir + fbest) if os.path.exists(wdir + 'best.pt') else None

        # save to cloud
        if opt.bucket:
            os.system('gsutil cp %s gs://%s/results' % (fresults, opt.bucket))
            os.system('gsutil cp %s gs://%s/weights' % (wdir + flast, opt.bucket))

    if not opt.evolve:
        plot_results()  # save as results.png
    print('%g epochs completed in %.3f hours.\n' % (epoch - start_epoch + 1, (time.time() - t0) / 3600))
    dist.destroy_process_group() if torch.cuda.device_count() > 1 else None
    torch.cuda.empty_cache()

    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=273)  # 500200 batches at bs 16, 117263 COCO images = 273 epochs
    parser.add_argument('--batch-size', type=int, default=16)  # effective bs = batch_size * accumulate = 16 * 4 = 64
    parser.add_argument('--accumulate', type=int, default=4, help='batches to accumulate before optimizing')
    parser.add_argument('--cfg', type=str, default='cfg/yolov3-spp.cfg', help='*.cfg path')
    parser.add_argument('--data', type=str, default='data/coco2017.data', help='*.data path')
    parser.add_argument('--multi-scale', action='store_true', help='adjust (67% - 150%) img_size every 10 batches')
    parser.add_argument('--img-size', nargs='+', type=int, default=[416], help='train and test image-sizes')
    parser.add_argument('--rect', action='store_true', help='rectangular training')
    parser.add_argument('--resume', action='store_true', help='resume training from last.pt')
    parser.add_argument('--nosave', action='store_true', help='only save final checkpoint')
    parser.add_argument('--testevery',type=int, default=1, help='only test final epoch')
    parser.add_argument('--evolve', action='store_true', help='evolve hyperparameters')
    parser.add_argument('--bucket', type=str, default='', help='gsutil bucket')
    parser.add_argument('--cache-images', action='store_true', help='cache images for faster training')
    parser.add_argument('--weights', type=str, default='weights/ultralytics68.pt', help='initial weights')
    parser.add_argument('--arc', type=str, default='default', help='yolo architecture')  # default, uCE, uBCE
    parser.add_argument('--name', default='', help='renames results.txt to results_name.txt if supplied')
    parser.add_argument('--device', default='', help='device id (i.e. 0 or 0,1 or cpu)')
    parser.add_argument('--adam', action='store_true', help='use adam optimizer')
    parser.add_argument('--single-cls', action='store_true', help='train as single-class dataset')
    parser.add_argument('--lr', type=float, default=0.001 ,help='learning rate')
    parser.add_argument('--momentum', type=float, default=0.95 ,help='momentum')
    parser.add_argument('--decay', type=float, default=0.001 ,help='decay')
    
    opt = parser.parse_args()
    opt.weights = last if opt.resume else opt.weights
    print(opt)
    device = torch_utils.select_device(opt.device, apex=mixed_precision, batch_size=opt.batch_size)
    if device.type == 'cpu':
        mixed_precision = False

    # scale hyp['obj'] by img_size (evolved at 320)
    # hyp['obj'] *= opt.img_size[0] / 320.

    tb_writer = None
    if not opt.evolve:  # Train normally
        try:
            # Start Tensorboard with "tensorboard --logdir=runs", view at http://localhost:6006/
            from torch.utils.tensorboard import SummaryWriter

            tb_writer = SummaryWriter()
        except:
            pass

        train()  # train normally

    else:  # Evolve hyperparameters (optional)
#        opt.testevery = True  # only test final epoch
        opt.nosave = True  # only save final checkpoint
        if opt.bucket:
            os.system('gsutil cp gs://%s/evolve.txt .' % opt.bucket)  # download evolve.txt if exists

        for _ in range(1):  # generations to evolve
            if os.path.exists('evolve.txt'):  # if evolve.txt exists: select best hyps and mutate
                # Select parent(s)
                parent = 'single'  # parent selection method: 'single' or 'weighted'
                x = np.loadtxt('evolve.txt', ndmin=2)
                n = min(8, len(x))  # number of previous results to consider
                x = x[np.argsort(-fitness(x))][:n]  # top n mutations
                w = fitness(x) - fitness(x).min()  # weights
                if parent == 'single' or len(x) == 1:
                    # x = x[random.randint(0, n - 1)]  # random selection
                    x = x[random.choices(range(n), weights=w)[0]]  # weighted selection
                elif parent == 'weighted':
                    x = (x * w.reshape(n, 1)).sum(0) / w.sum()  # weighted combination

                # Mutate
                method = 3
                s = 0.3  # 30% sigma
                np.random.seed(int(time.time()))
                g = np.array([1, 1, 1, 1, 1, 1, 1, 0, .1, 1, 0, 1, 1, 1, 1, 1, 1, 1])  # gains
                ng = len(g)
                if method == 1:
                    v = (np.random.randn(ng) * np.random.random() * g * s + 1) ** 2.0
                elif method == 2:
                    v = (np.random.randn(ng) * np.random.random(ng) * g * s + 1) ** 2.0
                elif method == 3:
                    v = np.ones(ng)
                    while all(v == 1):  # mutate until a change occurs (prevent duplicates)
                        r = (np.random.random(ng) < 0.1) * np.random.randn(ng)  # 10% mutation probability
                        v = (g * s * r + 1) ** 2.0
                for i, k in enumerate(hyp.keys()):  # plt.hist(v.ravel(), 300)
                    hyp[k] = x[i + 7] * v[i]  # mutate

            # Clip to limits
            keys = ['lr0', 'iou_t', 'momentum', 'weight_decay', 'hsv_s', 'hsv_v', 'translate', 'scale', 'fl_gamma']
            limits = [(1e-5, 1e-2), (0.00, 0.70), (0.60, 0.98), (0, 0.001), (0, .9), (0, .9), (0, .9), (0, .9), (0, 3)]
            for k, v in zip(keys, limits):
                hyp[k] = np.clip(hyp[k], v[0], v[1])

            # Train mutation
            results = train()

            # Write mutation results
            print_mutation(hyp, results, opt.bucket)

            # Plot results
            # plot_evolution_results(hyp)
