import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from GilAAADataset import *
from engine import train_one_epoch, evaluate
import utils
import pickle
import cv2
import transforms as T
from torchvision.transforms import functional as F

from gil_eval import *

def get_instance_segmentation_model(num_classes):
    # load an instance segmentation model pre-trained on COCO
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)

    # get the number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # now get the number of input features for the mask classifier
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    # and replace the mask predictor with a new one
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,
                                                       hidden_layer,
                                                       num_classes)

    return model

def main(mode):
    #dataset
    dataset = GilAAADataset('AAAGilDatasetPos', get_transform(train=True))
    dataset_test = GilAAADataset('AAAGilDatasetPos', get_transform(train=False))

    #k-fold
    # split the dataset in train and test set
    # torch.manual_seed(1)
    # indices = torch.randperm(len(dataset)).tolist()
    # dataset = torch.utils.data.Subset(dataset, indices[:-30])
    # dataset_test = torch.utils.data.Subset(dataset_test, indices[-30:])

    # leave subject out
    indices1 = list(range(0, 234))
    indices2 = list(range(234, 314))
    np.random.shuffle(indices1)
    np.random.shuffle(indices2)
    dataset = torch.utils.data.Subset(dataset, indices1)
    dataset_test = torch.utils.data.Subset(dataset_test, indices2)

    # define training and validation data loaders
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=2, shuffle=True, num_workers=4,
        collate_fn=utils.collate_fn)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=1, shuffle=False, num_workers=4,
        collate_fn=utils.collate_fn)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # our dataset has two classes only - background and ...
    num_classes = 2

    # get the model using our helper function
    model = get_instance_segmentation_model(num_classes)
    # move model to the right device
    model.to(device)

    if 'train' in mode:
        # construct an optimizer
        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.SGD(params, lr=0.005,
                                    momentum=0.9, weight_decay=0.0005)

        # and a learning rate scheduler which decreases the learning rate by
        # 10x every 3 epochs
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                       step_size=3,
                                                       gamma=0.1)

        # let's train it for 10 epochs
        num_epochs = 10

        for epoch in range(num_epochs):
            # train for one epoch, printing every 10 iterations
            train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10)
            # update the learning rate
            lr_scheduler.step()
            # evaluate on the test dataset
            evaluate(model, data_loader_test, device=device)

        torch.save(model.state_dict(), './pretrained/pretrained_weight.pth')
        torch.save(dataset_test, './pretrained/dataset_test.pth')


    if 'test' in mode:
        model.load_state_dict(torch.load('./pretrained/pretrained_weight.pth'))

        # img, _ = dataset_test[13]
        # print(img.shape)
        check_dir = '../AAAGilDatasetPos/'
        subject = '05390853_20200821'
        img_idx = 96


        img_name = 'AAAGilDatasetPos/raw/' + subject + '_%04d.png'%img_idx
        mask_name = 'AAAGilDatasetPos/mask/' + subject + '_%04d.png'%img_idx
        #
        # img = cv2.imread('AAAGilDatasetPos/img/img_0001_0010.png', 1)
        # seg = cv2.imread('AAAGilDatasetPos/mask/mask_0001_001 0.png', 1)

        img = Image.open(img_name).convert("RGB")
        mask_gt = Image.open(mask_name).convert("RGB")
        img = F.to_tensor(img)
        # put the model in evaluation mode
        model.eval()
        with torch.no_grad():
            prediction = model([img.to(device)])

        im = Image.fromarray(img.mul(255).permute(1, 2, 0).byte().numpy())
        # im.show()
        mask = Image.fromarray(prediction[0]['masks'][0, 0].mul(255).byte().cpu().numpy())
        # mask.show()
        # mask_gt.show()

        img_in = np.array(im)
        img_mask = np.array(mask)

        img_mask_gt = np.array(mask_gt)

        img_mask[img_mask > 50] = 255
        img_mask_gt_gray = cv2.cvtColor(img_mask_gt, cv2.COLOR_BGR2GRAY)

        # cv2.imshow('input', img_in)
        # cv2.imshow('mask result', img_mask)
        # cv2.imshow('mask gt', img_mask_gt)

        # segment evaluation
        overlap, jaccard, dice, fn, fp = eval_segmentation(img_mask, img_mask_gt_gray)
        print('[segmentation evaluation] overlab:%.4f jaccard:%.4f dice:%.4f fn:%.4f fp:%.4f'%(overlap, jaccard, dice, fn, fp))


        img_result = np.concatenate([img_mask, img_mask_gt_gray], axis=1)

        img_overlap = img_mask_gt.copy()
        img_overlap[:, :, 0] = 0
        img_overlap[:,:,1] = img_mask
        cv2.imshow('result', img_result)
        cv2.imshow('overlap', img_overlap)
        cv2.waitKey(0)

    if 'eval' in mode:
        model.load_state_dict(torch.load('./pretrained/pretrained_weight.pth'))
        dataset_test = torch.load('./pretrained/dataset_test.pth')

        num_test = len(dataset_test.indices)
        mat_eval = np.zeros((num_test, 5), np.float32)


        for i in range(num_test):
            img_name = 'AAAGilDatasetPos/raw/' + dataset_test.dataset.imgs[dataset_test.indices[i]]
            mask_name = 'AAAGilDatasetPos/mask/' + dataset_test.dataset.imgs[dataset_test.indices[i]]

            img = Image.open(img_name).convert("RGB")
            mask_gt = Image.open(mask_name).convert("RGB")
            img_rgb = np.array(img)
            img = F.to_tensor(img)

            model.eval()
            with torch.no_grad():
                prediction = model([img.to(device)])

            im = Image.fromarray(img.mul(255).permute(1, 2, 0).byte().numpy())
            mask = Image.fromarray(prediction[0]['masks'][0, 0].mul(255).byte().cpu().numpy())


            img_mask = np.array(mask)
            img_mask_gt = np.array(mask_gt)
            img_mask[img_mask >= 150] = 255
            img_mask[img_mask < 150] = 0
            img_mask_gt_gray = cv2.cvtColor(img_mask_gt, cv2.COLOR_BGR2GRAY)

            overlap, jaccard, dice, fn, fp = eval_segmentation(img_mask, img_mask_gt_gray)
            print('[segmentation evaluation] overlab:%.4f jaccard:%.4f dice:%.4f fn:%.4f fp:%.4f' % (
            overlap, jaccard, dice, fn, fp))

            mat_eval[i, :] = [overlap, jaccard, dice, fn, fp]


            img_gray  = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)

            img_result = np.concatenate([img_gray, img_mask, img_mask_gt_gray], axis=1)


            img_overlap = img_mask_gt.copy()
            img_overlap[:, :, 0] = 0
            img_overlap[:, :, 1] = img_mask



            # cv2.imwrite('result/' + dataset_test.dataset.imgs[dataset_test.indices[i]].replace('.png', '_raw.png'), img_gray)
            # cv2.imwrite('result/' + dataset_test.dataset.imgs[dataset_test.indices[i]].replace('.png', '_mask_predict.png'), img_mask)
            # cv2.imwrite('result/' + dataset_test.dataset.imgs[dataset_test.indices[i]].replace('.png', '_mask_gt.png'), img_mask_gt_gray)
            # cv2.imwrite('result/' + dataset_test.dataset.imgs[dataset_test.indices[i]].replace('.png', '_overlap.png'), img_overlap)
            # cv2.imwrite('result/' + dataset_test.dataset.imgs[dataset_test.indices[i]].replace('.png', '_all.png'), img_result)

            cv2.imshow('result', img_result)
            cv2.imshow('overlap', img_overlap)
            cv2.waitKey(0)

        print(mat_eval.mean(axis=0))



if __name__ == '__main__':
    main('eval')