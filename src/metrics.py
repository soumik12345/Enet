import torch


def nanmean(x):
    '''Arithmatic Mean ignoring NaNs'''
    return torch.mean(x[x == x])


def fast_hist(true, pred, num_classes):
    '''Confusion Matrix'''
    mask = (true >= 0) & (true < num_classes)
    hist = torch.bincount(
        num_classes * true[mask] + pred[mask],
        minlength=num_classes ** 2,
    ).reshape(num_classes, num_classes).float()
    return hist


def overall_pixel_accuracy(hist):
    '''Overall Pixel Accuracy'''
    correct = torch.diag(hist).sum()
    total = hist.sum()
    overall_acc = correct / (total + 1e-10)
    return overall_acc


def per_class_pixel_accuracy(hist):
    '''Per Class Pixel Accuracy'''
    correct_per_class = torch.diag(hist)
    total_per_class = hist.sum(dim=1)
    per_class_acc = correct_per_class / (total_per_class + 1e-10)
    avg_per_class_acc = nanmean(per_class_acc)
    return avg_per_class_acc


def jaccard_index(hist):
    '''Jaccard Index'''
    A_inter_B = torch.diag(hist)
    A = hist.sum(dim=1)
    B = hist.sum(dim=0)
    jaccard = A_inter_B / (A + B - A_inter_B + 1e-10)
    avg_jacc = nanmean(jaccard)
    return avg_jacc


def dice_coefficient(hist):
    '''Dice Coefficient'''
    A_inter_B = torch.diag(hist)
    A = hist.sum(dim=1)
    B = hist.sum(dim=0)
    dice = (2 * A_inter_B) / (A + B + 1e-10)
    avg_dice = nanmean(dice)
    return avg_dice