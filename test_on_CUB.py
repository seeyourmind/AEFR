import torch

classifier = torch.load('save/Classifier-CUB.pth')

with torch.no_grad():    
    acc_seen = classifier.val_gzsl(classifier.test_seen_feature, classifier.test_seen_label, classifier.seenclasses)
    acc_novel = classifier.val_gzsl(classifier.test_novel_feature, classifier.test_novel_label, classifier.novelclasses)
    H = H = (2 * acc_seen * acc_novel) / (acc_seen + acc_novel)
    print(f'CUB: U = {acc_novel}, S = {acc_seen}, H = {H}')