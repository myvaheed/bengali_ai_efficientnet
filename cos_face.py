import torch
from torch import nn
from torch.nn import Parameter
from torch.autograd import Variable
import torch.functional as F
 
class OneHot(nn.Module):
    def __init__(self, depth):
        super(OneHot,self).__init__()
        self.depth = depth
        self.ones = torch.sparse.torch.eye(depth)
    def forward(self, X_in):
        X_in = X_in.long()
        return Variable(self.ones.index_select(0,X_in.data))
    def __repr__(self):
        return self.__class__.__name__ + "({})".format(self.depth)

class CosFace2(nn.Module):
    r"""Implement of large margin cosine distance: :
    Args:
        in_features: size of each input sample
        out_features: size of each output sample
        s: norm of input feature
        m: margin
    """

    def __init__(self, in_features, out_features, s=30.0, m=0.35):
        super(CosFace2, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.one_hot = OneHot(out_features)
        self.weight = Parameter(torch.Tensor(in_features, out_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, input, label):
        y = self.one_hot(label)
        x = F.normalize(input, dim=1)
        self.weight = F.normalize(self.weight, dim=0)
        logits = torch.mm(x, self.weight)
        target_logits = logits - self.m
        logits = logits * (1 - y) + target_logits * y
        logits *= self.s
        return F.softmax(logits)

    def __repr__(self):
        return self.CosFace.__name__ + '(' \
               + 'in_features=' + str(self.in_features) \
               + ', out_features=' + str(self.out_features) \
               + ', s=' + str(self.s) \
               + ', m=' + str(self.m) + ')'
