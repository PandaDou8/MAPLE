import torch

# 定义了三个函数，用于处理张量（Tensor）数据。这些函数主要用于排序、计数和获取Top-K元素

# 接受一个输入张量列表，并返回一个索引张量，该索引张量按照输入张量的顺序对输入张量进行排序
def multikey_argsort(inputs, descending=False, break_tie=False):
    if break_tie:
        order = torch.randperm(len(inputs[0]), device=inputs[0].device)
    else:
        order = torch.arange(len(inputs[0]), device=inputs[0].device)
    for key in inputs[::-1]:
        # 改
        # index = key[order].argsort(stable=True, descending=descending)
        index = key[order].argsort(descending=descending)
        order = order[index]
    return order

# 计算输入张量中每个元素的计数，并返回一个计数张量
def bincount(input, minlength=0):
    if input.numel() == 0:
        return torch.zeros(minlength, dtype=torch.long, device=input.device)

    sorted = (input.diff() >= 0).all()
    if sorted:
        if minlength == 0:
            minlength = input.max() + 1
        range = torch.arange(minlength + 1, device=input.device)
        index = torch.bucketize(range, input)
        return index.diff()

    return input.bincount(minlength=minlength)

# 从输入张量中获取多个Top-K元素，例如在推荐系统中非常有用
def variadic_topks(input, size, ks, largest=True, break_tie=False):
    index2sample = torch.repeat_interleave(size)
    if largest:
        index2sample = -index2sample
    order = multikey_argsort((index2sample, input), descending=largest, break_tie=break_tie)

    range = torch.arange(ks.sum(), device=input.device)
    offset = (size - ks).cumsum(0) - size + ks
    range = range + offset.repeat_interleave(ks)
    index = order[range]

    return input[index], index
