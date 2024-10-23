# # from data.isbi_dataset import MnistDataSet
# from .mnist_dataset import MnistDataSet           # 与上面相同的
# from data.eval_dataset import MnistEvalDataSet
# from torch.utils import data

# def CreateTrainDataLoader(args):
#     mnist_dataset = MnistDataSet(args.data_dir)

#     train_dataset = data.DataLoader(dataset=mnist_dataset,
#                                     batch_size=args.batch_size,
#                                     shuffle=True)

#     return train_dataset


# def CreateEvalDataLoader(args):
#     mnist_dataset = MnistEvalDataSet(args.data_dir)

#     evaluation_dataset = data.DataLoader(mnist_dataset,
#                                          batch_size=args.batch_size,
#                                          shuffle=False)
#     return evaluation_dataset
