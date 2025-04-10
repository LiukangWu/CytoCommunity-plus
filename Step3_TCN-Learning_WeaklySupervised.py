import torch
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.loader import DenseDataLoader
from torch_geometric.nn import DenseGraphConv, dense_mincut_pool
from torch_geometric.data import InMemoryDataset
import torch_geometric.transforms as T
import os
import shutil
import numpy as np
import datetime
import csv
import time


## Hyperparameters
Num_TCN = 4
Num_Run = 10
Num_Epoch = 400
Num_Class = 2
Embedding_Dimension = 128
LearningRate = 0.001
MiniBatchSize =2
beta = 0.9


# Output folder
ThisStep_OutputFolderName = "./Step3_Output/"
if os.path.exists(ThisStep_OutputFolderName):
    shutil.rmtree(ThisStep_OutputFolderName)
os.makedirs(ThisStep_OutputFolderName)

## Load dataset from the constructed Dataset.
LastStep_OutputFolderName = "./Step2_Output/"
MaxNumNodes_filename = LastStep_OutputFolderName + "MaxNumNodes.txt"
max_nodes = np.loadtxt(MaxNumNodes_filename, dtype = 'int64', delimiter = "\t").item()

class SpatialOmicsImageDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super(SpatialOmicsImageDataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return ['SpatialOmicsImageDataset.pt']

    def download(self):
        pass
    
    def process(self):
        # Read data_list into huge `Data` list.
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

dataset = SpatialOmicsImageDataset(LastStep_OutputFolderName, transform=T.ToDense(max_nodes))


class Net(torch.nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channels=Embedding_Dimension):
        super(Net, self).__init__()

        self.conv1 = DenseGraphConv(in_channels, hidden_channels)
        num_cluster1 = Num_TCN   #This is a hyperparameter.
        self.pool1 = Linear(hidden_channels, num_cluster1)

        self.conv3 = DenseGraphConv(hidden_channels, hidden_channels)

        self.lin1 = Linear(hidden_channels, hidden_channels)
        self.lin2 = Linear(hidden_channels, out_channels)

    def forward(self, x, adj, mask=None):

        x = F.relu(self.conv1(x, adj, mask))
        s = self.pool1(x)  #here s is a non-softmax tensor.
        x, adj, mc1, o1 = dense_mincut_pool(x, adj, s, mask)
        #Save important clustering results_1.
        ClusterAssignTensor_1 = s
        ClusterAdjTensor_1 = adj

        x = self.conv3(x, adj)

        x = x.mean(dim=1)
        x = F.relu(self.lin1(x))
        x = self.lin2(x)
        return F.log_softmax(x, dim=-1), mc1, o1, ClusterAssignTensor_1, ClusterAdjTensor_1


def train(epoch):
    model.train()
    loss_all = 0
    loss_CE_all = 0
    loss_MinCut_all = 0

    data_loading_time = 0.0  # Data loading time
    training_time = 0.0  # Training time

    for data in train_loader:
        data_load_start = time.time()
        data = data.to(device, non_blocking=True)
        
        ## ​Correct Label Values to Ensure Valid Range
        #data.y[data.y >= Num_Class] = Num_Class - 1
        
        data_load_end = time.time()
        data_loading_time += data_load_end - data_load_start

        # Start recording time for training
        train_start = time.time()
        
        optimizer.zero_grad()
        out, mc_loss, o_loss, _, _ = model(data.x, data.adj, data.mask)
        loss_CE = F.nll_loss(out, data.y.view(-1))
        loss_MinCut = mc_loss + o_loss
        loss = loss_CE * (1 - beta) + loss_MinCut * beta
        loss.backward()
        optimizer.step()
        
        train_end = time.time()
        training_time += train_end - train_start

        # Accumulate Loss
        loss_all += data.y.size(0) * loss.item()
        loss_CE_all += data.y.size(0) * loss_CE.item()
        loss_MinCut_all += data.y.size(0) * loss_MinCut.item()

    print(f"Epoch {epoch}: Data Loading Time: {data_loading_time:.4f}s, Training Time: {training_time:.4f}s")
    return (
        loss_all / len(train_dataset),
        loss_CE_all / len(train_dataset),
        loss_MinCut_all / len(train_dataset),
    )


print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
for run_ind in range(1, Num_Run+1):  #Multiple runs of weakly-supervised learning.
    print(f'This is Run: {run_ind:02d}')
    RunFolderName = ThisStep_OutputFolderName + "Run" + str(run_ind)
    if os.path.exists(RunFolderName):
        shutil.rmtree(RunFolderName)
    os.makedirs(RunFolderName)  #Creating the Run folder.

    train_dataset = dataset
    train_loader = DenseDataLoader(
        train_dataset,
        batch_size=MiniBatchSize,
        shuffle=True,
        pin_memory=True,  # Improve the speed of data transfer to the GPU
    )
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Net(dataset.num_features, Num_Class).to(device)  #Initialize model for each run.
    optimizer = torch.optim.Adam(model.parameters(), lr=LearningRate)

    filename_0 = RunFolderName + "/Epoch_TrainLoss.csv"
    headers_0 = ["Epoch", "TrainLoss", "TrainLoss_CE", "TrainLoss_MinCut"]
    with open(filename_0, "w", newline='') as f0:
        f0_csv = csv.writer(f0)
        f0_csv.writerow(headers_0)

    for epoch in range(1, Num_Epoch+1):     #Specify the number of epoch for training in each run.
        train_loss, train_loss_CE, train_loss_MinCut = train(epoch)

        #print(f'Epoch: {epoch:03d}, Train Loss: {train_loss:.4f}')
        with open(filename_0, "a", newline='') as f0:
            f0_csv = csv.writer(f0)
            f0_csv.writerow([epoch, train_loss, train_loss_CE, train_loss_MinCut])
    print(f"Final train loss is {train_loss:.4f} with loss_CE of {train_loss_CE:.4f} and loss_MinCut of {train_loss_MinCut:.4f}")

    #Extract the soft clustering matrix using the trained model of each run.
    all_sample_loader = DenseDataLoader(dataset, batch_size=1)
    EachSample_num = 0

    for EachData in all_sample_loader:
        EachData = EachData.to(device, non_blocking=True)
        TestModelResult = model(EachData.x, EachData.adj, EachData.mask)

        ClusterAssignMatrix1 = TestModelResult[3][0, :, :]
        ClusterAssignMatrix1 = torch.softmax(ClusterAssignMatrix1, dim=-1)  #checked, consistent with the function built in "dense_mincut_pool".
        ClusterAssignMatrix1 = ClusterAssignMatrix1.detach().cpu().numpy()
        filename1 = RunFolderName + "/ClusterAssignMatrix1_" + str(EachSample_num) + ".csv"
        np.savetxt(filename1, ClusterAssignMatrix1, delimiter=',')

        ClusterAdjMatrix1 = TestModelResult[4][0, :, :]
        ClusterAdjMatrix1 = ClusterAdjMatrix1.detach().cpu().numpy()
        filename2 = RunFolderName + "/ClusterAdjMatrix1_" + str(EachSample_num) + ".csv"
        np.savetxt(filename2, ClusterAdjMatrix1, delimiter=',')

        NodeMask = EachData.mask.cpu().numpy()
        NodeMask = np.array(NodeMask)
        filename3 = RunFolderName + "/NodeMask_" + str(EachSample_num) + ".csv"
        np.savetxt(filename3, NodeMask.T, delimiter=',', fmt='%i')  #save as integers.

        EachSample_num = EachSample_num + 1
    
    print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))  #end of each run.

print("Step3 done!")
print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))


