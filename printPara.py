from gcn import GCN
import torch
import Parameter as P

if __name__ == '__main__':
    gcnModel = GCN()
    gcnModel.load_state_dict(torch.load(P.GcnModel_FilePath))
    print(gcnModel.parameters())
