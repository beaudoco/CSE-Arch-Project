import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torchq.layers import SethLayer0
from torchq.encoding import GeneralEncoder, encoder_op_list_name_dict
from torchq.module import QuantumModule
from torchq.devices import QuantumDevice
from torchq.datasets.mnist import MNIST
from torchq.measurement import MeasureAll
from torchq.operators import PauliZ
from torchq.plugins.qiskit_processor import QiskitProcessor
from torch.optim.lr_scheduler import CosineAnnealingLR
import multiprocessing
import os,glob
from qiskit import IBMQ

IBMQ.save_account('51a2a5d55d3e1d9683ab4f135fe6fbb84ecf3221765e19adb408699d43c6eaa238265059c3c2955ba59328634ffbd88ba14d5386c947d22eb9a826e40811d626', overwrite=True)

class QFCModel(QuantumModule):
    def __init__(self):
        super().__init__()
        self.n_wires = 4
        self.q_device = QuantumDevice(n_wires=self.n_wires)
        self.encoder = GeneralEncoder(
            encoder_op_list_name_dict['4x4_ryzxy'])

        self.arch = {'n_wires': self.n_wires, 'n_blocks': 2, 'n_layers_per_block': 2}
        self.q_layer = SethLayer0(self.arch)

        self.measure = MeasureAll(PauliZ)

    def forward(self, x, use_qiskit=False):
        bsz = x.shape[0]
        x = F.avg_pool2d(x, 6).view(bsz, 16)

        if use_qiskit:
            x = self.qiskit_processor.process_parameterized(
                self.q_device, self.encoder, self.q_layer, self.measure, x)
        else:
            self.encoder(self.q_device, x)
            self.q_layer(self.q_device)
            x = self.measure(self.q_device)

        x = x.reshape(bsz, 4)

        return x

def grad_calc(param):
    # print("hello \n")
    param[0].copy_(param[0] + np.pi * 0.5)
    out1 = model(param[2], True)
    param[0].copy_(param[0] - np.pi)
    out2 = model(param[2], True)
    param[0].copy_(param[0] + np.pi * 0.5)
    grad = 0.5 * (out1 - out2)
    file = open("gradients/grad-{0}.txt".format(param[1]), 'w')
    file.write(grad)
    file.close()
    # np.save("gradients/grad-{0}.npy".format(param[1]), asarray(flatten))
    # grad_list.append(grad)

def tmp_prog(x):
    print("hello")
    return 4

def shift_and_run(model, inputs, use_qiskit=False):
    param_list = []
    count = 0
    for param in model.parameters():
        param_list.append((param, count, inputs))
        count += 1
    grad_list = []
    if __name__ == '__main__':
        # pool = multiprocessing.Pool()
        # # pool = multiprocessing.Pool(multiprocessing.cpu_count())
        # pool = multiprocessing.Pool(1)
        pool.map(grad_calc, param_list)
        # pool.close()

    # for param in param_list:
    #     grad_calc(param)


    folder_path = '/gradients'
    for filename in glob.glob(os.path.join(folder_path, '*.txt')):
        with open(filename, 'r') as f:
            text = f.read()
            grad_list.append(text)
            f.close()
    return model(inputs, use_qiskit), grad_list

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
model = QFCModel().to(device)

# model = Q2Model().to(device)
processor_real_qc = QiskitProcessor(use_real_qc=True, backend_name='ibmq_quito')
model.set_qiskit_processor(processor_real_qc)

n_epochs = 15
optimizer = optim.Adam(model.parameters(), lr=5e-3, weight_decay=1e-4)
scheduler = CosineAnnealingLR(optimizer, T_max=n_epochs)

dataset = MNIST(
    root='./mnist_data',
    train_valid_split_ratio=[0.9, 0.1],
    digits_of_interest=[3, 6],
    n_test_samples=300,
    n_train_samples=500
)

dataflow = dict()
for split in dataset:
    sampler = torch.utils.data.RandomSampler(dataset[split])
    dataflow[split] = torch.utils.data.DataLoader(
        dataset[split],
        batch_size=64,
        sampler=sampler,
        num_workers=8,
        pin_memory=True)

grads_bp = []
grads_ps = []

def train_and_return_grad(dataflow, model, device, optimizer):
    for feed_dict in dataflow['train']:
        inputs = feed_dict['image'].to(device)
        targets = feed_dict['digit'].to(device)
        
        # calculate gradients via back propagation
        outputs = model(inputs)
        prediction = outputs.reshape(-1, 2, 2).sum(-1).squeeze()
        loss = F.nll_loss(F.log_softmax(prediction, dim=1), targets)
        optimizer.zero_grad()
        loss.backward()
        grad_bp = []
        for i, param in enumerate(model.q_layer.parameters()):
            grad_bp.append(param.grad.item())

        # calculate gradients via parameters shift rules
        with torch.no_grad():
            outputs, grad_list = shift_and_run(model, inputs, True)
        outputs.requires_grad=True
        prediction = outputs.reshape(-1, 2, 2).sum(-1).squeeze()
        loss = F.nll_loss(F.log_softmax(prediction, dim=1), targets)
        optimizer.zero_grad()
        loss.backward()
        grad_ps = []
        for i, param in enumerate(model.q_layer.parameters()):
            param.grad = torch.sum(grad_list[i] * outputs.grad).to(dtype=torch.float32, device=param.device).view(param.shape)
            grad_ps.append(param.grad.item())

        optimizer.step()
        print(f"loss: {loss.item()}", end='\r')
        grads_bp.append(grad_bp)
        grads_ps.append(grad_ps)

def valid_test(dataflow, split, model, device, qiskit=False):
    target_all = []
    output_all = []
    with torch.no_grad():
        for feed_dict in dataflow[split]:
            inputs = feed_dict['image'].to(device)
            targets = feed_dict['digit'].to(device)

            outputs = model(inputs, use_qiskit=qiskit)
            prediction = F.log_softmax(outputs.reshape(-1, 2, 2).sum(-1).squeeze(), dim=1)

            target_all.append(targets)
            output_all.append(prediction)
        target_all = torch.cat(target_all, dim=0)
        output_all = torch.cat(output_all, dim=0)

    _, indices = output_all.topk(1, dim=1)
    masks = indices.eq(target_all.view(-1, 1).expand_as(indices))
    size = target_all.shape[0]
    corrects = masks.sum().item()
    accuracy = corrects / size
    loss = F.nll_loss(output_all, target_all).item()

    print(f"{split} set accuracy: {accuracy}")
    print(f"{split} set loss: {loss}")

if __name__ == '__main__':
    for epoch in range(1, n_epochs + 1):
        # train
        print(f"Epoch {epoch}:")
        pool = multiprocessing.Pool()
        # # pool = multiprocessing.Pool(multiprocessing.cpu_count())
        pool = multiprocessing.Pool(1)
        train_and_return_grad(dataflow, model, device, optimizer)
        pool.close()
        print(optimizer.param_groups[0]['lr'])
        # valid
        valid_test(dataflow, 'valid', model, device)
        scheduler.step()

# test
valid_test(dataflow, 'test', model, device, qiskit=False)

# if __name__ == '__main__':
#     pool = multiprocessing.Pool()
#         # pool = multiprocessing.Pool(multiprocessing.cpu_count())
#     pool = multiprocessing.Pool(1)
#     listnames = [1,2,3,4,5,6]
#     pool.map(tmp_prog, listnames)
#     pool.close()