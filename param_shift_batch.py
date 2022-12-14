from cgitb import handler
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
import torch.multiprocessing as mp
import os
import glob
from qiskit import IBMQ
from joblib import Parallel, delayed

IBMQ.save_account(
    'b0ebd9d2f9e2475350fe3996cc65bf1381d1f3cf66baac95a7d91be4edd193b003623e2627ff84423043841bb8c7f8780e54a87572270d9076d93525a02cf49e', overwrite=True)


class QFCModel(QuantumModule):
    def __init__(self):
        super().__init__()
        self.n_wires = 4
        self.q_device = QuantumDevice(n_wires=self.n_wires)
        self.encoder = GeneralEncoder(
            encoder_op_list_name_dict['4x4_ryzxy'])

        self.arch = {'n_wires': self.n_wires,
                     'n_blocks': 2, 'n_layers_per_block': 2}
        self.q_layer = SethLayer0(self.arch)

        self.measure = MeasureAll(PauliZ)

    def forward(self, x, use_qiskit=False):
        bsz = x.shape[0]
        x = F.avg_pool2d(x, 6).view(bsz, 16)

        if use_qiskit:
            # print("In FWD")
            x = self.qiskit_processor.process_parameterized(
                self.q_device, self.encoder, self.q_layer, self.measure, x)
            # print("Done")
        else:
            # print("Step 1: \n")
            self.encoder(self.q_device, x)
            # print("Step 2: \n")
            self.q_layer(self.q_device)
            # print("Step 3: \n")
            x = self.measure(self.q_device)
            # print("Step 4: \n")

        x = x.reshape(bsz, 4)
        # print("Step 5: \n")
        return x

    def get_transpiled_circ(self, x, use_qiskit=False):
        bsz = x.shape[0]
        x = F.avg_pool2d(x, 6).view(bsz, 16)
        circ, binds = self.qiskit_processor.preprocess_parameterized(
            self.q_device, self.encoder, self.q_layer, self.measure, x)
        return circ, binds

    def batch_execute(self, x, circ_all, binds_all, use_qiskit=False):
        bsz = x.shape[0]
        all_out = []
        data = self.qiskit_processor.batch_transpiled_parameterized(
            self.q_device, x, circ_all, binds_all)
        for x in data:
            all_out.append(x.reshape(bsz, 4))
        return all_out


def grad_calc(param):
    with torch.no_grad():
        param[0].copy_(param[0] + np.pi * 0.5)
    # print("hello \n")
    out1 = model(param[2], param[3])
    # print("hello 2\n")
    with torch.no_grad():
        param[0].copy_(param[0] - np.pi)
    # print("hello \n")
    out2 = model(param[2], param[3])
    with torch.no_grad():
        param[0].copy_(param[0] + np.pi * 0.5)
    grad = 0.5 * (out1 - out2)
    # file = open("gradients/grad-{0}.txt".format(param[1]), 'w')
    # file.write(grad)
    # file.close()
    return grad
    # np.save("gradients/grad-{0}.npy".format(param[1]), asarray(flatten))
    # grad_list.append(grad)


def handler(result):
    print(result)


def batch_circ(param_list):
    circ_all = []
    bind_all = []

    for param in param_list:
        with torch.no_grad():
            param[0].copy_(param[0] + np.pi * 0.5)
        circ, bind = model.get_transpiled_circ(param[2], param[3])
        circ_all.append(circ)
        bind_all.append(bind)
        with torch.no_grad():
            param[0].copy_(param[0] - np.pi)
        circ, bind = model.get_transpiled_circ(param[2], param[3])
        circ_all.append(circ)
        bind_all.append(bind)
        with torch.no_grad():
            param[0].copy_(param[0] + np.pi * 0.5)
    return circ_all, bind_all


def shift_and_run(model, inputs, use_qiskit=False):
    param_list = []
    count = 0
    for param in model.parameters():
        param_list.append((param, count, inputs, use_qiskit))
        count += 1
    grad_list = []
    # procs = []
    # for param in param_list:
    #     proc = mp.Process(target=grad_calc, args=(param, use_qiskit))
    #     procs.append(proc)
    #     proc.start()

    # for proc in procs:
    #     proc.join()

    # for param in param_list:
    #     grad_list.append(grad_calc(param))

    circ_all, bind_all = batch_circ(param_list)
    out_all = model.batch_execute(inputs, circ_all, bind_all, use_qiskit)
    for i in range(len(out_all) - 1):
        grad = 0.5*(out_all[i] - out_all[i+1])
        grad_list.append(grad)

    # results = Parallel(n_jobs=5)(delayed(grad_calc)(param)
    #                              for param in param_list)

    # for res in results:
    #     grad_list.append(res)

    # if __name__ == '__main__' and use_qiskit:
    #     print(use_qiskit)
    #     # pool=mp.Pool(processes=len(param_list))
    #     with mp.Pool(processes=len(param_list)) as pool:
    #         proclist = [ pool.apply_async(grad_calc, args, callback=handler) for args in param_list ]
    #     pool.close()
    #     for res in proclist:
    #         print(res)
    #         grad_list.append(res)

    return model(inputs, use_qiskit), grad_list


use_cuda = torch.cuda.is_available()
# device = torch.device("cuda" if use_cuda else "cpu")
device = torch.device("cpu")
model = QFCModel().to(device)
processor_real_qc = QiskitProcessor(
    use_real_qc=True, backend_name='ibmq_manila')
model.set_qiskit_processor(processor_real_qc)

# model.share_memory()

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
        sampler=sampler)

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
        outputs.requires_grad = True
        prediction = outputs.reshape(-1, 2, 2).sum(-1).squeeze()
        loss = F.nll_loss(F.log_softmax(prediction, dim=1), targets)
        optimizer.zero_grad()
        loss.backward()
        grad_ps = []
        for i, param in enumerate(model.q_layer.parameters()):
            param.grad = torch.sum(grad_list[i] * outputs.grad).to(
                dtype=torch.float32, device=param.device).view(param.shape)
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
            prediction = F.log_softmax(
                outputs.reshape(-1, 2, 2).sum(-1).squeeze(), dim=1)

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


# if __name__ == '__main__':
for epoch in range(1, n_epochs + 1):
    # train
    print(f"Epoch {epoch}:")
    train_and_return_grad(dataflow, model, device, optimizer)
    print(optimizer.param_groups[0]['lr'])
    # valid
    valid_test(dataflow, 'valid', model, device)
    scheduler.step()

# test
valid_test(dataflow, 'test', model, device, qiskit=False)
