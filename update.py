from afsnet import _Graph_q,_Graph_pi
import torch
def update_afs(model,dataloaders,thres_m,thres_s):
    model.eval()
    for i in range(99):
        _Graph_pi.add_tensor_list(i)
        _Graph_q.add_tensor_list(i)
    for i, (inputs, labels) in enumerate(dataloaders):
        inputs = inputs
        _ = model(inputs.cuda())
    mean_q=[]
    mean_pi=[]
    var=[]
    for i in range(99):
        mean_q.append(torch.mean(_Graph_q.get_tensor_list(i)))
        mean_pi.append(torch.mean(_Graph_pi.get_tensor_list(i)))
        var.append(torch.var(_Graph_q.get_tensor_list(i)))
    layer_index=0
    for name,layer in model.named_children():
        if not name.find('layer')=='-1':
            for bottle_name, bottle_layer in layer.named_children():
                if mean_pi[layer_index]>thres_m:
                    bottle_layer.m1=1
                else:
                    bottle_layer.m1=0
                if var[layer_index]>thres_s:
                    bottle_layer.s1=1
                else:
                    bottle_layer.s1=0
                layer_index+=1
                if mean_pi[layer_index]>thres_m:
                    bottle_layer.m2=1
                else:
                    bottle_layer.m2=0
                if var[layer_index]>thres_s:
                    bottle_layer.s2=1
                else:
                    bottle_layer.s2=0
                layer_index+=1
                if mean_pi[layer_index]>thres_m:
                    bottle_layer.m3=1
                else:
                    bottle_layer.m3=0
                if var[layer_index]>thres_s:
                    bottle_layer.s3=1
                else:
                    bottle_layer.s3=0
                layer_index+=1

    mean_q.clear()
    mean_pi.clear()
    var.clear()
    for i in range(99):
        _Graph_pi.clear_tensor_list(i)
        _Graph_q.clear_tensor_list(i)