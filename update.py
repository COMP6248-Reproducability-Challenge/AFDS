import torch
def update_param(layer_outputs_source, ratio):
    # winner take all function
    mean_para = []
    var_para = []
    for i in range(len(layer_outputs_source)):
        inactive_channels = layer_outputs_source[i].size()[1] - round(layer_outputs_source[i].size()[1] * ratio)
        inactive_idx = (-layer_outputs_source[i]).topk(inactive_channels, 1)[1]
        layer_outputs_source[i].scatter_(1, inactive_idx, 0)

    new_output_source = []
    new = []
    # reindexing q_l of each batch
    # the order of index is the order of afs layer
    for i in range(0, 100):
        for j in range(i, layer_outputs_source.__len__(), 100):
            new_output_source.append(layer_outputs_source[j])

    # cat all batch
    batch_num = new_output_source.__len__() / 100
    for i in range(0, len(new_output_source), int(batch_num)):
        new1 = new_output_source[i]
        for a in range(1, int(batch_num)):
            new1 = torch.cat((new1, new_output_source[i + a]))
        new.append(new1)

    # compute mean and var
    for i in range(new.__len__()):
        mean_para.append(new[i].mean(dim=0))
        var_para.append(new[i].std())

    # the last element is the last fc layer(classifier), we dont need it
    mean_para.pop()
    var_para.pop()

    # compute p_l for updating m_l
    pro = []
    for i in range(mean_para.__len__()):
        mask = mean_para[i].gt(0)
        a = mean_para[i][mask]
        pro.append(a.size()[0] / mean_para[i].size()[0])
    return (mean_para, var_para, pro)

#updata s_l

def update_sl(model,threshold_s,var_para):
    model_child = model.named_children()
    i=0
    for name, layer in model_child:
        if name=='layer1':
            for num in range(3):
                if var_para[i]>threshold_s:
                    layer[num].s1= 1
                else:
                    layer[num].s1 = 0
                i = i+1
                if var_para[i]>threshold_s:
                    layer[num].s2= 1
                else:
                    layer[num].s2 = 0
                i = i+1
                if var_para[i]>threshold_s:
                    layer[num].s3 = 1
                else:
                    layer[num].s3 = 0
                i = i+1
        elif name=='layer2':
            for num in range(4):
                if var_para[i]>threshold_s:
                    layer[num].s1= 1
                else:
                    layer[num].s1 = 0
                i = i+1
                if var_para[i]>threshold_s:
                    layer[num].s2= 1
                else:
                    layer[num].s2 = 0
                i = i+1
                if var_para[i]>threshold_s:
                    layer[num].s3 = 1
                else:
                    layer[num].s3 = 0
                i = i+1

        elif name=='layer3':
            for num in range(23):
                if var_para[i]>threshold_s:
                    layer[num].s1= 1
                else:
                    layer[num].s1 = 0
                i = i+1
                if var_para[i]>threshold_s:
                    layer[num].s2= 1
                else:
                    layer[num].s2 = 0
                i = i+1
                if var_para[i]>threshold_s:
                    layer[num].s3 = 1
                else:
                    layer[num].s3 = 0
                i = i+1

        elif name=='layer4':
            for num in range(3):
                if var_para[i]>threshold_s:
                    layer[num].s1= 1
                else:
                    layer[num].s1 = 0
                i = i+1
                if var_para[i]>threshold_s:
                    layer[num].s2= 1
                else:
                    layer[num].s2 = 0
                i = i+1
                if var_para[i]>threshold_s:
                    layer[num].s3 = 1
                else:
                    layer[num].s3 = 0
                i = i+1


def update_gama_l(model,mean_para):
    # update gamma_l
    model_child = model.named_children()
    i = 0
    for name, layer in model_child:
        if name == 'layer1':
            for num in range(3):
                layer[num].gama1 = mean_para[i]
                i = i + 1
                layer[num].gama2 = mean_para[i]
                i = i + 1
                layer[num].gama3 = mean_para[i]
                i = i + 1
        elif name == 'layer2':
            for num in range(4):
                layer[num].gama1 = mean_para[i]
                i = i + 1
                layer[num].gama2 = mean_para[i]
                i = i + 1
                layer[num].gama3 = mean_para[i]
                i = i + 1

        elif name == 'layer3':
            for num in range(23):
                layer[num].gama1 = mean_para[i]
                i = i + 1
                layer[num].gama2 = mean_para[i]
                i = i + 1
                layer[num].gama3 = mean_para[i]
                i = i + 1

        elif name == 'layer4':
            for num in range(3):
                layer[num].gama1 = mean_para[i]
                i = i + 1
                layer[num].gama2 = mean_para[i]
                i = i + 1
                layer[num].gama3 = mean_para[i]
                i = i + 1

def update_m_l(model,pro,threshold_m):
    model_child = model.named_children()
    i=0
    for name,layer in model_child:
        if name=='layer1':
            for num in range(3):
                if pro[i]>threshold_m:
                    layer[num].m1= 1
                else:
                    layer[num].m1 = 0
                i = i+1
                if pro[i]>threshold_m:
                    layer[num].m2= 1
                else:
                    layer[num].m2 = 0
                i = i+1
                if pro[i]>threshold_m:
                    layer[num].m3 = 1
                else:
                    layer[num].m3 = 0
                i = i+1
        elif name=='layer2':
            for num in range(4):
                if pro[i]>threshold_m:
                    layer[num].m1= 1
                else:
                    layer[num].m1 = 0
                i = i+1
                if pro[i]>threshold_m:
                    layer[num].m2= 1
                else:
                    layer[num].m2 = 0
                i = i+1
                if pro[i]>threshold_m:
                    layer[num].m3 = 1
                else:
                    layer[num].m3 = 0
                i = i+1

        elif name=='layer3':
            for num in range(23):
                if pro[i]>threshold_m:
                    layer[num].m1= 1
                else:
                    layer[num].m1 = 0
                i = i+1
                if pro[i]>threshold_m:
                    layer[num].m2= 1
                else:
                    layer[num].m2 = 0
                i = i+1
                if pro[i]>threshold_m:
                    layer[num].m3 = 1
                else:
                    layer[num].m3 = 0
                i = i+1

        elif name=='layer4':
            for num in range(3):
                if pro[i]>threshold_m:
                    layer[num].m1= 1
                else:
                    layer[num].m1 = 0
                i = i+1
                if pro[i]>threshold_m:
                    layer[num].m2= 1
                else:
                    layer[num].m2 = 0
                i = i+1
                if pro[i]>threshold_m:
                    layer[num].m3 = 1
                else:
                    layer[num].m3 = 0
                i = i+1