from typing import NewType
from django.shortcuts import render
import torch
import torchvision
from .forms import UploadFileForm
from django.core.files.storage import FileSystemStorage

import shutil

# Create your views here.
new_net = torchvision.models.resnet18()
def getModel(new_net):
    new_net.fc = torch.nn.Linear(512, 4)
    new_net.load_state_dict(torch.load('./homepage/static/MLmodels/fp.pth', map_location="cpu"))

def test(src):
    trans = torchvision.transforms.Compose([
        torchvision.transforms.Resize((270,270)),
        torchvision.transforms.ToTensor(),
    ])
    ori_test = torchvision.datasets.ImageFolder(root = src, transform = trans)
    new_net.eval()
    for num, value in enumerate(ori_test):
        data, label = value
        result =torch.nn.functional.softmax(new_net(data.unsqueeze(0)))
        # print(result[0][0].item(), result[0][1].item(), result[0][2].item(), result[0][3].item())
        prediction = torch.argmax(new_net(data.unsqueeze(0))).item()
        # print(prediction)
    return {'sis':round(result[0][0].item()*100,2), 'me':round(result[0][1].item()*100,2), 'fa':round(result[0][2].item()*100,2), 'ma':round(result[0][3].item()*100,2), 'predict':prediction}
getModel(new_net)


def index(request):
    global pre

    li = []
    dic = test('./homepage/static/images/tr1')
    dic['name'] = 'train set1'
    dic['src'] = "/static/image/tr1.jpeg"
    li.append(dic)

    dic = test('./homepage/static/images/tr2')
    dic['name'] = 'train set2'
    dic['src'] = "/static/image/tr2.jpg"
    li.append(dic)

    dic = test('./homepage/static/images/te1')
    dic['name'] = 'test set1'
    dic['src'] = "/static/image/te1.jpg"
    li.append(dic)

    dic = test('./homepage/static/images/te2')
    dic['name'] = 'test set2'
    dic['src'] = "/static/image/te2.png"
    li.append(dic)

    dic = test('./homepage/static/images/te3')
    dic['name'] = 'test set3'
    dic['src'] = "/static/image/te3.jpg"
    li.append(dic)
 

    if request.method == 'POST':
        image = request.FILES.get('getImage','')
        if image:
            fs = FileSystemStorage()
            shutil.rmtree('./media/')

            filename  = fs.save("0/" + image.name, image)
            uploaded_file_url = fs.url(filename)

            dict = test('./media/')
            dict['src'] = uploaded_file_url
            name = {0:'누나', 1:'나', 2:'아빠', 3:'엄마'}
            dict['predict'] = name[dict['predict']]
            dict['li'] = li

            return render(request, 'index.html', dict)

    # form = UploadFileForm()
    return render(request, 'index.html', {'li':li})