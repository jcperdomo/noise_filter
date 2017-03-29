from django.shortcuts import render
from django.http import HttpResponse
from django.views.decorators.csrf import csrf_exempt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.dates import DateFormatter
from .models import Images

IMAGE_SIZE = 28
from IPython import embed

from os.path import join
import json
import numpy as np
# TODO this will uncompress mnist every time a request comes in, very messy
from .nnetmnist import NNetMnist

# checkpoint_path = join("..", "checkpoint")
# embed()
checkpoint_path = "checkpoint"
nn = NNetMnist(dirname=checkpoint_path)
nn.fit(epochs=40, skip_if_trained=True, verbose=True)

def index(request):
    return render(request, 'noise_filter/index.html', {})

def get_db_image(image_id):
    return Images.objects.filter(id=image_id)[0].get_array().reshape(IMAGE_SIZE, IMAGE_SIZE)

def get_image_array(version, image_id):
    if version == "normal":
        return get_db_image(image_id)
    else:
        #TODO:
        return get_db_image(image_id)


def get_random_image(request):
    # TODO : CHECK FOR IDS RANGE Amazon
    image_id = np.random.randint(10000)
    # im = Images.objects.filter(id=image_id)[0].get_array().reshape(IMAGE_SIZE, IMAGE_SIZE)
    return HttpResponse(str(image_id))


def display_image(request, version, image_id):
    image_array = get_image_array(version, image_id)
    fig = Figure()
    ax = fig.add_subplot(111)
    ax.imshow(image_array)
    canvas = FigureCanvas(fig)
    response = HttpResponse(content_type='image/png')
    canvas.print_png(response)
    return response

@csrf_exempt
def predict(request, image_id):
    if request.method == "GET":
        im = Images.objects.filter(id=image_id)[0].get_array()
        # embed()
        return HttpResponse("test")
    else:
        return HttpResponse("Use GET to send data")
