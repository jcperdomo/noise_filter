from django.shortcuts import render
from django.http import HttpResponse
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.dates import DateFormatter
from .models import Images

IMAGE_SIZE = 28
NUM_LABELS = 10
from IPython import embed

from os.path import join
import numpy as np

# TODO this will uncompress mnist every time a request comes in, very messy
from .nnetmnist import NNetMnist

checkpoint_path = "checkpoint"
nn = NNetMnist(dirname=checkpoint_path)
nn.fit(epochs=40, skip_if_trained=True, verbose=True)

def index(request):
    return render(request, 'noise_filter/index.html', {})

def get_db_image(image_id):
    try:
        return Images.objects.filter(id=image_id)[0].get_array()
    except:
        return None

def get_image_array(version, image_id, display=False):
    im = get_db_image(image_id)
    if im is None:
        return None

    if version == "noised":
        im = add_noise(im, image_id)

    if display:
        im = im.reshape(IMAGE_SIZE, IMAGE_SIZE)
    return im

def add_noise(im, image_id, epsilon=0.07):
    label = int(Images.objects.filter(id=image_id)[0].label)
    y = np.zeros((1, NUM_LABELS), dtype=np.float32)
    y[0, label] = 1
    grad = nn.gradient(im.reshape((1, len(im))), y)
    noise_vector = grad / np.linalg.norm(grad)
    return im + epsilon * noise_vector


def get_random_image(request):
    image_id = np.random.choice(list(Images.objects.all().values_list('id', flat=True)))
    return HttpResponse(str(image_id))


def display_image(request, version, image_id):
    image_array = get_image_array(version, image_id, display=True)
    if image_array is None:
        return HttpResponse("no image with id {}".format(image_id))
    fig = Figure()
    ax = fig.add_subplot(111)
    ax.imshow(image_array)
    canvas = FigureCanvas(fig)
    response = HttpResponse(content_type='image/png')
    canvas.print_png(response)
    return response

@csrf_exempt
def predict(request, version, image_id):
    if request.method == "GET":
        im = get_image_array(version, image_id)
        if im is None:
            return HttpResponse("No image with id {}".format(image_id))
        im = im.reshape((1, len(im)))
        p = nn.predict(im).tolist()[0]

        res = {
            "prediction": p,
            "label": int(np.argmax(p))
        }

        return JsonResponse(res)
    else:
        return HttpResponse("Use GET to send data")
