from django.shortcuts import render
from django.http import HttpResponse
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.dates import DateFormatter
from .models import Images
from django.conf import settings

IMAGE_SIZE = 28
NUM_LABELS = 10
from IPython import embed

from os.path import join
import numpy as np

# TODO this will uncompress mnist every time a request comes in, very messy
from .nnetmnist import NNetMnist

# checkpoint_path = join(settings.BASE_DIR, "checkpoint")
# checkpoint_path = "/opt/python/current/app/checkpoint"
checkpoint_path = ""
nn = NNetMnist(dirname=checkpoint_path)
nb_epochs = 40
# train on server if available
if not settings.RUNNING_DEVSERVER:
    nb_epochs = 2000
nn.fit(epochs=nb_epochs, skip_if_trained=True, verbose=True)

def index(request):
    return render(request, 'noise_filter/index.html', {})

def get_db_image(image_id):
    try:
        res = Images.objects.filter(id=image_id)[0]
        return res.get_array(), int(res.label)
    except:
        return None, None

def get_image_array(version, image_id, display=False):
    im, true_label = get_db_image(image_id)
    if im is None:
        return None, None

    if version == "noised":
        im = add_noise(im, image_id)

    if display:
        im = im.reshape(IMAGE_SIZE, IMAGE_SIZE)
    return im, true_label

def add_noise(im, image_id, epsilon=2.0):
    label = int(Images.objects.filter(id=image_id)[0].label)
    y = np.zeros((1, NUM_LABELS), dtype=np.float32)
    y[0, label] = 1
    grad = nn.gradient(im.reshape((1, len(im))), y)
    noise_vector = grad / np.linalg.norm(grad)
    # return im + epsilon * noise_vector
    return im + 0.07 * np.sign(grad)


def get_random_image(request):
    image_id = np.random.choice(list(Images.objects.all().values_list('id', flat=True)))
    return HttpResponse(str(image_id))


def display_image(request, version, image_id):
    image_array, _ = get_image_array(version, image_id, display=True)
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
        im, true_label = get_image_array(version, image_id)
        if im is None:
            return HttpResponse("No image with id {}".format(image_id))
        im = im.reshape((-1, IMAGE_SIZE*IMAGE_SIZE))
        p = [round(n*100) for n in nn.predict(im).astype(float).tolist()[0]]

        res = {
            "prediction": p,
            "label_predicted": int(np.argmax(p)),
            "label_true": true_label,
            "prediction_args": np.argsort(p)[::-1].tolist()
        }

        return JsonResponse(res)
    else:
        return HttpResponse("Use GET to send data")
