from django.shortcuts import render
from django.http import HttpResponse
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.dates import DateFormatter
from .models import Images

IMAGE_SIZE = 28

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

def display_image(request, version, image_id):
    image_array = get_image_array(version, image_id)
    fig = Figure()
    ax = fig.add_subplot(111)
    ax.imshow(image_array)
    canvas = FigureCanvas(fig)
    response = HttpResponse(content_type='image/png')
    canvas.print_png(response)
    return response
