from django.shortcuts import render
from django.http import HttpResponse
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.dates import DateFormatter
from .models import Images

def index(request):
    return render(request, 'noise_filter/index.html', {})

def image(request, image_id):
    print "Image, ", image_id
    im = Images.objects.filter(id=image_id)[0].get_array().reshape(28, 28)
    fig = Figure()
    ax = fig.add_subplot(111)
    ax.imshow(im)
    canvas = FigureCanvas(fig)
    response = HttpResponse(content_type='image/png')
    canvas.print_png(response)
    return response
