from django.conf.urls import url

from . import views

urlpatterns = [
    url(r'^$', views.index, name='index'),
    url(r'^noise_filter/getRandomImage$', views.rand_image, name='rand_image'),
    url(r'^predict/(?P<image_id>[0-9]+)/', views.predict, name='predict')
]
