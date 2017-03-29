from django.conf.urls import url

from . import views

urlpatterns = [
    url(r'^$', views.index, name='index'),
    url(r'^(?P<version>normal|noised)/(?P<image_id>[0-9]+)/$', views.display_image, name='image'),
    url(r'^getRandomImage$', views.get_random_image, name='getRandomImage'),
    url(r'^predict/(?P<version>normal|noised)/(?P<image_id>[0-9]+)/', views.predict, name='predict'),
]
