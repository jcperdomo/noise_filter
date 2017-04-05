from __future__ import unicode_literals
import numpy as np
from django.db import models
# TODO hacky way to make it work for python 2 and 3
try:
	from _pickle import loads
except:
	from cPickle import loads

# Create your models here
# TODO should this be plural?
class Images(models.Model):
    data = models.TextField()
    label = models.FloatField()

    def get_array(self):
        return loads(self.data.encode("latin-1"))
