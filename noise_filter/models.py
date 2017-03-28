from __future__ import unicode_literals
import numpy as np
from django.db import models
from cPickle import loads

# Create your models here.
class Images(models.Model):
    data = models.TextField()
    label = models.FloatField()

    def get_array(self):
        return loads(str(self.data))
