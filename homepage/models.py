from django.db import models


class UploadFileModel(models.Model):
    title = models.CharField(default="", max_length=20)
    photo = models.ImageField(upload_to="")

# Create your models here.
