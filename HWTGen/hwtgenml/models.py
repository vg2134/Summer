import os

from django.db import models
from django.utils import timezone
from django.contrib.auth.models import User
from django.conf import settings

# Create your models here.
from HWTGen.settings import MEDIA_ROOT


class Vendor(models.Model):
    vendor_name = models.CharField(max_length=250, blank=False, null=False, default="")
    vendor_shortname = models.CharField(max_length=50, blank=False, null=False, default="")

    def __str__(self):
        return self.vendor_name


class VendorCollection(models.Model):
    title = models.CharField(max_length=250, blank=False, null=False, default="")
    collection_shortname = models.CharField(max_length=50, blank=False, null=False, default="")
    uuid = models.CharField(max_length=100, blank=False, null=False, default="")  # same as Caption ID
    api_url = models.CharField(max_length=250, blank=False, null=False, default="")
    num_items = models.IntegerField(blank=False, null=False, default=0)
    collection_downloaded = models.BooleanField(blank=False, null=False, default=False)
    vendor = models.ForeignKey(Vendor, on_delete=models.CASCADE)

    def __str__(self):
        return self.title


class Collection(models.Model):
    user = models.ForeignKey(to=User, on_delete=models.CASCADE, db_constraint=False, related_name='collections')
    name = models.TextField(max_length=250)
    # description = models.CharField(max_length=250)
    estimated_time = models.CharField(max_length=250)
    is_new = models.BooleanField(default=True)
    create_time = models.DateTimeField(auto_now_add=True)
    update_time = models.DateTimeField(auto_now=True)


def upload_to(instance, filename):
    return '/'.join([settings.COLLECTION_ROOT, instance.collection.user.username, instance.collection.name, filename])


class CollectionImage(models.Model):
    collection = models.ForeignKey(to=Collection, on_delete=models.CASCADE, db_constraint=False, related_name="images")
    name = models.CharField(max_length=250)
    file_path = models.ImageField(upload_to=upload_to)

    def delete(self, using=None, keep_parents=False):
        self.file_path.delete(save=False)
        super().delete()

    def __str__(self):
        return self.name


class BoostrapModel(models.Model):
    now_collection = models.ForeignKey(to=Collection, related_name='b_model', db_constraint=False,
                                       on_delete=models.CASCADE, null=True, blank=True, editable=False)
    user = models.ForeignKey(to=User, related_name='b_model', db_constraint=False, on_delete=models.CASCADE, null=True,
                             blank=True, editable=False)
    hw = models.FileField(upload_to='BoostrapModels')
    sol = models.FileField(upload_to='BoostrapModels')
    lf = models.FileField(upload_to='BoostrapModels')
    estimated_time = models.CharField(max_length=250, editable=False)
    create_time = models.DateTimeField(auto_now_add=True)
    update_time = models.DateTimeField(auto_now=True)


class UserModel(models.Model):
    now_collection = models.ForeignKey(to=Collection, related_name='u_model', db_constraint=False,
                                       on_delete=models.CASCADE, null=True, blank=True, editable=False)
    user = models.ForeignKey(to=User, related_name='u_model', db_constraint=False, on_delete=models.CASCADE, null=True,
                             blank=True, editable=False)
    hw = models.FileField(upload_to='UserModels')
    sol = models.FileField(upload_to='UserModels')
    lf = models.FileField(upload_to='UserModels')
    estimated_time = models.CharField(max_length=250, editable=False)
    is_open = models.BooleanField(default=None, null=True, blank=True)
    create_time = models.DateTimeField(auto_now_add=True)
    update_time = models.DateTimeField(auto_now=True)


def model_upload_to(instance, filename):
    return "/".join([settings.MODEL_ROOT, instance.user.username, filename.split('/')[-1]])


class Caption(models.Model):
    name = models.CharField(max_length=250)
    collection = models.ForeignKey(to=Collection, on_delete=models.CASCADE, db_constraint=False,
                                   related_name='captions')


class CollectionText(models.Model):
    image = models.ForeignKey(to=CollectionImage, on_delete=models.CASCADE, db_constraint=False,
                              related_name='collection_texts')
    initial_text = models.TextField()


# class SubCaption(models.Model):
#     caption = models.ForeignKey(to=Caption, on_delete=models.CASCADE, db_constraint=False, related_name='sub_captions')
#     confidence_level = models.FloatField(default=0)


class CaptionImage(models.Model):
    caption = models.ForeignKey(to=Caption, on_delete=models.CASCADE, db_constraint=False, related_name='images')
    image = models.ForeignKey(to=CollectionImage, on_delete=models.CASCADE, db_constraint=False,
                              related_name='captions')
    recognized_text = models.TextField()
    confidence_level = models.IntegerField(default=0)
    initial_text = models.TextField()
    # accuracy = models.CharField(max_length=250, default='', null=True, blank=True)
    saved_text = models.TextField()
