from django.db import models
from django.contrib.auth.models import User
from django.utils import timezone
from autoslug import AutoSlugField
from phonenumber_field.modelfields import PhoneNumberField
from django.core.validators import MaxLengthValidator
from django.conf import settings
from hwtgenml.models import Vendor

# Create your models here.
class Profile(models.Model):
    ROLE_TYPES = (
        ('Admin', 'Admin'),
        ('Librarian', 'Librarian'),
        ('User', 'User')
    )
    
    user = models.OneToOneField(User, on_delete=models.CASCADE)
    phonenumber = PhoneNumberField(db_column='PhoneNumber', max_length=50, blank=True, null=False, default="", verbose_name="Phone Number", validators=[MaxLengthValidator(50, message="Phone number cannot exceed 50 characters.")])
    role = models.CharField(max_length=50, blank=False, null=False, default='User', choices=ROLE_TYPES)
    slug = AutoSlugField(populate_from='user')
    created_date = models.DateTimeField(default=timezone.now)

    def __str__(self):
        return str(self.user.username)

    def get_absolute_url(self):
        return "/users/{}".format(self.slug)

def post_save_user_model_receiver(sender, instance, created, *args, **kwargs):
    if created:
        try:
            Profile.objects.create(user=instance)
        except:
            pass

class VendorAuth(models.Model):
    authenticated_user = models.ForeignKey(User, on_delete=models.CASCADE)
    auth_token = models.CharField(max_length=250, blank=False, null=False, default="")
    vendor = models.ForeignKey(Vendor, on_delete=models.CASCADE)