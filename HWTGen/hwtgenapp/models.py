from django.db import models
from django.core.validators import EmailValidator, MaxLengthValidator, RegexValidator
from phonenumber_field.modelfields import PhoneNumberField

# Create your models here.
class Topic(models.Model):
    topicid = models.IntegerField(primary_key=True)
    topicname = models.CharField(max_length=50, blank=True, null=True)
    topicorderid = models.IntegerField(blank=True, null=True)
    description = models.TextField(blank=True, null=True)
    imagepath = models.CharField(max_length=150, blank=True, null=True)
    url = models.CharField(max_length=150, blank=True, null=True)
    createdby = models.CharField(max_length=50, blank=True, null=True)
    createdate = models.DateTimeField(blank=True, null=True)
    lastupdatedate = models.DateTimeField(blank=True, null=True)
    sitemapenabled = models.BooleanField(null=False, default=False)
    supertopicid = models.ForeignKey('self', blank=True, null=True, on_delete=models.CASCADE)

    def get_absolute_url(self):
        return self.url + '/'

    def __str__(self):
        return self.topicname

class TopicDetail(models.Model):
    topicdetailid = models.IntegerField(primary_key=True)
    title = models.CharField(max_length=50, blank=True, null=True)
    topicorderid = models.IntegerField(blank=True, null=True)
    content = models.TextField(blank=True, null=True)
    createdby = models.CharField(max_length=50, blank=True, null=True)
    createdate = models.DateTimeField(blank=True, null=True)
    lastupdatedate = models.DateTimeField(blank=True, null=True)
    sitemapenabled = models.BooleanField(null=False, default=False)
    topicid = models.ForeignKey(Topic, blank=True, null=True, on_delete=models.CASCADE)

    def get_absolute_url(self):
        return '/' + str(self.topicid) + '/' + self.title + '/'

    def __str__(self):
        return self.title
    
class ContactInfo(models.Model):
    valid_name = RegexValidator(r'^[A-Za-z ,.''-]+$', 'First and last name cannot contain numbers.')
    username = models.CharField(max_length=150, blank=True, null=False, validators=[MaxLengthValidator(150, message="Username cannot exceed 150 characters.")])
    firstname = models.CharField(max_length=150, blank=False, null=False, error_messages={'blank': 'First name is a required field.'}, validators=[valid_name])
    lastname = models.CharField(max_length=150, blank=False, null=False, error_messages={'blank': 'Last name is a required field.'}, validators=[valid_name])
    email = models.EmailField(max_length=254, blank=False, null=True, error_messages={'blank': 'Email is a required field.'}, validators=[EmailValidator(message='Please enter a valid email address.', code='invalid')])
    phonenumber = PhoneNumberField(max_length=50, blank=True, null=False, validators=[MaxLengthValidator(50, message="Phone number cannot exceed 50 characters.")])
    message = models.TextField(max_length=1000, blank=False, null=False, validators=[MaxLengthValidator(1000, message="Message cannot exceed 1000 characters.")])
    createdate = models.DateTimeField(blank=True, null=True)