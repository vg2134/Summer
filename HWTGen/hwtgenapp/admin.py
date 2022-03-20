from django.contrib import admin
from .models import Topic, TopicDetail, ContactInfo

# Register your models here.
admin.site.register(Topic)
admin.site.register(TopicDetail)
admin.site.register(ContactInfo)