from django.contrib import admin
from django.urls import path, include
from django.conf.urls import (handler400, handler403, handler404, handler500)
from hwtgenapp import views as hwtgenapp_views

handler400 = 'hwtgenapp.views.bad_request'
handler403 = 'hwtgenapp.views.permission_denied'
handler404 = 'hwtgenapp.views.page_not_found'
handler500 = 'hwtgenapp.views.server_error'

urlpatterns=[
    path('', hwtgenapp_views.index, name='home'),
    path('contact-us/', hwtgenapp_views.contact_us, name='contact-us'),
    path('about/', hwtgenapp_views.about, name='about'),
    path('legal/', hwtgenapp_views.legal, name='legal'),
    path('sponsors/<str:sponsorname>', hwtgenapp_views.sponsors, name='sponsors'),
]