from django.urls import path, re_path
# from django.conf.urls import (handler400, handler403, handler404, handler500)
from django.conf import settings
from django.conf.urls.static import static
from hwtgenml import views as hwtgenml_views

handler400 = 'hwtgenml.views.bad_request'
handler403 = 'hwtgenml.views.permission_denied'
handler404 = 'hwtgenml.views.page_not_found'
handler500 = 'hwtgenml.views.server_error'

urlpatterns = [
    path('connect_collection_api/', hwtgenml_views.connect_collection_api, name='connect_collection_api'),
    # path('new_collection/', hwtgenml_views.new_collection, name='new_collection'),
    path('upload_images/', hwtgenml_views.upload_images, name='upload_images'),
    path('update_collection/', hwtgenml_views.update_collection, name='update_collection'),
    path('update_images/', hwtgenml_views.update_images, name='update_images'),
    path('monitor_collection/', hwtgenml_views.monitor_collection, name='monitor_collection'),
    path('choose_collection_online/', hwtgenml_views.choose_collection_online, name='choose_collection_online'),
    path('copy_collection/', hwtgenml_views.copy_collection, name='copy_collection'),
    path('new_model/', hwtgenml_views.new_model, name='new_model'),
    path('copy_model/', hwtgenml_views.copy_model, name='copy_model'),
    path('open_model/', hwtgenml_views.open_model, name='open_model'),
    path('choose_model/', hwtgenml_views.choose_model, name='choose_model'),
    path('retrain_model/', hwtgenml_views.retrain_model, name='retrain_model'),
    path('choose_caption/', hwtgenml_views.choose_caption, name='choose_caption'),
    path('monitor_model/', hwtgenml_views.monitor_model, name='monitor_model'),
    path('new_caption/', hwtgenml_views.new_caption, name='new_caption'),
    path('make_caption/', hwtgenml_views.make_caption, name='make_caption'),
    path('make_exist_caption/', hwtgenml_views.make_exist_caption, name='make_exist_caption'),
    path('update_caption/', hwtgenml_views.update_caption, name='update_caption'),
    path('change_caption/', hwtgenml_views.change_caption, name='change_caption'),
    path('test_caption/', hwtgenml_views.test_caption, name='test_caption'),
    path('test_exist_caption/', hwtgenml_views.test_exist_caption, name='test_exist_caption'),
    path('test_model_exist/', hwtgenml_views.test_model_exist, name='test_model_exist'),
    path('test_model/', hwtgenml_views.test_model, name='test_model'),
    path('add_images/', hwtgenml_views.add_images, name='add_images'),
    path('download_images/', hwtgenml_views.download_images, name='download_images'),
    path('update_caption_threshold/', hwtgenml_views.update_caption_threshold, name='update_caption_threshold'),
    path('choose_boostrap_model/', hwtgenml_views.choose_boostrap_model, name='choose_boostrap_model'),
    # path('start_test/', hwtgenml_views.start_test, name='start_test'),
]  # + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
