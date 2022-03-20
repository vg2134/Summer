import requests
from django.core.paginator import Paginator, PageNotAnInteger, InvalidPage, EmptyPage
from django.shortcuts import render, get_object_or_404
from django.shortcuts import redirect
from django.contrib.auth.decorators import login_required
from tqdm import tqdm
from django.contrib.auth import get_user_model
from django.template import RequestContext
import sys
from torch.autograd import Variable
from e2e.e2e_model import E2EModel
from e2e import visualization
from utils.continuous_state import init_model
import codecs
import os
from threading import Thread
import queue
import mimetypes
from django.core.files import File
from io import BytesIO
from urllib.request import urlopen
from e2e import e2e_postprocessing
from celery.result import AsyncResult
from hwtgenml import tasks
import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torch.nn import CTCLoss

from hw import hw_dataset
from hw import cnn_lstm
from hw.hw_dataset import HwDataset, warp_image

from utils.dataset_wrapper import DatasetWrapper
from utils import safe_load, augmentation

import numpy as np
import cv2
import sys
import json
import os
import time
import traceback
import pandas as pd
from utils import string_utils, error_rates
import time
import random
import yaml
import torch
import json
import cv2
import numpy as np
import os
import yaml
from django.views.generic.edit import FormView
from django.contrib import messages
from django.urls import reverse
from django.contrib.auth.models import User
from django.conf import settings
from django.db.models import Q
from pathlib import Path
from django.http.response import HttpResponse, HttpResponseRedirect, JsonResponse
from django.core.files.storage import FileSystemStorage
import subprocess
import logging
import datetime
import os
import copy
import numpy as np
import json
from hwtgenml.models import *
from users.models import VendorAuth

# from atgenml.forms import


User = get_user_model()
logger = logging.getLogger(__name__)


# Create your views here.
def run_script(script_name, script_path, script_args):
    logger.info("run ML script")

    venvpath = str(Path(r'/Users/vipulgoyal/Documents/ITP/py3.8.10venv/bin', 'activate'))  # Python venv3.8.10 # ORIGINAL
    path = str(settings.BASE_DIR) + script_path
    scriptpath = str(Path(path, script_name))

    logger.info("PATH: " + str(path))
    logger.info(venvpath)
    logger.info(scriptpath)

    arg_str = ''
    for arg in script_args:
        arg_str += '"'
        arg_str += str(arg)
        arg_str += '" '

    cmd = 'source ' + venvpath + '; python ' + scriptpath + ' "' + script_args[0] + '" "' + script_args[1] + '" "' + \
          script_args[2] + '"'

    logger.info("CMD: " + cmd)
    subprocess.run(cmd, capture_output=False, shell=True, executable='/bin/bash')


# HTTP Error 400
def bad_request(request, Exception):
    response = render('400.html', context_instance=RequestContext(request))
    # response = render_to_response('400.html', context_instance=RequestContext(request))
    response.status_code = 400
    return response


# HTTP Error 403
def permission_denied(request, Exception):
    response = render('403.html', context_instance=RequestContext(request))
    # response = render_to_response('403.html', context_instance=RequestContext(request))
    response.status_code = 403
    return response


# HTTP Error 404
def page_not_found(request, Exception):
    response = render('404.html', context_instance=RequestContext(request))
    # response = render_to_response('404.html', context_instance=RequestContext(request))
    response.status_code = 404
    return response


# HTTP Error 500
def server_error(request):
    response = render('500.html', context_instance=RequestContext(request))
    # response = render_to_response('500.html', context_instance=RequestContext(request))
    response.status_code = 500
    return response


def connect_collection_api(request):
    request.session.modified = True
    auth = 'Token token=gxo76k0a1epy8qrv'
    nypl_titles = requests.get('http://api.repo.nypl.org/api/v1/collections', headers={'Authorization': auth})
    print(nypl_titles)
    if len(nypl_titles.json()['nyplAPI']["response"]['collection']):
        request.session['auth'] = auth
        request.session['nypl_titles'] = nypl_titles.json()['nyplAPI']["response"]['collection']
        print(len(request.session['nypl_titles']))
        return HttpResponse(status=200)
    else:
        request.session['nypl_titles'] = []
        return HttpResponse(status=500)


def choose_collection_online(request):
    if not request.session.get('nypl_titles') or not len(request.session.get('nypl_titles')):
        response = render(request, 'hwtgenml/collection_online.html',
                          {"collections": ['please connect collection!'],
                           'exists': request.session.get('exists')})
        request.session['exists'] = False
        response.status_code = 200
        return response
    # if not request.session.get('nypl_titles'):
    #     nypl_titles = requests.get('http://api.repo.nypl.org/api/v1/collections', headers={'Authorization': auth})
    #     request.session['nypl_titles'] = nypl_titles.json()['nyplAPI']["response"]['collection']
    if request.user.is_anonymous:
        return HttpResponseRedirect(reverse('login'))
    choose_collections = [collection for collection in request.session['nypl_titles'] if
                          len(Collection.objects.filter(name=collection['title']))]
    leave_collections = [collection for collection in request.session['nypl_titles'] if
                         not len(Collection.objects.filter(name=collection['title']))]
    response = render(request, 'hwtgenml/collection_online.html',
                      {"choose_collections": choose_collections, "collections": leave_collections,
                       'exists': request.session.get('exists')})
    request.session['exists'] = False
    response.status_code = 200
    return response


def download_images(request):
    auth = 'gxo76k0a1epy8qrv'
    uuid = eval(request.POST.get('collection'))['uuid']
    collection = Collection()
    collection.user = request.user
    collection.name = eval(request.POST.get('collection'))['title']
    collection.estimated_time = "calculating..."
    collection.save()
    tasks.download_images.delay(collection=collection, user=request.user, uuid=uuid)
    return HttpResponseRedirect(reverse('monitor_collection'))


def upload_images(request):
    COLLECTION_DIR_NAME = request.POST['collection_name']
    COLLECTION_FILES = request.FILES.getlist('collection_images')
    COLLECTION_DESCRIPTION = request.POST['collection_description']
    try:
        Collection.objects.get(Q(user=request.user) & Q(name=COLLECTION_DIR_NAME))
        request.session['exists'] = True
        return HttpResponseRedirect(reverse('new_collection'))
    except Collection.DoesNotExist:
        collection = Collection()
        collection.user = request.user
        collection.name = COLLECTION_DIR_NAME
        collection.description = COLLECTION_DESCRIPTION
        collection.save()
        for file in COLLECTION_FILES:
            image = CollectionImage()
            image.collection = collection
            image.name = file.name
            image.file_path = file
            image.save()
        return HttpResponseRedirect(reverse('new_collection'))


def update_collection(request):
    if request.user.is_anonymous:
        return HttpResponseRedirect(reverse('login'))
    collections = Collection.objects.filter(user=request.user).values('name')
    response = render(request, 'hwtgenml/update_collection.html', {"collections": collections})
    response.status_code = 200
    return response


def update_images(request):
    collection = Collection.objects.get(name=request.POST.get('collection_name'))
    tasks.update_image.delay(collection=collection)
    return HttpResponseRedirect(reverse('monitor_collection'))


def monitor_collection(request):
    if request.user.is_anonymous:
        return HttpResponseRedirect(reverse('login'))
    collections = Collection.objects.all()
    response = render(request, 'hwtgenml/monitor_collection.html',
                      {"new_collections": [collection for collection in collections if collection.is_new],
                       "update_collections": [collection for collection in collections if not collection.is_new],
                       "login_user": request.user.username})
    response.status_code = 200
    return response


def copy_collection(request):
    collection_name = request.POST.get('collection_name')
    collection_owner = request.POST.get('user_name')
    original_collection = Collection.objects.get(Q(name=collection_name) & Q(user__username=collection_owner))
    if len(request.user.collections.filter(name=original_collection.name)):
        copied_collection = request.user.collections.get(name=original_collection.name)
    else:
        copied_collection = copy.deepcopy(original_collection)
        copied_collection.pk = None
        copied_collection.user = request.user
        copied_collection.save()
    for image in original_collection.images.all():
        copied_image = copy.deepcopy(image)
        copied_image.pk = None
        copied_image.collection = copied_collection
        copied_image.save()
    return HttpResponseRedirect(reverse('monitor_collection'))


def new_model(request):
    if request.user.is_anonymous:
        return HttpResponseRedirect(reverse('login'))
    user_collections = Collection.objects.all()
    response = render(request, 'hwtgenml/new_model.html', {"user_collections": user_collections})
    response.status_code = 200
    return response


def choose_boostrap_model(request):
    collection_name = request.POST.get('collection_name')
    b_model = BoostrapModel.objects.first()
    model = copy.deepcopy(b_model)
    model.pk = None
    model.user = request.user
    model.now_collection = Collection.objects.get(name=collection_name)
    model.hw = b_model.hw
    model.lf = b_model.lf
    model.sol = b_model.sol
    model.estimated_time = "calculating"
    model.save()
    # caption, b = Caption.objects.get_or_create(name=collection_name,
    #                                            collection=Collection.objects.get(name=collection_name))
    tasks.generate_initial_text.delay(collection=model.now_collection, model=model)
    return HttpResponseRedirect(reverse('monitor_model'))


def open_model(request):
    if request.user.is_anonymous:
        return HttpResponseRedirect(reverse('login'))
    user_collections = Collection.objects.all()
    response = render(request, 'hwtgenml/open_model.html', {"user_collections": user_collections})
    response.status_code = 200
    return response


def choose_model(request):
    collection_name = request.POST.get('collection_name')
    if not len(UserModel.objects.all()):
        b_model = BoostrapModel.objects.first()
        model = UserModel()
        model.user = request.user
        model.now_collection = Collection.objects.get(name=collection_name)
        model.hw = b_model.hw.file
        model.lf = b_model.lf.file
        model.sol = b_model.sol.file
        model.estimated_time = "calculating"
        model.is_open = True
        model.save()
        # tasks.train_internal_model(collection=model.now_collection, model=model)
        caption, b = Caption.objects.get_or_create(name=collection_name,
                                                   collection=Collection.objects.get(name=collection_name))
        tasks.start_test.delay(collection=caption.collection, caption=caption, b=b, model=model)
    else:
        u_model = UserModel.objects.first()
        model = copy.deepcopy(u_model)
        model.pk = None
        model.user = request.user
        model.now_collection = Collection.objects.get(name=collection_name)
        model.estimated_time = "calculating"
        model.is_open = True
        model.save()
        caption, b = Caption.objects.get_or_create(name=collection_name,
                                                   collection=Collection.objects.get(name=collection_name))
        tasks.start_test(collection=caption.collection, caption=caption, b=b, model=model)
    return HttpResponseRedirect(reverse('monitor_model'))


def copy_model(request):
    pass


def retrain_model(request):
    if request.user.is_anonymous:
        return HttpResponseRedirect(reverse('login'))
    captions = Caption.objects.all()
    for model in UserModel.objects.all():
        print(model.estimated_time)
        if model.estimated_time:
            if model.estimated_time == 'calculating' or float(model.estimated_time) != float(0):
                request.session['processing'] = True
                return HttpResponseRedirect(reverse('monitor_model'))
    response = render(request, 'hwtgenml/retrain_model.html', {"captions": captions})
    response.status_code = 200
    return response


def choose_caption(request):
    caption_name = request.POST.get('caption_name')
    caption = Caption.objects.get(name=caption_name)
    if not len(UserModel.objects.all()):
        b_model = BoostrapModel.objects.first()
        model = UserModel()
        model.user = request.user
        model.now_collection = Collection.objects.get(name=caption_name)
        model.hw = b_model.hw.file
        model.lf = b_model.lf.file
        model.sol = b_model.sol.file
        model.estimated_time = "calculating"
        model.is_open = False
        model.save()
    else:
        u_model = UserModel.objects.first()
        model = copy.deepcopy(u_model)
        model.pk = None
        model.user = request.user
        model.now_collection = Collection.objects.get(name=caption_name)
        model.estimated_time = "calculating"
        model.is_open = False
        model.save()
    # request.session['now_caption'] = caption.id
    tasks.retrain_func.delay(model=model, caption=caption)
    return HttpResponseRedirect(reverse('monitor_model'))


def monitor_model(request):
    if request.user.is_anonymous:
        return HttpResponseRedirect(reverse('login'))
    boostrap_models = BoostrapModel.objects.exclude(user=None)
    open_models = UserModel.objects.filter(is_open=True)
    retrain_models = UserModel.objects.filter(is_open=False)
    response = render(request, 'hwtgenml/monitor_model.html',
                      {'boostrap_models': boostrap_models, 'open_models': open_models, 'retrain_models': retrain_models,
                       'processing': request.session.get('processing')})
    request.session['processing'] = False
    response.status_code = 200
    return response


def new_caption(request):
    if request.user.is_anonymous:
        return HttpResponseRedirect(reverse('login'))
    collections = Collection.objects.filter(user=request.user)
    response = render(request, 'hwtgenml/new_caption.html',
                      {'collections': collections, 'caption_exists': request.session.get('caption_exists')})
    request.session['caption_exists'] = False
    response.status_code = 200
    return response


def make_caption(request):
    collection_name = request.POST.get("collection_name")
    caption_name = request.POST.get("caption_name")
    if len(Caption.objects.filter(Q(collection__user=request.user) & Q(name=caption_name))):
        request.session['caption_exists'] = True
        return HttpResponseRedirect(reverse('new_caption'))
    collection = Collection.objects.get(Q(user=request.user) & Q(name=collection_name))
    caption = Caption()
    caption.collection = collection
    caption.name = caption_name
    caption.save()
    for image in collection.images.all():
        caption_image = CaptionImage()
        caption_image.caption = caption
        caption_image.image = image
        caption_image.content = ""
        caption_image.initial_text = caption_image.content
        caption_image.save()
    response = render(request, 'hwtgenml/change_caption.html', {'caption_images': caption.images.all()})
    response.status_code = 200
    return response


def update_caption(request):
    if request.user.is_anonymous:
        return HttpResponseRedirect(reverse('login'))
    captions = Caption.objects.all()
    response = render(request, 'hwtgenml/update_caption.html',
                      {'captions': captions})
    response.status_code = 200
    return response


def update_caption_threshold(request):
    caption_name = request.POST.get('caption_name')
    response = render(request, 'hwtgenml/update_caption_threshold.html',
                      {'caption': Caption.objects.get(name=caption_name),
                       'thresholds': [str(i) + "%" for i in sorted(list(range(0, 100, 10)), reverse=True)]})
    response.status_code = 200
    return response


def make_exist_caption(request):
    # sub_caption_id = request.POST.get("sub_caption_id")
    # sub_caption = SubCaption.objects.get(id=sub_caption_id)
    try:
        caption_id = request.POST.get('caption_id')
        threshold = int(request.POST.get('threshold').strip('%'))
    except:
        caption_id = request.GET.get('caption_id')
        threshold = int(request.GET.get('threshold'))
    caption = Caption.objects.get(id=caption_id)

    caption_images = caption.images.filter(
        Q(confidence_level__gte=threshold) & Q(confidence_level__lte=threshold + 10)).order_by('image__name')
    paginator = Paginator(caption_images, 8)
    page_num = request.GET.get('page', '1')
    try:
        Page = paginator.page(page_num)
    except(PageNotAnInteger, EmptyPage, InvalidPage):
        Page = paginator.page('1')
    if Page.paginator.num_pages >= 13:
        ifEllipsis = 1
        range1 = range(1, 13)
        range2 = range(1, 15)
        range3 = range(1, 14)
        lastButOne = Page.paginator.num_pages - 1
    else:
        ifEllipsis = 0
    response = render(request, 'hwtgenml/change_caption.html',
                      {'threshold': threshold,
                       'Page': Page,
                       'pagerange': paginator.page_range,
                       'caption_id': caption_id,
                       'ifEllipsis': ifEllipsis,
                       'range1': range1,
                       'range2': range2,
                       'range3': range3,
                       'lastButOne': lastButOne,
                       })
    response.status_code = 200
    return response


def change_caption(request):
    caption_image_id = request.POST.get('caption_image_id')
    caption_image = CaptionImage.objects.get(id=caption_image_id)
    content = request.POST.get('content')
    caption_image.saved_text = content
    caption_image.save()
    threshold = int(request.POST.get('threshold').strip('%'))
    caption = caption_image.caption
    caption_id = caption.id
    caption_images = caption.images.filter(
        Q(confidence_level__gte=threshold) & Q(confidence_level__lte=threshold + 10)).order_by('image__name')
    paginator = Paginator(caption_images, 8)
    page_num = request.POST.get('page', '1')
    try:
        Page = paginator.page(page_num)
    except(PageNotAnInteger, EmptyPage, InvalidPage):
        Page = paginator.page('1')
    if Page.paginator.num_pages >= 13:
        ifEllipsis = 1
        range1 = range(1, 13)
        range2 = range(1, 15)
        range3 = range(1, 14)
        lastButOne = Page.paginator.num_pages - 1
    else:
        ifEllipsis = 0
    response = render(request, 'hwtgenml/change_caption.html',
                      {'threshold': threshold,
                       'Page': Page,
                       'pagerange': paginator.page_range,
                       'caption_id': caption_id,
                       'ifEllipsis': ifEllipsis,
                       'range1': range1,
                       'range2': range2,
                       'range3': range3,
                       'lastButOne': lastButOne,
                       })
    response.status_code = 200
    return response


def test_caption(request):
    if request.user.is_anonymous:
        return HttpResponseRedirect(reverse('login'))
    collections = Collection.objects.filter(user=request.user)
    response = render(request, 'hwtgenml/test_caption.html',
                      {'collections': collections, 'caption_exists': request.session.get('caption_exists'),
                       'model_choiced': request.session.get('now_model')})
    request.session['caption_exists'] = False
    response.status_code = 200
    return response


def test_exist_caption(request):
    if request.user.is_anonymous:
        return HttpResponseRedirect(reverse('login'))
    captions = Caption.objects.filter(collection__user=request.user)
    response = render(request, 'hwtgenml/test_exist_caption.html',
                      {'captions': captions,
                       'model_choiced': request.session.get('now_model')})
    response.status_code = 200
    return response


def test_model(request):
    collection_name = request.POST.get('collection_name')
    caption_name = request.POST.get('caption_name')
    if len(Caption.objects.filter(Q(name=caption_name) & Q(collection__user=request.user))):
        request.session['caption_exists'] = True
        return HttpResponseRedirect(reverse('test_caption'))
    collection = Collection.objects.get(Q(user=request.user) & Q(name=collection_name))
    model = UserModel.objects.get(id=request.session['now_model'])
    caption = Caption()
    caption.collection = collection
    caption.name = caption_name
    caption.save()
    for image in collection.images.all():
        caption_image = CaptionImage()
        caption_image.image = image
        caption_image.caption = caption
        caption_image.content = ""
        caption_image.initial_text = caption_image.content
        caption_image.save()
    recognize_results = tasks.start_test.delay(collection, model, caption)
    response = render(request, 'hwtgenml/change_caption.html', {'caption_images': caption.images.all()})
    response.status_code = 200
    return response


def test_model_exist(request):
    caption_name = request.POST.get('caption_name')
    caption = Caption.objects.get(Q(name=caption_name) & Q(collection__user=request.user))
    model = UserModel.objects.get(id=request.session['now_model'])
    recognize_results = tasks.start_test.delay(caption.collection, model, caption)
    response = render(request, 'hwtgenml/change_caption.html', {'caption_images': caption.images.all()})
    response.status_code = 200
    return response


def add_images(request):
    path = '/root/images'
    images = os.listdir(path)
    collection = Collection.objects.first()
    collection.images.all().delete()
    for image in tqdm(images):
        im_path = os.path.join(path, image)
        im = open(im_path, 'rb')
        collection_image = CollectionImage()
        collection_image.collection = collection
        collection_image.name = os.path.basename(im_path)
        collection_image.file_path = File(im)
        collection_image.save()
    return HttpResponse('success')
