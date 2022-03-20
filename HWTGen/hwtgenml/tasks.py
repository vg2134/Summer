import mimetypes
import traceback
from io import BytesIO
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from celery import task
import requests
import pandas as pd
import queue
from threading import Thread
from django.core.files import File
from django.shortcuts import render, get_object_or_404
from django.shortcuts import redirect
from django.contrib.auth.decorators import login_required
from django.contrib.auth import get_user_model
from django.template import RequestContext
import sys
from torch.autograd import Variable
from tqdm import tqdm

from e2e.e2e_model import E2EModel
from e2e import visualization
from utils.continuous_state import init_model
import codecs
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
from utils import string_utils, error_rates
from time import *
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
import time
import numpy as np
import json
from hwtgenml.models import *
from users.models import VendorAuth


# from atgenml.forms import

def calculate_accuracy(r_text, text):
    try:
        vectorizer = TfidfVectorizer()
        texts = [r_text, text]
        vectorizer.fit(texts)
        tfidf_ = vectorizer.transform(texts)
        similarity_matrix = cosine_similarity(tfidf_, tfidf_)
        return similarity_matrix[0][1]
    except:
        return 1


@task(bind=True)
def generate_initial_text(self, collection, model):
    sol, lf, hw = init_model(model)
    sol.cuda()
    lf.cuda()
    hw.cuda()
    char_set_path = os.path.join(settings.MEDIA_ROOT, 'char_set.json')

    with open(char_set_path) as f:
        char_set = json.load(f)
    idx_to_char = {}
    for k, v in char_set['idx_to_char'].items():
        idx_to_char[int(k)] = v

    char_to_idx = char_set['char_to_idx']

    e2e = E2EModel(sol, lf, hw)
    e2e.cuda()
    dtype = torch.FloatTensor
    e2e.eval()
    recognize_results = dict()
    all_length = len(collection.images.all())
    for im in tqdm(collection.images.all()):
        begin_time = time.time()
        image = im
        org_img = cv2.imread(image.file_path.path)

        target_dim1 = 512
        try:
            s = target_dim1 / float(org_img.shape[1])
        except:
            continue

        pad_amount = 128
        org_img = np.pad(org_img, ((pad_amount, pad_amount), (pad_amount, pad_amount), (0, 0)), 'constant',
                         constant_values=255)
        before_padding = org_img

        target_dim0 = int(org_img.shape[0] * s)
        target_dim1 = int(org_img.shape[1] * s)

        full_img = org_img.astype(np.float32)
        full_img = full_img.transpose([2, 1, 0])[None, ...]
        full_img = torch.from_numpy(full_img)
        full_img = full_img / 128 - 1

        img = cv2.resize(org_img, (target_dim1, target_dim0), interpolation=cv2.INTER_CUBIC)
        img = img.astype(np.float32)
        img = img.transpose([2, 1, 0])[None, ...]
        img = torch.from_numpy(img)
        img = img.cuda()
        img = img / 128 - 1
        out = e2e.forward({
            "resized_img": img,
            "full_img": full_img,
            "resize_scale": 1.0 / s
        }, use_full_img=True)
        out = e2e_postprocessing.results_to_numpy(out)

        if out is None:
            print("No Results")

            continue

        # take into account the padding
        out['sol'][:, :2] = out['sol'][:, :2] - pad_amount
        for l in out['lf']:
            l[:, :2, :2] = l[:, :2, :2] - pad_amount

        out = dict(out)

        # Postprocessing Steps
        out['idx'] = np.arange(out['sol'].shape[0])
        out = e2e_postprocessing.trim_ends(out)
        e2e_postprocessing.filter_on_pick(out, e2e_postprocessing.select_non_empty_string(out))
        out = e2e_postprocessing.postprocess(out,
                                             sol_threshold=0.1,
                                             lf_nms_params={
                                                 "overlap_range": [0, 6],
                                                 "overlap_threshold": 0.5
                                             }
                                             )
        order = e2e_postprocessing.read_order(out)
        e2e_postprocessing.filter_on_pick(out, order)

        output_strings_ = []

        output_strings_, decoded_raw_hw = e2e_postprocessing.decode_handwriting(out, idx_to_char)
        output_strings = output_strings_[0]
        decoded_raw_hw = decoded_raw_hw[0]
        recognized_text = ' '.join(output_strings)
        initial_text, b = CollectionText.objects.get_or_create(image=im, initial_text=recognized_text)
        initial_text.save()
        end_time = time.time()
        run_time = end_time - begin_time
        all_length -= 1
        model.estimated_time = (run_time / 60 / 60) * all_length
        model.save()
    model.estimated_time = 0
    model.save()
    return recognize_results


@task(bind=True)
def train_internal_model(self, collection, model):
    hw_network_config = {
        'num_of_outputs': 197,
        'num_of_channels': 3,
        'cnn_out_size': 1024,
    }
    char_set_path = os.path.join(settings.MEDIA_ROOT, 'char_set.json')

    with open(char_set_path) as f:
        char_set = json.load(f)
    idx_to_char = {}
    for k, v in char_set['idx_to_char'].items():
        idx_to_char[int(k)] = v
    hw = cnn_lstm.create_model(hw_network_config)
    hw.cuda()
    hw.load_state_dict(safe_load.torch_state(model.hw.path))
    criterion = CTCLoss()
    dtype = torch.cuda.FloatTensor
    lowest_loss = np.inf
    sum_loss = 0.0
    steps = 0.0
    optimizer = torch.optim.Adam(hw.parameters(), lr=0.0002)
    hw.train()
    train_images = list()
    for image in collection.images.all()[:10]:
        try:
            img = image.file_path.path
            text = image.collection_texts.first().initial_text
            train_images.append({'img': img, 'text': text})
        except:
            continue
    all_length = len(train_images) * 10
    for i in range(1):
        for img_dict in tqdm(train_images):
            begin_time = time.time()
            img = cv2.imread(img_dict['img'])
            if img.shape[0] != 60:
                percent = float(60) / img.shape[0]
                img = cv2.resize(img, (0, 0), fx=percent, fy=percent, interpolation=cv2.INTER_CUBIC)
            img = augmentation.apply_random_color_rotation(img)
            img = augmentation.apply_tensmeyer_brightness(img)
            img = warp_image(img)
            img = img.astype(np.float32)
            img = img / 128.0 - 1.0
            gt_label = string_utils.str2label_single(img_dict['text'], char_set['char_to_idx'])
            img = img[np.newaxis, :, :, :]
            img = img.transpose([0, 3, 1, 2])
            line_image = Variable(torch.from_numpy(img).type(dtype), requires_grad=False).cuda()
            gt_label = torch.from_numpy(gt_label.astype(int)).cuda()
            # print(line_image.shape)
            label = Variable(gt_label, requires_grad=False).cuda()
            label_length = Variable(torch.IntTensor([len(gt_label)]), requires_grad=False).cuda()
            preds = hw(line_image).cpu()

            output_batch = preds.permute(1, 0, 2)
            out = output_batch.data.cpu().numpy()
            print(out[0, ...])
            for i, gt_line in enumerate([img_dict['text']]):
                logits = out[i, ...]
                pred, raw_pred = string_utils.naive_decode(logits)
                pred = pred[0]
                raw_pred = raw_pred[0]
                pred_str = string_utils.label2str_single(pred, idx_to_char, False)
                cer = error_rates.cer(gt_line, pred_str)
                sum_loss += cer
                steps += 1

            batch_size = preds.size(1)
            preds_size = Variable(torch.IntTensor([preds.size(0)] * batch_size)).cuda()

            loss = criterion(preds, label, preds_size, label_length)

            optimizer.zero_grad()
            loss.backward()
            print('loss:', loss.item())
            optimizer.step()
            end_time = time.time()
            run_time = end_time - begin_time
            all_length -= 1
            model.estimated_time = (run_time / 60 / 60) * all_length
            # torch.save(hw.state_dict(), model.hw.path)
            # model.save()
    model.estimated_time = 0
    # model.save()


@task(bind=True)
def start_test(self, collection, model, caption, b):
    sol, lf, hw = init_model(model)
    sol.cuda()
    lf.cuda()
    hw.cuda()
    char_set_path = os.path.join(settings.MEDIA_ROOT, 'char_set.json')

    with open(char_set_path) as f:
        char_set = json.load(f)
    idx_to_char = {}
    for k, v in char_set['idx_to_char'].items():
        idx_to_char[int(k)] = v

    char_to_idx = char_set['char_to_idx']

    e2e = E2EModel(sol, lf, hw)
    e2e.cuda()
    dtype = torch.FloatTensor
    e2e.eval()

    recognize_results = dict()
    print(len(caption.collection.images.all()))
    if b:
        for image in caption.collection.images.all():
            caption_image = CaptionImage()
            caption_image.image = image
            caption_image.caption = caption
            caption_image.recognized_text = ""
            caption_image.saved_text = caption_image.recognized_text
            caption_image.save()
    all_length = len(caption.collection.images.all())
    for im in tqdm(caption.images.all()):
        begin_time = time.time()
        image = im.image
        org_img = cv2.imread(image.file_path.path)

        target_dim1 = 512
        try:
            s = target_dim1 / float(org_img.shape[1])
        except:
            continue

        pad_amount = 128
        org_img = np.pad(org_img, ((pad_amount, pad_amount), (pad_amount, pad_amount), (0, 0)), 'constant',
                         constant_values=255)
        before_padding = org_img

        target_dim0 = int(org_img.shape[0] * s)
        target_dim1 = int(org_img.shape[1] * s)

        full_img = org_img.astype(np.float32)
        full_img = full_img.transpose([2, 1, 0])[None, ...]
        full_img = torch.from_numpy(full_img).cuda()
        full_img = full_img / 128.0 - 1.0

        img = cv2.resize(org_img, (target_dim1, target_dim0), interpolation=cv2.INTER_CUBIC)
        img = img.astype(np.float32)
        img = img.transpose([2, 1, 0])[None, ...]
        img = torch.from_numpy(img).cuda()
        img = img / 128.0 - 1.0
        out = e2e.forward({
            "resized_img": img,
            "full_img": full_img,
            "resize_scale": 1.0 / s
        }, use_full_img=True)
        out = e2e_postprocessing.results_to_numpy(out)

        if out is None:
            print("No Results")

            continue

        # take into account the padding
        out['sol'][:, :2] = out['sol'][:, :2] - pad_amount
        for l in out['lf']:
            l[:, :2, :2] = l[:, :2, :2] - pad_amount

        out = dict(out)

        # Postprocessing Steps
        out['idx'] = np.arange(out['sol'].shape[0])
        out = e2e_postprocessing.trim_ends(out)
        e2e_postprocessing.filter_on_pick(out, e2e_postprocessing.select_non_empty_string(out))
        out = e2e_postprocessing.postprocess(out,
                                             sol_threshold=0.1,
                                             lf_nms_params={
                                                 "overlap_range": [0, 6],
                                                 "overlap_threshold": 0.5
                                             }
                                             )
        order = e2e_postprocessing.read_order(out)
        e2e_postprocessing.filter_on_pick(out, order)

        output_strings_ = []

        print("-----------------test--------------------")
        output_strings_, decoded_raw_hw = e2e_postprocessing.decode_handwriting(out, idx_to_char)
        output_strings = output_strings_[1]
        decoded_raw_hw = decoded_raw_hw[1]
        im.recognized_text = ' '.join(output_strings)
        im.confidence_level = calculate_accuracy(im.recognized_text, ' '.join(output_strings_[0])) * 100
        print(im.confidence_level)
        im.save()
        # if len(im.saved_text):
        #     acc = calculate_accuracy(im.recognized_text, im.saved_text)
        #     im.accuracy = acc
        end_time = time.time()
        run_time = end_time - begin_time
        all_length -= 1
        model.estimated_time = (run_time / 60 / 60) * all_length
        model.save()
    model.estimated_time = 0
    model.save()
    return recognize_results


@task(bind=True)
def retrain_func(self, model, caption):
    hw_network_config = {
        'num_of_outputs': 197,
        'num_of_channels': 3,
        'cnn_out_size': 1024,
    }
    char_set_path = os.path.join(settings.MEDIA_ROOT, 'char_set.json')

    with open(char_set_path) as f:
        char_set = json.load(f)
    idx_to_char = {}
    for k, v in char_set['idx_to_char'].items():
        idx_to_char[int(k)] = v
    hw = cnn_lstm.create_model(hw_network_config)
    hw.cuda()
    hw_path = model.hw.path
    hw_state = safe_load.torch_state(hw_path)
    hw.load_state_dict(hw_state)
    criterion = CTCLoss()
    dtype = torch.FloatTensor
    lowest_loss = np.inf
    sum_loss = 0.0
    steps = 0.0
    optimizer = torch.optim.Adam(hw.parameters(), lr=0.0002)
    hw.train()
    all_retrain_images = list()
    for image in caption.images.all():
        if len(image.saved_text):
            all_retrain_images.append(image)
    print(all_retrain_images)
    all_length = len(all_retrain_images) * 10
    for i in range(10):
        for ix, x in enumerate(all_retrain_images):
            if len(x.saved_text):
                begin_time = time.time()
                img = cv2.imread(x.image.file_path.path)
                if img.shape[0] != 60:
                    percent = float(60) / img.shape[0]
                    img = cv2.resize(img, (0, 0), fx=percent, fy=percent, interpolation=cv2.INTER_CUBIC)
                img = augmentation.apply_random_color_rotation(img)
                img = augmentation.apply_tensmeyer_brightness(img)
                img = warp_image(img)
                img = img.astype(np.float32)
                img = img / 128.0 - 1.0
                gt_label = string_utils.str2label_single(x.saved_text, char_set['char_to_idx'])
                img = img[np.newaxis, :, :, :]
                img = img.transpose([0, 3, 1, 2])
                line_image = Variable(torch.from_numpy(img).type(dtype), requires_grad=False).cuda()
                gt_label = torch.from_numpy(gt_label.astype(int)).cuda()
                # print(line_image.shape)
                label = Variable(gt_label, requires_grad=False).cuda()
                label_length = Variable(torch.IntTensor([len(gt_label)]), requires_grad=False).cuda()
                preds = hw(line_image).cpu()

                output_batch = preds.permute(1, 0, 2)
                out = output_batch.data.cpu().numpy()
                for i, gt_line in enumerate([x.saved_text]):
                    logits = out[i, ...]
                    pred, raw_pred = string_utils.naive_decode(logits)
                    pred = pred[0]
                    raw_pred = raw_pred[0]
                    pred_str = string_utils.label2str_single(pred, idx_to_char, False)
                    cer = error_rates.cer(gt_line, pred_str)
                    sum_loss += cer
                    steps += 1

                batch_size = preds.size(1)
                preds_size = Variable(torch.IntTensor([preds.size(0)] * batch_size)).cuda()

                loss = criterion(preds, label, preds_size, label_length)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                end_time = time.time()
                run_time = end_time - begin_time
                all_length -= 1
                model.estimated_time = (run_time / 60 / 60) * all_length
                torch.save(hw.state_dict(), model.hw.path)
                model.save()
    model.estimated_time = 0
    model.save()
    # print("Train Loss", sum_loss / steps)

    # model.save()


def parseData(resp):
    if resp.status_code != 200:
        raise Exception("not 200!")
    response = json.loads(resp.text)
    if 'nyplAPI' in response:
        data = response['nyplAPI']
        if "response" in data:
            data = data['response']
        else:
            raise ValueError("nyplAPI not in response!")
        return data
    else:
        print(response.text)
        raise ValueError("not 200 or invalid response!")


def fetchData(url):
    token = '9o3m4fb9yb7wv800'
    headers = {'Authorization': 'Token token=%s' % token}
    resp = requests.get(url, headers=headers)
    return parseData(resp)


remain_images = 0


@task(bind=True)
def download_images(self, collection, uuid, user):
    page = 0
    captures = []
    error_uris = []
    error_count = 0

    # Start the loop
    while (True):
        try:
            turl = "http://api.repo.nypl.org/api/v1/items/{uuid}?page={page}".format(uuid=uuid, page=page)
            # print(turl)
            # turl = "http://api.repo.nypl.org/api/v1/items/{uuid}?page={page}&per_page={pp}".format(uuid=uuid,
            #                                                                                        page=page,
            #                                                                                        pp=per_page)
            data = fetchData(turl)
            nres = len(data['capture'])
            for item in data['capture']:
                capture = {"type": item['typeOfResource'], "uuid": item['uuid'], "imageid": item['imageID']}
                captures.append(capture)
            print("({}:{})".format(page, len(captures)), end="")
            collection.estimated_time = 'calculating...'
            collection.save()
            end_page = 1
            if page >= end_page:
                raise StopIteration()
            # if nres < per_page or (end_page and page >= end_page):
            #     raise StopIteration()

        except (StopIteration, ValueError):
            print("\nDone! Reached the end of collection(%d captures at page %d)." % (len(captures), page))
            if len(error_uris):
                print("Some error occurred for the following uris, you can ")
                for uri in error_uris:
                    print(uri[0], uri[1])
            break
        except Exception:
            error_uris.append((turl, traceback.format_exc()))
            print("\nException while processing %s" % turl, file=sys.stderr)
            time.sleep(5)
            error_count += 1
            if error_count > 20:
                print("Too many errors, stopping. we are at page %d now." % page)
                for uri in error_uris:
                    print(uri[0], uri[1])
                break
        finally:
            page += 1

    # Save our result
    df = pd.DataFrame(captures)
    df.to_csv('captures.csv')
    # df = pd.read_csv('captures.csv')
    # captures = df.to_dict(orient='index').values()
    global remain_images
    remain_images = len(df)
    print(remain_images)

    quality = "w"
    threadnum = 5
    imgslimit = None  # maximum images

    def down():
        while (True):
            begin_time = time.time()
            try:
                if stopFlag:
                    break
                cap = q.get(timeout=10)
                uri = "http://images.nypl.org/index.php?id={imgid}&t={tq}&download=1&suffix={uuid}.001".format(
                    imgid=cap['imageid'], tq=quality, uuid=cap['uuid'])
                r = requests.get(uri, timeout=10)

                extension = mimetypes.guess_extension(r.headers['content-type'])
                extension = ".jpg" if not extension else extension
                filename = "{}{}".format(cap['imageid'], extension)
                folder = "./imgs/%s" % (cap['imageid'][-2:])  # use folder named its last 2 digits
                # if not os.path.exists(folder):
                #     os.makedirs(folder)

                # with open(os.path.join(folder, filename), 'wb') as f:
                #     f.write(r.content)

                io = BytesIO(r.content)

                collection_image = CollectionImage()
                collection_image.collection = collection
                collection_image.file_path.save("{}_{}.jpg".format(user.id, int(time.time())), File(io))
                collection_image.name = collection_image.file_path.path.split('/')[-1]
                collection_image.save()
                end_time = time.time()
                run_time = end_time - begin_time
                global remain_images
                remain_images -= 1
                collection.estimated_time = str(remain_images * run_time / 60 / 60)
                collection.save()
                donenum = len(captures) - q.qsize()
                if donenum % 19 == 0:
                    print("last: %14s, %5d, %.1f%%" % (filename, donenum, donenum / len(captures) * 100))
            except TimeoutError:
                continue
            except queue.Empty:
                collection.estimated_time = 0
                collection.save()
                break

    q = queue.Queue()
    threads = []
    stopFlag = False

    for capture in captures:
        q.put(capture)
        if imgslimit and q.qsize() == imgslimit:  # apply the imgslimit
            break

    for i in range(threadnum):
        thread = Thread(target=down)
        threads.append(thread)
        thread.setDaemon(True)
        thread.start()

    try:
        for thread in threads:
            thread.join()  # wait until they have done
    except KeyboardInterrupt:
        stopFlag = True


@task(bind=True)
def update_image(self, collection):
    collection.is_new = False
    # turl = "http://api.repo.nypl.org/api/v1/items/{uuid}?page={page}"
    turl = "http://api.repo.nypl.org/api/v1/items/recent"
    captures = []
    data = fetchData(turl)
    nres = len(data['capture'])
    for item in data['capture']:
        capture = {"type": item['typeOfResource'], "uuid": item['uuid'], "imageid": item['imageID']}
        captures.append(capture)
    print(pd.DataFrame(captures))
    pass
