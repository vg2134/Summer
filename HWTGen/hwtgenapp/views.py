from django.shortcuts import render
from django.template import RequestContext
from django.conf import settings
from hwtgenapp.forms import ContactInfoForm
from django.db.models import Q
from hwtgenapp.models import Topic, TopicDetail, ContactInfo
from django.core.mail import send_mail
from pathlib import Path
import subprocess

# Create your views here.
def index(request):
    # intro_text = TopicDetail.objects.values_list('content').get(pk=4)
    # latestimageseq = TopicImage.objects.all().aggregate(Max('imageser'))['imageser__max']
    # presentation_slides = TopicImage.objects.filter(topicdetailid = 6, imageser = latestimageseq).order_by('orderid')

    #return render(request, 'archemywebapp/index.html', {'banner_image': banner_image, 'top_slogan': top_slogan, 'sub_slogan': sub_slogan, 'intro_text': intro_text, 'catalogue_link': catalogue_link, 'presentation_anchor': presentation_slides})
    return render(request, "hwtgenapp/index.html", {'is_home_page':True})
    #return render(request, "hwtgenapp/index.html", { 'intro_text': intro_text, 'presentation_slides':presentation_slides })

def about(request):
    topicdetails = TopicDetail.objects.filter(Q(topicid = 2) & Q(sitemapenabled = 1)).order_by('topicorderid')
    topic = Topic.objects.get(pk = 2)
    cards_scope = 'public'
    if request.user.is_authenticated:
        cards_scope = 'private'
    return render(request, 'hwtgenapp/about.html', {'topicdetails':topicdetails, 'cards_scope':cards_scope, 'topic':topic})

def legal(request):
    topicdetails = TopicDetail.objects.filter(Q(topicid = 3) & Q(sitemapenabled = 1)).order_by('topicorderid')
    topic = Topic.objects.get(pk = 3)
    cards_scope = 'public'
    if request.user.is_authenticated:
        cards_scope = 'private'
        
    return render(request, 'hwtgenapp/legal.html', {'topicdetails':topicdetails, 'cards_scope':cards_scope, 'topic':topic})

def sponsors(request, sponsorname):
    if sponsorname == 'nypl':
        return render(request, 'hwtgenapp/sponsors/nypl_sponsor_page.html', { 'sponsorname':sponsorname })

def contact_us(request):
    if request.method == 'POST':
        form = ContactInfoForm(request.POST)
        if form.is_valid():
            #form.save()
            cards_scope = 'public'
            email_content = []
            firstname = form.cleaned_data['firstname']
            lastname = form.cleaned_data['lastname']
            phonenumber = form.cleaned_data['phonenumber']
            sender_email = form.cleaned_data['email']
            from_email = settings.DEFAULT_FROM_EMAIL 
            to_email = settings.DEFAULT_FROM_EMAIL
            email_subject = "Voixla Contact Request"
            content = form.cleaned_data['message'] 
            email_message = f"Sender: {email_subject}\nReply-to: {sender_email}\nPhone Number: {phonenumber}\n\nFirstname: {firstname}\nLastname: {lastname}\n\nMessage: {content}"
            
            if request.user.is_authenticated:
                cards_scope = 'private'
                contactinfo = ContactInfo(username=request.user.username, firstname=firstname, lastname=lastname, email=from_email, phonenumber=phonenumber, message=email_message)
                contactinfo.save()
            else:
                contactinfo = ContactInfo(firstname=firstname, lastname=lastname, email=from_email, phonenumber=phonenumber, message=email_message)
                contactinfo.save()

            # REPLACE WITH CALL TO email script (see below)
            send_mail(
                email_subject,
                email_message,
                from_email,
                [to_email],
                fail_silently=False,
            )
            
#             email_content.append(email_subject)
#             email_content.append(from_email)
#             email_content.append(email_message)
#             output = script_function(email_content)
            
            return render(request, 'hwtgenapp/contact_success.html', {'cards_scope':cards_scope})
    else:
        form = ContactInfoForm()
    return render(request, 'hwtgenapp/contact_info.html', {'form':form})

def script_function(post_from_form):
    venvpath = str(Path(r'/home/archemy/python_virtualenv/python3.8.3/bin', 'activate')) # Python venv3.8.3
    #venvpath = str(Path(r'/home/metacomp/python_virtualenv/py3.8.5/bin', 'activate')) # Python venv3.8.5
    path = str(settings.BASE_DIR) + "/scripts"
    emailscriptpath = str(Path(path, 'sendmail.py'))
        
    cmd = 'source ' + venvpath + '; python ' + emailscriptpath + ' "' + post_from_form[0] + '" "' + post_from_form[1] + '" "' + post_from_form[2] + '"'
    return subprocess.run(cmd, capture_output=True, shell=True, executable='/bin/bash', timeout=1800)

# HTTP Error 400
def bad_request(request, Exception):
    response = render('400.html', context_instance=RequestContext(request))
    #response = render_to_response('400.html', context_instance=RequestContext(request))
    response.status_code = 400
    return response

# HTTP Error 403
def permission_denied(request, Exception):
    response = render('403.html', context_instance=RequestContext(request))
    #response = render_to_response('403.html', context_instance=RequestContext(request))
    response.status_code = 403
    return response

# HTTP Error 404
def page_not_found(request, Exception):
    response = render('404.html', context_instance=RequestContext(request))
    #response = render_to_response('404.html', context_instance=RequestContext(request))
    response.status_code = 404
    return response

# HTTP Error 500
def server_error(request):
    response = render('500.html', context_instance=RequestContext(request))
    #response = render_to_response('500.html', context_instance=RequestContext(request))
    response.status_code = 500
    return response
