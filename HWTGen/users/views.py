from django.shortcuts import render, redirect, get_object_or_404
from django.http import HttpResponse, HttpResponseRedirect
from django.template import RequestContext
from django.contrib import messages
from django.contrib.auth import authenticate, login, logout, get_user_model
from django.conf import settings
from django.views.generic.edit import FormView
from django.views.decorators.csrf import csrf_exempt
from django.contrib.auth.tokens import default_token_generator
from django.utils.encoding import force_bytes
from django.utils.http import urlsafe_base64_encode, urlsafe_base64_decode
from django.core.mail import BadHeaderError
from django.template import loader
from users.forms import UserRegisterForm, PasswordResetRequestForm, SetNewPasswordForm, ChangePasswordForm
from pathlib import Path
import subprocess
import logging

# Create your views here.
User = get_user_model()
logger = logging.getLogger(__name__)


def script_function(post_from_form):
    logger.info("run email script")
    venvpath = str(Path(r'/Users/vipulgoyal/Documents/ITP/py3.8.10venv/bin', 'activate'))  # Python venv3.9.0
    path = str(settings.BASE_DIR) + "/scripts"
    emailscriptpath = str(Path(path, 'sendmail.py'))

    logger.info("PATH: " + str(path))
    logger.info(venvpath)
    logger.info(emailscriptpath)

    cmd = 'source ' + venvpath + '; python ' + emailscriptpath + ' "' + post_from_form[0] + '" "' + post_from_form[
        1] + '" "' + post_from_form[2] + '"'
    return subprocess.run(cmd, capture_output=True, shell=True, executable='/bin/bash', timeout=1800)


def login_dropdown(request):
    next = request.GET.get('next', '/')
    try:
        username = request.POST['username']
        password = request.POST['password']
        auth_user = authenticate(request, username=username, password=password)
        try:
            login(request, auth_user)
            return HttpResponseRedirect(next)
        except:
            messages.error(request, 'Invalid credentials')
            return HttpResponseRedirect(next)
    except (KeyError):
        messages.error(request, 'Invalid credentials')
        # return render(request, '/', { 'message': "Invalid username or password. Please try again." })


def register(request):
    if request.method == 'POST':
        form = UserRegisterForm(request.POST)
        if form.is_valid():
            form.save()
            username = form.cleaned_data.get('username')
            try:  # create settings for newly created user
                user = get_object_or_404(User, username=username)
            except:
                pass

            messages.success(request, f'Your account has been created! You can now login!')
            return redirect('login')
    else:
        form = UserRegisterForm()
    return render(request, 'users/register.html', {'form': form})


class ResetPasswordRequest(FormView):
    """
    Request for a password reset (forgot password) form display
    """
    template_name = "Users/password_reset.html"
    success_url = 'password-reset-processed/'
    form_class = PasswordResetRequestForm

    @csrf_exempt
    def post(self, request, *args, **kwargs):
        args = {}
        if request.method == 'POST':
            form = PasswordResetRequestForm(request.POST)
            if form.is_valid():
                # username = form.cleaned_data["username"]
                email = form.cleaned_data["email"]

                try:
                    if User.objects.filter(email=email).exists():
                        try:
                            user = User.objects.get(email=email)
                            if user is not None:
                                post_from_form = []
                                sitenamestr = u'\u00A9'
                                sitename = "(New) Cadillac Database" + sitenamestr
                                c = {
                                    'email': user.email,
                                    'domain': request.META['HTTP_HOST'],
                                    'site_name': sitename,
                                    'uid': urlsafe_base64_encode(force_bytes(user.pk)),
                                    'user': user,
                                    'token': default_token_generator.make_token(user),
                                    'protocol': 'http',
                                }
                                emailtemplatename = 'Users/password_reset_email.html'
                                subject = 'Request Username or Reset Password'
                                email = loader.render_to_string(emailtemplatename, c)
                                post_from_form.append(subject)
                                post_from_form.append(user.email)
                                post_from_form.append(email)
                                # dummy_script_function(post_from_form) # CHANGE TO SCRIPT
                                # logger.info(post_from_form)
                                output = script_function(post_from_form)
                                # logger.info("Mail sent", str(output))
                        except User.DoesNotExist:
                            user = None
                            messages.error(request, 'Invalid username, please try again.')
                            return self.form_invalid(form)
                    else:
                        messages.error(request, 'Invalid username, please try again.')
                        return self.form_invalid(form)
                except BadHeaderError:
                    return HttpResponse('Invalid header found.')
                return render(request, 'Users/password_reset_processed.html')
        else:
            form = PasswordResetRequestForm()
        args['form'] = form
        return render(request, "Users/password_reset.html", args)


def PasswordResetProcessed(request):
    return render(request, 'Users/password_reset_processed.html')


class PasswordResetConfirm(FormView):
    """
    Password reset (after requested for a password reset) form display
    """
    template_name = "Users/password_reset_change.html"
    success_url = 'password-reset-success/'
    form_class = SetNewPasswordForm

    def post(self, request, uidb64=None, token=None, *arg, **kwargs):

        # View that checks the hash in a password reset link and presents a
        # form for entering a new password.

        UserModel = get_user_model()
        form = self.form_class(request.POST)
        assert uidb64 is not None and token is not None  # checked by URLconf
        try:
            uid = urlsafe_base64_decode(uidb64)
            user = UserModel._default_manager.get(pk=uid)
        except (TypeError, ValueError, OverflowError, UserModel.DoesNotExist):
            user = None

        if user is not None and default_token_generator.check_token(user, token):
            if form.is_valid():
                new_password1 = form.cleaned_data['new_password1']
                new_password = form.cleaned_data['new_password2']
                user.set_password(new_password)
                user.save()
                return render(request, 'Users/password_reset_success.html')
            else:
                # messages.error(request, 'Password reset was unsuccessful.')
                return self.form_invalid(form)
        else:
            messages.error(request, 'Password change was unsuccessful.')
            return self.form_invalid(form)


def PasswordResetSuccess(request):
    return render(request, 'Users/password_reset_success.html')


class ChangePassword(FormView):
    """
    Change password (from manage user profile) form display
    """
    form_class = ChangePasswordForm
    template_name = "Users/password_change.html"
    success_url = 'password-change-success/'

    def post(self, request, *arg, **kwargs):
        args = {}
        if request.method == 'POST':
            form = ChangePasswordForm(request.POST)
            if form.is_valid():
                current_user = User.objects.get(username=request.user.username)
                try:
                    if current_user is not None:  # need to add error-handling to this if-condition
                        # change user password
                        new_password1 = form.cleaned_data['new_password1']
                        new_password = form.cleaned_data['new_password2']
                        current_user.set_password(new_password)
                        current_user.save()
                except BadHeaderError:
                    return HttpResponse('Invalid header found.')
            else:
                messages.error(request, 'Invalid input.')
        else:
            form = ChangePasswordForm()
        args['form'] = form
        return render(request, "Users/password_change.html", args)


def PasswordChangeSuccess(request):
    return render(request, 'Users/password_change_success.html')


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
