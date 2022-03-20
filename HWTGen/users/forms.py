from django import forms
from django.contrib.auth.models import User
from django.contrib.auth.forms import UserCreationForm
from django.utils.translation import ugettext_lazy as _
from django.core.exceptions import ValidationError
from crispy_forms.bootstrap import StrictButton
from crispy_forms.helper import FormHelper
from crispy_forms.layout import Layout, HTML, ButtonHolder

class UserRegisterForm(UserCreationForm):
    email = forms.EmailField()
 
    class Meta:
        model = User
        fields = ['username', 'first_name', 'last_name', 'email', 'password1', 'password2']
 
class UserUpdateForm(forms.ModelForm):
    email = forms.EmailField()
 
    class Meta:
        model = User
        fields = ['username', 'email']
 
class PasswordResetRequestForm(forms.Form):
    """
    Password Reset Form - request to reset password
    """
    email = forms.CharField(
        label = "Email Address",
        max_length = 254,
        widget = forms.TextInput,
        required = True,
    )

    def __init__(self, *args, **kwargs):
        super(PasswordResetRequestForm, self).__init__(*args, **kwargs)

        field_username = self.fields.get('email')
        field_username.widget.attrs['placeholder'] = 'Email Address*'
        field_username.label = ''

        self.helper = FormHelper()
        self.helper.form_class = 'form-horizontal' 
        self.helper.attrs ={'novalidate':''}
        self.helper.form_id='id-passwordresetrequest-form'
        self.helper.form_method='post'
        self.helper.form_action='ResetPassword_private'
        self.helper.layout=Layout(
            'email',
        )
        
    def clean_email(self):
        email = self.cleaned_data.get('email')
        try:
            User.objects.get(email=email)
        except User.DoesNotExist:
            raise forms.ValidationError("Invalid email, please try again.")
        return email

class SetNewPasswordForm(forms.Form):
    """
    Set Password Form - associated with PasswordResetRequestForm
    """
    new_password1 = forms.CharField(
        label = "Password",
        max_length = 128,
        widget = forms.PasswordInput(render_value=False),
        required = True
    )
    new_password2 = forms.CharField(
        label = "Confirm Password",
        max_length = 128,
        widget = forms.PasswordInput(render_value=False),
        required = True
    )

    def __init__(self, *args, **kwargs):
        super(SetNewPasswordForm, self).__init__(*args, **kwargs)

        field_password1 = self.fields.get('new_password1')
        field_password1.widget.attrs['placeholder'] = 'Password*'
        field_password1.label = ''
        field_password2 = self.fields.get('new_password2')
        field_password2.widget.attrs['placeholder'] = 'Confirm Password*'
        field_password2.label = ''

        self.helper = FormHelper()
        self.helper.form_class = 'form-horizontal' 
        self.helper.attrs ={'novalidate':'', 'role': 'form'}
        self.helper.form_id='id-setnewpassword-form'
        self.helper.form_method='post'
        self.helper.form_action=''
        self.helper.layout=Layout(
            'new_password1', 'new_password2',
        )

    def clean_new_password2(self):
        """
        Method to check that password1 corresponds to the
        confirmation password2 during form validation
        """
        password2 = self.cleaned_data.get('new_password2')
        password1 = self.cleaned_data.get('new_password1')
        if password1 != password2:
            raise forms.ValidationError(
                "The passwords did not match, please try again.")
        return password2

class ChangePasswordForm(forms.Form):
    """
    Change Password Form - associated with ManageUserProfile
    """

    new_password1 = forms.CharField(
        label = "New Password",
        max_length = 128,
        widget = forms.PasswordInput(render_value=False),
        required = True
    )
    new_password2 = forms.CharField(
        label = "Confirm New Password",
        max_length = 128,
        widget = forms.PasswordInput(render_value=False),
        required = True
    )

    def __init__(self, *args, **kwargs):
        super(ChangePasswordForm, self).__init__(*args, **kwargs)

        field_password = self.fields.get('new_password1')
        field_password.widget.attrs['placeholder'] = 'New Password*'
        field_password.label = ''
        field_password1 = self.fields.get('new_password2')
        field_password1.widget.attrs['placeholder'] = 'Confirm New Password*'
        field_password1.label = ''

        self.helper = FormHelper()
        self.helper.form_class = 'form-horizontal' 
        self.helper.attrs ={'novalidate':''}
        self.helper.form_id='id-changepassword-form'
        self.helper.form_method='post'
        self.helper.form_action='ChangePassword_private'
        self.helper.layout=Layout(
            'new_password1', 'new_password2',
            ButtonHolder(
                StrictButton(
                    '<span class="glyphicon glyphicon-ok" \
                    aria-hidden="true"></span> %s' % "Submit",
                    type='submit',
                    css_class='btn btn-primary',
                ),
                #HTML('<a class="btn btn-primary" href="/ManageUserProfile/">Back to Manage Profile</a>'),
                HTML('<a class="btn btn-primary" href="/Index/">Cancel</a>')
            ),
            HTML('<h4 class="form-signin-heading">Back to Manage Profile? <a href="/ManageUserProfile/">Click here</a>.</h4>'),
        )

    def clean_new_password2(self):
        """
        Method to check that the password corresponds to the
        confirmation password during form validation
        """
        password2 = self.cleaned_data.get('new_password2')
        password1 = self.cleaned_data.get('new_password1')
        if password1 != password2:
            raise forms.ValidationError(
                "The passwords did not match, please try again.")
        return password2
  
    def clean(self):
        username = self.cleaned_data.get('username')
        if User.objects.filter(username=username).exists():
            user = User.objects.get(username=username)
            if user.is_active == True:
                raise ValidationError("Username already assigned, please select another.")
            
        if 'password1' in self.cleaned_data and 'password2' in self.cleaned_data:
            if self.cleaned_data['password1'] != self.cleaned_data['password2']:
                raise ValidationError(_("The two password fields did not match."))
            
        if 'email' in self.cleaned_data and 'email1' in self.cleaned_data:
            if self.cleaned_data['email'] != self.cleaned_data['email1']:
                raise ValidationError(_("The two email fields did not match."))
        return self.cleaned_data
    
