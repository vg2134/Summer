from django import forms
from phonenumber_field.widgets import PhoneNumberInternationalFallbackWidget
from hwtgenapp.models import ContactInfo

class ContactInfoForm(forms.ModelForm):    
    class Meta:
        model = ContactInfo
        fields = ['firstname', 'lastname', 'email', 'phonenumber', 'message']
        widgets = {'phonenumber':PhoneNumberInternationalFallbackWidget()}
        labels = {
            'firstname': 'First Name',
            'lastname': 'Last Name',
            'phonenumber': 'Phone Number'
        }
        help_texts = {
            'phonenumber': 'Precede phone number with + country code (e.g., +15556789 for U.S. numbers).',
        }
