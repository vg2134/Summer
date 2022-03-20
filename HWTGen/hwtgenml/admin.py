from django.contrib import admin
from hwtgenml.models import *

# Register your models here.
admin.site.register(Vendor)
admin.site.register(VendorCollection)


class CollectionTextAdmin(admin.ModelAdmin):
    list_display = ['image', 'initial_text']


admin.site.register(BoostrapModel)
admin.site.register(CaptionImage)
admin.site.register(Caption)
admin.site.register(UserModel)
admin.site.register(Collection)
admin.site.register(CollectionImage)
admin.site.register(CollectionText, CollectionTextAdmin)
