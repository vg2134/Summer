from django.urls import path
from django.contrib.auth import views as auth_views
from django.conf.urls import (handler400, handler403, handler404, handler500)
from users import views as user_views

handler400 = 'users.views.bad_request'
handler403 = 'users.views.permission_denied'
handler404 = 'users.views.page_not_found'
handler500 = 'users.views.server_error'

urlpatterns=[
    path('register/', user_views.register, name='register'),
    #path('login/', user_views.login_dropdown, name='login'),
    path('login/', auth_views.LoginView.as_view(template_name='users/login.html'), name='login'),
    path('logout/', auth_views.LogoutView.as_view(template_name='users/logout.html'), name='logout'),
    path('change-password/', user_views.ChangePassword.as_view(), name="change-password"),
    path('password-reset-processed/', user_views.PasswordResetProcessed, name="password-reset-processed"),
    path('password-reset-success/', user_views.PasswordResetSuccess, name="password-reset-success"),
    path('password-change-success/', user_views.PasswordChangeSuccess, name="password-change-success"),
    path('password-reset-confirm/<uidb64>/<token>/',  user_views.PasswordResetConfirm.as_view(), name="password-reset-confirm"),
    path('password-reset/', user_views.ResetPasswordRequest.as_view(), name="password-reset"),    
]