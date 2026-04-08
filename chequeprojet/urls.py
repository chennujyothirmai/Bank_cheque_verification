from django.contrib import admin
from django.urls import path, re_path
from users import views as uviews
from admins import views as adviews
from django.conf import settings
from django.conf.urls.static import static
from django.views.static import serve

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', uviews.basefunction, name='basefunction'),
    path('userlogin/', uviews.userlogin, name='userlogin'),    
    path('register/', uviews.register, name='register'),
    path('logout/', uviews.logout_view, name='logout'),
    path('userhome/', uviews.userhome, name='userhome'),
    path("ChequeSamples/", uviews.cheque_samples, name="ChequeSamples"),
    path("prediction/", uviews.prediction, name="prediction"),
    path("model_evaluation/", uviews.model_evaluation, name="model_evaluation"),

    # Admin
    path("admin-login/", adviews.adminlogin, name="adminlogin"),
    path("admin-home/", adviews.adminhome, name="adminhome"),
    path("admin-logout/", adviews.adminlogout, name="adminlogout"),
    path('admin-users/', adviews.admin_users_list, name='admin_users_list'),
    path('activate-user/<int:user_id>/', adviews.activate_user, name='activate_user'),
    path('block-user/<int:user_id>/', adviews.block_user, name='block_user'),
    path('unblock-user/<int:user_id>/', adviews.unblock_user, name='unblock_user'),
    path('delete-user/<int:user_id>/', adviews.delete_user, name='delete_user'),

    # Media and Static serving FIX for Render
    re_path(r'^media/(?P<path>.*)$', serve, {'document_root': settings.MEDIA_ROOT}),
    re_path(r'^static/(?P<path>.*)$', serve, {'document_root': settings.STATIC_ROOT}),
]

if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
    urlpatterns += static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)
