"""
URL configuration for chequeprojet project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/5.2/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path
from users import views as uviews
from admins import views as adviews
from django.conf import settings
from django.conf.urls.static import static

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


 





    # ==================== ADMIN VIEWS ====================
    path("admin-login/", adviews.adminlogin, name="adminlogin"),
    path("admin-home/", adviews.adminhome, name="adminhome"),
    path("admin-logout/", adviews.adminlogout, name="adminlogout"),
    path('admin-users/', adviews.admin_users_list, name='admin_users_list'),
    path('activate-user/<int:user_id>/', adviews.activate_user, name='activate_user'),
    path('block-user/<int:user_id>/', adviews.block_user, name='block_user'),
    path('unblock-user/<int:user_id>/', adviews.unblock_user, name='unblock_user'),
    path('delete-user/<int:user_id>/', adviews.delete_user, name='delete_user'),
]

if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
    urlpatterns += static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)
