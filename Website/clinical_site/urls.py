"""clinical_site URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/4.0/topics/http/urls/
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
from django.urls import path, include
from django.contrib.auth import views as auth_views

import django_sb_admin.views
from django.views.generic import TemplateView
from django.urls import path, re_path

from clinical_site import settings
from django.contrib.auth import views as auth_views

from pages import views as v

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', include('pages.urls')),
    path('logout/', auth_views.LogoutView.as_view(next_page=settings.LOGOUT_REDIRECT_URL), name='logout_success'),
    path('logout_success/', v.logout_success, name='logout_success'),

    # Required paths for bootstrap template
    re_path(r'^$', django_sb_admin.views.start, name='sb_admin_start'),
    re_path(r'^dashboard/$', django_sb_admin.views.dashboard, name='sb_admin_dashboard'),
    re_path(r'^charts/$', django_sb_admin.views.charts, name='sb_admin_charts'),
    re_path(r'^tables/$', django_sb_admin.views.tables, name='sb_admin_tables'),
    re_path(r'^forms/$', django_sb_admin.views.forms, name='sb_admin_forms'),
    re_path(r'^bootstrap-elements/$', django_sb_admin.views.bootstrap_elements, name='sb_admin_bootstrap_elements'),
    re_path(r'^bootstrap-grid/$', django_sb_admin.views.bootstrap_grid, name='sb_admin_bootstrap_grid'),
    re_path(r'^rtl-dashboard/$', django_sb_admin.views.rtl_dashboard, name='sb_admin_rtl_dashboard'),
    re_path(r'^blank/$', django_sb_admin.views.blank, name='sb_admin_blank'),
]
