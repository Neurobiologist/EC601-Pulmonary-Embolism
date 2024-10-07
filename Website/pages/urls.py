from django.urls import path, include
from . import views as v
from django.contrib.auth import views as auth_views


urlpatterns = [
    path('', v.index, name='index'),
    path("accounts/", include("django.contrib.auth.urls")),
    path('register/', v.register, name='register'),
    path('success/', v.success, name='success'),
    path('dashboard/', v.dashboard, name='dashboard'),
    path('patient_notes/', v.patient_notes, name='patient_notes'),
    path('ctpa_results/', v.ctpa_results, name='ctpa_results'),
]
