from django.shortcuts import render
from django.contrib.auth.decorators import login_required
from django.contrib.auth import logout
from pages.models import Patient, CTPA, Medication

# Create your views here.
def index(request):
    return render(request, 'pages/index.html')

def register(request):
    return render(request, 'registration/register.html')

def success(request):
    return render(request, 'registration/success.html')

def logout(request):
    logout(request)
    #return render(request, 'registration/logout.html')

def logout_success(request):
    return render(request, 'registration/logout_success.html')

@login_required
def dashboard(request):
    patient = Patient.objects.get(pk=1) # Example for prototype
    medication_list = Medication.objects.filter(patient=patient)
    context = {'pt': patient,
               'medication_list': medication_list}
    return render(request, 'pages/dashboard.html', context)

@login_required
def patient_notes(request):
    patient = Patient.objects.get(pk=1) # Example for prototype
    context = {'pt': patient}
    return render(request, 'pages/patient_notes.html', context)

@login_required
def ctpa_results(request):
    patient = Patient.objects.get(pk=1) # Example for prototype
    scan = CTPA.objects.get(pk=1) # Real CTPA example
    context = {'pt': patient,
               'img': scan}
    return render(request, 'pages/ctpa_results.html', context)
