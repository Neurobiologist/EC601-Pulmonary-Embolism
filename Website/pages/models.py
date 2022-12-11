from django.conf import settings
from django.db import models
from django.utils import timezone
from dateutil.relativedelta import relativedelta

# Create your models here.

class Patient(models.Model):
    patient_id = models.CharField(max_length=50, unique=True, blank=False)
    last_name = models.CharField(max_length=100, blank=False)
    first_name = models.CharField(max_length=100, blank=False)
    date_of_birth = models.DateField()
    sex = models.CharField(max_length=50)
    height = models.IntegerField(default=0)
    weight = models.DecimalField(default=0.00, max_digits=5, decimal_places=2)
    tobacco_status = models.CharField(max_length=50, blank=False, default='Never')
    tobacco_start = models.IntegerField(null=True)
    tobacco_end = models.IntegerField(null=True)
    created_date = models.DateTimeField(auto_now_add=True)
    last_updated = models.DateTimeField(auto_now=True, null=True)

    # Calculate Risk Factors for PE
    ## BMI
    @property
    def bmi(self):
        wt_kg = float(self.weight) * 0.453592
        ht_m2 = (float(self.height)/100)**2
        bmi = wt_kg/ht_m2
        return round(bmi, 1)

    ## Overweight
    @property
    def is_overweight(self):
        return 1 if self.bmi >= 25 else 0

    ## Current or former smoker
    @property
    def is_smoker(self):
        return 1 if self.tobacco_status == 'Current' else 0

    @property
    def was_smoker(self):
        return 1 if self.tobacco_status == 'Former' else 0

    @property
    def years_smoked(self):
        if self.is_smoker:
            return timezone.now().year-self.tobacco_start
        if self.was_smoker:
            return self.tobacco_end-self.tobacco_start
        return 0

    ## Age
    @property
    def age(self):
        calendar_diff = relativedelta(timezone.make_naive(timezone.now()),
        self.date_of_birth)
        return calendar_diff.years

    def publish(self):
        self.last_updated = timezone.now()
        self.save()

    def __str__(self):
        return self.patient_id

class Medication(models.Model):
    patient = models.ForeignKey(Patient, on_delete=models.CASCADE)
    medication_name = models.CharField(max_length=50, blank=False)
    SIG = models.CharField(max_length=300)
    status = models.CharField(max_length=50)

class CTPA(models.Model):
    patient = models.ForeignKey(Patient, on_delete=models.PROTECT)
    scan = models.ImageField()
    probability_of_PE = models.DecimalField(default=0.0000000, max_digits=8, decimal_places=7)
