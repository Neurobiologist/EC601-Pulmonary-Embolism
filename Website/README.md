# Website: CT Pulmonary Angiogram Results & Interpretation

We have created a clinician-facing web interface with Django to provide patient CT Pulmonary Angiogram (CTPA) results and interpretation in a user-friendly way. Our goal is to provide the results of our algorithm in an accessible manner that is easily integrated into existing hospital workflows.

Our website features three main pages:

* **Dashboard**: Provides an overview of patient data, including basic medical history and a list of medications (including medication name, Signatura [SIG], and active/inactive status)
* **Patient Chart**: Medical chart detailing the patient's medical history and clinical data
* **CTPA Results**: Displays CTPA results, including predicted probability of PE along with class activation maps (CAMs) for further interpretation and analysis by the radiologist

We also integrated a few additional features:
* The medication list is searchable and sortable. This makes it easier to obtain signatura for a particular medication or sort medications by active/inactive status.
* The sidebar navigation is collapsible to maximize the view of charts or scans.

The CAMs displayed on our website are a vital resource to the radiologist. By providing insight into how the algorithm outputs the final prediction, we provide an additional data point to consider during diagnosis. This complements hermeneutic analysis necessary in medicine, and allows the physician to contextualize or better interpret the scalar probability value. The CAMs may even draw attention to clinically important areas of the scan that would have otherwise gone unnoticed by the radiologist, many of whom are inundated with work and subject to human error.

It is our goal to provide algorithm results in an intuitive and user-friendly way that promotes understanding and adoption in clinical settings.
