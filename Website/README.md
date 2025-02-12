# Website: CT Pulmonary Angiogram Results & Interpretation

## Overview

We have created a clinician-facing web interface with Django to provide patient CT Pulmonary Angiogram (CTPA) results and interpretation in a user-friendly way. Our goal is to provide the results of our algorithm in an accessible manner that is easily integrated into existing hospital workflows.

Our website features three main pages:

* **Dashboard**: Provides an overview of patient data, including basic medical history and a list of medications (including medication name, Signatura [SIG], and active/inactive status)
* **Patient Notes**: Dynamic webpage indicating patient risk factors for PE based on dashboard data
* **CTPA Results**: Displays CTPA results, including predicted probability of PE along with class activation maps (CAMs) for further interpretation and analysis by the radiologist

## Website Access Instructions

The website prototype can be accessed [here](https://ctpa.pythonanywhere.com/).

Username: radiologist

Password: password123

## Website Features

This prototype was built with the following features:
* SQLite database with artificial patient data (for demonstration only)
* Patient chart populated with example data from the database
* Risk factor calculations (based on the scientific literature) using the <code>@property</code> decorator, which uses logic to get computed model attributes without modifying the database
  * This allows the database to store patient data exclusively, while model properties are used to determine patient risk factors
  * These calculated risk factors are then highlighted in the "Patient Notes" to help the clinician contextualize the CTPA results
  * When patient data is updated in the database, the model properties (risk factor assesment) is updated automatically
* CTPA results with probability of PE and CAMs to aid in interpretation of the data

Additionally, Bootstrap template features include the following highlights:
* The medication list is searchable and sortable. This makes it easier to obtain signatura for a particular medication or sort medications by active/inactive status.
* The sidebar navigation is collapsible to maximize the view of charts or scans.


## Conclusion & Future Directions

The CAMs displayed on our website are a vital resource to the radiologist. By providing insight into how the algorithm outputs the final prediction, we provide an additional data point to consider during diagnosis. This complements hermeneutic analysis necessary in medicine, and allows the physician to contextualize or better interpret the scalar probability value. The CAMs may even draw attention to clinically important areas of the scan that would have otherwise gone unnoticed by the radiologist, many of whom are inundated with work and subject to human error.

Future directions include integration with existing electronic medical records (EMRs); HIPAA compliance and security measures (including upgrades to a fully HIPAA-compliant database); and expansion or automation of the risk factor calculations based on advancements in our understanding of PE risk factors in the scientific literature.

It is our goal to provide algorithm results in an intuitive and user-friendly way that promotes understanding and adoption in clinical settings.
