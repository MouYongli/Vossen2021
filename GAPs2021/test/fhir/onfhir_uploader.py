import os
import pandas as pd
import json
from fhirpy import SyncFHIRClient
import requests

########################################################
# Create Organization and ImagingStudy
########################################################
# group = {
#     "id": "sl",
#     "name": "Skin Lesion",
#     "type": "person",
#     "actual": True,
#     "resourceType": "Group"
# }
# url = 'http://192.168.96.129:8090/fhir/Group/sl'
# headers = {'content-type': 'application/json'}
# r = requests.put(url, headers=headers, data=json.dumps(group))
# print(r.text)
# #
# imagestudy = {
#     "id": "isic",
#     "status":"available",
#     # "subject": {"reference":"Group/sl"},
#     "resourceType": "ImagingStudy",
# }
# url = 'http://192.168.96.129:8090/fhir/ImagingStudy/isic'
# headers = {'content-type': 'application/json'}
# r = requests.put(url, headers=headers, data=json.dumps(imagestudy))
# print(r.text)


########################################################
# Create Patient and Media to HAPI
########################################################
# patient = {
#     "gender":"female",
#     "meta":{"profile":["https://www.medizininformatik-initiative.de/fhir/core/StructureDefinition/Patient"]},
#     "resourceType":"Patient",
#     "birthDate":"1966-07-01"
# }
# url = 'http://192.168.96.129:8090/fhir/Patient/'
# headers = {'content-type': 'application/json'}
# r = requests.post(url, headers=headers, data=json.dumps(patient))
# print(r.text)
# obj = json.loads(r.text)
# media = {
#     "id":"0000000-isic-image",
#     "resourceType":"Media",
#     "status":"completed",
#     "subject":{"reference":"Patient/{}".format(obj['id'])},
#     # "encounter":{"reference":"ImagingStudy/isic"},
#     "content":{
#         "contentType":"image/jpeg",
#         "url":"http://menzel.informatik.rwth-aachen.de:9000/isic/0000000.jpg"
#     },
#     "bodySite":{"text":"anterior torso"},
#     "note":[{"text":"NV"}]
# }
# url = 'http://192.168.96.129:8090/fhir/Media'
# headers = {'content-type': 'application/json'}
# r = requests.post(url, headers=headers, data=json.dumps(media))
# print(r.text)

patient = {
    "gender":"female",
    "id":"0000025",
    "meta":{"profile":["https://www.medizininformatik-initiative.de/fhir/core/StructureDefinition/Patient"]},
    "resourceType":"Patient",
    "birthDate":"1966-07-01"
}
url = 'http://192.168.96.129:8080/fhir/Patient/0000025'
headers = {'content-type': 'application/json'}
r = requests.put(url, headers=headers, data=json.dumps(patient))
print(r.text)
media = {
    "id":"0000025-isic-image",
    "resourceType":"Media",
    "status":"completed",
    "subject":{"reference":"Patient/0000025"},
    # "encounter":{"reference":"ImagingStudy/isic"},
    "content":{
        "contentType":"image/jpeg",
        "url":"http://menzel.informatik.rwth-aachen.de:9000/isic/0000025.jpg"
    },
    "bodySite":{"text":"anterior torso"},
    "note":[{"text":"NV"}]
}
url = 'http://192.168.96.129:8080/fhir/Media/0000025-isic-image'
headers = {'content-type': 'application/json'}
r = requests.put(url, headers=headers, data=json.dumps(media))
print(r.text)

########################################################
# Example: Get and Delete Patient
########################################################
# url = 'http://192.168.96.129:8080/fhir/Patient'
# r = requests.get(url)
# print(r.text)
#
# url = 'http://192.168.96.129:8080/fhir/Patient/0000000'
# r = requests.delete(url)
# print(r.text)


# print("==========================")
# client = SyncFHIRClient('http://{}:{}/fhir'.format('192.168.96.129', '8080'))
# patients = client.resources('Patient')  # Return lazy search set
# patients_data = []
# for patient in patients:
#     patient_birthDate = None
#     try:
#         patient_birthDate = patient.birthDate
#     except:
#         pass
#     # patinet_id, gender, birthDate
#     patients_data.append([patient.id, patient.gender, patient_birthDate])
# patients_df = pd.DataFrame(patients_data, columns=["patient_id", "gender", "birthDate"])
# print(patients_df)
# media_list = client.resources('Media').include('Patient', 'subject')
# media_data = []
# for media in media_list:
#     media_bodySite = None
#     media_reasonCode = None
#     media_note = None
#     try:
#         media_bodySite = media.bodySite.text
#     except:
#         pass
#     try:
#         media_reasonCode = media.reasonCode[0].text
#     except:
#         pass
#     try:
#         media_note = media.note[0].text
#     except:
#         pass
#     media_data.append([media.subject.id, media.id, media_bodySite, media_reasonCode, media_note, media.content.url])
# media_df = pd.DataFrame(media_data, columns=["patient_id", "media_id", "bodySite", "reasonCode", "note", "image_url"])
# data_df = pd.merge(patients_df, media_df, on='patient_id', how='outer')
# data_df = data_df[data_df['note'].notna()].reset_index()
# print(data_df)

