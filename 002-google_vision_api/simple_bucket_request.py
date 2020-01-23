
# Base on: "ML Study Jam — Vision API" by Evan Fang
# Medium Link: https://towardsdatascience.com/ml-study-jam-detect-labels-faces-and-landmarks-in-images-with-the-cloud-vision-api-a80e89feb66f 

# Google Vision REST API Docs: https://cloud.google.com/vision/docs/reference/rest/v1/AnnotateImageRequest#Image

import requests
import pprint
import os

# Load credentials
try:
    GOOGLE_BUCKET_API_KEY = os.environ["GOOGLE_BUCKET_API_KEY"]
    GCS_IMAGE_URI = os.environ["GCS_IMAGE_URI"]
except:
    raise Exception("ERROR: credentials not found.")

pp = pprint.PrettyPrinter(indent=3)

url = f"https://vision.googleapis.com/v1/images:annotate?key={GOOGLE_BUCKET_API_KEY}"

request = {
    "requests": [{
        "image": { "source": {"gcsImageUri" : GCS_IMAGE_URI} },
        "features": [ { "type": "LABEL_DETECTION", "maxResults": 10 } ]
        }
    ]
}

# Make POST request to Google API Vision
r = requests.post(url=url, json=request)
res_body = r.json()
pp.pprint(res_body)

##
# {  'responses': [  {  'labelAnnotations': [  {  'description': 'Event',
#                                                 'mid': '/m/081pkj',
#                                                 'score': 0.89533305,
#                                                 'topicality': 0.89533305},
#                                              {  'description': 'Fireworks',
#                                                 'mid': '/m/0g6b5',
#                                                 'score': 0.83875424,
#                                                 'topicality': 0.83875424},
#                                              {  'description': 'Friendship',
#                                                 'mid': '/m/019_nn',
#                                                 'score': 0.8311425,
#                                                 'topicality': 0.8311425},
#                                              {  'description': 'Fête',
#                                                 'mid': '/m/07bb2_',
#                                                 'score': 0.8242853,
#                                                 'topicality': 0.8242853},
#                                              {  'description': 'Fun',
#                                                 'mid': '/m/0ds99lh',
#                                                 'score': 0.8220452,
#                                                 'topicality': 0.8220452},
#                                              {  'description': "New year's eve",
#                                                 'mid': '/m/01pl3y',
#                                                 'score': 0.7846255,
#                                                 'topicality': 0.7846255},
#                                              {  'description': 'Crowd',
#                                                 'mid': '/m/03qtwd',
#                                                 'score': 0.76387537,
#                                                 'topicality': 0.76387537},
#                                              {  'description': 'Interaction',
#                                                 'mid': '/m/01ckgp',
#                                                 'score': 0.76290643,
#                                                 'topicality': 0.76290643},
#                                              {  'description': 'Smile',
#                                                 'mid': '/m/019nj4',
#                                                 'score': 0.75412655,
#                                                 'topicality': 0.75412655},
#                                              {  'description': 'Holiday',
#                                                 'mid': '/m/03gkl',
#                                                 'score': 0.74658704,
#                                                 'topicality': 0.74658704}]}]}
#