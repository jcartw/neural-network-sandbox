# From: "Tutorial: Google Vision API with Python and Heroku" by Evan Fang
# Medium Link: https://towardsdatascience.com/tutorial-google-vision-api-with-python-and-heroku-3b8d3ff5f6ef

# Note: make sure to import environment variables

from google.cloud import vision

pics = ['https://i.imgur.com/2EUmDJO.jpg', 'https://i.imgur.com/FPMomNl.png']

client = vision.ImageAnnotatorClient()
image = vision.types.Image()

for pic in pics:
    image.source.image_uri = pic
    response = client.face_detection(image=image)

    print('=' * 79)
    print('File: {pic}')
    for face in response.face_annotations:
        likelihood = vision.enums.Likelihood(face.surprise_likelihood)
        vertices = [f'({v.x},{v.y})' for v in face.bounding_poly.vertices]
        print(f'Face surprised: {likelihood.name}')
        print(f'Face bounds: {",".join(vertices)}')