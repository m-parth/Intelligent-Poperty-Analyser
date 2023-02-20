from django.shortcuts import render
import numpy as np
from joblib import load

model = load('./JupyterNotebooks/ipa.pkl')

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()


def predictor(request):
    return render(request, 'user.html')

def formInfo(request):
    area = request.GET['area']
    bedrooms = request.GET['bedrooms']
    bathrooms = request.GET['bathrooms']
    stories = request.GET['stories']
    mainroad = request.GET['mainroad']
    guestroom = request.GET['guestroom']
    basement = request.GET['basement']
    hotwaterheating = request.GET['hotwaterheating']
    airconditioning = request.GET['airconditioning']
    parking = request.GET['parking']
    prefarea = request.GET['prefarea']
    furnishing = request.GET['furnishing']

    unfurnished = 0
    semi_furnished = 0
    furnished = 0
    if furnishing == "unfurnished":
        unfurnished = 1
    elif furnishing == "semi-furnished":
        semi_furnished = 1
    else:
        furnished = 1

    x = [area, bedrooms, bathrooms, stories, mainroad,guestroom,basement, hotwaterheating, airconditioning, parking, prefarea, furnished, semi_furnished, unfurnished]
    x1 = np.array(x, dtype=int)
    xr=x1.reshape(1, -1)
    scaler.fit(xr)
    x_scaled = scaler.transform(xr)

    y_pred = model.predict(x_scaled)


    return render(request, 'user.html', {'result': y_pred})
