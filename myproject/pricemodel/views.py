from django.shortcuts import render


import os
#import joblib
#import pickle
from keras.models import load_model
import numpy as np
from django.shortcuts import render
from .forms import URL
import numpy as np
import pandas as pd
import urllib.parse
import tensorflow as tf

# Previous pre-processing unit 
def convert_url(raw_url):
    # Tokenize the URL
    tokenized_url = urllib.parse.quote(raw_url)

    # Standardize the URL
    standardized_url = urllib.parse.urlsplit(tokenized_url).geturl()

    # Truncate or pad the URL
    max_url_length = 200  # Maximum length of the padded URL

    if len(standardized_url) > max_url_length:
        # Truncate the URL if it is longer than the maximum length
        truncated_url = standardized_url[-max_url_length:]
        padded_url = [ord(char) for char in truncated_url]
    else:
        # Pad the URL with zeros if it is shorter than the maximum length
        padded_url = [0] * (max_url_length - len(standardized_url)) + [ord(char) for char in standardized_url]
    
    return padded_url

def predict_price(request):
    if request.method == 'POST':
        form = URL(request.POST)
        if form.is_valid():
            # Extract the URL input
            url = form.cleaned_data.get('url')

            # Load the trained PRISM_Model
            model_path = os.path.join(os.path.dirname(__file__), 'models', 'PRISM_Model.h5')
            # with open(model_path, 'rb') as file:
            #     model = pickle.load(file)
            
            model = load_model(model_path)

            url = convert_url(url)
            url = np.array(url)
            url = url.reshape(1, -1)

            #predict PRISM_Model
            
            url = model.predict(url)

            #print("Return val: ", url)

            if url >= 0.5:
                 url1 = "Malicious URL"
            else:
                 url1 = "Benign URL"
            # url = "Some text"
            # Prepare the response
            context = {
                'form': form,
                'url': url1,  # Add the URL to the context for rendering in the template
            }
            return render(request, 'index.html', context)
    else:
        form = URL()

    context = {'form': form}
    return render(request, 'index.html', context)




























'''
import os
import joblib
import numpy as np
from django.shortcuts import render
from .forms import PricePredictionForm

def predict_price(request):
    if request.method == 'POST':
        form = PricePredictionForm(request.POST)
        if form.is_valid():
            # Load the trained linear regression model
            model_path = os.path.join(os.path.dirname(_file_), 'models', 'linear_regression_model.pkl')
            model = joblib.load(model_path)

            # Extract input data from the form
            new_data = np.array(list(form.cleaned_data.values())).reshape(1, -1)

            # Perform prediction
            predicted_price = model.predict(new_data)[0]

            # Prepare the response
            context = {
                'form': form,
                'predicted_price': round(predicted_price, 2),
            }
            return render(request, 'index.html', context)
    else:
        form = PricePredictionForm()

    context = {'form': form}
    return render(request, 'index.html', context)
'''