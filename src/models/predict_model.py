from flask import Flask, render_template, request, jsonify
import os
import numpy as np
import yaml
import joblib 

webapp_root = "webapp"
params_path = "params.yaml"

static_dir = os.path.join(webapp_root, "static")
template_dir = os.path.join(webapp_root, "templates")

def read_params(config_path):
    with open(config_path) as yaml_file:
        config = yaml.safe_load(yaml_file)
    return config
    
def predict(data):
    config = read_params(params_path)
    model_dir_path = config["model_dir"]
    model = joblib.load(model_dir_path)
    prediction = model.predict(data).tolist()[0]
    return prediction 
