import pickle
import pandas as pd
import numpy as np
from data_processing import process_data
import asyncio
import aiohttp
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

SERVER_URL = os.getenv('SERVER_URL')
SLEEP_TIME = int(os.getenv('SLEEP_TIME'))

try:
    with open("./savedmodel.sav", "rb") as f:
        model = pickle.load(f)
        print('model loaded')
except Exception as err:
    print(f"Unexpected {err=}, {type(err)=}")

# Asynchronous function to fetch data from the server
async def fetch_data(session):
    async with session.get(SERVER_URL) as response:
        return await response.json()

async def post_data(session, url, payload):
    async with session.post(url, json=payload) as response:
        return await response.json()

# Asynchronous function to perform prediction and return the result
async def perform_prediction(session):
    try:
        json_data = await fetch_data(session)
        if 'data' in json_data:
            for machine_id, readings in json_data['data'].items():
                df = pd.DataFrame(readings)
                input_data = process_data(df)
                y_pred_prob = model.predict(input_data)
                y_pred = np.argmax(y_pred_prob, axis=1)
                class_labels = ['comp1', 'comp2', 'comp3', 'comp4', 'none']
                predicted_class = class_labels[y_pred[0]]

                post_data_payload = {
                    'id': machine_id,
                    'prediction': predicted_class
                }
                print(post_data_payload)

                post_response = await post_data(session, SERVER_URL, post_data_payload)
                print(f'Posted prediction for {machine_id}: {post_response}')
        else:
            print("Error: 'data' key not found in JSON response")
    except Exception as e:
        print(f'Error: {str(e)}')

# Asynchronous function to periodically request data every hour
async def periodic_task():
    async with aiohttp.ClientSession() as session:
        while True:
            await perform_prediction(session)
            await asyncio.sleep(SLEEP_TIME)  # Use environment variable for sleep time

if __name__ == "__main__":
    asyncio.run(periodic_task())
