import os
import json
import requests
from loguru import logger
from typing import Callable

class RemoteMixteraDataset:
    def __init__(self, dataset_url) -> None:
        self.dataset_url = dataset_url
    
    @classmethod
    def from_url(cls, dataset_url):
        return cls(dataset_url)

    def __repr__(self) -> str:
        return f"RemoteMixteraDataset({self.dataset_url})"
    
    def read_keys(self):
        response = requests.get(f"{self.dataset_url}/keys")
        return response.json()
    
    def find_by_key(self, key):
        response = requests.get(f"{self.dataset_url}/key/{key}")
        return response.json()

    def read_values(self, keys):
        response = requests.post(f"{self.dataset_url}/data", json={
            "fids": keys,
            "streaming": False
        })
        res = response.json()
        print(res)
        return res

    def stream_values(self, keys):
        response = requests.post(f"{self.dataset_url}/data", json={
            "fids": keys,
            "streaming": True
        }, stream=True)
        return response.iter_lines()