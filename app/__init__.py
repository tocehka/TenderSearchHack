from flask import Flask, session, redirect, url_for, request 
from markupsafe import escape 
import json 
from dotenv import load_dotenv 
import os

app = Flask(__name__)
from app import routes
