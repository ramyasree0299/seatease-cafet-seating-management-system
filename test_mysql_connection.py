#!/usr/bin/env python3

import mysql.connector
import os

mydb = mysql.connector.connect(
	host="localhost",
	user="root",
	password="<password>",
    database="seatease"
)
