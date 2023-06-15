#!/usr/bin/env python3

#import mysql.connector
#import os
#
#mydb = mysql.connector.connect(
#	host="localhost",
#	user="root",
#	password="<password>",
#   database="seatease"
#)


import sqlite3
import time
import pandas as pd

def createDatabasesAndTables():	
	# Create both the tables
	conn_cursor.execute('''CREATE TABLE IF NOT EXISTS table_metadata ([table_id] TEXT PRIMARY KEY, [station] TEXT,[camera] TEXT)''')
	conn_cursor.execute('''CREATE TABLE IF NOT EXISTS table_occupancy ([table_id] TEXT PRIMARY KEY, [total_seats] INTEGER, [seats_occupied] INTEGER, [seats_unoccupied] INTEGER,[entry_time] CURRENT_TIMESTAMP)''')
	conn.commit()


def insertRecordsInTables():
	#Insert into table_metadata table
	conn_cursor.execute("""INSERT or IGNORE INTO table_metadata VALUES ('Choix_1','Choix','Camera_1')""")	
	conn_cursor.execute("""INSERT or IGNORE INTO table_metadata VALUES ('Choix_2','Choix','Camera_1')""")	
	conn_cursor.execute("""INSERT or IGNORE INTO table_metadata VALUES ('Choix_3','Choix','Camera_1')""")	
	conn_cursor.execute("""INSERT or IGNORE INTO table_metadata VALUES ('Distintly_South_1','Distintly_South','Camera_2')""")	
	conn_cursor.execute("""INSERT or IGNORE INTO table_metadata VALUES ('Distintly_South_2','Distintly_South','Camera_2')""")	
	conn_cursor.execute("""INSERT or IGNORE INTO table_metadata VALUES ('Distintly_South_3','Distintly_South','Camera_2')""")	
	
	# Insert into table_occupancy table
	conn_cursor.execute("""INSERT or IGNORE INTO table_occupancy VALUES ('Choix_1',6,0,6,{time})""".format(time = time.time()))
	conn_cursor.execute("""INSERT or IGNORE INTO table_occupancy VALUES ('Choix_2',6,0,6,{time})""".format(time = time.time()))
	conn_cursor.execute("""INSERT or IGNORE INTO table_occupancy VALUES ('Choix_3',6,0,6,{time})""".format(time = time.time()))
	conn_cursor.execute("""INSERT or IGNORE INTO table_occupancy VALUES ('Distintly_South_1',10,0,10,{time})""".format(time = time.time()))
	conn_cursor.execute("""INSERT or IGNORE INTO table_occupancy VALUES ('Distintly_South_2',6,0,6,{time})""".format(time = time.time()))
	conn_cursor.execute("""INSERT or IGNORE INTO table_occupancy VALUES ('Distintly_South_3',8,0,8,{time})""".format(time = time.time()))
	
	conn.commit()


def selectDataFromTables():
	conn_cursor.execute('''SELECT * from table_metadata''')
	df_table_metadata = pd.DataFrame(conn_cursor.fetchall())
	print(df_table_metadata)
	conn_cursor.execute('''SELECT * from table_occupancy''')
	df_table_occupancy = pd.DataFrame(conn_cursor.fetchall())
	print(df_table_occupancy)
	conn.commit()
	

conn = sqlite3.connect('seatease_database') 
conn_cursor = conn.cursor()	









				