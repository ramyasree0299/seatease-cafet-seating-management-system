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
	conn_cursor.execute("DROP TABLE table_occupancy")
	conn_cursor.execute('''CREATE TABLE IF NOT EXISTS table_metadata ([table_id] TEXT PRIMARY KEY, [station] TEXT,[camera] TEXT,[x1] INTEGER, [y1] INTEGER,[x2] INTEGER,[y2] INTEGER)''')
	conn_cursor.execute('''CREATE TABLE IF NOT EXISTS table_occupancy ([table_id] TEXT PRIMARY KEY, [total_seats] INTEGER, [seats_occupied] INTEGER, [seats_unoccupied] INTEGER,[entry_time] CURRENT_TIMESTAMP)''')
	conn.commit()


def loadTablesData():
	#Insert into table_metadata table
	conn_cursor.execute("""INSERT or IGNORE INTO table_metadata VALUES ('Choix_1','Choix','Camera_1',375, 142, 146, 145)""")		
	conn_cursor.execute("""INSERT or IGNORE INTO table_metadata VALUES ('Choix_2','Choix','Camera_1',15, 148, 249, 120)""")	
	# Insert into table_occupancy table
	conn_cursor.execute("""INSERT or IGNORE INTO table_occupancy VALUES ('Choix_1',4,0,4,{time})""".format(time = time.time()))
	conn_cursor.execute("""INSERT or IGNORE INTO table_occupancy VALUES ('Choix_2',4,0,4,{time})""".format(time = time.time()))
	conn.commit()


def showTableMetadata():
	table_metadata_map = dict()
	table_detect = dict()
	conn_cursor.execute('''SELECT table_id,station,camera,x1,y1,x2,y2 from table_metadata''')
	result_table_metadata = conn_cursor.fetchall()
	for table_metadata in result_table_metadata:
		table_metadata_map[table_metadata[0]] = (table_metadata[3], table_metadata[4], table_metadata[5],table_metadata[6])
		table_detect[table_metadata[0]] = dict()
		table_detect[table_metadata[0]]["chairCount"] = 0
		table_detect[table_metadata[0]]["prevChairCount"] = 0
		table_detect[table_metadata[0]]["personCount"] = 0
		table_detect[table_metadata[0]]["prevPersonCount"] = 0
		
	print(table_detect,table_metadata_map)
	return (table_metadata_map,table_detect)

def showTableOccupancy():
	table_occupancy_map = dict()
	conn_cursor.execute('''SELECT * from table_occupancy''')
	result_table_occupancy = conn_cursor.fetchall()
	for result in result_table_occupancy:
		table_occupancy_map[result[0]] = result[1]
	print(table_occupancy_map)
	return table_occupancy_map
	



def updateTableOccupancyTable(occ,unocc,table):
	conn_cursor.execute("""UPDATE table_occupancy SET seats_occupied = {occupied}, seats_unoccupied = {unoccupied} WHERE table_id = \'{table_id}\'""".format(occupied=occ,unoccupied=unocc,table_id=table))
	conn.commit()
	

conn = sqlite3.connect('seatease_database',check_same_thread=False) 
conn_cursor = conn.cursor()	

