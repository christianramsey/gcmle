import tensorflow as tf

# our data doesn't have column names or an explicit y,
# we will define the header here to get ready to ingest the data
CSV_COLUMNS  = \
('ontime,dep_delay,taxiout,distance,avg_dep_delay,avg_arr_delay' + \
 'carrier,dep_lat,dep_lon,arr_lat,arr_lon,origin,dest').split(',')
LABEL_COLUMN = 'ontime'

