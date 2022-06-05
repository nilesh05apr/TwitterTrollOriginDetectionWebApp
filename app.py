import sqlite3
import streamlit as st
import pandas as pd
from get_troll_tweets import get_troll_pred_tweets
from rank_bm25 import BM25Okapi
import numpy as np
import json
# Security
#passlib,hashlib,bcrypt,scrypt
import hashlib
import plotly.figure_factory as ff
import plotly.express as px
from resolve_location import check_if_loc_exists
import pandas as pd

import folium
from folium.plugins import MarkerCluster
import streamlit.components.v1 as components

import base64
import uuid
import re
import pickle
import json
import matplotlib.pyplot as plt


@st.cache
def convert_df(df):
    # IMPORTANT: Cache the conversion to prevent computation on every rerun
    return df.to_csv().encode('utf-8')


#@st.cache(suppress_st_warning=True)
def get_tweets(text):
    pred_df,tq = get_troll_pred_tweets(text,max_tweets=100)
    return pred_df,tq

def get_troll_origin(df, troll_serial_no):
	"""This function returns the origin of the troll"""
	text = df[df["Sl_No"] == int(troll_serial_no)]["text"].copy()
	corpus = df.drop([int(troll_serial_no)], axis=0)["text"].copy().to_list()
	tokenized_corpus = [doc.split(" ") for doc in corpus]
	bm25 = BM25Okapi(tokenized_corpus)

	tokenized_query = text.item().split()
	scores = bm25.get_scores(tokenized_query)
	top_matched_index = np.argmax(scores)
	return df.iloc[top_matched_index]

def form_callback2():
    #st.session_state.pred_df

    troll_df = get_troll_origin(st.session_state.pred_df, st.session_state.troll_index)
    st.session_state.troll_df = troll_df

    st.write("Here is troll origin")

    st.write("created_at : ", st.session_state.troll_df["created_at"])
    st.write("tweet : ", st.session_state.troll_df["text"])
    st.write("user_id : ", st.session_state.troll_df["user_id"])
    st.write("user_location : ", st.session_state.troll_df["user_location"])
    st.write("user_name : ", st.session_state.troll_df["user_name"])
	
def reset_start():
    st.session_state.clear()

    with st.form(key='my_form'):
        text_input = st.text_input(label='Enter keyword to search in twitter', key='keyword')
        if 'keyword' not in st.session_state:
            st.session_state["keyword"] = text_input
        submit_button = st.form_submit_button(label='Submit', on_click=form_callback)

def form_callback3():

    troll_index = st.session_state["option"]
    troll_df = get_troll_origin(st.session_state.pred_df, troll_index)
    st.session_state.troll_df = troll_df
    
    st.write("Here is troll origin")
    st.write("created_at : ", st.session_state.troll_df["created_at"])
    st.write("tweet : ", st.session_state.troll_df["text"])
    st.write("user_id : ", st.session_state.troll_df["user_id"])
    st.write("user_location : ", st.session_state.troll_df["user_location"])
    st.write("user_name : ", st.session_state.troll_df["user_name"])

    st.button("Search for another keyword", on_click=reset_start)


def convert_df(df):
    return df.to_csv().encode('utf-8')

def download_button(object_to_download, download_filename, button_text, pickle_it=False):
    """
    Generates a link to download the given object_to_download.

    Params:
    ------
    object_to_download:  The object to be downloaded.
    download_filename (str): filename and extension of file. e.g. mydata.csv,
    some_txt_output.txt download_link_text (str): Text to display for download
    link.
    button_text (str): Text to display on download button (e.g. 'click here to download file')
    pickle_it (bool): If True, pickle file.

    Returns:
    -------
    (str): the anchor tag to download object_to_download

    Examples:
    --------
    download_link(your_df, 'YOUR_DF.csv', 'Click to download data!')
    download_link(your_str, 'YOUR_STRING.txt', 'Click to download text!')

    """
    if pickle_it:
        try:
            object_to_download = pickle.dumps(object_to_download)
        except pickle.PicklingError as e:
            st.write(e)
            return None

    else:
        if isinstance(object_to_download, bytes):
            pass

        elif isinstance(object_to_download, pd.DataFrame):
            object_to_download = object_to_download.to_csv(index=False)

        # Try JSON encode for everything else
        else:
            object_to_download = json.dumps(object_to_download)

    try:
        # some strings <-> bytes conversions necessary here
        b64 = base64.b64encode(object_to_download.encode()).decode()

    except AttributeError as e:
        b64 = base64.b64encode(object_to_download).decode()

    button_uuid = str(uuid.uuid4()).replace('-', '')
    button_id = re.sub('\d+', '', button_uuid)

    custom_css = f""" 
        <style>
            #{button_id} {{
                background-color: rgb(255, 255, 255);
                color: rgb(38, 39, 48);
                padding: 0.25em 0.38em;
                position: relative;
                text-decoration: none;
                border-radius: 4px;
                border-width: 1px;
                border-style: solid;
                border-color: rgb(230, 234, 241);
                border-image: initial;

            }} 
            #{button_id}:hover {{
                border-color: rgb(246, 51, 102);
                color: rgb(246, 51, 102);
            }}
            #{button_id}:active {{
                box-shadow: none;
                background-color: rgb(246, 51, 102);
                color: white;
                }}
        </style> """

    dl_link = custom_css + f'<a download="{download_filename}" id="{button_id}" href="data:file/txt;base64,{b64}">{button_text}</a><br></br>'

    return dl_link

def form_callback():

    #st.write("Showing tweets for keyword : ", st.session_state.keyword)

    pred_df,tq = get_tweets(st.session_state.keyword)
    st.session_state.pred_df = pred_df
    st.markdown("<h1 style='text-align: center; color: #42b9f5;'>Twitter Troll Origin Detection Application</h1>", 
    unsafe_allow_html=True)
    N = 15
    sub_df = pred_df.iloc[0:len(pred_df)%N]
    st.session_state.sub_df = sub_df

    st.title("Showing tweets for keyword : "+st.session_state.keyword)

    st.table(st.session_state.sub_df)

    # --------------- csv download ----------------------
    # st.markdown(download_button(pred_df, 
    #                             'tweets.csv', 
    #                             'Download Troll Tweets'), 
    #                             unsafe_allow_html=True)
    csv = convert_df(pred_df)
        
    #print(type(csv))
    button = st.download_button(
        label="Download Troll Tweets",
        data=csv,
        file_name='tweets.csv',
        mime='text/csv',)
    # ------------------------------------------------- Map visualization -----------------------------------
    ## Add this from here
    pred_df['user_location'] = pred_df['user_location'].fillna("Unknown")
    temp_df = pred_df.apply(lambda row: check_if_loc_exists(row.user_location), axis='columns', result_type='expand')
    pred_df = pd.concat([pred_df, temp_df], axis='columns')
    pred_df.rename(columns={0:'latitude', 1:'longitude'}, inplace = True)

    #empty map
    world_map= folium.Map(tiles="cartodbpositron")
    marker_cluster = MarkerCluster().add_to(world_map)

    #for each coordinate, create circlemarker of user percent
    for i in range(len(pred_df)):
        lat = pred_df.iloc[i]['latitude']
        long = pred_df.iloc[i]['longitude']
        radius=5
        popup_text = """user_name : {}<br>
                    user_location : {}<br>"""
        popup_text = popup_text.format(pred_df.iloc[i]['user_name'], pred_df.iloc[i]['user_location'])
    
        folium.CircleMarker(location = [lat, long], radius=radius, popup= popup_text, fill =True).add_to(marker_cluster)

    st.title("Trolls in map")
    MAP_HEIGHT = 500

    components.html(world_map._repr_html_(), height=MAP_HEIGHT+10)

    # Top troll locations
    pred_df['user_location'] = pred_df['user_location'].replace('', 'Unknown')
    top_locations = pred_df.groupby(['user_location'])['user_location'].count().sort_values(ascending=False).reset_index(name="count")
    st.markdown('Top locations of trolls as below')
    st.table(top_locations)


    #print(top_locations['count'])
    counts = list(top_locations['count'])
    locations = list(top_locations['user_location'])
    chart_data = pd.DataFrame({'value':counts,'labels':locations})
    fig = px.bar(chart_data,color='labels')

    # fig = ff.create_distplot(hist_data,group_labels,bin_size=[0.1,0.25])
    st.plotly_chart(fig)         

    #----------------------------------------------------------------------------------------------------------
    #----------------------------------------------------------------------------------------------------------
   #Bar graph

    st.markdown("No of tweets classified as troll and non-troll")

    countTroll = max(0,len(pred_df))
    countNonTroll = max(0,tq - countTroll)
    hist_data = [countNonTroll,countTroll]

    group_labels = ["NonTroll","Troll"]
    chart_data = pd.DataFrame({'value':hist_data,'labels':group_labels})
    fig = px.bar(chart_data,color='labels')

    # fig = ff.create_distplot(hist_data,group_labels,bin_size=[0.1,0.25])
    st.plotly_chart(fig)         
    
    labels = ['Troll','Non-Troll']
    sizes = [countTroll,countNonTroll]
    explode = (0.2,0.2)
    fig1, ax1 = plt.subplots(figsize=(2,2))
    ax1.pie(
        sizes, 
        explode=explode, 
        labels=labels, 
        autopct='%1.1f%%',
        shadow=True, 
        startangle=90)
    ax1.axis('equal')
    st.pyplot(fig1)




    if "option" not in st.session_state:
        st.session_state["option"] = 0
    st.session_state["option"] = st.selectbox('Select serial number of tweet for which you want to findout troll origin.',
                                                st.session_state.pred_df["Sl_No"],
                                                on_change=form_callback3) 



def main():
    """TRoll detection login"""

    if "initial_state" not in st.session_state:
        st.set_page_config(page_title="Troll detection", page_icon=None, layout='wide', initial_sidebar_state='auto')
        st.session_state["initial_state"] = True

    st.session_state.placeholder = st.empty()



    st.markdown("<h1 style='text-align: center; color: #42b9f5;'>Twitter Troll Origin Detection Application</h1>", 
    unsafe_allow_html=True)


    with st.form(key='my_form'):
        text_input = st.text_input(label='Enter keyword to search in twitter', key='keyword')
        submit_button = st.form_submit_button(label='Submit', on_click=form_callback)



if __name__ == '__main__':
	main()
