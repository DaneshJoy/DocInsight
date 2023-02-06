import json
import os
import glob
import time
import shutil
from random import randint
import requests

import pandas as pd
import streamlit as st
from streamlit_custom_notification_box import custom_notification_box

from utils.utils import timed_alert
from utils.html_codes import *
from utils.config import Paths, Urls
from utils.utils import set_state_if_absent
from utils.ai import get_embedding


st.set_page_config(page_title="Doc. Insight", page_icon="📎", layout="wide",
                   menu_items={'About': "### Doc. Insight app!"})


def process_docs():
    with st.spinner('Preparing Processor...'):
        from haystack.utils import convert_files_to_docs, clean_wiki_text
        from haystack.nodes import PreProcessor

    with st.spinner('Creating Passages...'):
        all_docs = convert_files_to_docs(dir_path=Paths.TMP_DIR,
                                         clean_func=clean_wiki_text,
                                         split_paragraphs=True)
        for doc_txt in all_docs:
            doc_filename = f"{os.path.basename(doc_txt.meta['name']).split('.')[0]}.txt"
            with open(os.path.join(Paths.TXT_DIR, doc_filename), 'w', encoding="utf-8") as f:
                f.write(doc_txt.content)

    with st.spinner('Processing Docs...'):
        # %% Preprocess
        preprocessor = PreProcessor(
                clean_empty_lines=True,
                clean_whitespace=True,
                clean_header_footer=False,
                split_by="word",
                split_length=200,
                split_overlap=20,
                split_respect_sentence_boundary=True,
        )
        chunks = preprocessor.process(all_docs)
        print(f"n_docs_input: {len(all_docs)}\nn_docs_output: {len(chunks)}")

    with st.spinner('Saving Docs...'):
        chunk_paths = []
        for chunk in chunks:
            chunk_filename = f"{chunk.meta['name']}_{chunk.meta['_split_id']}.txt"
            chunk_path = os.path.join(Paths.CHK_DIR, chunk_filename)
            chunk_paths.append(chunk_path)
            with open(chunk_path, 'w', encoding="utf-8") as f:
                f.write(chunk.content)

    return len(all_docs), len(chunks), chunk_paths


def chunks_to_df(chunk_paths):
    data = []
    for chunk_path in chunk_paths:
        # print('---reading', chunk_path)
        with open(chunk_path, 'r', encoding="utf8", errors="ignore") as f:
            content = f.read()
            # print('---content:', content)
            data.append([chunk_path, content])
        df = pd.DataFrame(data, columns=['Filename', 'Content'])
    return pd.DataFrame(data, columns=['Filename', 'Content'])


def apply_get_embedding(df):
    df['Embedding'] = df['Content'].apply(get_embedding)
    return df


def file_exists(file_name):
    existing_files = os.listdir(Paths.DOC_DIR)
    temp_files = os.listdir(Paths.TMP_DIR)
    for doc in [*existing_files, *temp_files]:
        if doc == file_name:
            return True
    return False


def save_uploaded_file(uploaded_file):
    save_path = os.path.join(Paths.TMP_DIR, uploaded_file.name)
    with open(save_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return save_path


if 'uploader_key' not in st.session_state:
    st.session_state.uploader_key = str(randint(1000, 100000000))


def upload_doc():
    uploaded_any = False
    uploaded_files = st.sidebar.file_uploader("📤 Upload a document file",
                                              type=['txt', 'doc', 'docx', 'pdf'],
                                              accept_multiple_files=True,
                                              key=st.session_state.uploader_key)

    if uploaded_files:
        css = """
            .css-fis6aj {
                display: none;
            }"""
        st.sidebar.markdown(f'<style>{css}</style>', unsafe_allow_html=True)

        with progress_container:
            placeholder = st.empty()
            placeholder.write(' 🚧 >>> Processing documents ...')
            process_progress = st.progress(0)
        for i, file in enumerate(uploaded_files):
            # file_details = {"FileName":file.name, "FileType":file.type}

            if not file_exists(file.name):
                save_path = save_uploaded_file(file)
                uploaded_any = True
            elif not uploaded_any:
                with progress_container:
                    timed_alert(f'Uploaded file already exists:\n "{file.name}"',
                                type_='error')
                return

            process_progress.progress((i + 1) * 100 // len(uploaded_files))

        num_docs, num_chunks, chunk_paths = process_docs()
        chunks_df = chunks_to_df(chunk_paths)
        chunks_df.to_csv(os.path.join(os.getcwd(), 'users', st.session_state["username"], 'docs.csv'))
        with st.spinner('Calculating Embeddings...'):
            full_df = apply_get_embedding(chunks_df)
            full_df.to_csv(os.path.join(os.getcwd(), 'users', st.session_state["username"], 'processed_docs.csv'))

        placeholder.empty()
        process_progress.empty()
        with progress_container:
            if num_docs > 0:
                message = (f"Processed {num_docs} document(s)"
                           f"and created {num_chunks} passages")

                custom_notification_box(icon='add_task',
                                        textDisplay=message,
                                        externalLink='', url='#',
                                        styles=NOTIF_STYLE, key="qwerty")
                time.sleep(2)

        uploaded_files = None
        if 'uploader_key' in st.session_state.keys():
            st.session_state.pop('uploader_key')
        # with progress_container:
        #     timed_alert('Document processing finished')

        for f in os.listdir(Paths.TMP_DIR):
            shutil.move(os.path.join(Paths.TMP_DIR, f), os.path.join(Paths.DOC_DIR, f))

    return uploaded_any


def upload_link():
    url = st.sidebar.text_input("Enter a url to crawl", value="")
    if url is not None or url != '':
        st.sidebar.markdown(f"{url}")
        # st.sidebar.button('Crawl', on_click=partial(send_link_to_api, None))

    # from selenium import webdriver
    # options = webdriver.ChromeOptions()
    # or from selenium.webdriver.chrome.options import Options
    # options.binary_location = '/home/appuser/.wdm/drivers/chromedriver/linux64/109.0.5414/chromedriver'
    if url != '':
        with st.spinner('Please wait...'):
            from haystack.nodes import Crawler
        with progress_container:
            placeholder = st.empty()
            placeholder.write(' 🚧 >>> Processing URL ...')

        with st.spinner('Please wait ...'):
            # webdriver_options=["--disable-gpu", "--no-sandbox", "--single-process"])
            crawler = Crawler(output_dir=Paths.URL_DIR, crawler_depth=1)
            sub_urls = crawler._extract_sublinks_from_url(base_url=url)
            with st.sidebar.expander(f"Found {len(sub_urls)} sub-urls", expanded=False):
                for _url in sub_urls:
                    st.markdown(_url)
            if len(sub_urls) > 0:
                clicked = st.sidebar.button('📚 Get Content')
                if clicked:
                    crawler.crawl(output_dir=Paths.URL_DIR, urls=url, crawler_depth=1)
            # crawler._extract_sublinks_from_url -> already_found_links: Optional[List] = None
            # st.sidebar.write(f"Found {len(sub_urls)} sub-urls")
            # for u in sub_urls:
            #     st.sidebar.markdown(f"{u}")

            placeholder.empty()


if st.session_state['authentication_status']:

    for _ in range(2):
        st.sidebar.write("")

    if not os.path.exists(Paths.TMP_DIR):
        os.makedirs(Paths.TMP_DIR)
    if not os.path.exists(Paths.DOC_DIR):
        os.makedirs(Paths.DOC_DIR)
    if not os.path.exists(Paths.TXT_DIR):
        os.makedirs(Paths.TXT_DIR)
    if not os.path.exists(Paths.CHK_DIR):
        os.makedirs(Paths.CHK_DIR)
    if not os.path.exists(Paths.URL_DIR):
        os.makedirs(Paths.URL_DIR)

    uploaded_contents = os.listdir(Paths.DOC_DIR)

    st.markdown('## Processed Documents')
    st.markdown("---")

    progress_container = st.container()

    cc = st.columns(2)
    for i, f in enumerate(uploaded_contents):
        with cc[i % 2]:
            with st.expander(f"{f}", expanded=False):
                with open(os.path.join(Paths.DOC_DIR, f), "rb") as file:
                    btn = st.download_button(
                            label="⬇️ Download PDF",
                            data=file,
                            file_name=f,
                            mime='application/octet-stream',
                            key=f'd_pdf_{i}'
                    )

    uploaded = upload_doc()
    if uploaded:
        st.experimental_rerun()

    st.sidebar.markdown("---")

    upload_link()

    st.sidebar.markdown("---")

else:
    st.markdown('Please <a href="/" target="_self">login</a> to access this page',
                unsafe_allow_html=True)