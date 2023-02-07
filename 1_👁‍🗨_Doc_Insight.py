import ast
import os
import requests
from functools import partial

import numpy as np
import pandas as pd
import streamlit as st
from annotated_text import annotated_text
import streamlit_authenticator as stauth
import openai
from scipy.spatial.distance import cosine
import tiktoken


from utils.utils import auth
from utils.utils import timed_alert
from utils.html_codes import *
from utils.api import send_question_to_api
from utils.utils import set_state_if_absent
from utils.ai import get_embedding, vector_similarity


if 'sidebar_state' not in st.session_state:
    st.session_state.sidebar_state = 'collapsed'

if 'layout' not in st.session_state:
    st.session_state.layout = 'centered'

st.set_page_config(page_title="Doc. Insight", page_icon="📎",
                   layout=st.session_state.layout,
                   menu_items={'About': "### Doc. Insight app!"},
                   initial_sidebar_state=st.session_state.sidebar_state)

st.markdown(HIDE_ST, unsafe_allow_html=True)

COMPLETIONS_MODEL = "text-davinci-003"
COMPLETIONS_API_PARAMS = {
    "temperature": 0.0,
    "max_tokens": 300,
    "model": COMPLETIONS_MODEL,
}

MAX_SECTION_LEN = 500
SEPARATOR = "\n* "
ENCODING = "cl100k_base"  # encoding for text-embedding-ada-002

# openai.api_key = os.getenv("OPENAI_API_KEY")
openai.api_key_path = 'api.txt'
# print(openai.api_key)
header = """Answer the question as truthfully as possible using the provided context, \
and include the parts of the context that are used to generate the answer after the answer starting with "\nRef:". \
If the answer is not contained within the text below, say "I don't know."\n\nContext:\n"""



def logout(authenticator):
    for _ in range(15):
        st.sidebar.write("")
    st.sidebar.markdown('---')
    st.sidebar.write(f'Logged in as "*{st.session_state["name"]}*"')
    authenticator.logout('Logout', 'sidebar')


@st.cache(suppress_st_warning=True, show_spinner=False)
def str_vector_to_float(str_vector):
    return np.array(ast.literal_eval(str_vector), dtype='float32')


@st.cache(suppress_st_warning=True, show_spinner=False)
def find_top_k_similar_vectors(df, input_vector, k):
    # vectors = df['Embedding'].to_list()
    df_ = df.copy(deep=True)
    df_['Embedding'] = df_['Embedding'].apply(str_vector_to_float)
    df_['Score'] = df_['Embedding'].apply(lambda x: np.dot(x, input_vector))
    top_k_indexes = df_['Score'].nlargest(k).index
    return df_.iloc[top_k_indexes]


@st.cache(suppress_st_warning=True, show_spinner=False)
def get_separator_len():
    encoding = tiktoken.get_encoding(ENCODING)
    separator_len = len(encoding.encode(SEPARATOR))
    return separator_len


@st.cache(suppress_st_warning=True, show_spinner=False)
def construct_prompt(question: str, df: pd.DataFrame) -> str:
    """
    Fetch relevant
    """

    question_embedding = get_embedding(question)
    found_contents = find_top_k_similar_vectors(df, question_embedding, 3)

    scores, contents = found_contents['Score'], found_contents['Content']
    most_relevant_document_sections = contents.to_list()
    print('Scores:', scores)
    # print('Contents:', most_relevant_document_sections)

    chosen_sections = []
    chosen_sections_len = 0
    chosen_sections_indexes = []

    for document_section in most_relevant_document_sections:
        # # Add contexts until we run out of space.
        # chosen_sections_len += document_section.tokens + get_separator_len()
        # if chosen_sections_len > MAX_SECTION_LEN:
        #     break

        chosen_sections.append(SEPARATOR + document_section.replace("\n", " "))

    # Useful diagnostic information
    print(f"Selected {len(chosen_sections)} document sections:")
    # print("\n".join(chosen_sections_indexes))

    return (header + "".join(chosen_sections) + "\n\n Q: " + question + "\n A:",
            scores, contents)


@st.cache(suppress_st_warning=True, show_spinner=False)
def answer_query_with_context(
        query: str,
        df: pd.DataFrame,
        show_prompt: bool = False
) -> str:

    with st.spinner('Finding related docs...'):
        prompt, scores, contents = construct_prompt(query, df)

    if show_prompt:
        print(prompt)

    with st.spinner('Generating answer...'):
        response = openai.Completion.create(
                prompt=prompt,
                **COMPLETIONS_API_PARAMS
        )

    return response["choices"][0]["text"].strip(" \n"), scores, contents


# @st.cache(suppress_st_warning=True)
# def get_answer(refs, question):
#     chosen_sections = []
#     for i in refs.keys():
#         chosen_sections.append(refs[i][f'Ref {i+1}'])
#     # chosen_sections = [d['content'] for d in refs['documents']]
#     chosen_sections = '\n'.join(chosen_sections)
#     prompt = header + "".join(chosen_sections) + "\n\n Q: " + question + "\n A:"
#     response = openai.Completion.create(prompt=prompt, **COMPLETIONS_API_PARAMS)
#     full_ans = response["choices"][0]["text"].strip(" \n")
#     return full_ans


@st.cache(suppress_st_warning=True)
def get_doc_dataset(dataset_path):
    df = pd.read_csv(dataset_path)
    return df


def main():
    authenticator = auth()
    if authenticator:
        set_state_if_absent('prev_question', '')
        set_state_if_absent('question', '')
        st.markdown(SHOW_BAR, unsafe_allow_html=True)
        # set_state_if_absent('question', None)

        st.title('📎 Doc. Insight')
        st.markdown("##### Intelligent Question-Answering on Documents")
        # st.markdown('---')
        question_text = st.text_input(label="Query", value="", placeholder="Enter your question",
                                      label_visibility='hidden')
        clicked = st.button('Answer')

        print('Q:', question_text)
        # if clicked or question_text:
        if clicked and question_text:
            with st.spinner('Preparing Dataset...'):
                dataset_path = os.path.join(os.getcwd(), 'users', st.session_state["username"], 'processed_docs.csv')
                df = get_doc_dataset(dataset_path)

            full_ans, scores, contents = answer_query_with_context(question_text, df)
            ans_part = full_ans.split('Ref:')[0]
            st.markdown('**Answer:**')
            # st.write(ans_part)
            annotated_text((ans_part, 'ANS', "#28404A"))
            st.markdown('---')
            if "I don't know" not in ans_part:
                ref_part = full_ans.split('Ref:')[1].strip()
                st.markdown('**Reference:**')
                st.markdown(f"- _{ref_part}_")
                # for c in contents:
                #     dd = c.find(ref_part)
                #     print(c)
                #     if dd != -1:
                #         p = c[dd:dd+len(ref_part)]
                #         annotated_text(c[:dd], (ref_part, "REF"), c[dd+len(ref_part)+1:])
                #         break
                with st.expander('Related Contents', expanded=False):
                    for idx, (score, doc) in enumerate(zip(scores, contents)):
                        part = {idx+1: {'Score': round(score, 4), 'Content': doc}}
                        st.write(part)


        st.session_state.sidebar_state = 'auto'
        st.session_state.layout = 'wide'
        logout(authenticator)
        st.markdown(SHOW_BAR, unsafe_allow_html=True)
    else:
        st.markdown(HIDE_BAR, unsafe_allow_html=True)
        st.session_state.sidebar_state = 'collapsed'
        st.session_state.layout = 'centered'


main()
