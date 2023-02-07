@ECHO OFF

::set the codepage to UTF-8 
chcp 65001 >NUL

CALL conda activate deploy
streamlit run "1_👁‍🗨_Doc_Insight.py"
PAUSE