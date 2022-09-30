#Importation des packages-----------------------------------------------------------------------------------------------
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import streamlit.components.v1 as components
import shap
import seaborn as sns
import plotly.graph_objects as go
from streamlit import caching
from datetime import date
import plotly.express as px
import lightgbm as lgbm
from sklearn.pipeline import make_pipeline, Pipeline
import lightgbm as lgbm
from IPython.core.display import display, HTML
from scipy.stats import gaussian_kde
from streamlit_option_menu import option_menu
import re
import time
import requests
import json
import json
from st_aggrid import AgGrid
from st_aggrid.grid_options_builder import GridOptionsBuilder
from PIL import Image
import pandas_profiling
from streamlit_pandas_profiling import st_profile_report
#import dask.dataframe as dd
#Fin Importation des packages-----------------------------------------------------------------------------------------------

#@st.cache(suppress_st_warning=True)
def local_css(file_name):
    with open(file_name) as f:
        st.markdown('<style>{}</style>'.format(f.read()), unsafe_allow_html=True)
# --------------------------------------------------------------------------------------------------------------------
# visualize the first prediction's explanation (use matplotlib=True to avoid Javascript)
@st.cache(suppress_st_warning=True)
def st_shap(plot, height=None):
    shap_html = f"<head>{shap.getjs()}</head><body>{plot.html()}</body>"
    components.html(shap_html, height=height)
# --------------------------------------------------------------------------------------------------------------------
def caract_entree():
    with st.spinner('Chargement de la base de données ⌛'):
        SK_ID_CURR = st.sidebar.selectbox('Sélectionner un client', df_train1['SK_ID_CURR'])

        data = {
            'SK_ID_CURR': SK_ID_CURR,
        }

        df = pd.DataFrame(data, index=[0])
        return df


# --------------------------------------------------------------------------------------------------------------------

def path_to_image_html(path):
    '''
     This function essentially convert the image url to
     '<img src="'+ path + '"/>' format. And one can put any
     formatting adjustments to control the height, aspect ratio, size etc.
     within as in the below example.
    '''

    return '<img src="' + path + '" style=max-height:60px;margin-left:auto;margin-right:auto;display:block;"/>'


# ----------------------------------------------------------------------------------------------------------------
def path_to_image_url(path):
    '''
     This function essentially convert the image url to
     '<img src="'+ path + '"/>' format. And one can put any
     formatting adjustments to control the height, aspect ratio, size etc.
     within as in the below example.
    '''

    return '<div class ="image" ><img class="url" src="' + path + '""/></div>'

#Supprimer le cache chaque jour-----------------------------------------------------------------------------------------
def cache_clear_dt(dummy):
    clear_dt = date.today()
    return clear_dt
if cache_clear_dt("dummy")<date.today():
    caching.clear_cache()
# ----------------------------------------------------------------------------------------------------------------
def try_read_df(file):
    with st.spinner('Chargement de la base de données ⌛'):
        df = pd.read_csv(file)
    with st.spinner('Optimisation de la mémoire ⌛'):
        return reduce_mem_usage(df)

@st.cache(allow_output_mutation=True, show_spinner=False, ttl =3600)
def load_data(file):
    with st.spinner('Chargement de la base de données ⌛'):
        df = pd.read_csv(file, compression='gzip')

    with st.spinner('Optimisation de la mémoire ⌛'):
        return reduce_mem_usage(df)

# ----------------------------------------------------------------------------------------------------------------
@st.cache(allow_output_mutation=True, show_spinner=False, ttl =3600)
def try_read_desc(file):
    with st.spinner('Chargement des données ⌛'):
        df_desc = reduce_mem_usage(pd.read_csv(file, encoding= 'unicode_escape', usecols=['Row','Description']))
        return df_desc
# ----------------------------------------------------------------------------------------------------------------
#@st.cache(allow_output_mutation=True, show_spinner=False)
def get_explainer():
    lgbm_clf = pickle.load(open(dirname + 'Model/best_lgbm_over.pkl', 'rb'))
    explainer = shap.TreeExplainer(lgbm_clf.steps[0][1])
    nb_row = index_client+1
    shap_values = explainer.shap_values(df_train1.iloc[index_client:nb_row, 2:-2])
    expected_value = explainer.expected_value
    return explainer, shap_values, expected_value
# ----------------------------------------------------------------------------------------------------------------
# visualize the first prediction's explanation (use matplotlib=True to avoid Javascript)
def st_shap(plot, height=None):
    shap_html = f"<head>{shap.getjs()}</head><body>{plot.html()}</body>"
    components.html(shap_html, height=height)
# ----------------------------------------------------------------------------------------------------------------
# Plot distribution of multiple features, with TARGET = 1/0 on the same graph
def plot_distribution_comp(var, id_client, nrow=2, ncol=2):
    sns.set_style("darkgrid", {"grid.color": ".6", "grid.linestyle": ":"})
    i = 0
    j=0
    t1 = df_train1.loc[df_train1['TARGET'] != 0]
    t0 = df_train1.loc[df_train1['TARGET'] == 0]



    for feature in var:
        # sns.set_style('whitegrid')
        plt.figure()
        fig, ax = plt.subplots(figsize=(16, 5))
        sns.set_style("dark")
        i += 1

        sns.kdeplot(t1[feature], bw_adjust=0.5, label="In default", color='red', shade=True)
        sns.kdeplot(t0[feature], bw_adjust=0.5, label="No default", color='blue', shade=True)
        plt.ylabel('Density plot', fontsize=8)
        plt.xlabel(feature, fontsize=8)
        #locs, labels = plt.xticks()
        plt.tick_params(axis='both', which='major', labelsize=10)
        client = df_train1[feature][df_train1['SK_ID_CURR'] == str(id_client)].values[0]
        #var = (df_train1[feature][df_train1[feature] == str(id_client)].count()) / ((df_train1[feature].count()))
        #plt.text(client, var, int(client), fontsize=8)
        plt.axvline(client, c='yellow', linewidth=0.9,  alpha=0.8)
        #plt.title(feature, fontsize=9)

        plt.legend(fontsize=10)

        if col_selected[j] in df_description['col_name'].values:
            chaine = df_description['Description'][df_description['col_name'] == col_selected[j]].head(1).values[0]
            if (len(chaine)>100):
                title = str(chaine)[0:100]+'...'
            else :
                title = chaine
            #st.sidebar.write(col_selected[j]+' : '+title)
            plt.title(title,fontsize=11)
        j=j+1

        density = gaussian_kde(df_train1[feature])
        #max_density = density(df_train1[feature]).max()
        max_features = df_train1[feature].max()
        x = client
        y = density(x)

        plt.annotate(var_code+'\n'+str(x), xy=(x, y), xytext=(max_features/1.05, y/1.5), fontsize=10,
                     #arrowprops=dict(facecolor='green', shrink=0.01),
                     #bbox=dict(boxstyle="round4,pad=.5", fc="0.8"),
                     #arrowprops=dict(arrowstyle="->", connectionstyle="angle,angleA=0,angleB=80,rad=20")
                     #bbox=dict(boxstyle ="round", fc ="0.8", edgecolor='blue',color='white'),
                     bbox=dict(facecolor='white', fc ="0.99", edgecolor='black', boxstyle='round'),
                     arrowprops=dict(arrowstyle = "->",connectionstyle = "angle,angleA=90,angleB=180,rad=0",color='black')
                     )

        st.pyplot(fig)
# ----------------------------------------------------------------------------------------------------------------
def gauge(col, col_st):
    fig = go.Figure()

    if (float(risk) > threshold):
        fig.add_trace(go.Indicator(

            value=y_pred[0][1] * 100,
            delta={'reference': threshold, 'increasing': {'color': "red"}, 'decreasing': {'color': "green"}},
            gauge={

                'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue",'visible': True},
                'bar': {'color': "red"},
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': threshold}
            },
            domain={'row': 0, 'column': 0}))

        col_st.error('❌ Crédit refusé')

    else:
        fig.add_trace(go.Indicator(

            value=y_pred[0][1] * 100,
            delta={'reference': threshold, 'increasing': {'color': "red"}, 'decreasing': {'color': "green"}},
            gauge={

                'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue",'visible': True},
                'bar': {'color': "green"},
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': threshold}
            },
            domain={'row': 0, 'column': 0}))

        col_st.success('✔️ Crédit accordé')

    fig.update_layout(
        # paper_bgcolor = "lightgray",
        # font = {'color': "darkblue", 'family': "Arial"},
        grid={'rows': 1, 'columns': 1, 'pattern': "independent"},
        autosize=True,
        # width=1000,
        template={'data': {'indicator': [{
            'title': {'text': "Threshold = "+str(threshold)+" %", 'font': {'size': 20}},
            'mode': "number+delta+gauge",
            'delta': {'reference': 100}}]
        }})

    col.plotly_chart(fig, use_container_width=True)

# ----------------------------------------------------------------------------------------------------------------
def streamlit_menu(example=1):
    if example == 1:
        # 1. as sidebar menu
        with st.sidebar:
            selected = option_menu(
                menu_title="HOME CREDIT",  # required
                options=["Data Client", "Analyse", "Shapley", "Description",
                         #Rapport
                         ],  # required
                icons=['files',"bar-chart-line-fill", "bar-chart-steps", "zoom-in",
                       #"graph-up-arrow"
                       ],  # optional
                menu_icon="menu-button-wide-fill",  # optional
                default_index=0,  # optional
            )
        return selected

    if example == 2:
        # 2. horizontal menu w/o custom style
        selected = option_menu(
            menu_title=None,  # required
            options=["Data Client", "Analyse", "Shapley", "Description"],  # required
            icons=['files', "bar-chart-line-fill", "activity", "zoom-in"],  # optional
            menu_icon="cast",  # optional
            default_index=0,  # optional
            orientation="horizontal",
        )
        return selected

    if example == 3:
        # 2. horizontal menu with custom style
        selected = option_menu(
            menu_title=None,  # required
            options=["Home", "Projects", "Contact"],  # required
            icons=["house", "book", "envelope"],  # optional
            menu_icon="cast",  # optional
            default_index=0,  # optional
            orientation="horizontal",
            styles={
                "container": {"padding": "0!important", "background-color": "#fafafa"},
                "icon": {"color": "orange", "font-size": "25px"},
                "nav-link": {
                    "font-size": "25px",
                    "text-align": "left",
                    "margin": "0px",
                    "--hover-color": "#eee",
                },
                "nav-link-selected": {"background-color": "green"},
            },
        )
        return selected


#Prédiction via FastApi
#@st.cache(allow_output_mutation=True, show_spinner=False)
def get_predictions(df):
    df = df.to_dict('records')[0]
    df = json.dumps(df)
    headers = {'Content-Type': 'application/json'}
    response = requests.request("POST", 'http://127.0.0.1:8000/predict/', headers=headers, data=df)
    df_json = response.json()
    return df_json

#@st.cache(allow_output_mutation=True, show_spinner=False)
def report(df):
    pr = df.profile_report()
    st_profile_report(pr)


# Reduce Memory Usage
def reduce_mem_usage(df):
    """ iterate through all the columns of a dataframe and modify the data type
        to reduce memory usage.
    """
    start_mem = df.memory_usage().sum() / 1024 ** 2
    #print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))

    for col in df.columns:
        col_type = df[col].dtype

        if col_type != object and col_type.name != 'category' and 'datetime' not in col_type.name:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
        elif 'datetime' not in col_type.name:
            df[col] = df[col].astype('category')

    #end_mem = df.memory_usage().sum() / 1024 ** 2
    #print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    #print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))

    return df

def undummify(df, prefix_sep="_"):
    cols2collapse = {
        item.split(prefix_sep)[0]: (prefix_sep in item) for item in df.columns
    }
    series_list = []
    for col, needs_to_collapse in cols2collapse.items():
        if needs_to_collapse:
            undummified = (
                df.filter(like=col)
                .idxmax(axis=1)
                .apply(lambda x: x.split(prefix_sep, maxsplit=1)[1])
                .rename(col)
            )
            series_list.append(undummified)
        else:
            series_list.append(df[col])
    undummified_df = pd.concat(series_list, axis=1)
    return undummified_df


def funct_grid_option(df):
    gb = GridOptionsBuilder.from_dataframe(df)

    gb.configure_pagination()
    gb.configure_side_bar()
    gb.configure_default_column(groupable=True, value=True, enableRowGroup=True, aggFunc="sum", editable=True)
    gridOptions = gb.build()
    return gridOptions