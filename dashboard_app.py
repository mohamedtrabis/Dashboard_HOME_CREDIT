# Chemin des fichiers
#dirname = "../"
dirname = ""

#Importer les fonctions-------------------------------------------------------------------------------------------------
exec(open(dirname+"function.py",encoding='utf-8').read())
#Fin Importation les fonctions------------------------------------------------------------------------------------------

#Config streamlit affichage---------------------------------------------------------------------------------------------
from PIL import Image
st.set_page_config(
    page_title="Home Credit",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon = 'Image/logo_home_credit.gif',
)
#Fin Config streamlit affichage-----------------------------------------------------------------------------------------

#s = st.session_state
st.set_option('deprecation.showPyplotGlobalUse', False)

#Feuille de style css---------------------------------------------------------------------------------------------------
local_css(dirname+"style.css")
#Fin Feuille de style css-----------------------------------------------------------------------------------------------

pd.options.display.float_format = '{:,.2f}'.format

#Image logo entreprise--------------------------------------------------------------------------------------------------
image_logo = Image.open("Image/home credit.jpg")
#newsize = (464, 283)

left, mid ,right = st.columns([1,1, 1])

with mid:
    #image_logo = image_logo.resize(newsize)
    st.image(image_logo, '')
#Fin Image logo entreprise----------------------------------------------------------------------------------------------

#st.header("Dashboard 📈")
st.markdown("<div id='dash'><h1>Dashboard  📈</h1></div>", unsafe_allow_html=True)

#Importation des fichier data-------------------------------------------------------------------------------------------
#file = dirname+r"db/df_train1_all.gz"
file = dirname+r"db/df_train1_2000.gz"
file_desc = dirname+'db/HomeCredit_columns_description.csv'

#Chargement des données
#s_time = time.time()
df_train1 = load_data(file)
#e_time = time.time()
#st.write("Read without chunks: ", (e_time - s_time), "seconds")

description = try_read_desc(file_desc)
#Importation des fichier data-------------------------------------------------------------------------------------------

#st.write(df_train1.memory_usage().sum() / 1024 ** 2)

#Nettoyage du Dataframe description-------------------------------------------------------------------------------------
description = description.drop_duplicates(subset='Row', keep='last')
df_train1  = df_train1.rename(columns = lambda x:re.sub(' ', '_', x))

df_col_train = pd.DataFrame(list(zip(df_train1.iloc[:, 2:-2].columns)),columns=['col_name'])
#Remplacer les chaines de caractères pour merger avec la dataframe de description
df_col_train['Row'] = df_col_train['col_name'].str.replace(
    r'\_MEAN|_STD|PREV_|POS_|INS_|BUREAU_|APPROVED_|APPROVED_|REFUSED_|REFUSED_|_Completed|_Rare|_Active|Active_|_Industry|_Married|_Separated|_Civil marriage|_Single / not married|_Civil_marriage|_Higher_education|_Closed|CLOSED_', '',regex=True)

df_col_train['Row'] = df_col_train['Row'].replace(['EXT_SOURCE', 'DPD', 'DBD'],
                                                  ['EXT_SOURCE_3', 'SK_DPD', 'SK_DPD'])

df_description = pd.merge(df_col_train, description, how="inner", on="Row")
#Fin du nettoyage du Dataframe description------------------------------------------------------------------------------

#Affichage sidebar client et seuil--------------------------------------------------------------------------------------
left_, mid_ ,right_ = st.columns([1,1, 1])

with mid_:
    #SK_ID_CURR = st.sidebar.text_input("Entrer le code client", 100007)
    #SK_ID_CURR = st.sidebar.selectbox('Sélectionner un client', df_train1['SK_ID_CURR'])
    #data = {
    #    'SK_ID_CURR': SK_ID_CURR,
    #}

    input_df = caract_entree()

threshold = st.sidebar.number_input("Seuil de solvabilité en %",min_value=0.00, max_value=100.00, value=75.00, step=1.00)
#Fin Affichage sidebar client et seuil----------------------------------------------------------------------------------

#Choix barre menu à gauche----------------------------------------------------------------------------------------------
#Barre de menu : 1=sidebar menu, 2=horizontal menu, 3=horizontal menu w/ custom menu
EXAMPLE_NO = 1
selected = streamlit_menu(example=EXAMPLE_NO)
#Fin Choix barre menu à gauche------------------------------------------------------------------------------------------

#Préparer les données client--------------------------------------------------------------------------------------------
donnee_entree = pd.concat([input_df, df_train1])

donnee_entree = donnee_entree[:1]

donnee_entree['SK_ID_CURR'] = donnee_entree['SK_ID_CURR'].apply(str)

var_code = donnee_entree['SK_ID_CURR'][0]

df_train1['AGE'] = abs(np.around(df_train1['DAYS_BIRTH']/365,2))
df_train1['YEARS_EMPLOYED'] = abs(np.around(df_train1['DAYS_EMPLOYED']/365,2))

col = ['SK_ID_CURR', 'CODE_GENDER','AGE',
       'FLAG_OWN_CAR', 'FLAG_OWN_REALTY', 'CNT_CHILDREN', 'AMT_INCOME_TOTAL',
       'AMT_CREDIT', 'AMT_ANNUITY', 'AMT_GOODS_PRICE', 'YEARS_EMPLOYED']

donnee_sortie = df_train1[col].copy()

var_code = donnee_entree['SK_ID_CURR'][0]

donnee_sortie['SK_ID_CURR'] = donnee_sortie['SK_ID_CURR'].apply(str)
donnee_sortie = donnee_sortie[(donnee_sortie['SK_ID_CURR'] == var_code)]

#st.table(donnee_sortie.assign(hack='').set_index('hack'))
#st.dataframe(donnee_sortie.assign(hack='').set_index('hack'))
#st.dataframe(donnee_sortie)

#extraire le code client et les informations relatives au client
donnee_entree['SK_ID_CURR'] = donnee_entree['SK_ID_CURR'].apply(str)
df_train1['SK_ID_CURR'] = df_train1['SK_ID_CURR'].apply(str)

var_code = donnee_entree['SK_ID_CURR'][0]
pred_client = df_train1[df_train1['SK_ID_CURR'] == var_code]

if len(pred_client)==0:
    st.write('No data found')
    st.stop()
#Fin Préparer les données client----------------------------------------------------------------------------------------

#Gestion session--------------------------------------------------------------------------------------------------------
#initialize session state
if "load_state" not in st.session_state :
    st.session_state.load_state = False

if "tab" not in st.session_state:
    st.session_state.tab = 'AMT_CREDIT'

def form_callback():
    if len(col_descr) != 0 :
        st.session_state['tab'] = col_descr
#Fin Gestion session----------------------------------------------------------------------------------------------------

# Suuprimer les message warning-----------------------------------------------------------------------------------------
st.set_option('deprecation.showPyplotGlobalUse', False)
# Fin Suuprimer les message warning-------------------------------------------------------------------------------------

# Requete via FASTAPI pour la prédiction client-------------------------------------------------------------------------

# Importer le modèle entrainé lightGBM
lgbm_clf = pickle.load(open(dirname + 'Model/best_lgbm_over.pkl', 'rb'))

# Prediction resultat
#tab = ['No Default', 'Default']
y_pred = lgbm_clf.predict_proba(pred_client.iloc[:, 2:-2])

#Prédiction via l'API FastAPI
#with st.spinner('Chargement des données de FastAPI ⌛'):
#    y_pred = get_predictions(pred_client.iloc[:, 2:-2])
# Fin Requete via FASTAPI pour la prédiction client---------------------------------------------------------------------

#Calculer le rique du prêt----------------------------------------------------------------------------------------------
risk = "{:,.0f}".format(y_pred[0][1]*100)
pred_0 = "{:,.0f}".format(y_pred[0][0]*100)
#Fin Calculer le rique du prêt------------------------------------------------------------------------------------------

#Tableau client transposé-----------------------------------------------------------------------------------------------
if selected!='Description':

    with st.expander("Informations Client : "+var_code, expanded=True):
        col_0, col_1, col_3, col_4  = st.columns([1,5,7,1])
        col_5, col_6, col_7  = st.columns([1,20,1])

        #Working with checkbox
        if len(pred_client)!=0 and selected!='Description':
            donnee_sortie['CODE_GENDER'] = donnee_sortie['CODE_GENDER'].map({0: 'M', 1: 'F'})
            donnee_sortie['FLAG_OWN_CAR'] = donnee_sortie['FLAG_OWN_CAR'].map({0: 'N', 1: 'Y'})
            donnee_sortie['FLAG_OWN_REALTY'] = donnee_sortie['FLAG_OWN_REALTY'].map({0: 'Y', 1: 'N'})
            donnee_sortie['Features'] = "Data"
            df = donnee_sortie.set_index('Features').T
            df=df.reset_index()
            df=df.rename(columns={"index": "Features"})
            #col_1.write(HTML(df.to_html(escape=False)))
            with col_1:
                gridOptions = funct_grid_option(df)
                AgGrid(df,gridOptions=gridOptions, enable_enterprise_modules=True)
            gauge(col_3, col_6)

#Fin Tableau client transposé-------------------------------------------------------------------------------------------

#Onglet Rapport---------------------------------------------------------------------------------------------------------
#if selected == 'Rapport':
#    report(df_train1[col])
#Fin Onglet Rapport-----------------------------------------------------------------------------------------------------

#Onglet Data Client-----------------------------------------------------------------------------------------------------
if selected == "Description" or selected == "Data Client":

    with st.expander("Description des variables", expanded=True):

        form_descr = st.form(key="form_descr")
    #if colonne_descr:
        #select_desc = st.selectbox('Sélectionner une variables', description['Row'], key='tab')

        form_descr.markdown("<div id='shapley'><h3>Description des variables</h3></div></br>", unsafe_allow_html=True)

        col_descr = form_descr.multiselect('Sélectionner une ou plusieurs variables', description['Row'])
        #t.sidebar.write(description['Description'][description['Row']==select_desc].head(1).values[0])

        submit_descr = form_descr.form_submit_button(label="submit 🔎" )

        if submit_descr :

            #st.session_state.load_state = True
            #form_descr.session_state.Trueload_state = True
            if (len(col_descr) == 0):
                form_descr.error('❌ Sélectionner au moins une variable')
                st.stop()

            Row_list=[]
            for index, rows in description.iterrows():
                for i in range(len(col_descr)):
                    if (rows.Row == col_descr[i] and col_descr[i] in description['Row'].values):
                        my_list = [rows.Row, rows.Description]


                        Row_list.append(my_list)
            if(len(Row_list)!=0):
                df_dscr = pd.DataFrame(Row_list)
                df_dscr.columns = ['Variable', 'Description']
                form_descr.write(HTML(df_dscr.to_html(index=False, escape=False)))
                form_descr.markdown("</br>", unsafe_allow_html=True)


        #st.write(HTML(description[['Row','Description']][description['Row']==select_desc].head(1).to_html(index = False, escape=False)))
        #st.markdown("</br>", unsafe_allow_html=True)

#Fin Onglet Data Client-------------------------------------------------------------------------------------------------

#Onglet Analyse---------------------------------------------------------------------------------------------------------
if selected == "Analyse":
    with st.expander("Analyse Client : "+var_code, expanded=True):
        my_form = st.form(key="form")
#if plot_client and len(pred_client)!=0:

        my_form.markdown("<div id='shapley'><h3>Analyse Client : "+var_code+"</h3></div></br>", unsafe_allow_html=True)


        var = ['All','EXT_SOURCE_MEAN', 'AMT_CREDIT', 'DAYS_BIRTH','FLAG_OWN_CAR','FLAG_OWN_REALTY',
               'AMT_ANNUITY', 'POS_CNT_INSTALMENT_FUTURE_MEAN', 'AMT_GOODS_PRICE',
               'PREV_CNT_PAYMENT_MEAN','BUREAU_AMT_CREDIT_SUM_DEBT_MEAN', 'DAYS_EMPLOYED',
               'APPROVED_AMT_ANNUITY_MEAN', 'PREV_APP_CREDIT_PERC_MEAN',
               'DAYS_ID_PUBLISH', 'ACTIVE_DAYS_CREDIT_MEAN', 'INS_AMT_PAYMENT_MEAN', 'CODE_GENDER',
               'PREV_DAYS_LAST_DUE_1ST_VERSION_MEAN', 'DAYS_LAST_PHONE_CHANGE','INS_DAYS_ENTRY_PAYMENT_MEAN',
               'BUREAU_AMT_CREDIT_SUM_MEAN', 'INS_DAYS_INSTALMENT_MEAN','PREV_AMT_DOWN_PAYMENT_MEAN']

        col_selected = my_form.multiselect('Sélectionner une ou plusieurs variables', var)

        # all forms end with a submit button, that is how the user can trigger
        submit = my_form.form_submit_button(label="submit 🔎")
        #clear = my_form.form_submit_button(label="Clear", on_click=clear_form)
        with st.spinner('Chargement des graphiques ⌛'):
            if submit:

                #st.session_state.load_state = True
                #st.session_state.select_1 = col_selected[0]

                if (len(col_selected) == 0):
                    my_form.error('❌ Sélectionner au moins une variable')
                    st.stop()

                elif (col_selected[0] == 'All'):
                    col_selected = var[1:]
                    plot_distribution_comp(col_selected, int(var_code), nrow=int(np.ceil(len(col_selected) / 2)), ncol=2)

                elif (len(col_selected)<3) :
                    plot_distribution_comp(col_selected, int(var_code), nrow=1, ncol=len(col_selected))

                elif (len(col_selected)>0) :
                    plot_distribution_comp(col_selected, int(var_code), nrow=int(np.ceil(len(col_selected)/2)), ncol=2)
    with st.expander("Description des variables", expanded=True):
        Row_list = []
        for i in range(len(col_selected)):
            if col_selected[i] in df_description['col_name'].values:
                desc = df_description[['col_name','Description']][df_description['col_name'] == col_selected[i]].head(1).values[0]

                Row_list.append(desc)

        if(len(Row_list)!=0):
            df_dscr = pd.DataFrame(Row_list)
            df_dscr.columns = ['Variable', 'Description']

            AgGrid(df_dscr,height=200, width='100%')
            #st.write(HTML(df_dscr.to_html(index=False, escape=False)))
            st.markdown("</br>",unsafe_allow_html=True)
#Fin Onglet Analyse-----------------------------------------------------------------------------------------------------

#Fin Onglet Shapley-----------------------------------------------------------------------------------------------------
if selected == "Shapley":
    with st.expander("Analyse Shapley client: "+var_code, expanded=True):
#if shapley:
        # Initialize SHAP Tree explainer
        with st.spinner('Chargement des données Shapley ⌛'):
            index_client = df_train1[df_train1['SK_ID_CURR'] == var_code].index.values[0]

            explainer, shap_values, expected_value = get_explainer()

            st.markdown("<div id='shapley'><h3>Analyse Shapley Client: "+var_code+"</h3></div></br>", unsafe_allow_html=True)

            shap1, shap2 = st.columns(2)
            shap1.markdown("<div id='graph_shap'><h4>Decision plot</h4></div>", unsafe_allow_html=True)
            # decision_plot
            decision = shap.decision_plot(base_value=explainer.expected_value[1],
                                          shap_values=shap_values[1][0],
                                          features=df_train1.iloc[index_client:, 2:-2],
                                          feature_names=df_train1.iloc[:, 2:-2].columns.tolist())

            shap1.pyplot(decision)
            # st.pyplot(fig,bbox_inches='tight',dpi=300,pad_inches=0)

            shap2.markdown("<div id='graph_shap'><h4>Waterfall Plot</h4></div>", unsafe_allow_html=True)
            # Summarize the effects of all the features
            #shap2.pyplot(shap.summary_plot(shap_values, pred_client.iloc[:, 2:-2]))

            shap2.pyplot(shap.plots._waterfall.waterfall_legacy(expected_value[1],
                                                                shap_values[1][0],
                                                                feature_names = df_train1.iloc[:, 2:-2].columns,
                                                                max_display = 20))

            st.markdown("<div id='graph_shap'><h4>Force plot</h4></div>", unsafe_allow_html=True)
            # force_plot
            st_shap(shap.force_plot(expected_value[1], shap_values[1][0],
                                    df_train1.iloc[index_client, 2:-2]))

    with st.expander("Description des variables Shapley", expanded=True):

        #Extraire les variables les plus importantes
        vals = np.abs(shap_values).mean(0)
        feature_importance = pd.DataFrame(list(zip(pred_client.iloc[:, 2:-2].columns, vals[0])),
                                          columns=['col_name', 'feature_importance_vals'])
        feature_importance.sort_values(by=['feature_importance_vals'], ascending=False, inplace=True)

        feature_importance = feature_importance.head(20)

        feature_importance = pd.merge(feature_importance, df_description, how="inner", on="col_name")
        feature_importance.rename(columns={'col_name':'Variable'}, inplace=True)

        st.markdown("<div id='shapley'><h3>Description des variables Shapley</h3></div></br>",
                            unsafe_allow_html=True)

        gridOptions = funct_grid_option(feature_importance[['Variable', 'Description']])

        AgGrid(feature_importance[['Variable', 'Description']],gridOptions=gridOptions, enable_enterprise_modules=True)
        #st.write(HTML(feature_importance[['Variable', 'Description']].to_html(index=False, escape=False)))
        st.markdown("</br>", unsafe_allow_html=True)
        #st.balloons()
# Fin Onglet Shapley----------------------------------------------------------------------------------------------------
