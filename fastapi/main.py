# This is a sample Python script.
# 1. Library imports
import pandas as pd
import pycaret
from pycaret.regression import load_model, predict_model
from pydantic import BaseModel
from fastapi import FastAPI
import uvicorn
import joblib,os
import pickle
import json
from fastapi.encoders import jsonable_encoder

# 2. Create the app object
app = FastAPI()

#. Load trained model
model = open('../Model/best_lgbm_over.pkl', 'rb')
lgbm_clf = joblib.load(model)

#lgbm_clf = pickle.load(open('../lgbm_clf.pkl', 'rb'))

class inputs(BaseModel):
    NAME_CONTRACT_TYPE: int
    CODE_GENDER: int
    FLAG_OWN_CAR: int
    FLAG_OWN_REALTY: int
    CNT_CHILDREN: int
    AMT_INCOME_TOTAL: float
    AMT_CREDIT: float
    AMT_ANNUITY: float
    AMT_GOODS_PRICE: float
    REGION_POPULATION_RELATIVE: float
    DAYS_BIRTH: int
    DAYS_EMPLOYED: float
    DAYS_REGISTRATION: float
    DAYS_ID_PUBLISH: int
    CNT_FAM_MEMBERS: float
    REGION_RATING_CLIENT: int
    REGION_RATING_CLIENT_W_CITY: int
    HOUR_APPR_PROCESS_START: int
    OBS_30_CNT_SOCIAL_CIRCLE: float
    DEF_30_CNT_SOCIAL_CIRCLE: float
    OBS_60_CNT_SOCIAL_CIRCLE: float
    DEF_60_CNT_SOCIAL_CIRCLE: float
    DAYS_LAST_PHONE_CHANGE: float
    AMT_REQ_CREDIT_BUREAU_HOUR: float
    AMT_REQ_CREDIT_BUREAU_DAY: float
    AMT_REQ_CREDIT_BUREAU_WEEK: float
    AMT_REQ_CREDIT_BUREAU_MON: float
    AMT_REQ_CREDIT_BUREAU_QRT: float
    AMT_REQ_CREDIT_BUREAU_YEAR: float
    REGION: int
    NAME_TYPE_SUITE_Children: float
    NAME_TYPE_SUITE_Family: float
    NAME_TYPE_SUITE_Rare: float
    NAME_TYPE_SUITE_Spouse_partner: float
    NAME_TYPE_SUITE_Unaccompanied: float
    NAME_INCOME_TYPE_Commercial_associate: float
    NAME_INCOME_TYPE_Pensioner: float
    NAME_INCOME_TYPE_Rare: float
    NAME_INCOME_TYPE_State_servant: float
    NAME_INCOME_TYPE_Working: float
    NAME_EDUCATION_TYPE_Higher_education: float
    NAME_EDUCATION_TYPE_Incomplete_higher: float
    NAME_EDUCATION_TYPE_Lower_secondary: float
    NAME_EDUCATION_TYPE_Secondary_secondary_special: float
    NAME_FAMILY_STATUS_Civil_marriage: float
    NAME_FAMILY_STATUS_Married: float
    NAME_FAMILY_STATUS_Separated: float
    NAME_FAMILY_STATUS_Single_not_married: float
    NAME_FAMILY_STATUS_Widow: float
    OCCUPATION_TYPE_Accountants: float
    OCCUPATION_TYPE_Core_staff: float
    OCCUPATION_TYPE_Drivers: float
    OCCUPATION_TYPE_HR_staff: float
    OCCUPATION_TYPE_High_skill_tech_staff: float
    OCCUPATION_TYPE_Laborers: float
    OCCUPATION_TYPE_Low_skill_Laborers: float
    OCCUPATION_TYPE_Managers: float
    OCCUPATION_TYPE_Medicine_staff: float
    OCCUPATION_TYPE_Realty_agents: float
    OCCUPATION_TYPE_Sales_staff: float
    OCCUPATION_TYPE_Secretaries: float
    ORGANIZATION_TYPE_Agriculture: float
    ORGANIZATION_TYPE_Business_Entity: float
    ORGANIZATION_TYPE_Construction: float
    ORGANIZATION_TYPE_Education: float
    ORGANIZATION_TYPE_Finance: float
    ORGANIZATION_TYPE_Government: float
    ORGANIZATION_TYPE_HotelRestaurant: float
    ORGANIZATION_TYPE_House: float
    ORGANIZATION_TYPE_Industry: float
    ORGANIZATION_TYPE_Other: float
    ORGANIZATION_TYPE_Public: float
    ORGANIZATION_TYPE_Security: float
    ORGANIZATION_TYPE_Self_employed: float
    ORGANIZATION_TYPE_Services: float
    ORGANIZATION_TYPE_Trade: float
    ORGANIZATION_TYPE_Transport: float
    ORGANIZATION_TYPE_XNA: float
    EXT_SOURCE_MEAN: float
    BUREAU_DAYS_CREDIT_MEAN: float
    BUREAU_DAYS_CREDIT_ENDDATE_MEAN: float
    BUREAU_DAYS_CREDIT_UPDATE_MEAN: float
    BUREAU_CREDIT_DAY_OVERDUE_MEAN: float
    BUREAU_AMT_CREDIT_SUM_MEAN: float
    BUREAU_AMT_CREDIT_SUM_DEBT_MEAN: float
    BUREAU_AMT_CREDIT_SUM_OVERDUE_MEAN: float
    BUREAU_AMT_CREDIT_SUM_LIMIT_MEAN: float
    BUREAU_CNT_CREDIT_PROLONG_MEAN: float
    BUREAU_CREDIT_ACTIVE_Active_MEAN: float
    BUREAU_CREDIT_ACTIVE_Closed_MEAN: float
    BUREAU_CREDIT_ACTIVE_Sold_BadDebt_MEAN: float
    BUREAU_CREDIT_CURRENCY_currency_1_MEAN: float
    BUREAU_CREDIT_CURRENCY_currency_2_MEAN: float
    BUREAU_CREDIT_CURRENCY_currency_3_MEAN: float
    BUREAU_CREDIT_CURRENCY_currency_4_MEAN: float
    BUREAU_CREDIT_TYPE_Car_loan_MEAN: float
    BUREAU_CREDIT_TYPE_Consumer_credit_MEAN: float
    BUREAU_CREDIT_TYPE_Credit_card_MEAN: float
    BUREAU_CREDIT_TYPE_Microloan_MEAN: float
    BUREAU_CREDIT_TYPE_Mortgage_MEAN: float
    BUREAU_CREDIT_TYPE_Other_MEAN: float
    BUREAU_COUNT: float
    BUREAU_COUNT_CAT: float
    ACTIVE_DAYS_CREDIT_MEAN: float
    ACTIVE_DAYS_CREDIT_UPDATE_MEAN: float
    ACTIVE_CREDIT_DAY_OVERDUE_MEAN: float
    ACTIVE_AMT_CREDIT_SUM_MEAN: float
    ACTIVE_AMT_CREDIT_SUM_OVERDUE_MEAN: float
    ACTIVE_CNT_CREDIT_PROLONG_MEAN: float
    CLOSED_DAYS_CREDIT_MEAN: float
    CLOSED_DAYS_CREDIT_ENDDATE_MEAN: float
    CLOSED_DAYS_CREDIT_UPDATE_MEAN: float
    CLOSED_CREDIT_DAY_OVERDUE_MEAN: float
    CLOSED_AMT_CREDIT_SUM_MEAN: float
    CLOSED_AMT_CREDIT_SUM_DEBT_MEAN: float
    CLOSED_AMT_CREDIT_SUM_OVERDUE_MEAN: float
    CLOSED_CNT_CREDIT_PROLONG_MEAN: float
    POS_MONTHS_BALANCE_MEAN: float
    POS_MONTHS_BALANCE_SIZE: float
    POS_CNT_INSTALMENT_MEAN: float
    POS_CNT_INSTALMENT_FUTURE_MEAN: float
    POS_SK_DPD_MEAN: float
    POS_SK_DPD_DEF_MEAN: float
    POS_NAME_CONTRACT_STATUS_Active_MEAN: float
    POS_NAME_CONTRACT_STATUS_Completed_MEAN: float
    POS_NAME_CONTRACT_STATUS_Rare_MEAN: float
    df_POS_CASH_balance_COUNT: float
    INS_NUM_INSTALMENT_VERSION_NUNIQUE: float
    INS_NUM_INSTALMENT_NUMBER_MEAN: float
    INS_DAYS_INSTALMENT_MEAN: float
    INS_DAYS_ENTRY_PAYMENT_MEAN: float
    INS_AMT_INSTALMENT_MEAN: float
    INS_AMT_PAYMENT_MEAN: float
    INS_DPD_MEAN: float
    INS_DBD_MEAN: float
    INS_PAYMENT_PERC_MEAN: float
    INS_PAYMENT_DIFF_MEAN: float
    INS_COUNT: float
    PREV_AMT_ANNUITY_MEAN: float
    PREV_AMT_APPLICATION_MEAN: float
    PREV_AMT_CREDIT_MEAN: float
    PREV_APP_CREDIT_PERC_MEAN: float
    PREV_AMT_DOWN_PAYMENT_MEAN: float
    PREV_AMT_GOODS_PRICE_MEAN: float
    PREV_HOUR_APPR_PROCESS_START_MEAN: float
    PREV_RATE_DOWN_PAYMENT_MEAN: float
    PREV_DAYS_DECISION_MEAN: float
    PREV_CNT_PAYMENT_MEAN: float
    PREV_SELLERPLACE_AREA_MEAN: float
    PREV_DAYS_FIRST_DUE_MEAN: float
    PREV_DAYS_LAST_DUE_1ST_VERSION_MEAN: float
    PREV_DAYS_LAST_DUE_MEAN: float
    PREV_DAYS_TERMINATION_MEAN: float
    PREV_NFLAG_INSURED_ON_APPROVAL_MEAN: float
    PREV_NAME_CONTRACT_TYPE_Cash_loans_MEAN: float
    PREV_NAME_CONTRACT_TYPE_Consumer_loans_MEAN: float
    PREV_NAME_CONTRACT_TYPE_Revolving_loans_MEAN: float
    PREV_NAME_CONTRACT_TYPE_XNA_MEAN: float
    PREV_NAME_CONTRACT_TYPE_nan_MEAN: float
    PREV_NAME_CASH_LOAN_PURPOSE_Other_MEAN: float
    PREV_NAME_CASH_LOAN_PURPOSE_XAP_MEAN: float
    PREV_NAME_CASH_LOAN_PURPOSE_XNA_MEAN: float
    PREV_NAME_CASH_LOAN_PURPOSE_nan_MEAN: float
    PREV_NAME_CONTRACT_STATUS_Approved_MEAN: float
    PREV_NAME_CONTRACT_STATUS_Canceled_MEAN: float
    PREV_NAME_CONTRACT_STATUS_Refused_MEAN: float
    PREV_NAME_CONTRACT_STATUS_Unused_offer_MEAN: float
    PREV_NAME_CONTRACT_STATUS_nan_MEAN: float
    PREV_NAME_PAYMENT_TYPE_Cash_through_the_bank_MEAN: float
    PREV_NAME_PAYMENT_TYPE_Rare_MEAN: float
    PREV_NAME_PAYMENT_TYPE_XNA_MEAN: float
    PREV_NAME_PAYMENT_TYPE_nan_MEAN: float
    PREV_CODE_REJECT_REASON_CLIENT_MEAN: float
    PREV_CODE_REJECT_REASON_HC_MEAN: float
    PREV_CODE_REJECT_REASON_LIMIT_MEAN: float
    PREV_CODE_REJECT_REASON_Rare_MEAN: float
    PREV_CODE_REJECT_REASON_SCO_MEAN: float
    PREV_CODE_REJECT_REASON_XAP_MEAN: float
    PREV_CODE_REJECT_REASON_nan_MEAN: float
    PREV_NAME_TYPE_SUITE_Children_MEAN: float
    PREV_NAME_TYPE_SUITE_Family_MEAN: float
    PREV_NAME_TYPE_SUITE_Other_B_MEAN: float
    PREV_NAME_TYPE_SUITE_Rare_MEAN: float
    PREV_NAME_TYPE_SUITE_Spouse_partner_MEAN: float
    PREV_NAME_TYPE_SUITE_Unaccompanied_MEAN: float
    PREV_NAME_TYPE_SUITE_nan_MEAN: float
    PREV_NAME_CLIENT_TYPE_New_MEAN: float
    PREV_NAME_CLIENT_TYPE_Refreshed_MEAN: float
    PREV_NAME_CLIENT_TYPE_Repeater_MEAN: float
    PREV_NAME_CLIENT_TYPE_XNA_MEAN: float
    PREV_NAME_CLIENT_TYPE_nan_MEAN: float
    PREV_NAME_GOODS_CATEGORY_Audio_Video_MEAN: float
    PREV_NAME_GOODS_CATEGORY_Clothing_and_Accessories_MEAN: float
    PREV_NAME_GOODS_CATEGORY_Computers_MEAN: float
    PREV_NAME_GOODS_CATEGORY_Construction_Materials_MEAN: float
    PREV_NAME_GOODS_CATEGORY_Consumer_Electronics_MEAN: float
    PREV_NAME_GOODS_CATEGORY_Furniture_MEAN: float
    PREV_NAME_GOODS_CATEGORY_Mobile_MEAN: float
    PREV_NAME_GOODS_CATEGORY_Photo_Cinema_Equipment_MEAN: float
    PREV_NAME_GOODS_CATEGORY_Rare_MEAN: float
    PREV_NAME_GOODS_CATEGORY_XNA_MEAN: float
    PREV_NAME_GOODS_CATEGORY_nan_MEAN: float
    PREV_NAME_PORTFOLIO_Cash_MEAN: float
    PREV_NAME_PORTFOLIO_POS_MEAN: float
    PREV_NAME_PORTFOLIO_Rare_MEAN: float
    PREV_NAME_PORTFOLIO_XNA_MEAN: float
    PREV_NAME_PORTFOLIO_nan_MEAN: float
    PREV_NAME_PRODUCT_TYPE_XNA_MEAN: float
    PREV_NAME_PRODUCT_TYPE_walk_in_MEAN: float
    PREV_NAME_PRODUCT_TYPE_x_sell_MEAN: float
    PREV_NAME_PRODUCT_TYPE_nan_MEAN: float
    PREV_CHANNEL_TYPE_AP_Cash_loan__MEAN: float
    PREV_CHANNEL_TYPE_Contact_center_MEAN: float
    PREV_CHANNEL_TYPE_Country_wide_MEAN: float
    PREV_CHANNEL_TYPE_Credit_and_cash_offices_MEAN: float
    PREV_CHANNEL_TYPE_Rare_MEAN: float
    PREV_CHANNEL_TYPE_Regional_Local_MEAN: float
    PREV_CHANNEL_TYPE_Stone_MEAN: float
    PREV_CHANNEL_TYPE_nan_MEAN: float
    PREV_NAME_SELLER_INDUSTRY_Clothing_MEAN: float
    PREV_NAME_SELLER_INDUSTRY_Connectivity_MEAN: float
    PREV_NAME_SELLER_INDUSTRY_Construction_MEAN: float
    PREV_NAME_SELLER_INDUSTRY_Consumer_electronics_MEAN: float
    PREV_NAME_SELLER_INDUSTRY_Furniture_MEAN: float
    PREV_NAME_SELLER_INDUSTRY_Industry_MEAN: float
    PREV_NAME_SELLER_INDUSTRY_Rare_MEAN: float
    PREV_NAME_SELLER_INDUSTRY_XNA_MEAN: float
    PREV_NAME_SELLER_INDUSTRY_nan_MEAN: float
    PREV_NAME_YIELD_GROUP_XNA_MEAN: float
    PREV_NAME_YIELD_GROUP_high_MEAN: float
    PREV_NAME_YIELD_GROUP_low_action_MEAN: float
    PREV_NAME_YIELD_GROUP_low_normal_MEAN: float
    PREV_NAME_YIELD_GROUP_middle_MEAN: float
    PREV_NAME_YIELD_GROUP_nan_MEAN: float
    PREV_PRODUCT_COMBINATION_Card_Street_MEAN: float
    PREV_PRODUCT_COMBINATION_Card_X_Sell_MEAN: float
    PREV_PRODUCT_COMBINATION_Cash_MEAN: float
    PREV_PRODUCT_COMBINATION_Cash_Street_high_MEAN: float
    PREV_PRODUCT_COMBINATION_Cash_Street_low_MEAN: float
    PREV_PRODUCT_COMBINATION_Cash_Street_middle_MEAN: float
    PREV_PRODUCT_COMBINATION_Cash_X_Sell_high_MEAN: float
    PREV_PRODUCT_COMBINATION_Cash_X_Sell_low_MEAN: float
    PREV_PRODUCT_COMBINATION_Cash_X_Sell_middle_MEAN: float
    PREV_PRODUCT_COMBINATION_POS_household_with_interest_MEAN: float
    PREV_PRODUCT_COMBINATION_POS_household_without_interest_MEAN: float
    PREV_PRODUCT_COMBINATION_POS_industry_with_interest_MEAN: float
    PREV_PRODUCT_COMBINATION_POS_industry_without_interest_MEAN: float
    PREV_PRODUCT_COMBINATION_POS_mobile_with_interest_MEAN: float
    PREV_PRODUCT_COMBINATION_POS_mobile_without_interest_MEAN: float
    PREV_PRODUCT_COMBINATION_POS_other_with_interest_MEAN: float
    PREV_PRODUCT_COMBINATION_POS_others_without_interest_MEAN: float
    PREV_PRODUCT_COMBINATION_nan_MEAN: float
    PREV_PRODUCT_COMBINATION_CATS_CARD_MEAN: float
    PREV_PRODUCT_COMBINATION_CATS_CASH_MEAN: float
    PREV_PRODUCT_COMBINATION_CATS_POS_MEAN: float
    PREV_PRODUCT_COMBINATION_CATS_nan_MEAN: float
    PREV_PRODUCT_COMBINATION_POS_WITH_OTHER_MEAN: float
    PREV_PRODUCT_COMBINATION_POS_WITH_WITH_MEAN: float
    PREV_PRODUCT_COMBINATION_POS_WITH_WITHOUT_MEAN: float
    PREV_PRODUCT_COMBINATION_POS_WITH_nan_MEAN: float
    PREV_PRODUCT_COMBINATION_POS_TYPE_OTHER_MEAN: float
    PREV_PRODUCT_COMBINATION_POS_TYPE_household_MEAN: float
    PREV_PRODUCT_COMBINATION_POS_TYPE_industry_MEAN: float
    PREV_PRODUCT_COMBINATION_POS_TYPE_mobile_MEAN: float
    PREV_PRODUCT_COMBINATION_POS_TYPE_posother_MEAN: float
    PREV_PRODUCT_COMBINATION_POS_TYPE_nan_MEAN: float
    PREV_PRODUCT_COMBINATION_CASH_TYPE_OTHER_MEAN: float
    PREV_PRODUCT_COMBINATION_CASH_TYPE_street_MEAN: float
    PREV_PRODUCT_COMBINATION_CASH_TYPE_xsell_MEAN: float
    PREV_PRODUCT_COMBINATION_CASH_TYPE_nan_MEAN: float
    APPROVED_AMT_ANNUITY_MEAN: float
    APPROVED_AMT_APPLICATION_MEAN: float
    APPROVED_AMT_CREDIT_MEAN: float
    APPROVED_APP_CREDIT_PERC_MEAN: float
    APPROVED_AMT_DOWN_PAYMENT_MEAN: float
    APPROVED_AMT_GOODS_PRICE_MEAN: float
    APPROVED_HOUR_APPR_PROCESS_START_MEAN: float
    APPROVED_RATE_DOWN_PAYMENT_MEAN: float
    APPROVED_DAYS_DECISION_MEAN: float
    APPROVED_CNT_PAYMENT_MEAN: float
    APPROVED_SELLERPLACE_AREA_MEAN: float
    APPROVED_DAYS_FIRST_DUE_MEAN: float
    APPROVED_DAYS_LAST_DUE_1ST_VERSION_MEAN: float
    APPROVED_DAYS_LAST_DUE_MEAN: float
    APPROVED_DAYS_TERMINATION_MEAN: float
    APPROVED_NFLAG_INSURED_ON_APPROVAL_MEAN: float

# Press Maj+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

# Define predict function
@app.post('/predict/')
async def predict(data: inputs):
    # get data from request body
    data_dict = data.dict()
    data_df = pd.DataFrame.from_dict([data_dict])
    #pred_name = pd.DataFrame(data)
    #data1 = pd.DataFrame(data)
    #data = [data]
    # convert to pd DF since sklearn cannot predict from dict
    #inputDF = pd.DataFrame(inputData.to_dict())
    #pred = lgbm_clf.predict(data)
    #pred = int(pred)
    #y_pred = predict_model(model, data=df)
    #print (pred)
    #return {pred}
    #return {'message': int(pred)}

# Press the green button in the gutter to run the script.

    pred = lgbm_clf.predict_proba(data_df).tolist()

    return pred

@app.post('/predict_all/')
async def predict_all(data: inputs):
    # get data from request body
    #data_dict = data.dict()
    #data_df = pd.DataFrame.from_dict([data_dict])


    #pred = lgbm_clf.predict_proba(data_df).tolist()

    return data

if __name__ == '__main__':
    print_hi('PyCharm')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
    uvicorn.run(app, host='127.0.0.1', port=8000)