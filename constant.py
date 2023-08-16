SQL1 = """select case SEX when 1 then 1 else 0 END as male
            , case SEX when 0 then 1 else 0 END as female
            , AGE
            , cast(educa as SIGNED INTEGER) as educa
            , case when income_ann <= 3 then 1
            when income_ann <= 8 then 2
            when income_ann <= 15 then 3
            when income_ann <= 30 then 4
            when income_ann <= 50 then 5
            when income_ann <= 100 then 6
            else 7 end as income_ann
            , OCC_CODE
            , ACTS
            , OPEN_DAY
            , CRED_LIMIT
            , ROUND(USE_RATE, 3)
            , HI_PURCHSE
            , OCT_COUNT
            , AGE_CNT as DUE_CNT
            , AGE_MAX as DUE_MA_CNT
            , ROUND(DET_CRE,3)
            , cast(DUE_CRE*CRED_LIMIT as signed integer) AS DUE_AMT
            , cast(PMT_DUE*DUE_CRE*CRED_LIMIT as signed integer) AS PMT_AMT
            , PMT_DAY
            , CT_MP_Y
            , CASH_CT
            , cast(CASH_AV as signed integer)
            , M_PRECIOUS
            , M_SPORTSCLUB
            , M_FURNITURE
            , M_AUTOSERVICE
            , RES
            from a00_ccb_res
            where EDUCA != ''
                and INCOME_ANN != ''
                and OCC_CATGRY != ''
                and occ_code !='';"""

