import streamlit as st
import pandas as pd
import numpy as np
import pickle

st.title("Bebek İsmi Popülerlik Tahmini :baby:")
model=pickle.load(open('isim_model.pkl','rb'))
pivot=pickle.load(open('isim_pivot.pkl','rb'))

isim=st.selectbox('İsim seçin',sorted(pivot.index.tolist()))
yil_sayisi=st.number_input('Kaç yıl sonrasına kadar tahmin edilsin?',1,10,5)
if st.button('Tahmin et'):
    seri=pivot.loc[isim]
    gecmis=seri.values.astype(float).tolist()
    yil=int(seri.index[-1])
    tahminler={}
    for i in range(int(yil_sayisi)):
        pencere=np.array(gecmis[-5:])
        m=pencere.mean()
        if m==0:
            t=0.0
        else:
            t=float(model.predict([pencere/m])[0])*m
        t=max(t,0)
        gecmis.append(t)
        yil=yil+1
        tahminler[yil]=round(t)
    st.subheader('Geçmiş ve tahmin')
    gecmis_df=pd.DataFrame({'Geçmiş':seri})
    tahmin_df=pd.DataFrame({'Tahmin':pd.Series(tahminler)})
    st.line_chart(gecmis_df.join(tahmin_df,how='outer'))
    st.write(tahmin_df)
