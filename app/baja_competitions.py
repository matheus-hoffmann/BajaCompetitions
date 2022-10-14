import pickle
import pandas as pd


class BajaCompetitions(object):
    def __init__(self):
        self.mms_seguranca = pickle.load(open('../persistence/mms_Seguranca.pkl', 'rb'))
        self.mms_projeto = pickle.load(open('../persistence/mms_Projeto.pkl', 'rb'))
        self.mms_dinamicas = pickle.load(open('../persistence/mms_Dinamicas.pkl', 'rb'))
        self.mms_enduro = pickle.load(open('../persistence/mms_Enduro.pkl', 'rb'))

    def data_preparation(self, df: pd.DataFrame):
        _df = df.copy()

        # Rescaling inputs
        _df['Seguranca'] = self.mms_seguranca.transform(_df['Seguranca'].values.reshape((-1, 1)))
        _df['Projeto'] = self.mms_projeto.transform(_df['Projeto'].values.reshape((-1, 1)))
        _df['Dinamicas'] = self.mms_dinamicas.transform(_df['Dinamicas'].values.reshape((-1, 1)))
        _df['Enduro'] = self.mms_enduro.transform(_df['Enduro'].values.reshape((-1, 1)))

        return _df
