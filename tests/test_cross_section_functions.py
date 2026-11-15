import unittest
import numpy as np
import pandas as pd
from lite.expressions import calculate, parse, ExpressionError, GUIDE

class CrossSectionFunctionsTests(unittest.TestCase):
    def setUp(self):
        index=pd.MultiIndex.from_product([['A','B','C','D'],pd.date_range('2020-01-01',periods=3)],names=['instrument','datetime'])
        self.frame=pd.DataFrame({'$close':[1,0,4, 3,0,np.nan, 100,0,10, 7,0,np.nan]},index=index,dtype=float)
        self.member=pd.Series(True,index=index)
        self.member.loc[('C',pd.Timestamp('2020-01-01'))]=False
    def values(self,expr,date):
        return calculate(expr,self.frame,self.member).xs(pd.Timestamp(date),level='datetime')
    def test_cross_section_and_membership(self):
        np.testing.assert_allclose(self.values('MEAN($close)','2020-01-01'),[11/3,11/3,np.nan,11/3],equal_nan=True)
        std=np.std([1,3,7],ddof=1)
        np.testing.assert_allclose(self.values('STD($close)','2020-01-01'),[std,std,np.nan,std],equal_nan=True)
        np.testing.assert_allclose(self.values('ZSCORE($close)','2020-01-01'),[(1-11/3)/std,(3-11/3)/std,np.nan,(7-11/3)/std],equal_nan=True)
        np.testing.assert_allclose(self.values('SCALE($close)','2020-01-01'),[1/11,3/11,np.nan,7/11],equal_nan=True)
    def test_missing_zero_and_singleton(self):
        self.assertTrue(self.values('ZSCORE($close)','2020-01-02').isna().all())
        self.assertTrue(self.values('SCALE($close)','2020-01-02').isna().all())
        # Missing instruments do not get an imputed mean.
        np.testing.assert_allclose(self.values('MEAN($close)','2020-01-03'),[7,np.nan,7,np.nan],equal_nan=True)
        one=self.member.copy();one.loc[('C',pd.Timestamp('2020-01-03'))]=False
        self.assertTrue(calculate('STD($close)',self.frame,one).xs(pd.Timestamp('2020-01-03'),level='datetime').isna().all())
    def test_pow_bounded_and_equivalent(self):
        for power in [-4,-1,0,2,4]:
            a=calculate(f'POW($close,{power})',self.frame,self.member)
            b=calculate(f'$close ** {power}',self.frame,self.member)
            pd.testing.assert_series_equal(a,b,check_names=False)
        for expr in ['POW($close,5)','POW($close,-5)','POW($close,0.5)','POW($close,$close)']:
            with self.subTest(expr=expr),self.assertRaises(ExpressionError):parse(expr,['$close'])
        for name in ['ZSCORE','MEAN','STD','SCALE','POW']:self.assertIn(name+'(',GUIDE)
