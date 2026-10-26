"""Real-engine checks: VOLT_EXERCISE_PRICER=target/release/exercise_pricer python -m unittest discover -s demo -p 'test_exercise*.py'."""
import copy
import os
from pathlib import Path
import unittest
from exercise_risk_service import calculate

@unittest.skipUnless(os.environ.get('VOLT_EXERCISE_PRICER'), 'set VOLT_EXERCISE_PRICER to the release binary')
class ExerciseRiskServiceTests(unittest.TestCase):
    def setUp(self):
        self.pricer = Path(os.environ['VOLT_EXERCISE_PRICER']).resolve(strict=True)
        self.now = 1767225600000000000
        self.grid = {'space_steps': 400, 'time_steps': 400}
        self.frame = {'schema':'volt.exercise_quote_frame.v1','generation':1,'source':'synthetic', 'as_of_ns':str(self.now), 'quotes':[]}
        for i, quantity in enumerate([2,-1]):
            self.frame['quotes'].append({'underlying':'XYZ','bid':10.44,'ask':10.46,'quote_ns':str(self.now),'spot_ns':str(self.now), 'quantity':quantity,'multiplier':100,
              'option':{'row_id':str(i),'contract_id':'XYZ-C-'+str(i),'side':'call','spot':100.,'strike':100.,'valuation_time':self.now,'exercise':{'style':'american','first_exercise':self.now,'expiry':self.now+31536000000000000},'volatility':.2,'rate':.05,'dividend_yield':0.,'cash_dividends':[]}})
    def run_frame(self, frame=None, cache=None, workers=4):
        return calculate(frame or self.frame,self.pricer,self.grid,30,cache,workers)
    def test_signed_units_and_serial_parallel_agree(self):
        a=self.run_frame(workers=1);b=self.run_frame(workers=4)
        self.assertEqual(a['rows'],b['rows']);self.assertEqual(a['portfolio'],b['portfolio'])
        row=a['rows'][0];risk=row['risk'];self.assertAlmostEqual(row['iv']['volatility'],.2,delta=.001)
        self.assertAlmostEqual(a['portfolio']['value'],100*risk['valuation']['price'])
        self.assertAlmostEqual(a['portfolio']['delta_dollars'],10000*risk['delta'])
        self.assertAlmostEqual(a['portfolio']['gamma_1pct'],50*risk['gamma'])
    def test_stock_update_retains_quote_paired_iv(self):
        cache={};first=self.run_frame(cache=cache);moved=copy.deepcopy(self.frame);later=self.now+60_000_000_000
        moved.update(generation=2,as_of_ns=str(later))
        for q in moved['quotes']:
            q.update(calibration_spot=100.,calibration_spot_ns=str(self.now),calibration_ns=str(self.now),spot_ns=str(later))
            q['option'].update(spot=101.,valuation_time=later)
        second=self.run_frame(moved,cache)
        self.assertEqual(len(cache),2)
        for a,b in zip(first['rows'],second['rows']):
            self.assertEqual(a['iv'],b['iv']);self.assertGreater(b['risk']['valuation']['price'],a['risk']['valuation']['price'])
            self.assertEqual(b['calibration_ns'],str(self.now));self.assertEqual(b['risk_asof_ns'],str(later))
    def test_rejects_future_or_duplicate_observations(self):
        frame=copy.deepcopy(self.frame);frame['quotes'][0]['spot_ns']=str(self.now+1)
        with self.assertRaisesRegex(ValueError,'observations'):self.run_frame(frame)
        frame=copy.deepcopy(self.frame);frame['quotes'][1]['option']['contract_id']=frame['quotes'][0]['option']['contract_id']
        with self.assertRaisesRegex(ValueError,'unique'):self.run_frame(frame)
    def test_unidentified_position_has_unavailable_portfolio_risk(self):
        frame=copy.deepcopy(self.frame);frame['quotes'][0].update(bid=150.,ask=151.)
        result=self.run_frame(frame)
        self.assertIsNone(result['rows'][0]['risk']);self.assertIsNone(result['portfolio']['value'])
        self.assertEqual(result['portfolio']['unpriced_positions'],['XYZ-C-0'])
