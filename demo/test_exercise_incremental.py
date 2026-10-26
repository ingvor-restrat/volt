"""Real Rust pricer -> incremental adapter -> persistent Rust Hypercube tests."""
import copy
import os
from pathlib import Path
import unittest
from exercise_incremental import IncrementalExerciseRisk, NativePublisher

@unittest.skipUnless(os.environ.get('VOLT_EXERCISE_PRICER') and os.environ.get('VOLT_EXERCISE_CUBE'), 'set both native binary paths')
class IncrementalTests(unittest.TestCase):
    def setUp(self):
        self.now = 1767225600000000000
        self.native = NativePublisher(Path(os.environ['VOLT_EXERCISE_CUBE']))
        self.engine = IncrementalExerciseRisk(Path(os.environ['VOLT_EXERCISE_PRICER']), self.native,
                                             {'space_steps': 200, 'time_steps': 200})
    def tearDown(self):
        self.native.close()
    def event(self, underlying, generation, now=None):
        now = self.now if now is None else now
        return {'schema':'volt.exercise_ingest.v1', 'action':'replace_underlying', 'generation':generation,
                'as_of_ns':str(now), 'underlying':underlying, 'frame':{
                'schema':'volt.exercise_quote_frame.v1','source':'synthetic','quotes':[
                    {'underlying':underlying,'bid':10.44,'ask':10.46,'quote_ns':str(now),'spot_ns':str(now),
                     'quantity':-2,'multiplier':100,'option':{'row_id':underlying,'contract_id':underlying+'-C',
                     'side':'call','spot':100.,'strike':100.,'valuation_time':now,'exercise':{'style':'american',
                     'first_exercise':self.now,'expiry':self.now+31536000000000000},'volatility':.2,
                     'rate':.05,'dividend_yield':0.,'cash_dividends':[]}}]}}
    def spot(self, generation=3):
        return {'schema':'volt.exercise_ingest.v1','action':'spot','generation':generation,
                'as_of_ns':str(self.now+60_000_000_000),'underlying':'ABC','spot':101.,
                'spot_ns':str(self.now+60_000_000_000)}
    def test_two_underlying_native_roundtrip_updates_one_and_retains_other_clock(self):
        self.engine.submit(self.event('ABC',1));before=self.engine.submit(self.event('XYZ',2))
        after=self.engine.submit(self.spot());a={r['underlying']:r for r in before['rows']};b={r['underlying']:r for r in after['rows']}
        self.assertEqual(a['XYZ'],b['XYZ']);self.assertEqual(a['ABC']['iv'],b['ABC']['iv'])
        self.assertGreater(b['ABC']['risk']['valuation']['price'],a['ABC']['risk']['valuation']['price'])
        self.assertEqual(after['underlyings']['XYZ']['risk_age_at_publication_seconds'],60.)
        self.assertEqual(after['underlyings']['ABC']['risk_age_at_publication_seconds'],0.)
        self.assertEqual(after['incremental']['repriced_underlyings'],['ABC'])
        self.assertEqual(after['hypercube']['entity_count'],2)
        expected=sum(r['position_units']*r['risk']['delta']*r['spot'] for r in after['rows'])
        self.assertAlmostEqual(after['portfolio']['delta_dollars'],expected)
        cells={(v['node'],v['key']):v['value'] for v in after['hypercube']['values']}
        self.assertEqual(cells['value','ABC-C'],-200*b['ABC']['risk']['valuation']['price'])
        self.assertEqual(len(self.engine.iv_caches['ABC']),1)
    def test_failures_preserve_state_and_generation_can_retry(self):
        self.engine.submit(self.event('ABC',1));before=copy.deepcopy(self.engine.snapshot)
        bad=self.spot(2);bad['spot']=-1
        with self.assertRaises(ValueError):self.engine.submit(bad)
        self.assertEqual(self.engine.snapshot,before)
        duplicate=self.event('XYZ',2);duplicate['frame']['quotes'][0]['option']['contract_id']='ABC-C'
        with self.assertRaisesRegex(ValueError,'another underlying'):self.engine.submit(duplicate)
        self.assertEqual(self.engine.snapshot,before)
        good=self.spot(2);self.assertEqual(self.engine.submit(good)['generation'],2)
        with self.assertRaisesRegex(ValueError,'stale'):self.engine.submit(good)
    def test_native_failure_rolls_back_calibration_and_retries(self):
        self.engine.submit(self.event('ABC',1));before=copy.deepcopy(self.engine.snapshot)
        original=self.native.publish
        def fail(_):raise ValueError('injected publication failure')
        self.native.publish=fail
        with self.assertRaisesRegex(ValueError,'publication failure'):self.engine.submit(self.spot(2))
        self.assertEqual(self.engine.snapshot,before)
        self.native.publish=original
        self.assertEqual(self.engine.submit(self.spot(2))['generation'],2)
    def test_source_currency_and_contract_definition_conflicts_rejected(self):
        self.engine.submit(self.event('ABC',1));before=copy.deepcopy(self.engine.snapshot)
        bad=self.event('XYZ',2);bad['frame']['currency']='EUR'
        with self.assertRaisesRegex(ValueError,'currency'):self.engine.submit(bad)
        bad=self.event('ABC',2);bad['frame']['quotes'][0]['option']['strike']=101.
        with self.assertRaisesRegex(ValueError,'definition'):self.engine.submit(bad)
        self.assertEqual(self.engine.snapshot,before)
    def test_dividend_event_requires_new_complete_calibration_and_removal_works(self):
        event=self.event('ABC',1);event['frame']['quotes'][0]['option']['cash_dividends']=[{'ex_date':self.now+30_000_000_000,'amount':.1}]
        self.engine.submit(event)
        with self.assertRaisesRegex(ValueError,'dividend crossing'):self.engine.submit(self.spot(2))
        stale=self.event('ABC',2,self.now+60_000_000_000)
        stale['frame']['quotes'][0]['calibration_ns']=str(self.now)
        with self.assertRaisesRegex(ValueError,'fresh quote'):self.engine.submit(stale)
        self.engine.submit(self.event('XYZ',2))
        out=self.engine.submit({'schema':'volt.exercise_ingest.v1','action':'remove_underlying','generation':3,'as_of_ns':str(self.now),'underlying':'ABC'})
        self.assertEqual(out['hypercube']['entity_count'],1);self.assertEqual(set(out['underlyings']),{'XYZ'})

if __name__=='__main__':unittest.main()
