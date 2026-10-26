"""Actual HTTP -> dirty-underlying Rust pricing -> persistent Hypercube round-trip."""
import json
import http.client
import os
import threading
from types import SimpleNamespace
import unittest
from urllib.error import HTTPError
from urllib.request import Request, urlopen
from exercise_incremental_service import make_server
import test_exercise_incremental as fixtures

@unittest.skipUnless(os.environ.get('VOLT_EXERCISE_PRICER') and os.environ.get('VOLT_EXERCISE_CUBE'), 'set both native binary paths')
class HttpTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.IncrementalTests()
        self.fixture.setUp()
        self.engine = self.fixture.engine
        self.token_env = 'VOLT_INCREMENTAL_TEST_TOKEN'
        self.old_token = os.environ.get(self.token_env)
        os.environ[self.token_env] = 'test-token'
        self.args = SimpleNamespace(host='127.0.0.1', port=0, token_env=self.token_env, allow_origin=[])
        self.server = make_server(self.args, self.engine)
        self.thread = threading.Thread(target=lambda:self.server.serve_forever(poll_interval=.02), daemon=True)
        self.thread.start()
        self.url = 'http://127.0.0.1:'+str(self.server.server_address[1])
    def tearDown(self):
        self.server.shutdown();self.server.server_close();self.thread.join(timeout=5)
        self.fixture.tearDown()
        if self.old_token is None:os.environ.pop(self.token_env,None)
        else:os.environ[self.token_env]=self.old_token
    def request(self, path, data=None, headers=None):
        h={'Authorization':'Bearer test-token','Content-Type':'application/json',**(headers or {})}
        request=Request(self.url+path, data=None if data is None else json.dumps(data).encode(), headers=h)
        try:
            response=urlopen(request, timeout=30)
        except HTTPError as error:
            response=error
        with response:
            body=response.read()
            try:body=json.loads(body)
            except ValueError:pass
            return response.status,body,dict(response.headers)
    def test_actual_two_underlying_http_roundtrip_and_rejected_update_retains_snapshot(self):
        for name,generation in [('ABC',1),('XYZ',2)]:
            status,snapshot,_=self.request('/api/options-risk/event',self.fixture.event(name,generation))
            self.assertEqual(status,200);self.assertEqual(snapshot['hypercube']['generation'],generation)
        status,after,_=self.request('/api/options-risk/event',self.fixture.spot())
        self.assertEqual(status,200);self.assertEqual(after['underlyings']['XYZ']['risk_age_at_publication_seconds'],60.)
        self.assertEqual(after['underlyings']['ABC']['risk_age_at_publication_seconds'],0.)
        self.assertEqual(after['hypercube']['entity_count'],2)
        self.assertTrue(all(c['calibration_stock_lineage_complete'] for c in after['underlyings'].values()))
        bad=self.fixture.spot(4);bad['spot_ns']=str(self.fixture.now+120_000_000_000)
        self.assertEqual(self.request('/api/options-risk/event',bad)[0],400)
        self.assertEqual(self.request('/api/options-risk/snapshot')[1],after)
        self.assertEqual(self.request('/api/options-risk/event',self.fixture.spot())[0],409)
    def test_auth_origin_request_cap_and_nonlocal_guard(self):
        self.assertEqual(self.request('/api/options-risk/snapshot',headers={'Authorization':''})[0],401)
        self.assertEqual(self.request('/api/options-risk/snapshot',headers={'Origin':'https://external.example'})[0],403)
        status,_,headers=self.request('/api/options-risk/snapshot',headers={'Origin':'http://localhost:3000'})
        self.assertEqual(status,200);self.assertEqual(headers['Access-Control-Allow-Origin'],'http://localhost:3000')
        connection = http.client.HTTPConnection('127.0.0.1', self.server.server_address[1], timeout=5)
        connection.putrequest('POST', '/api/options-risk/event')
        for key, value in [('Authorization','Bearer test-token'), ('Content-Type','application/json'), ('Content-Length',str(2*1024*1024+1))]:
            connection.putheader(key, value)
        connection.endheaders()
        self.assertEqual(connection.getresponse().status,413)
        connection.close()
        os.environ.pop(self.token_env)
        args=SimpleNamespace(host='0.0.0.0',port=0,token_env=self.token_env,allow_origin=[])
        with self.assertRaisesRegex(ValueError,'nonlocal'):make_server(args,self.engine)
    def test_get_retains_previous_while_busy_post_rejects_overlap(self):
        self.assertEqual(self.request('/api/options-risk/event',self.fixture.event('ABC',1))[0],200)
        previous=self.request('/api/options-risk/snapshot')[1]
        entered,release=threading.Event(),threading.Event()
        original=self.engine.submit
        def slow(event):
            entered.set()
            if not release.wait(5):raise TimeoutError('test synchronization failed')
            return original(event)
        self.engine.submit=slow
        response=[]
        worker=threading.Thread(target=lambda:response.append(self.request('/api/options-risk/event',self.fixture.spot(2))))
        worker.start()
        try:
            self.assertTrue(entered.wait(5))
            self.assertEqual(self.request('/api/options-risk/snapshot')[1],previous)
            self.assertEqual(self.request('/api/options-risk/event',self.fixture.spot(3))[0],429)
        finally:
            release.set();worker.join(timeout=30)
        self.assertEqual(response[0][0],200)
        self.assertEqual(self.request('/api/options-risk/snapshot')[1]['generation'],2)

if __name__=='__main__':unittest.main()
