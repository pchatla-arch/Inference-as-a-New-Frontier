import math
import sys
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from improvement_accounting import (RoundDemand,token_demand,bytes_per_verified_improvement,
                                     weighted_movement_cost,quality_gain_per_joule)

class AccountingTests(unittest.TestCase):
    def test_example(self):
        x=token_demand([RoundDemand(32,2,512,64,256,128)]*3)
        self.assertEqual(x.logical_decode_tokens,61440)
        self.assertEqual(x.logical_prompt_tokens,147456)
    def test_score_only_judge(self):
        x=token_demand([RoundDemand(1,3,100,0,20,10)])
        self.assertEqual(x.logical_decode_tokens,100)
        self.assertEqual(x.logical_prompt_tokens,350)
    def test_empty_and_extras(self):
        x=token_demand([],extra_decode=4,extra_prompt=8)
        self.assertEqual((x.logical_decode_tokens,x.logical_prompt_tokens),(4,8))
    def test_invalid_inputs(self):
        for v in [-1,float('nan'),float('inf')]:
            with self.assertRaises(ValueError):RoundDemand(1,v,0,0,0,0)
        with self.assertRaises(TypeError):RoundDemand(True,1,0,0,0,0)
    def test_zero_verified_is_undefined(self):
        self.assertIsNone(bytes_per_verified_improvement({'HBM':1234},0)['HBM'])
        self.assertIsNone(weighted_movement_cost({'HBM':1234},{'HBM':1e-12},0))
    def test_bytes_not_weighted_cost(self):
        self.assertEqual(bytes_per_verified_improvement({'HBM':100,'network':20},2),
                         {'HBM':50,'network':10})
        self.assertAlmostEqual(weighted_movement_cost({'HBM':100},{'HBM':1e-12},2),5e-11)
    def test_missing_coefficient(self):
        with self.assertRaises(ValueError):weighted_movement_cost({'HBM':100},{},2)
    def test_negative_improvement_retained(self):
        energy={k:2 for k in ['generation','evaluation','tools','training','synchronization']}
        self.assertEqual(quality_gain_per_joule(-.1,energy),-.01)
    def test_zero_energy_rejected(self):
        energy={k:0 for k in ['generation','evaluation','tools','training','synchronization']}
        with self.assertRaises(ValueError):quality_gain_per_joule(0,energy)
    def test_capacity_example(self):
        kv=2*80*8*128*2
        self.assertEqual(kv,327680)
        self.assertEqual(kv*8192/2**30,2.5)
        self.assertEqual(kv*8192*64/2**30,160)
        self.assertAlmostEqual(2*70e9*64/(70e9*2+64*8192*kv),28.736490031837075,places=5)

if __name__=='__main__':unittest.main()
